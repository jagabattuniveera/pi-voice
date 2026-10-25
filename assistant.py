#!/usr/bin/env python3
"""
Offline voice assistant for Raspberry Pi 5.

  mic -> wake word (openWakeWord) -> VAD recording (webrtcvad)
      -> STT (faster-whisper) -> LLM (Ollama, streamed)
      -> TTS (Piper, sentence by sentence) -> speaker

Env vars (all optional):
  AUDIO_IN / AUDIO_OUT   sounddevice device index or name substring
  LLM_MODEL              default qwen2.5:1.5b
  WHISPER_MODEL          default base.en  (tiny.en = faster, small.en = more accurate)
  WAKE_WORD              default hey_jarvis   (set to "none" for push-to-talk via Enter)
  WAKE_THRESHOLD         default 0.5
"""
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

BASE = Path(__file__).parent
SR_IN = 16000
BLOCK = 1280                      # 80 ms @ 16 kHz (openWakeWord frame size)
END_SILENCE_S = 0.9               # silence that ends an utterance
MAX_UTTERANCE_S = 20
NO_SPEECH_TIMEOUT_S = 6

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")
WAKE_WORD = os.getenv("WAKE_WORD", "hey_jarvis")
WAKE_THRESHOLD = float(os.getenv("WAKE_THRESHOLD", "0.7"))
VOICE = BASE / "voices" / os.getenv("PIPER_VOICE", "en_US-lessac-medium.onnx")

SYSTEM_PROMPT = (
    "You are a helpful voice assistant running locally on a Raspberry Pi. "
    "Reply in short, natural spoken sentences. No markdown, no lists, no emojis. "
    "Keep answers under three sentences unless asked for more."
)


def _dev(name):
    v = os.getenv(name)
    if v is None:
        return None
    return int(v) if v.isdigit() else v


IN_DEV, OUT_DEV = _dev("AUDIO_IN"), _dev("AUDIO_OUT")

# Optional ILI9488 status screen (no-op if the display/libraries aren't available)
from display_ui import VoiceUI  # noqa: E402

UI = VoiceUI()


# --------------------------------------------------------------------------- mic
class Mic:
    """Opens the mic at any rate/channel count it supports; delivers 16 kHz mono blocks."""

    def __init__(self):
        from math import gcd
        self._gcd = gcd
        self.noise = None
        self.q = queue.Queue()
        info = sd.query_devices(IN_DEV, "input")
        default_sr = int(info["default_samplerate"])
        rates = [SR_IN]
        for r in (48000, 44100, 32000, default_sr):
            if r not in rates:
                rates.append(r)
        chans = [1, 2] if info["max_input_channels"] >= 2 else [1]
        err = None
        for rate in rates:
            for ch in chans:
                try:
                    self._open(rate, ch)
                    print(f"[mic] opened at {rate} Hz, {ch} ch")
                    return
                except sd.PortAudioError as e:
                    err = e
        raise err

    def _open(self, rate, ch):
        g = self._gcd(SR_IN, rate)
        self.up, self.down = SR_IN // g, rate // g
        self.rate = rate
        self.stream = sd.InputStream(
            samplerate=rate, channels=ch, dtype="int16",
            blocksize=int(rate * BLOCK / SR_IN), device=IN_DEV, callback=self._cb,
        )
        self.stream.start()

    def _cb(self, indata, frames, t, status):
        x = indata[:, 0]
        if self.rate != SR_IN:
            from scipy.signal import resample_poly
            x = resample_poly(x.astype(np.float32), self.up, self.down)
            x = np.clip(x, -32768, 32767)
        x = x.astype(np.int16)
        if len(x) < BLOCK:
            x = np.pad(x, (0, BLOCK - len(x)))
        x = x[:BLOCK].copy()
        # slow-rising minimum tracker = estimate of the room's background noise level
        rms = float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))
        if self.noise is None or rms < self.noise:
            self.noise = rms if self.noise is None else 0.3 * rms + 0.7 * self.noise
        else:
            self.noise = self.noise * 0.995 + rms * 0.005
        self.q.put(x)

    def read(self):
        return self.q.get()

    def flush(self):
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                return


def play_audio(audio, sr):
    """Play int16 mono audio, resampling to the output device's native rate/channels."""
    from math import gcd
    from scipy.signal import resample_poly

    info = sd.query_devices(OUT_DEV, "output")
    default_sr = int(info["default_samplerate"])
    rates = [default_sr] + [r for r in (48000, 44100, 32000, 16000, 22050) if r != default_sr]
    chans = [2, 1] if info["max_output_channels"] >= 2 else [1]
    last_err = None
    for rate in rates:
        data = audio
        if sr != rate:
            g = gcd(sr, rate)
            data = resample_poly(audio.astype(np.float32), rate // g, sr // g)
            data = np.clip(data, -32768, 32767).astype(np.int16)
        for ch in chans:
            out = np.column_stack([data] * ch) if ch > 1 else data
            try:
                sd.play(out, rate, device=OUT_DEV)
                sd.wait()
                return
            except sd.PortAudioError as e:
                last_err = e
    raise last_err


BEEP = os.getenv("BEEP", "0") == "1"


def beep(freq=880, dur=0.12):
    if not BEEP:
        return
    t = np.linspace(0, dur, int(22050 * dur), False)
    tone = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    try:
        play_audio(tone, 22050)
    except Exception as e:  # noqa: BLE001
        print("[beep] error:", e)


# ----------------------------------------------------------------------- wake word
class Waker:
    def __init__(self):
        self.model = None
        if WAKE_WORD.lower() == "none":
            return
        try:
            from openwakeword.model import Model
            self.model = Model(wakeword_models=[WAKE_WORD], inference_framework="onnx")
        except Exception as e:  # noqa: BLE001
            print(f"[wake] openWakeWord unavailable ({e}); using push-to-talk (Enter).")

    def wait(self, mic):
        UI.set_state("idle")
        if self.model is None:
            input("\nPress Enter to talk... ")
            mic.flush()
            return
        mic.flush()
        print(f"\nListening for wake word '{WAKE_WORD}'...")
        try:
            self.model.reset()
        except Exception:  # noqa: BLE001
            pass
        debug = os.getenv("WAKE_DEBUG") == "1"
        n = 0
        while True:
            blk = mic.read()
            scores = self.model.predict(blk)
            best = max(scores.values())
            if debug:
                n += 1
                if n % 6 == 0:
                    rms = float(np.sqrt(np.mean(blk.astype(np.float32) ** 2)))
                    print(f"  [debug] mic rms={rms:6.0f}  wake score={best:.2f}", end="\r")
            if best >= WAKE_THRESHOLD:
                return


# ----------------------------------------------------------------------- record
def save_wav(pcm, path="/tmp/last_utterance.wav"):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR_IN)
        w.writeframes(pcm.tobytes())


def record_utterance(mic):
    pcm = _record(mic)
    if pcm is not None:
        print(f"[rec] {len(pcm) / SR_IN:.1f}s, noise floor ~{mic.noise:.0f}")
        if os.getenv("WAKE_DEBUG") == "1":
            save_wav(pcm)
    return pcm


def _record(mic):
    vad = webrtcvad.Vad(3)
    pre = deque(maxlen=4)
    frames, started, silent, voiced_n = [], False, 0, 0
    t0 = time.time()
    blk_s = BLOCK / SR_IN
    while True:
        blk = mic.read()
        rms = float(np.sqrt(np.mean(blk.astype(np.float32) ** 2)))
        loud_enough = rms > max(mic.noise * 2.5, 120)
        voiced = loud_enough and sum(
            vad.is_speech(blk[i:i + 320].tobytes(), SR_IN) for i in range(0, BLOCK, 320)
        ) >= 2
        if not started:
            pre.append(blk)
            if voiced:
                started = True
                frames = list(pre)
            elif time.time() - t0 > NO_SPEECH_TIMEOUT_S:
                return None
        else:
            frames.append(blk)
            voiced_n += int(voiced)
            silent = 0 if voiced else silent + 1
            if silent * blk_s >= END_SILENCE_S or len(frames) * blk_s > MAX_UTTERANCE_S:
                break
    if voiced_n * blk_s < 0.5:
        print("[rec] too short, ignoring")
        return None
    return np.concatenate(frames)


# -------------------------------------------------------------------------- STT
print("Loading Whisper...")
stt = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=4)
JUNK = {"", "you", "thank you.", "thanks for watching!", "bye."}


def transcribe(pcm):
    audio = pcm.astype(np.float32) / 32768.0
    segs, _ = stt.transcribe(
        audio, language="en", beam_size=1, vad_filter=False,
        condition_on_previous_text=False,
    )
    text = " ".join(s.text.strip() for s in segs).strip()
    return "" if text.lower() in JUNK else text


# -------------------------------------------------------------------------- TTS
class Speaker:
    """text -> (Piper synth thread) -> audio -> (playback thread); overlaps the two."""

    def __init__(self):
        cfg = json.loads(Path(str(VOICE) + ".json").read_text())
        self.sr = cfg["audio"]["sample_rate"]
        self.text_q, self.audio_q = queue.Queue(), queue.Queue()
        threading.Thread(target=self._synth, daemon=True).start()
        threading.Thread(target=self._play, daemon=True).start()

    def say(self, text):
        text = re.sub(r"[*_#`>~|]", "", text)
        text = re.sub(r"[^\x00-\x7F]+", "", text).strip()   # strip emojis etc.
        if text:
            self.text_q.put(text)

    def _synth(self):
        while True:
            text = self.text_q.get()
            try:
                p = subprocess.run(
                    [sys.executable, "-m", "piper", "--model", str(VOICE), "--output-raw"],
                    input=text.encode(), capture_output=True, check=True,
                )
                self.audio_q.put(np.frombuffer(p.stdout, dtype=np.int16))
            except Exception as e:  # noqa: BLE001
                print("[tts] error:", e)
            finally:
                self.text_q.task_done()

    def _play(self):
        while True:
            audio = self.audio_q.get()
            try:
                play_audio(audio, self.sr)
            except Exception as e:  # noqa: BLE001
                print("[play] error:", e)
            finally:
                self.audio_q.task_done()

    def wait(self):
        self.text_q.join()
        self.audio_q.join()


# -------------------------------------------------------------------------- LLM
SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def chat(history, speaker):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": LLM_MODEL, "messages": msgs, "stream": True, "keep_alive": "30m",
            "options": {"num_ctx": 2048, "num_predict": 220, "temperature": 0.7},
        },
        stream=True, timeout=300,
    )
    r.raise_for_status()
    buf, full = "", ""
    for line in r.iter_lines():
        if not line:
            continue
        d = json.loads(line)
        tok = d.get("message", {}).get("content", "")
        buf += tok
        full += tok
        parts = SPLIT.split(buf)
        if len(parts) > 1:
            for s in parts[:-1]:
                print("  >", s)
                UI.set_state("speaking")
                UI.append_reply(s)
                speaker.say(s)
            buf = parts[-1]
    if buf.strip():
        print("  >", buf.strip())
        UI.set_state("speaking")
        UI.append_reply(buf)
        speaker.say(buf)
    return full.strip()


def warm_up():
    try:
        requests.post(f"{OLLAMA_URL}/api/generate",
                      json={"model": LLM_MODEL, "prompt": "hi", "stream": False,
                            "keep_alive": "30m", "options": {"num_predict": 1}}, timeout=300)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Cannot reach Ollama at {OLLAMA_URL}: {e}")


# --------------------------------------------------------------------------- main
def main():
    if "--list-devices" in sys.argv:
        print(sd.query_devices())
        return
    if not VOICE.exists():
        sys.exit(f"Missing Piper voice: {VOICE} (run setup.sh)")

    warm_up()
    mic, waker, speaker = Mic(), Waker(), Speaker()
    history = []
    speaker.say("Ready.")
    speaker.wait()

    while True:
        waker.wait(mic)
        UI.new_turn()
        UI.set_state("listening")
        beep()
        time.sleep(0.7 if BEEP else 0.3)   # let the wake phrase tail / Bluetooth beep die out
        mic.flush()
        pcm = record_utterance(mic)
        if pcm is None:
            continue

        UI.set_state("thinking")
        t = time.time()
        text = transcribe(pcm)
        print(f"You ({time.time() - t:.1f}s): {text}")
        if not text:
            continue
        UI.set_user(text)
        if re.search(r"\b(reset|forget) (the )?conversation\b", text.lower()):
            history.clear()
            speaker.say("Okay, starting fresh.")
            speaker.wait()
            continue

        history.append({"role": "user", "content": text})
        try:
            reply = chat(history, speaker)
        except Exception as e:  # noqa: BLE001
            print("[llm] error:", e)
            UI.set_state("error")
            speaker.say("Sorry, my language model had a problem.")
            history.pop()
            speaker.wait()
            continue
        history.append({"role": "assistant", "content": reply})
        history[:] = history[-12:]        # keep last 6 exchanges
        speaker.wait()                    # don't hear ourselves; resume after speech


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
