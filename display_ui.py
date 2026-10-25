"""
Status screen for the voice assistant on an ILI9488 (480x320) display.

Shows: state banner (loading / idle / listening / thinking / speaking / error),
what you said, and the assistant's reply as it streams in.

Everything is a safe no-op if the display can't be initialised, so the
assistant keeps working without it.

Env vars (all optional):
  LCD=0              disable the display entirely
  LCD_ROTATION=1     0-3 (1 = 480x320 landscape)
  LCD_INVERT=1       if colours look negative
  LCD_BGR=0          if red and blue are swapped
  LCD_SPEED=32000000 SPI clock in Hz (lower it if you see noise)
"""
import os
import threading
import time

from PIL import Image, ImageDraw, ImageFont

W, H = 480, 320
BG = (12, 14, 24)

STATES = {            # state: (banner colour, banner text)
    "loading":   ((90, 90, 100), "LOADING..."),
    "idle":      ((35, 90, 170), None),            # text filled in from wake word
    "listening": ((30, 150, 70), "LISTENING"),
    "thinking":  ((200, 130, 20), "THINKING"),
    "speaking":  ((120, 60, 190), "SPEAKING"),
    "error":     ((180, 40, 40), "ERROR"),
}


def _font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except OSError:
        try:
            return ImageFont.load_default(size)
        except TypeError:
            return ImageFont.load_default()


def _wrap(draw, text, font, max_w):
    lines = []
    for para in text.split("\n"):
        cur = ""
        for word in para.split():
            test = (cur + " " + word).strip()
            if draw.textlength(test, font=font) <= max_w:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = word
        lines.append(cur)
    return lines


class VoiceUI:
    def __init__(self):
        self.lcd = None
        self.state, self.user, self.reply = "loading", "", ""
        self._tick = 0
        self._lock = threading.Lock()
        self._dirty = threading.Event()

        wake = os.getenv("WAKE_WORD", "hey_jarvis")
        self._idle_text = "PRESS ENTER" if wake.lower() == "none" else f"SAY \"{wake.replace('_', ' ').upper()}\""

        if os.getenv("LCD", "1") == "0":
            return
        try:
            from ili9488 import ILI9488
            self.lcd = ILI9488(
                rotation=int(os.getenv("LCD_ROTATION", "1")),
                invert=os.getenv("LCD_INVERT") == "1",
                bgr=os.getenv("LCD_BGR", "1") == "1",
                speed_hz=int(os.getenv("LCD_SPEED", "32000000")),
            )
        except Exception as e:  # noqa: BLE001
            print(f"[lcd] disabled: {e}")
            return

        self.f_banner = _font(26, bold=True)
        self.f_label = _font(15, bold=True)
        self.f_text = _font(22)
        self.f_hint = _font(20)
        threading.Thread(target=self._loop, daemon=True).start()
        self._dirty.set()

    # ------------------------------------------------------------------ updates
    def set_state(self, state):
        if self.lcd is None:
            return
        with self._lock:
            self.state = state
        self._dirty.set()

    def new_turn(self):
        if self.lcd is None:
            return
        with self._lock:
            self.user, self.reply = "", ""
        self._dirty.set()

    def set_user(self, text):
        if self.lcd is None:
            return
        with self._lock:
            self.user = text
        self._dirty.set()

    def append_reply(self, text):
        if self.lcd is None:
            return
        with self._lock:
            self.reply = (self.reply + " " + text.strip()).strip()
        self._dirty.set()

    # ------------------------------------------------------------------ drawing
    def _loop(self):
        while True:
            self._dirty.wait(timeout=0.45)
            animate = self.state == "thinking"
            if not self._dirty.is_set() and not animate:
                continue
            self._dirty.clear()
            try:
                self._draw()
            except Exception as e:  # noqa: BLE001
                print("[lcd] draw error:", e)
                time.sleep(1)
            time.sleep(0.05)

    def _draw(self):
        with self._lock:
            state, user, reply = self.state, self.user, self.reply
        self._tick += 1

        color, label = STATES[state]
        if label is None:
            label = self._idle_text
        if state == "thinking":
            label += "." * (self._tick % 4)

        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)

        # banner
        d.rectangle([0, 0, W, 46], fill=color)
        d.text((14, 8), label, font=self.f_banner, fill="white")

        y, pad = 58, 14
        max_w = W - 2 * pad
        lh = self.f_text.getbbox("Ag")[3] + 5

        if not user and not reply:
            hint = "Ask me anything" if state in ("idle", "listening") else ""
            if hint:
                d.text((pad, H // 2), hint, font=self.f_hint, fill=(110, 115, 135))
        else:
            if user:
                d.text((pad, y), "YOU", font=self.f_label, fill=(130, 135, 155))
                y += 20
                lines = _wrap(d, user, self.f_text, max_w)[:3]
                for ln in lines:
                    d.text((pad, y), ln, font=self.f_text, fill=(255, 220, 120))
                    y += lh
                y += 8
            if reply:
                d.text((pad, y), "ASSISTANT", font=self.f_label, fill=(130, 135, 155))
                y += 20
                room = max(1, (H - y - 6) // lh)
                lines = _wrap(d, reply, self.f_text, max_w)[-room:]   # keep newest lines
                for ln in lines:
                    d.text((pad, y), ln, font=self.f_text, fill="white")
                    y += lh

        self.lcd.display(img)
