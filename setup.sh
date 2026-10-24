#!/usr/bin/env bash
# One-shot setup for the offline voice assistant on Raspberry Pi 5
# (Raspberry Pi OS Bookworm 64-bit, 8 GB recommended)
set -euo pipefail
cd "$(dirname "$0")"

LLM_MODEL="${LLM_MODEL:-qwen2.5:1.5b}"   # try llama3.2:3b for smarter, slower replies
VOICE="en_US-lessac-medium"
VOICE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"

echo "==> System packages"
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential libportaudio2 \
    portaudio19-dev alsa-utils curl

echo "==> Ollama (local LLM server)"
if ! command -v ollama >/dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
ollama pull "$LLM_MODEL"

echo "==> Python environment"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install faster-whisper sounddevice numpy requests webrtcvad-wheels piper-tts
# openWakeWord: install without deps to avoid tflite-runtime (no aarch64 wheel); use ONNX instead
pip install onnxruntime scipy scikit-learn tqdm
pip install --no-deps openwakeword
python - <<'EOF'
import openwakeword
openwakeword.utils.download_models()
EOF

echo "==> Piper voice"
mkdir -p voices
for ext in onnx onnx.json; do
  [ -f "voices/$VOICE.$ext" ] || curl -L -o "voices/$VOICE.$ext" "$VOICE_URL/$VOICE.$ext"
done

echo "==> Pre-download Whisper model"
python - <<'EOF'
from faster_whisper import WhisperModel
WhisperModel("base.en", device="cpu", compute_type="int8")
EOF

echo
echo "Done. Find your audio devices with:  .venv/bin/python assistant.py --list-devices"
echo "Run with:                           .venv/bin/python assistant.py"
