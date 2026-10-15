import numpy as np
import sounddevice as sd
from openwakeword.model import Model
from faster_whisper import WhisperModel

SR = 16000
CHUNK = 1280
WAKE_THRESHOLD = 0.5
SILENCE_SECONDS = 1.0
MAX_SECONDS = 10

wake = Model(wakeword_models=["hey jarvis"], inference_framework="onnx")
stt = WhisperModel("small", device="cpu", compute_type="int8")

def rms(a):
  return float(np.sqrt(np.mean(a.astype(np.float32) ** 2)))

def calibrate(stream, secs=1.0):
  vals = [rms(stream.read(CHUNK)[0][:, 0]) for _ in range(int(secs * SR / CHUNK))]
  return max(np.mean(vals) * 3, 300)

def record_until_silence(stream, threshold):
  frames, silent, spoke = [], 0, False
  silence_chunks = int(SILENCE_SECONDS * SR / CHUNK)
  for _ in range(int(MAX_SECONDS * SR / CHUNK)):
    audio = stream.read(CHUNK)[0][:, 0]
    frames.append(audio)
    if rms(audio) > threshold:
      spoke, silent = True, 0
    elif spoke:
      silent += 1
    if spoke and silent >= silence_chunks:
      break
  return np.concatenate(frames)

def transcribe(audio):
  f = audio.astype(np.float32) / 32768.0
  segments, _ = stt.transcribe(f, language="en", vad_filter=True)
  return " ".join(s.text.strip() for s in segments)

with sd.InputStream(samplerate=SR, channels=1, dtype="int16", blocksize=CHUNK) as stream:
  print("Calibrating background noise... stay quiet for a second.")
  threshold = calibrate(stream)
  print(f"Ready. Say 'hey jarvis', then you sentence. (Ctrl + C to stop )")
  try:
    while True:
      audio = stream.read(CHUNK)[0][:, 0]
      score = wake.predict(audio)["hey jarvis"]
      if score >= WAKE_THRESHOLD:
        print(">>>> Wake! Listening...")
        clip = record_until_silence(stream, threshold)
        print("You said:", transcribe(clip) or "(nothing heard)")
        wake.reset()
        print("Back to waiting...")
  except KeyboardInterrupt:
    print("Stopped.")