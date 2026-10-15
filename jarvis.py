import numpy as np
import sounddevice as sd
import ollama
from openwakeword.model import Model
from faster_whisper import WhisperModel
from piper import PiperVoice, SynthesisConfig

SR = 16000
CHUNK = 1280
WAKE_THRESHOLD = 0.5
SILENCE_SECONDS = 1.0
MAX_SECONDS = 10
MODEL = "qwen3:8b"
SYSTEM = (
  "You are JARVIS, a calm, polite, dry-witted personal assistant. "
  "Address the user as 'sir'. Keep replies short (1 to 3 sentences). "
  "Never use emojis, markdown, or lists, because your replies are spoken aloud."
  "You cannot take actions or access accounts yet. If asked, say so plainly."
)

wake = Model(wakeword_models=["hey jarvis"], inference_framework="onnx")
stt = WhisperModel("small", device="cpu", compute_type="int8")
voice = PiperVoice.load("voices/en_GB-alan-medium.onnx")
syn_config = SynthesisConfig(length_scale=0.85)
history = [{"role": "system", "content": SYSTEM}]

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

def is_just_wake_phrase(text):
  cleaned = "".join(c for c in text.lower() if c.isalpha())
  return cleaned in ("heyjarvis", "jarvis", "")

def ask_brain(text):
  history.append({"role": "user", "content": text})
  reply = ollama.chat(model=MODEL, messages=history, 
                      think=False, keep_alive="30m")
  answer = reply["message"]["content"].strip()
  history.append({"role": "assistant", "content": answer})
  return answer

def speak(text):
  for chunk in voice.synthesize(text, syn_config=syn_config):
    audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
    sd.play(audio, chunk.sample_rate)
    sd.wait()

with sd.InputStream(samplerate=SR, channels=1, dtype="int16", blocksize=CHUNK) as stream:
  print("Calibrating background noise... stay quiet for a second.")
  threshold = calibrate(stream)
  print("Ready. Say 'hey jarvis', then you request. (Ctrl + C to stop)")
  try:
    while True:
      audio = stream.read(CHUNK)[0][:, 0]
      score = max(wake.predict(audio).values())
      if score >= WAKE_THRESHOLD:
        print(">>> Wake! Listening...")
        text = ""
        for _ in range(2):
          text = transcribe(record_until_silence(stream, threshold))
          if not is_just_wake_phrase(text):
            break
        if is_just_wake_phrase(text):
          print("(nothing heard)")
        else:
          print("You:", text)
          # print("JARVIS: ", ask_brain(text))
          answer = ask_brain(text)
          print("JARVIS:", answer)
          speak(answer)
          stream.read(stream.read_available)
        wake.reset()
        print("Back to waiting...")
  except KeyboardInterrupt:
    print("Stopped")