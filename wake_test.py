import numpy as np
import sounddevice as sd
from openwakeword.model import Model

SAMPLE_RATE = 16000
CHUNK = 1280
THRESHOLD = 0.5

model = Model(wakeword_models=["hey jarvis"], inference_framework="onnx")

print("Listening... say 'hey jarvis' (Ctrl + C to stop)")
try:
  with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                      dtype="int16", blocksize=CHUNK) as stream:
    while True:
      audio, _ = stream.read(CHUNK)
      scores = model.predict(audio[:, 0])
      for name, score in scores.items():
        if score > 0.1:
          print(f"{name}: {score:.2f}")
        if score >= THRESHOLD:
          print(">>> WAKE!")
except KeyboardInterrupt:
  print("Stopped.")