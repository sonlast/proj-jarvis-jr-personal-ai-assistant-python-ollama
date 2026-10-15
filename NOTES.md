# JARVIS Jr. Notes

## Done
- Repo + .venv set up (Python 3.13.3)
- faster-whisper, sounddevice, openWakeWord, ollama, piper-tts installed and working on 3.13
- openWakeWord "hey jarvis" model loaded manually (.onnx files in the venv's resources/models folder)
- wake_test.py: detects "hey jarvis" live (threshold 0.5)
- voice_test.py: wake -> record until silence -> faster-whisper (small, CPU) transcribes
- Ollama + qwen3:8b running 100% on GPU; think=False, keep_alive=30m
- brain_test.py: text chat with JARVIS persona works
- jarvis.py: full voice loop works (wake -> transcribe -> qwen3:8b -> Piper en_GB-alan-medium, length_scale 0.85)
- Latency about 1-2 seconds from end of speech to start of reply

## In progress
- Nothing half-done

## Next
- First real tool: current time and date (fixes wrong-weekday guesses)
- Then: 7AM briefing script (scheduled job)

## Notes / later
- Back up the 3 .onnx wake word model files outside .venv (lost if the venv is rebuilt)
- Check for self-waking (his voice triggering the wake word); use a headset or add a cooldown after he speaks
- Add a "stop" command to interrupt him
- Move Whisper to the GPU (needs CUDA libraries) or try the "base" model for lower latency
- Tune wake threshold (0.5 now, drop to ~0.4 if he misses you)
- Occasional "(nothing heard)" after the wake phrase; improve the wake-phrase retry
- System prompt: he must say plainly that he can't take actions yet
- Commit regularly; keep voices/ and *.wav gitignored
- Goal: the full "Good morning, sir" briefing script