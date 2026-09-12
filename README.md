# whisper-transcriber

A one-click local transcription tool for Windows. Drop the folder anywhere, double-click, paste an audio or video path — you get plain text back. No installer, no commands to type, no Python setup, and no AI background needed: everything from building the environment to choosing which hardware runs the model is handled for you.

**The project here is the tool, not the model.** The listening is done by Whisper large-v3, an OpenAI model I neither trained nor published (see [Credits](#credits--models)). Everything around the model — provisioning, runtime selection, downloads, fallbacks, translation, the interface — is what this repository is about. I built it for my own files first, and kept pushing until it would survive landing on a Windows machine it had never seen: any GPU or none, a fresh system with nothing installed. That turned out to be the interesting part.

## What it does

- **One-click, zero commands.** Double-click `whisper_transcribe.bat`. On first run it finds or installs Python, builds its own environment, picks a runtime, and fetches what it needs. After that it just opens and asks for a file path. No command line, nothing to configure, no AI knowledge required — if you can double-click a file, you can use this.
- **Aggregates four runtimes, and picks the right one.** NVIDIA CUDA → Intel NPU → Intel iGPU → plain CPU. It probes what the machine actually has, recommends the fastest available, and lets you switch at any time (`/s`) — no need to know what a "runtime" even is. That's the point: one program that behaves correctly on whatever Windows machine it lands on.
- **Downloads that survive bad networks.** Models, dependencies and even precompiled caches are fetched on demand, with mirror pre-flights: a stalled connection gets detected and abandoned instead of waited on, and fallbacks cover networks where huggingface.co or PyPI is slow or unreachable.
- **Self-healing.** A broken dependency, missing GPU libraries, a driver update that knocks out the NPU — the tool repairs itself (`/fix`), retries, or falls back to the next runtime instead of dying with a stack trace.
- **Speaks the system's language.** The interface follows the Windows display language (Chinese or English), and it can never end up half-translated: a missing string falls back to the source text rather than crashing.
- **Optimized where it counts.** Streaming output on the CTranslate2 paths; VAD-based silence trimming; decode safeguards against Whisper's short-phrase repetition loops; a one-time NPU precompile that turns ~5.5-minute first loads into ~6-second ones afterwards.
- **Scriptable if you want it to be.** `whisper_transcribe.bat "a.mp3" "b.m4a"` transcribes and exits — handy for batch jobs and automation, though you'll never have to type a command. The model loads once per session and is reused across files, so a batch pays the load cost only once.

## Quick start

1. Get `whisper_transcribe.bat` + `whisper_transcribe.py` (that's the whole tool; the .bat is a thin launcher and all logic sits in the .py).
2. Double-click the .bat. First run provisions everything and walks you through a one-time runtime choice. Model weights download on first use (1.6–2.9 GB depending on runtime).
3. Paste a file path at the prompt. Done — the transcript is written as `<name>.txt` next to the source file.

At the prompt: `/s` switches runtime, `/fix` repairs dependencies, `/x` exits.

## Step by step

The quick start above skips a few details that are useful the first time. Here is the whole flow, in order:

1. **Get the files.** On this page: green **Code** button → **Download ZIP**, then unzip it (right-click the ZIP → *Extract All…*) and put the folder wherever you like — the Desktop is fine. Nothing needs to be installed beforehand.
2. **Start it.** Open the folder and double-click `whisper_transcribe.bat` — the only file you need to touch. Windows may stop it first, because the file came from the internet. Three ways past that, in the order they are worth trying:
   - Open a command prompt in the folder: type `cmd` in the Explorer address bar (the bar showing the folder path), press Enter, then type `whisper_transcribe.bat` and press Enter. Starting it from a prompt does not go through Windows' download check, and it runs with your normal permissions.
   - Double-click it and, if a warning about an unrecognized app appears, choose **More info → Run anyway**.
   - If the warning offers no way forward (newer Windows 11 builds with Smart App Control enabled), right-click `whisper_transcribe.bat` and choose **Run as administrator**.
3. **Let it set itself up (first time only).** A black window opens and explains what it is doing. Depending on your PC it may ask one or two things:
   - *Install Python automatically?* — if it asks, press Enter (yes). That's the background program the tool runs on, and it is a normal user-level install.
   - *Which "runtime" to use?* — this only decides what hardware does the listening. If you have no preference, press Enter to take the option tagged `[Recommended]`.
   Then it downloads the speech model (2–3 GB) with a progress bar — the slow part, and a one-time one. Later starts take seconds.
4. **Give it a file.** When you see the `>` prompt: drag an audio or video file from File Explorer into the window — the path fills in by itself — and press Enter. (Or right-click the file → *Copy as path*, and paste it; both work.) Common formats work as they are: mp3, m4a, wav, mp4, mkv.
5. **Take the text.** The transcript is saved next to your original file, with the same name and a `.txt` extension, and the window tells you the exact path. The text also appears on screen as it works.
6. **Another file, or done.** Paste another path to keep going; type `/x` and press Enter to finish (closing the window works too). Two more commands exist for later, but you won't need them: `/s` switches runtime, `/fix` repairs the setup if anything ever breaks.

## Requirements

- Windows 10/11
- Disk: ~3 GB for one runtime (models + environment); up to ~10 GB if you enable every runtime and keep the NPU compile cache
- Optional hardware: NVIDIA GPU, Intel NPU, Intel iGPU, or none of the above. The tool uses whatever is present; missing drivers are reported, never required.

## Engineering notes

The parts that took the real work — all of it on the tool side:

**Self-provisioning, and staying portable.** The whole point is that the folder travels. Python is detected by actually executing each candidate (`py -3.14/3.13/3.12/3.11`, then `python`, then common install dirs), not by trusting PATH. A virtualenv copied from another machine is "adopted" by rewriting its `pyvenv.cfg` home path rather than rebuilt. And the .bat must stay pure ASCII: `chcp 65001` plus non-ASCII comments makes cmd re-read the file at misaligned byte offsets, which corrupts parsing in ways that took a while to believe (a `->` inside a comment was read as a redirect and created a stray file).

**Loading CUDA on Windows is a DLL path problem.** CTranslate2 4.8.x resolves its `cublas64_12.dll` / `cudnn64_9.dll` dependencies from the extension module's own directory — PATH and `add_dll_directory` don't help (Python 3.8+ uses `LOAD_LIBRARY_SEARCH` semantics), and a CUDA 13 toolkit can't substitute. The tool pip-installs the `nvidia-*-cu12` wheels into the venv and relocates the DLLs next to `ctranslate2`.

**A download that never finishes.** On some networks the PyPI CDN accepts connections and then stalls — no error, no exit code, so "retry on failure" logic never fires. So the installer pre-flights mirrors instead: a hard 12-second cap, plus a "stalled if it stays under 5 KB/s for 5 seconds" rule, then picks the fastest one that actually moves bytes.

**Whisper loops on short phrases — handled in the decoder.** On short audio, large-v3 can emit "They're not." five times in a row; the cause is cross-segment conditioning (`condition_on_previous_text`). The tool disables it and adds a `no_repeat_ngram_size=4` backstop. It reproduced on both CUDA fp16 and CPU int8, so it's a decoding artifact, not a quantization one — and the fix also removed a stretch of garbled text the loop had displaced (the same run got slightly faster: 18.3s → 15.4s).

**NPU first-run compilation.** OpenVINO compiles the model ahead of time for the NPU: first compile measured 5.5 minutes; afterwards a cached blob loads in ~6.3 seconds. The cache is bound to the hardware generation — on a different generation OpenVINO rebuilds once by itself.

**Translation that can't break the app.** UI strings are keyed in Chinese with an English lookup table built on top; a missing entry falls back to the source string, so a gap in the translation can never crash a run — worst case, one line shows up in Chinese.

## Performance

Measured on a laptop (Core Ultra 7 + RTX 5070M), 7.6-second clip:

| Runtime | Model load | RTF |
|---|---|---|
| CUDA (CT2 fp16) | ~6.5 s (first run adds kernel warmup) | 0.16 |
| Intel NPU (cached blob) | ~6.3 s | 0.16 |
| Intel iGPU (OpenVINO int8) | ~2.5 s once cached | 0.29 |
| CPU (CT2 int8) | ~10.5 s | 0.91 |

RTF = transcription time ÷ audio duration; lower is better.

## Limitations

- Windows only.
- The OpenVINO path returns the transcript in one block when done; live streaming is CTranslate2-only.
- NPU enumeration occasionally fails after a driver update until reboot — the menu will fall back to the next runtime.
- The Intel NPU path is the least battle-tested of the four.

## Credits & models

- **Whisper large-v3** — the speech recognition model — is by OpenAI (MIT license). This tool is an independent project and is not affiliated with or endorsed by OpenAI.
- Weights are downloaded on first use from public repositories: [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) (CTranslate2 format, used by the CUDA/CPU paths) and [OpenVINO/whisper-large-v3-int8-ov](https://huggingface.co/OpenVINO/whisper-large-v3-int8-ov) (OpenVINO int8, for NPU/iGPU). Nothing model-related is bundled in this repository.

## License

MIT — see [LICENSE](LICENSE).
