# whisper-transcriber

A one-click local transcription tool for Windows. Drop the folder anywhere, double-click, drop in an audio or video file — you get plain text back. No installer, no commands to type, no Python setup, and no AI background needed: everything from building the environment to choosing which hardware runs the model is handled for you. The window is the default face of the tool; a console mode is one flag away for anyone who prefers it.

**The project here is the tool, not the model.** The listening is done by Whisper large-v3, an OpenAI model I neither trained nor published (see [Credits](#credits--models)). Everything around the model — provisioning, runtime selection, downloads, fallbacks, translation, the window, the interface — is what this repository is about. I built it for my own files first, and kept pushing until it would survive landing on a Windows machine it had never seen: any GPU or none, a fresh system with nothing installed. That turned out to be the interesting part.

## What it does

- **One-click, zero commands.** Double-click `whisper_transcribe.bat`. On first run it finds or installs Python, builds its own environment, picks a runtime, and fetches what it needs. After that it opens its window, and files go in by drag-and-drop, by the Browse button, or by pasting a path. No command line, nothing to configure, no AI knowledge required — if you can drag a file onto a window, you can use this.
- **A console mode that stays out of the way.** `whisper_transcribe.bat --cli` (or `no-gui`) runs the original text interface. It is also the safety net: if the window cannot start — no WebView2 runtime, a locked-down machine — the tool falls back to the console instead of failing, and pressing any key while the window is loading cancels it the same way.
- **Aggregates four runtimes, and picks the right one.** NVIDIA CUDA → Intel NPU → Intel iGPU → plain CPU. It probes what the machine actually has, recommends the fastest available, and lets you switch at any time (`/s`) — no need to know what a "runtime" even is. That's the point: one program that behaves correctly on whatever Windows machine it lands on.
- **Downloads that survive bad networks.** Models, dependencies and even precompiled caches are fetched on demand, with mirror pre-flights: a stalled connection gets detected and abandoned instead of waited on, and fallbacks cover networks where huggingface.co or PyPI is slow or unreachable.
- **Self-healing, with a way down rather than a dead end.** A broken dependency, missing GPU libraries, a driver update that knocks out the NPU — the tool repairs itself (`/fix`), retries, or walks a fallback ladder: retry the NPU with its other compiler, then the iGPU (same model file, nothing to download), then the CPU — fetching that model first if it has never been used. It only reports failure after the ladder is exhausted.
- **Speaks the system's language.** The interface follows the Windows display language (Chinese or English), and it can never end up half-translated: a missing string falls back to the source text rather than crashing.
- **Optimized where it counts.** Streaming output on the CTranslate2 paths; VAD-based silence trimming; decode safeguards against Whisper's short-phrase repetition loops; a one-time NPU precompile that turns ~5.5-minute first loads into ~6-second ones afterwards.
- **Scriptable if you want it to be.** `whisper_transcribe.bat "a.mp3" "b.m4a"` opens with both files already queued (and in console mode transcribes them and exits) — handy for batch jobs and automation, though you'll never have to type a command. The model loads once per session and is reused across files, so a batch pays the load cost only once.

## Quick start

1. Get `whisper_transcribe.bat` + `whisper_transcribe.py` + `whisper_gui.py` (that's the whole tool; the .bat is a thin launcher, the .py holds the logic, and `whisper_gui.py` is the window layer).
2. Double-click the .bat. First run provisions everything and walks you through a one-time runtime choice. Model weights download on first use (1.6–2.9 GB depending on runtime).
3. Drag a file (or several) into the window, or paste a path at the console prompt. Done — the transcript is written as `<name>.txt` next to the source file.

At the prompt: `/s` switches runtime, `/fix` repairs dependencies, `/x` exits.

## Step by step

The quick start above skips a few details that are useful the first time. Here is the whole flow, in order:

1. **Get the files.** On this page: green **Code** button → **Download ZIP**, then unzip it (right-click the ZIP → *Extract All…*) and put the folder wherever you like — the Desktop is fine. Nothing needs to be installed beforehand.
2. **Start it.** Open the folder and double-click `whisper_transcribe.bat` — the only file you need to touch. Windows may stop it first, because the file came from the internet. Three ways past that, in the order they are worth trying:
   - Open a command prompt in the folder: type `cmd` in the Explorer address bar (the bar showing the folder path), press Enter, then type `whisper_transcribe.bat` and press Enter. Starting it from a prompt does not go through Windows' download check, and it runs with your normal permissions.
   - Double-click it and, if a warning about an unrecognized app appears, choose **More info → Run anyway**.
   - If the warning offers no way forward (newer Windows 11 builds with Smart App Control enabled), right-click `whisper_transcribe.bat` and choose **Run as administrator**.
3. **Let it set itself up (first time only).** A console window opens and explains what it is doing. Depending on your PC it may ask one or two things:
   - *Install Python automatically?* — if it asks, press Enter (yes). That's the background program the tool runs on, and it is a normal user-level install.
   - *Which "runtime" to use?* — this only decides what hardware does the listening. If you have no preference, press Enter to take the option tagged `[Recommended]`.
   Then it downloads the speech model (2–3 GB) with a progress bar — the slow part, and a one-time one. Later starts take seconds.
4. **Give it a file.** When the window opens, drag an audio or video file onto it — or click **Browse**, or paste a path into the box and press Enter. Several files at once are fine: they queue up and are transcribed one after another, with the model loaded only once. Common formats work as they are: mp3, m4a, wav, mp4, mkv.
5. **Take the text.** The transcript is saved next to your original file, with the same name and a `.txt` extension, and the window shows the exact path. The text also appears on screen as it works.
6. **Another file, or done.** Add more files at any time and close the window when you are finished. The console behind it is the off switch — closing either one stops everything, and nothing is left running.

**Prefer the console?** `whisper_transcribe.bat --cli` (or `no-gui`) starts the text interface directly. You can also back out at startup: while the window is loading, the console says so, and a keypress cancels the window and drops into console mode. Both modes share the same environment, models, cache and settings, so switching between them costs nothing.

## Folder layout

A folder that has been started at least once:

```
whisper-transcriber/
├── whisper_transcribe.bat     the launcher — double-click this one
├── whisper_transcribe.py      core logic: provisioning, runtimes, transcription
├── whisper_gui.py             the window layer (uses whisper_transcribe.py as-is)
├── how-to-use.txt
├── .venv/                     created on the first run: the Python environment
├── models/
│   ├── ct2/                   faster-whisper large-v3 — used by the CUDA and CPU runtimes
│   └── ov/                    OpenVINO int8 large-v3 — used by the NPU and iGPU runtimes
├── cache/                     compiled kernels (NPU / iGPU paths)
├── transcripts/               only when the folder holding the source file is read-only
└── logs/                      only if the window layer hits an error (gui-error.log)
```

The transcript for `clip.mp3` is written beside it as `clip.txt`.

## Requirements

- Windows 10/11
- Disk: ~3 GB for one runtime (models + environment); up to ~10 GB if you enable every runtime and keep the NPU compile cache
- The window needs the Microsoft **WebView2 runtime** — it ships with current Windows 10/11, and if it is missing the tool runs in console mode instead. `whisper_transcribe.bat --cli` skips the window entirely.
- Optional hardware: NVIDIA GPU, Intel NPU, Intel iGPU, or none of the above. The tool uses whatever is present; missing drivers are reported, never required.

## Engineering notes

The parts that took the real work — all of it on the tool side:

**Two front ends, one core.** `whisper_gui.py` imports `whisper_transcribe.py` and does not modify it: the window layer owns presentation, the per-file queue, the progress bars and nothing else. The core's console output is captured in-process (a `sys.stdout`/`stderr` tee) and rendered in the window's log panel, so progress, warnings and errors live in exactly one place — and anything the core learns, both modes learn the same way.

**Startup you can back out of.** The window is created hidden, the page loads, and only then does it appear — so cancelling (press any key while it loads, or during the one-second grace period after it is ready) never flashes a window, and the console is always there to catch the user. No detached processes: the console hosts the app, and closing either one takes the whole thing down.

**Self-provisioning, and staying portable.** The whole point is that the folder travels. Python is detected by actually executing each candidate (`py -3.14/3.13/3.12/3.11`, then `python`, then common install dirs), not by trusting PATH. A virtualenv copied from another machine is "adopted" by rewriting its `pyvenv.cfg` home path rather than rebuilt. And the .bat must stay pure ASCII: `chcp 65001` plus non-ASCII comments makes cmd re-read the file at misaligned byte offsets, which corrupts parsing in ways that took a while to believe (a `->` inside a comment was read as a redirect and created a stray file).

**Loading CUDA on Windows is a DLL path problem.** CTranslate2 4.8.x resolves its `cublas64_12.dll` / `cudnn64_9.dll` dependencies from the extension module's own directory — PATH and `add_dll_directory` don't help (Python 3.8+ uses `LOAD_LIBRARY_SEARCH` semantics), and a CUDA 13 toolkit can't substitute. The tool pip-installs the `nvidia-*-cu12` wheels into the venv and relocates the DLLs next to `ctranslate2`.

**A download that never finishes.** On some networks the PyPI CDN accepts connections and then stalls — no error, no exit code, so "retry on failure" logic never fires. So the installer pre-flights mirrors instead: a hard 12-second cap, plus a "stalled if it stays under 5 KB/s for 5 seconds" rule, then picks the fastest one that actually moves bytes.

**A failure that has somewhere to go.** When transcription fails, the tool first separates "the environment broke" from "this device cannot run it": a broken dependency is reinstalled and retried in place, while a device problem walks down the runtimes — retry the NPU with its other compiler (which compiler produced the artifact decides whether the driver accepts it), then the iGPU, which shares the OpenVINO model, then the CPU, downloading that model first if it has never been fetched. Nothing is ever run against a model that is not there.

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

The model cache is what makes a batch cheap: the second file skips loading entirely (measured 0.5 s instead of ~4 s on the RTX 5090 machine).

## Limitations

- Windows only.
- The window needs the WebView2 runtime (bundled with current Windows); without it the tool falls back to console mode.
- Files are transcribed one after another rather than simultaneously. The model is loaded once and shared, and one transcription already keeps the device it runs on busy; running several at the same time was measured to be no faster on CUDA and only marginally faster on CPU.
- The OpenVINO path returns the transcript in one block when done; live streaming is CTranslate2-only.
- NPU enumeration occasionally fails after a driver update until reboot — the menu will fall back to the next runtime.
- The Intel NPU path is the least battle-tested of the four.

## Credits & models

- **Whisper large-v3** — the speech recognition model — is by OpenAI (MIT license). This tool is an independent project and is not affiliated with or endorsed by OpenAI.
- Weights are downloaded on first use from public repositories: [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) (CTranslate2 format, used by the CUDA/CPU paths) and [OpenVINO/whisper-large-v3-int8-ov](https://huggingface.co/OpenVINO/whisper-large-v3-int8-ov) (OpenVINO int8, for NPU/iGPU). Nothing model-related is bundled in this repository.
- The window is rendered by [pywebview](https://pywebview.flowrl.com/) on the WebView2 runtime.

## License

MIT — see [LICENSE](LICENSE).
