# whisper-transcriber

A one-click local transcription tool for Windows. Drop the folder anywhere, double-click, drop in an audio or video file, and you get plain text back. No installer, no commands to type, no Python setup, and no AI background needed: everything from building the environment to choosing which hardware runs the model is handled for you. The window is the default face of the tool; a console mode is one flag away for anyone who prefers it.

## What this repository is

Whisper large-v3 does the listening. It is an OpenAI model I neither trained nor published (see [Credits](#credits--models)), and nothing model-related is bundled here. This repository is everything around it: provisioning, runtime selection, downloads, fallbacks, translation, the window, the interface. I built it for my own files first, then kept pushing until it would survive landing on a Windows machine it had never seen, whether that machine has a fast GPU or nothing installed at all.

## What it does

- Double-click `whisper_transcribe.bat`. On first run it finds or installs Python, builds its own environment, picks a runtime, and fetches what it needs. After that it opens its window, and files go in by drag-and-drop, by the Browse button, or by pasting a path. No command line, nothing to configure, no AI knowledge required.
- `whisper_transcribe.bat --cli` (or `no-gui`) runs the original text interface. The console is also the safety net: if the window cannot start, because WebView2 is missing or the machine is locked down, the tool falls back to it instead of failing, and pressing any key while the window is loading cancels the window the same way.
- Four runtimes are aggregated, in that order of preference: NVIDIA CUDA, Intel NPU, Intel iGPU, or plain CPU. The tool probes what the machine actually has, recommends the fastest available option, and lets you switch at any time with `/s`, so you never have to know what "runtime" even means.
- Models, dependencies and even precompiled caches are fetched on demand, with mirror pre-flights. A stalled connection is detected and abandoned rather than waited on, and fallbacks cover networks where huggingface.co or PyPI is slow or unreachable.
- The tool repairs itself. A broken dependency, missing GPU libraries, or a driver update that knocks out the NPU is handled by `/fix`, by a retry, or by a fallback ladder: retry the NPU with its other compiler, then the iGPU (same model file, nothing to download), then the CPU, fetching that model first if it has never been used. It only reports failure after the ladder is exhausted.
- The interface follows the Windows display language (Chinese or English), and it can never end up half-translated: a missing string falls back to the source text rather than crashing.
- Output streams as it is produced on the CTranslate2 paths. Silence is trimmed with VAD, the decoder carries safeguards against Whisper's short-phrase repetition loops, and a one-time NPU precompile turns first loads of about 5.5 minutes into roughly 6 seconds afterwards. The model for the configured runtime also loads in the background as soon as the tool starts, so the first file you pick starts transcribing immediately instead of waiting on a load.
- `whisper_transcribe.bat "a.mp3" "b.m4a"` opens with both files already queued (and in console mode transcribes them and exits), which is handy for batch jobs and automation, though you will never have to type a command. The model loads once per session and is reused across files, so a batch pays the load cost only once.

## Quick start

1. Get `whisper_transcribe.bat` + `whisper_transcribe.py` + `whisper_gui.py` (that's the whole tool; the .bat is a thin launcher, the .py holds the logic, and `whisper_gui.py` is the window layer).
2. Double-click the .bat. First run provisions everything and walks you through a one-time runtime choice. Model weights download on first use (1.6–3 GB depending on runtime).
3. Drag a file (or several) into the window, or paste a path at the console prompt. The transcript is written as `<name>.txt` next to the source file.

At the prompt: `/s` switches runtime, `/fix` repairs dependencies, `/x` exits.

## Step by step

The quick start above skips a few details that are useful the first time. Here is the whole flow, in order:

1. Get the files. On this page: green **Code** button, then **Download ZIP**. Unzip it (right-click the ZIP, *Extract All…*) and put the folder wherever you like; the Desktop is fine. Nothing needs to be installed beforehand.
2. Start it. Open the folder and double-click `whisper_transcribe.bat`, the only file you need to touch. Windows may stop it first, because the file came from the internet. Three ways past that, in the order they are worth trying:
   - Open a command prompt in the folder: type `cmd` in the Explorer address bar (the bar showing the folder path), press Enter, then type `whisper_transcribe.bat` and press Enter. Starting it from a prompt does not go through Windows' download check, and it runs with your normal permissions.
   - Double-click it and, if a warning about an unrecognized app appears, choose **More info → Run anyway**.
   - If the warning offers no way forward (newer Windows 11 builds with Smart App Control enabled), right-click `whisper_transcribe.bat` and choose **Run as administrator**.
3. Let it set itself up (first time only). A console window opens and explains what it is doing. Depending on your PC it may ask one or two things:
   - *Install Python automatically?* If it asks, press Enter (yes). That is the background program the tool runs on, and it is a normal user-level install.
   - *Which "runtime" to use?* This only decides what hardware does the listening. If you have no preference, press Enter to take the option tagged `[Recommended]`.
   Then it downloads the speech model (1.6–3 GB) with a progress bar. That is the slow part, and a one-time one; later starts take seconds.
4. Give it a file. When the window opens, drag an audio or video file onto it, or click **Browse**, or paste a path into the box and press Enter. Several files at once are fine: they queue up and are transcribed one after another, with the model loaded only once. The console prompt takes several paths too, one per line or several on one line, with quoted paths for spaces. Common formats work as they are: mp3, m4a, wav, mp4, mkv.
5. Take the text. The transcript is saved next to your original file, with the same name and a `.txt` extension, and the window shows the exact path. The text also appears on screen as it works.
6. Another file, or done. Add more files at any time and close the window when you are finished. The console behind it is the off switch, and closing either one stops everything with nothing left running.

**Prefer the console?** `whisper_transcribe.bat --cli` (or `no-gui`) starts the text interface directly. You can also back out at startup: while the window is loading, the console says so, and a keypress cancels the window and drops into console mode. Both modes share the same environment, models, cache and settings, so switching between them costs nothing. At the prompt, several paths can go in at once, one per line or several on one line, and each is transcribed in turn.

## Folder layout

A folder that has been started at least once:

```
whisper-transcriber/
├── whisper_transcribe.bat     the launcher; double-click this one
├── whisper_transcribe.py      core logic: provisioning, runtimes, transcription
├── whisper_gui.py             the window layer (uses whisper_transcribe.py as-is)
├── how-to-use.txt
├── .venv/                     created on the first run: the Python environment
├── models/
│   ├── ct2/                   faster-whisper large-v3 (CUDA and CPU runtimes)
│   └── ov/                    OpenVINO int8 large-v3 (NPU and iGPU runtimes)
├── cache/                     compiled kernels (NPU / iGPU paths)
├── transcripts/               only when the folder holding the source file is read-only
└── logs/                      only if the window layer hits an error (gui-error.log)
```

The transcript for `clip.mp3` is written beside it as `clip.txt`.

## Requirements

- Windows 10/11
- Disk: ~3 GB for one runtime (models + environment); up to ~10 GB if you enable every runtime and keep the NPU compile cache
- The window needs the Microsoft **WebView2 runtime**. It ships with current Windows 10/11, and if it is missing the tool runs in console mode instead; `whisper_transcribe.bat --cli` skips the window entirely.
- Optional hardware: NVIDIA GPU, Intel NPU, Intel iGPU, or none of the above. The tool uses whatever is present; missing drivers are reported, never required.

## Engineering notes

The parts that took the real work, all of it on the tool side.

**Two front ends, one core.** `whisper_gui.py` imports `whisper_transcribe.py` and does not modify it: the window layer owns presentation, the per-file queue and the progress bars, and nothing else. The core's console output is captured in-process (a `sys.stdout`/`stderr` tee) and rendered in the window's log panel, so progress, warnings and errors live in exactly one place, and anything the core learns, both modes learn the same way.

**Startup you can back out of.** The window is created hidden, the page loads, and only then does it appear, so cancelling (any key while it loads, or during the one-second grace period after it is ready) never flashes a window, and the console is always there to catch the user. There are no detached processes: the console hosts the app, and closing either one takes the whole thing down.

Loading starts early. As soon as the tool has a configured runtime and a local model, it loads that model in the background, at startup and again after a runtime switch, so the time between pasting a path and seeing text is just the transcription. That preload is silent, so nobody has to wonder why it started loading on its own, and the load is still reported with its real duration the moment a file is actually transcribed. Loading and transcribing share one cache guarded by a lock, so a file dropped in mid-load waits for the load instead of starting a second one. On the console side, a line of input can hold several paths: they are split on whitespace with quote awareness (straight and CJK quotes), and adjacent fragments are re-joined as long as the result exists on disk, which keeps unquoted paths with spaces working.

**Self-provisioning, and staying portable.** The whole point is that the folder travels. Python is detected by actually executing each candidate (`py -3.14/3.13/3.12/3.11`, then `python`, then common install dirs), not by trusting PATH. A virtualenv copied from another machine is "adopted" by rewriting its `pyvenv.cfg` home path rather than rebuilt. And the .bat must stay pure ASCII: `chcp 65001` plus non-ASCII comments makes cmd re-read the file at misaligned byte offsets, which corrupts parsing in ways that took a while to believe (a `->` inside a comment was read as a redirect and created a stray file).

Loading CUDA on Windows is a DLL path problem. CTranslate2 4.8.x resolves its `cublas64_12.dll` / `cudnn64_9.dll` dependencies from the extension module's own directory; PATH and `add_dll_directory` don't help (Python 3.8+ uses `LOAD_LIBRARY_SEARCH` semantics), and a CUDA 13 toolkit can't substitute. The tool pip-installs the `nvidia-*-cu12` wheels into the venv and relocates the DLLs next to `ctranslate2`.

**A download that never finishes.** On some networks the PyPI CDN accepts connections and then stalls, with no error and no exit code, so "retry on failure" logic never fires. The installer pre-flights mirrors instead: a hard 12-second cap, a "stalled if it stays under 5 KB/s for 5 seconds" rule, and then the fastest mirror that actually moves bytes.

Failures have somewhere to go. When transcription fails, the tool first separates "the environment broke" from "this device cannot run it": a broken dependency is reinstalled and retried in place, while a device problem walks down the runtimes, retrying the NPU with its other compiler (which compiler produced the artifact decides whether the driver accepts it), then the iGPU, which shares the OpenVINO model, then the CPU, downloading that model first if it has never been fetched. Nothing is ever run against a model that is not there.

**Whisper loops on short phrases, handled in the decoder.** On short audio, large-v3 can emit "They're not." five times in a row; the cause is cross-segment conditioning (`condition_on_previous_text`). The tool disables it and adds a `no_repeat_ngram_size=4` backstop. It reproduced on both CUDA fp16 and CPU int8, so it is a decoding artifact rather than a quantization one, and the fix also removed a stretch of garbled text the loop had displaced. The 18.3 s to 15.4 s figures for that change come from a longer test file decoded through CTranslate2, not from the short sample used in the table below.

**NPU first-run compilation.** OpenVINO compiles the model ahead of time for the NPU: the first compile measured 5.5 minutes, and afterwards a cached blob loads in about 6.3 seconds. The cache is bound to the hardware generation, so on a different generation OpenVINO rebuilds once by itself.

**Translation that can't break the app.** UI strings are keyed in Chinese with an English lookup table built on top; a missing entry falls back to the source string, so a gap in the translation can never crash a run. Worst case, one line shows up in Chinese.

## Performance

Every number in the table below was measured on one machine, a laptop with a Core Ultra 7 and an RTX 5070M, on a 7.6-second clip:

| Runtime | Model load | RTF |
|---|---|---|
| CUDA (CT2 fp16) | ~6.5 s (first run adds kernel warmup) | 0.16 |
| Intel NPU (cached blob) | ~6.3 s | 0.16 |
| Intel iGPU (OpenVINO int8) | ~2.5 s once cached | 0.29 |
| CPU (CT2 int8) | ~10.5 s | 0.91 |

RTF = transcription time ÷ audio duration; lower is better.

The model cache is what makes a batch cheap: the second file skips loading entirely. That was measured on a different machine, a desktop with an RTX 5090, where the second file loaded in 0.5 s against about 4 s from cold.

## Limitations

- Windows only.
- The window needs the WebView2 runtime (bundled with current Windows); without it the tool falls back to console mode.
- Files are transcribed one after another rather than simultaneously. The model is loaded once and shared, and one transcription already keeps the device it runs on busy; running several at the same time was measured to be no faster on CUDA and only marginally faster on CPU.
- The OpenVINO path returns the transcript in one block when done; live streaming is CTranslate2-only.
- NPU enumeration occasionally fails after a driver update until reboot, in which case the menu falls back to the next runtime.
- The Intel NPU path is the least battle-tested of the four.

## Credits & models

- **Whisper large-v3**, the speech recognition model, is by OpenAI (MIT license). This tool is an independent project and is not affiliated with or endorsed by OpenAI.
- Weights are downloaded on first use from public repositories: [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) (CTranslate2 format, used by the CUDA/CPU paths) and [OpenVINO/whisper-large-v3-int8-ov](https://huggingface.co/OpenVINO/whisper-large-v3-int8-ov) (OpenVINO int8, for NPU/iGPU). Nothing model-related is bundled in this repository.
- The window is rendered by [pywebview](https://pywebview.flowrl.com/) on the WebView2 runtime.

## License

MIT. See [LICENSE](LICENSE).
