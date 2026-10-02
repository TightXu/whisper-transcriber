# whisper-transcriber — the 30-second version

A one-click Windows transcription tool: drop in an audio or video file, get a plain .txt back. Three files, no installer, nothing to configure.

## What it shows

- Four runtimes, tried in order: NVIDIA CUDA, Intel NPU, Intel iGPU, CPU. When one fails, it walks down the ladder instead of reporting an error.
- Speed, 7.6-second clip: CUDA and the NPU both at 0.16x real time (RTF 0.16, lower is better); CPU at 0.91x.
- The NPU compiles once: 5.5 minutes the first time, about 6.3 seconds from cache afterwards.
- It provisions itself. On a machine with no Python it installs one (about 470 MB) and pre-flights mirrors when the PyPI CDN stalls.
- The model loads once per session, so batches are cheap: a second file loaded in 0.5 s against 4 s cold.

## Where to start

- [README.md](README.md#start-here): the `## Start here` block, on what to look at first.
- [how-to-use.txt](how-to-use.txt): how to run it, step by step.

## What it does not claim

- Windows only. Files are transcribed one after another, since parallel runs measured no faster on CUDA.
- The numbers come from one laptop (Core Ultra 7, RTX 5070M) and are self-reported; the Intel NPU path is the least battle-tested of the four.
