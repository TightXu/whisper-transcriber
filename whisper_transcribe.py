# -*- coding: utf-8 -*-
"""
whisper_transcribe.py — 音频/视频 一键转文字（自包含版，v3 重构）

模型目录 models/ 下两种格式，对应不同 runtime：
  ct2/  (CTranslate2, Systran/faster-whisper-large-v3)  → CUDA / CPU
  ov/   (OpenVINO int8, OpenVINO/whisper-large-v3-int8-ov) → Intel NPU / 核显

状态文件 .venv/runtimes.json 保存：已下载的模型、当前 runtime、
首次配置是否完成。每次启动自动检测，第二次起直接读取。

用法：
  bat + 粘贴文件路径  →  转写
  bat "文件1" "文件2" →  逐个转写后退出
  /s                 →  调整 Runtime 菜单
  /x                 →  退出
"""
import os
import re
import sys
import json
import time
import glob
import shutil
import threading
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
CT2_DIR = os.path.join(MODELS_DIR, "ct2")
OV_DIR = os.path.join(MODELS_DIR, "ov")
VENV_DIR = os.path.join(SCRIPT_DIR, ".venv")
VENV_BIN = os.path.join(VENV_DIR, "Scripts" if os.name == "nt" else "bin")
VENV_PY = os.path.join(VENV_BIN, "python.exe" if os.name == "nt" else "python")
STATE_FILE = os.path.join(VENV_DIR, "runtimes.json")
TRANSCRIPTS_DIR = os.path.join(SCRIPT_DIR, "transcripts")
# 字节码缓存一律收进 .venv\pycache：程序目录里不该冒出 __pycache__。
# 环境变量是解释器启动时读的，setdefault 只对本进程之后拉起的子进程
# 有效；sys.pycache_prefix 才管得住本进程后续的 import。两个都设。
os.environ.setdefault("PYTHONPYCACHEPREFIX", os.path.join(VENV_DIR, "pycache"))
try:
    sys.pycache_prefix = os.path.join(VENV_DIR, "pycache")
except Exception:
    pass


def _drop_stray_pycache():
    """别的解释器 import 本模块时，它的 .pyc 会先落在程序目录的
    __pycache__ 里（写缓存发生在模块体执行之前，脚本内部拦不住），
    这里在启动时清掉，保证程序目录始终干净。只删纯 .pyc 的目录，
    别的内容一律不碰。"""
    d = os.path.join(SCRIPT_DIR, "__pycache__")
    if not os.path.isdir(d):
        return
    try:
        if all(n.endswith(".pyc") for n in os.listdir(d)):
            shutil.rmtree(d, ignore_errors=True)
    except OSError:
        pass


_drop_stray_pycache()

# Runtime 优先级 = 实测/预期速度序：cuda > npu > igpu > cpu。
# NPU 峰值最快但强依赖 cache/npu 编译缓存：无缓存首次编译约 5-30
# 分钟（视硬件，int8 本机实测 5.5 分钟），有缓存秒级；核显首次
# 编译仅几十秒，无此顾虑。
PRIORITY = ["cuda", "npu", "igpu", "cpu"]
RUNTIME_MODEL = {"cuda": "ct2", "cpu": "ct2", "npu": "ov", "igpu": "ov"}

# NPU 失败后用哪套编译器重编一次（DRIVER = 驱动侧编译器，PLUGIN = 插件
# 自带）。产出的 blob 驱动认哪个取决于两者版本，切换就是改这一行。
NPU_RETRY_COMPILER = "DRIVER"

# 解码参数：Whisper 在短句/韵律重复处易陷入"重复循环幻觉"。实测 large-v3
# 在 Aphantasia 音频上 cuda fp16 出 "They're not." ×5、cpu int8 出
# "Those were the images." ×4（两种精度都中招，与量化无关）。关掉跨段
# 上下文条件 + 限制 4-gram 重复后：循环消失、原本被它插入的一小段乱码
# 被纠正，耗时还略降（18.3s → 15.4s）。
CT2_DECODE = {
    "beam_size": 5,
    "vad_filter": True,
    "vad_parameters": {"min_silence_duration_ms": 500},
    "condition_on_previous_text": False,
    "no_repeat_ngram_size": 4,
}
# OV（npu/igpu）侧同款抑制；不动 beam 策略（保持管线默认），只加重复惩罚。
OV_DECODE = {
    "no_repeat_ngram_size": 4,
    "repetition_penalty": 1.05,
}

HF_TEST_URLS = ["https://huggingface.co", "https://hf-mirror.com"]


# ================================================================ i18n
# 界面语言自动跟随系统显示语言：中文系统显示中文，其它一律英文。
# WT_LANG 环境变量可强制指定（zh/en），供测试用。
# 文案约定：源码里的字符串是中文原文（键），英文译文查 _TR 表；
# 表缺项时退回中文原文，保证不会出现半翻译状态崩溃。
def _detect_lang():
    v = os.environ.get("WT_LANG", "").strip().lower()
    if v in ("zh", "en"):
        return v
    if os.name == "nt":
        try:
            import ctypes
            # 显示语言 LANGID 主 ID：0x04 = 中文（覆盖 zh-CN/TW/HK/SG）
            if ctypes.windll.kernel32.GetUserDefaultUILanguage() & 0xFF == 4:
                return "zh"
        except Exception:
            pass
    else:
        env = (os.environ.get("LC_ALL") or os.environ.get("LC_MESSAGES")
               or os.environ.get("LANG") or "").lower()
        if env.startswith("zh"):
            return "zh"
    return "en"


LANG = _detect_lang()


def tr(s):
    """翻译用户可见文案。名为 tr 而非惯例的 _，避免与
    `stem, _ = ...` 之类的解包变量互相遮蔽（实测会 TypeError）。"""
    if LANG == "zh":
        return s
    return _TR.get(s, s)


_TR = {
    # ---- venv / 依赖 ----
    "[setup] 使用源: %s":
        "[setup] Using index: %s",
    "[setup] 源 %s 失败，换下一个源重试 ...":
        "[setup] Index %s failed, trying the next one ...",
    "[setup] .venv 来自其它机器，尝试认领 ...":
        "[setup] .venv is from another machine, trying to claim it ...",
    "[setup] 认领成功。":
        "[setup] Claimed.",
    "[setup] 本机无匹配的同版本 Python，重建 .venv ...":
        "[setup] No matching local Python of the same version, "
        "rebuilding .venv ...",
    "[setup] 安装基础依赖 ...":
        "[setup] Installing base dependencies ...",
    "[ERROR] 依赖安装失败，请检查网络后重试。":
        "[ERROR] Dependency installation failed. Check your network "
        "and retry.",
    "[setup] 用 uv 创建 .venv ...":
        "[setup] Creating .venv with uv ...",
    "[setup] 用当前 Python 创建 .venv ...":
        "[setup] Creating .venv with the current Python ...",
    "[setup] 依赖缺失/损坏，重新安装 ...":
        "[setup] Dependencies missing/broken, reinstalling ...",
    "[ERROR] 重试仍失败: %s":
        "[ERROR] Retry still failed: %s",
    "换 NPU 编译器（%s）重新编译并重试 ...":
        "NPU failed; retrying with the %s compiler (recompiles) ...",
    "[ERROR] 换编译器后仍失败: %s":
        "[ERROR] Still failing after switching the compiler: %s",
    "退回 %s：先下载它用的模型 ...":
        "Falling back to %s: downloading its model first ...",
    "[ERROR] 模型下载失败，%s 不可用。":
        "[ERROR] Model download failed; %s is not available.",
    "退回 %s 重试...":
        "Falling back to %s ...",
    "[ERROR] %s 也失败: %s":
        "[ERROR] %s also failed: %s",
    "[setup] 检查 {} 方案依赖 ...":
        "[setup] Checking dependencies for the {} runtime ...",
    "[setup] 依赖正常，无需修复。":
        "[setup] Dependencies are fine, nothing to fix.",
    "[setup] 重装依赖 ...":
        "[setup] Reinstalling dependencies ...",
    "[ERROR] 依赖修复失败，请检查网络后重试。":
        "[ERROR] Dependency fix failed. Check your network and retry.",
    "[setup] 依赖修复完成。":
        "[setup] Dependencies fixed.",
    "[ERROR] 重装后仍异常，请尝试删除 .venv 后重新运行。":
        "[ERROR] Still broken after reinstall. Try deleting .venv and "
        "rerunning.",
    "\n[提示] 后台依赖检查失败：{} 方案所需的库缺失或损坏。输入 /fix 立即修复（重装）；直接转写也会自动修复。":
        "\n[hint] Background dependency check failed: libraries for the "
        "{} runtime are missing or broken.\n"
        "Run /fix to reinstall now; transcription also auto-repairs "
        "on failure.",
    # ---- 模型下载 ----
    "[setup] 使用 HF 端点: %s":
        "[setup] Using HF endpoint: %s",
    "[setup] 下载 %s -> models/%s（约 1.6~3GB）...":
        "[setup] Downloading %s -> models/%s (~1.6-3GB) ...",
    "[ERROR] 模型下载失败。":
        "[ERROR] Model download failed.",
    "[setup] 下载完成。":
        "[setup] Download complete.",
    # ---- CUDA 运行库 ----
    "[setup] 安装 CUDA 12 运行库（约 1.3GB）...":
        "[setup] Installing CUDA 12 runtime libs (~1.3GB) ...",
    "[setup] 运行库自动安装失败。手动方案：把 cudart64_12/cublas64_12/cublasLt64_12/cudnn64_9 四个 DLL 拷进 venv 的 site-packages\\ctranslate2\\。":
        "[setup] Automatic runtime-lib install failed. Manual fix: copy "
        "the four DLLs cudart64_12/cublas64_12/cublasLt64_12/cudnn64_9 "
        "into the venv's site-packages\\ctranslate2\\ directory.",
    "[setup] 安装完成但未检测到运行库，cuda 将退回 CPU。":
        "[setup] Install finished but libs not detected; cuda falls "
        "back to CPU.",
    "[setup] CUDA 运行库就绪。":
        "[setup] CUDA runtime libs ready.",
    "CUDA 12 运行库缺失（选中后自动安装，约 1.3GB）":
        "CUDA 12 runtime libs missing (installed automatically, ~1.3GB)",
    # ---- runtime 菜单 / 就绪度 ----
    "未检测到 NVIDIA 显卡":
        "No NVIDIA GPU detected",
    "未检测到 NPU":
        "No NPU detected",
    "未检测到 Intel 核显":
        "No Intel iGPU detected",
    "首次需编译（约 5-30 分钟）":
        "First use needs kernel compile (~5-30 min)",
    "预编译缓存已就绪":
        "Precompiled cache ready",
    "需下载模型(%.1fGB)":
        "Need model download (%.1fGB)",
    "就绪，可直接使用":
        "Ready to use",
    "；":
        "; ",
    "　[当前]":
        " [Current]",
    "　[Recommended]":
        " [Recommended]",
    "CUDA（NVIDIA 显卡）":
        "CUDA (NVIDIA GPU)",
    "Intel 核显":
        "Intel iGPU",
    "仅 CPU":
        "CPU only",
    "←/→ 或 ↑/↓ 移动   Enter 确认   Esc 取消":
        "←/→ or ↑/↓ to move   Enter to confirm   Esc to cancel",
    "首次使用 - 选择方案（速度优先级: cuda > npu > igpu > cpu）":
        "First-time setup - choose a runtime (speed order: "
        "cuda > npu > igpu > cpu)",
    "调整 Runtime":
        "Adjust Runtime",
    "未更改。":
        "No change.",
    # ---- 首次配置 / 状态 ----
    "[setup] 检测到 NVIDIA 显卡但缺 CUDA 12 运行库，选择 CUDA 方案时会自动安装。":
        "[setup] NVIDIA GPU found but CUDA 12 runtime libs missing; "
        "choosing CUDA auto-installs them.",
    "默认使用推荐方案 ...":
        "Using the recommended option ...",
    "[setup] 完成。当前 runtime: %s":
        "[setup] Done. Current runtime: %s",
    "        以后双击 bat 直接粘贴文件即可；/s 可随时调整 runtime。":
        "        From now on: double-click the bat and paste a file; "
        "/s switches runtime anytime.",
    "[setup] Runtime 已设为 %s":
        "[setup] Runtime set to %s",
    "[setup] 检测到配置来自其它机器，重新进入配置...":
        "[setup] Config is from another machine, re-running first-time "
        "setup ...",
    # ---- 转写 ----
    "[错误] 文件不存在: %s":
        "[ERROR] File not found: %s",
    "输入: %s":
        "Input: %s",
    "输出: %s":
        "Output: %s",
    "[提示] 源目录不可写入(%s)，已改存到: %s":
        "[hint] Source dir not writable (%s), saved to: %s",
    "加载模型 ...":
        "Loading model ...",
    "加载模型（%s, int8）...":
        "Loading model (int8 on %s) ...",
    "模型加载 %.1fs":
        "Model loaded in %.1fs",
    "音频时长 %.1f 秒":
        "Audio duration: %.1fs",
    "转写中 ... 实时输出：":
        "Transcribing ... live output:",
    "转写中（%s）...":
        "Transcribing on %s ...",
    "转写完成，共 %d 段":
        "Transcription done, %d segments",
    "识别语言: %s (置信度 %.3f)":
        "Detected language: %s (confidence %.3f)",
    "转写用时 %.1fs":
        "Transcribed in %.1fs",
    "转写完成":
        "Transcription done",
    "（首次使用需编译内核，约 5-30 分钟；之后秒级加载）":
        "(First use compiles kernels, ~5-30 min; near-instant afterwards)",
    "（首次使用需编译内核，几十秒；结果会缓存）":
        "(First use compiles kernels, tens of seconds; result will "
        "be cached)",
    "文本已保存: %s":
        "Saved: %s",
    # ---- 交互主循环 ----
    "  粘贴音频/视频文件（支持拖拽）→ 转写":
        "  Paste an audio/video file (drag & drop) -> transcribe",
    "  /s → 调整 Runtime      /x → 退出":
        "  /s -> Runtime settings      /x -> quit",
    "  当前 Runtime: %s":
        "  Current Runtime: %s",
    "当前 Runtime: %s":
        "Current Runtime: %s",
    "无效命令: %s（可用: /s 调整Runtime, /fix 修复依赖, /x 退出）":
        "Invalid command: %s (available: /s runtime, /fix fix deps, "
        "/x quit)",
    "（转写失败，可重试其它文件或 /s 调整 runtime）":
        "(Transcription failed; retry the file or /s to switch runtime)",
}


# ================================================================ 状态
def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _machine_sig():
    """来源机器唯一特征（同机型重装的概率极低，够用）：
    machine GUID > 机器名+用户名。取 16 位十六进制指纹。"""
    import hashlib
    raw = None
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["reg", "query",
                 r"HKLM\SOFTWARE\Microsoft\Cryptography",
                 "/v", "MachineGuid"],
                capture_output=True, text=True, timeout=10).stdout
            m = re.search(r"MachineGuid\s+REG_SZ\s+(\S+)", out)
            if m:
                raw = m.group(1)
        except Exception:
            pass
    if not raw:
        raw = "%s|%s" % (os.environ.get("COMPUTERNAME", "?"),
                         os.environ.get("USERNAME", "?"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _venv_base_ok():
    """检查 .venv 指向的 base Python 是否在本机存在（复制来的 venv 会失效）。"""
    try:
        with open(os.path.join(VENV_DIR, "pyvenv.cfg"), "r",
                  encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("home"):
                    home = line.split("=", 1)[1].strip()
                    return os.path.exists(home)
    except Exception:
        pass
    return False


def _fix_venv_base():
    """复制来的 venv：把 pyvenv.cfg 的 home 指向本机已有的同版本 Python。
    成功返回 True。"""
    local = os.environ.get("LOCALAPPDATA", "")
    roaming = os.environ.get("APPDATA", "")
    cands = []
    # uv 托管的 Python（Roaming\uv\python，注意不是 Local）
    cands += glob.glob(os.path.join(roaming, "uv", "python", "*"))
    cands += glob.glob(os.path.join(local, "uv", "python", "*"))
    # 官方安装器位置
    cands += glob.glob(os.path.join(local, "Programs", "Python",
                                     "Python31*"))
    cands += glob.glob("C:/Python31*/python.exe")
    # 程序目录内便携式 Python
    cands += glob.glob(os.path.join(SCRIPT_DIR, "python", "python.exe"))
    want = _venv_python_version()
    for c in cands:
        exe = c if os.path.basename(c).lower() == "python.exe" \
            else os.path.join(c, "python.exe")
        if not os.path.exists(exe):
            continue
        try:
            out = subprocess.run([exe, "-V"], capture_output=True, text=True,
                                 timeout=15)
            ver = (out.stdout or out.stderr).strip()   # 'Python 3.11.15'
            if want and want not in ver:
                continue
            home = os.path.dirname(os.path.abspath(exe))
            cfg = os.path.join(VENV_DIR, "pyvenv.cfg")
            with open(cfg, "r", encoding="utf-8") as f:
                lines = f.readlines()
            with open(cfg, "w", encoding="utf-8") as f:
                for line in lines:
                    if line.lower().startswith("home"):
                        f.write("home = %s\n" % home)
                    else:
                        f.write(line)
            return True
        except Exception:
            continue
    return False


def _venv_python_version():
    """从 pyvenv.cfg 读版本号（如 '3.11'）。uv 写 version_info，
    标准 venv 写 version，两者都认。"""
    try:
        with open(os.path.join(VENV_DIR, "pyvenv.cfg"), "r",
                  encoding="utf-8") as f:
            for line in f:
                low = line.lower()
                if low.startswith("version_info") or low.startswith("version"):
                    v = line.split("=", 1)[1].strip()
                    return ".".join(v.split(".")[:2])   # '3.14.6' -> '3.14'
    except Exception:
        pass
    return None


def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def models_present():
    """本地已有哪些模型。ct2: model.bin；ov: encoder bin。"""
    have = []
    if os.path.exists(os.path.join(CT2_DIR, "model.bin")):
        have.append("ct2")
    if (glob.glob(os.path.join(OV_DIR, "*encoder*.xml"))
            or os.path.exists(os.path.join(OV_DIR, "openvino_encoder_model.bin"))):
        have.append("ov")
    return have


# ================================================================ 硬件探测
_probe_cache = None


def probe(refresh=False):
    """探测硬件。返回 dict: nvidia, cuda_libs, npu, igpu（0/1）。
    cuda_libs 只有在 nvidia=1 时才有意义。
    结果按进程缓存：硬件不会热插拔，而每次探测都要构造 openvino
    Core（约 1 秒）；归位 CUDA DLL 后用 refresh=True 强制重探。"""
    global _probe_cache
    if _probe_cache is not None and not refresh:
        return _probe_cache
    d = {"nvidia": 0, "cuda_libs": 0, "npu": 0, "igpu": 0}
    try:
        import ctranslate2
        d["nvidia"] = 1 if ctranslate2.get_cuda_device_count() > 0 else 0
    except Exception:
        pass
    if d["nvidia"]:
        try:
            # Windows 上 CT2 扩展用 LOAD_LIBRARY_SEARCH 标志找依赖 DLL，
            # 实测只有"扩展模块自身目录"生效（add_dll_directory 注册目录
            # 与 PATH 对 CT2 的内部 LoadLibrary 都无效），因此按
            # ctranslate2 包目录里的文件存在性判定，与实际加载一致。
            ct2_dir = os.path.dirname(os.path.abspath(ctranslate2.__file__))
            need = ["cudart64_12.dll", "cublas64_12.dll"]
            cudnn = ["cudnn64_9.dll", "cudnn_ops64_9.dll"]   # 伞库或子库任一即可
            have = {n: os.path.exists(os.path.join(ct2_dir, n))
                    for n in need + cudnn}
            try:
                os.add_dll_directory(ct2_dir)
            except Exception:
                pass
            if all(have[n] for n in need) and any(have[n] for n in cudnn):
                d["cuda_libs"] = 1
        except Exception:
            pass
    try:
        import openvino as ov
        core = ov.Core()
        for x in core.get_available_devices():
            xu = x.upper()
            if xu.startswith("NPU"):
                d["npu"] = 1
            elif xu == "GPU" or xu.startswith("GPU."):
                try:
                    name = core.get_property(x, "FULL_DEVICE_NAME").upper()
                except Exception:
                    name = ""
                if "INTEL" in name:
                    d["igpu"] = 1
                    # 不提前 return：设备列表里 NPU 可能排在 GPU 之后，
                    # 继续遍历避免漏检。
    except Exception:
        pass
    _probe_cache = d
    return d


def best_runtime(d):
    """按 PRIORITY 顺序返回本机可用的最佳 runtime。"""
    avail = {"cuda": d["nvidia"] and d["cuda_libs"],
             "npu": d["npu"], "igpu": d["igpu"], "cpu": True}
    for rt in PRIORITY:
        if avail[rt]:
            return rt
    return "cpu"


# ================================================================ 依赖 / venv
PIP_MIRROR = "https://mirrors.aliyun.com/pypi/simple/"
#  (名称, index-url；None = 官方默认源)
PIP_INDEXES = [("official", None),
               ("aliyun", PIP_MIRROR),
               ("tsinghua", "https://pypi.tuna.tsinghua.edu.cn/simple/")]
PROBE_PKG = "faster-whisper"


def _index_probe(index_url):
    """预检一个 PyPI 源：curl 拉一个小页面。12s 硬上限，且连续 5s
    平均速度低于 5KB/s 即判为"停滞"——官方 CDN 的典型故障是连得上、
    下不动（进程不报错也不退出），仅靠"失败才换源"永远等不到。
    返回耗时秒；不可用返回 None。"""
    base = index_url or "https://pypi.org/simple/"
    url = base.rstrip("/") + "/" + PROBE_PKG + "/"
    try:
        out = subprocess.check_output(
            ["curl", "-s", "-o", "NUL" if os.name == "nt" else "/dev/null",
             "--max-time", "12", "--speed-time", "5", "--speed-limit", "5000",
             "-w", "%{time_total}", url],
            stderr=subprocess.DEVNULL)
        return float(out.decode("ascii", "ignore").strip())
    except Exception:
        return None


def _pick_pip_indexes():
    """预检所有源，返回尝试顺序：可用且最快的排第一，其余按原顺序跟上；
    一个都探不到时保持官方优先。"""
    if not shutil.which("curl"):
        return list(PIP_INDEXES)
    scored = []
    for name, url in PIP_INDEXES:
        t = _index_probe(url)
        if t is not None:
            scored.append((t, name, url))
    if not scored:
        return list(PIP_INDEXES)
    scored.sort(key=lambda x: x[0])
    picked = [(n, u) for _, n, u in scored]
    picked += [(n, u) for n, u in PIP_INDEXES if n not in [p[0] for p in picked]]
    return picked


def _pip_install_into(py, *pkgs, reinstall=False):
    """往指定解释器环境装包。uv 优先（uv 建的 venv 没有 pip）。
    reinstall=True：包目录被破坏但 dist-info 残留时，pip/uv 会误判
    "已安装"而跳过，须强制重装。
    源策略：先预检选源（含停滞判定），失败再依次换下一个源。"""
    extra = []
    if reinstall:
        extra = ["--reinstall"] if shutil.which("uv") else ["--force-reinstall"]
    use_uv = bool(shutil.which("uv"))
    if use_uv:
        cmd = ["uv", "pip", "install", "--python", py] + extra + list(pkgs)
        env_key = "UV_DEFAULT_INDEX"
    else:
        cmd = [py, "-m", "pip", "install"] + extra + list(pkgs)
        env_key = "PIP_INDEX_URL"
    for name, url in _pick_pip_indexes():
        env = dict(os.environ)
        if use_uv:
            env.setdefault("UV_HTTP_TIMEOUT", "30")
        if url:
            env[env_key] = url
        print(tr("[setup] 使用源: %s") % name)
        if subprocess.call(cmd, cwd=SCRIPT_DIR, env=env) == 0:
            return 0
        print(tr("[setup] 源 %s 失败，换下一个源重试 ...") % name)
    return 1


DEP_PKGS = ("faster-whisper", "ctranslate2", "openvino", "openvino-genai")


def _imports_for(rt):
    """按 runtime 返回所需引擎的 import 语句（只查真正用到的）：
    ct2 方案（cuda/cpu）不依赖 openvino，ov 方案（npu/igpu）不依赖
    ctranslate2 —— 未装另一个引擎属正常，不算损坏。"""
    return {"ct2": "import faster_whisper, ctranslate2",
            "ov": "import openvino, openvino_genai"}[RUNTIME_MODEL[rt]]


def _import_ok(imports):
    return subprocess.call([VENV_PY, "-c", imports],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL) == 0


def ensure_venv():
    """确保 .venv 就绪，返回 venv python 路径。
    含"来源机器指纹"机制：复制的 .venv 在新机器上认领（重指 base
    Python），避免重装 470MB 依赖。
    依赖完整性不在启动时同步校验（import 引擎要数秒）：交互模式下
    由后台线程按配置的 runtime 校验（_bg_dep_check），失败提示 /fix；
    转写失败另有自愈兜底。仅"本次新建 venv"时同步装依赖（首次装机，
    等待是预期内的）。"""
    created = False
    # --- 情况 A: .venv 不存在或残缺（无 pyvenv.cfg）→ 新建 ---
    # （残缺目录的删/改名让路逻辑统一在 _create_venv(force=True) 里）
    cfg_ok = os.path.exists(os.path.join(VENV_DIR, "pyvenv.cfg"))
    if not os.path.exists(VENV_PY) or not cfg_ok:
        _create_venv(force=True)
        created = True
    else:
        # --- 情况 B: .venv 存在（本机创建或复制而来）---
        sig = _machine_sig()
        state = load_state()
        # host_sig 语义 = "配置文件归属"。ensure_venv 不改写它，
        # 让 main() 能检测"配置来自其它机器"→ 重新进入首次配置。
        # venv 本身的认领状态单独记录在 venv_claimed。
        claimed = state.get("venv_claimed")
        base_ok = _venv_base_ok()
        if claimed is None and base_ok:
            # bat 引导新建的本地 venv（还没写过状态文件）：静默登记
            # 归属即可，不算"来自其它机器"。py 自建的 venv 在
            # _create_venv 里已登记，不会走到这里。
            state["venv_claimed"] = sig
            save_state(state)
        elif claimed != sig or not base_ok:
            # venv 带着别机的指纹（整目录拷来），或 base Python 已
            # 失效（卸载/移动）：认领（重指本机同版本 Python），
            # 失败才重建。
            print(tr("[setup] .venv 来自其它机器，尝试认领 ..."))
            if _fix_venv_base():
                print(tr("[setup] 认领成功。"))
            else:
                print(tr("[setup] 本机无匹配的同版本 Python，重建 .venv ..."))
                _create_venv(force=True)
            state = load_state()
            state["venv_claimed"] = sig
            save_state(state)

    if created:
        # 本次新建：同步装依赖（首次装机，等待是预期内的）
        print(tr("[setup] 安装基础依赖 ..."))
        if _pip_install_into(VENV_PY, *DEP_PKGS) != 0:
            print(tr("[ERROR] 依赖安装失败，请检查网络后重试。"))
            sys.exit(1)
    return VENV_PY


def _create_venv(force=False):
    if force and os.path.exists(VENV_DIR):
        # 强制重建：先让路（正在运行的进程会占用，失败则改名保留）
        try:
            shutil.rmtree(VENV_DIR)
        except Exception:
            try:
                os.rename(VENV_DIR, VENV_DIR + ".broken-%d" % int(time.time()))
            except Exception:
                pass
    if shutil.which("uv"):
        print(tr("[setup] 用 uv 创建 .venv ..."))
        subprocess.check_call(["uv", "venv", ".venv", "--python", "3.12"],
                              cwd=SCRIPT_DIR)
    else:
        print(tr("[setup] 用当前 Python 创建 .venv ..."))
        subprocess.check_call([sys.executable, "-m", "venv", ".venv"],
                              cwd=SCRIPT_DIR)
    state = load_state()
    state["venv_claimed"] = _machine_sig()
    save_state(state)


def hf_endpoint():
    """ping HF 官方与镜像，返回可用的那个；都不通返回镜像（让报错冒出来）。"""
    for url in HF_TEST_URLS:
        try:
            rc = subprocess.call(
                ["curl", "-s", "-o", "NUL" if os.name == "nt" else "/dev/null",
                 "-m", "8", url],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if rc == 0:
                return url
        except Exception:
            pass
    return "https://hf-mirror.com"


def download_model(kind):
    """从 HF 下载模型（kind: ct2 / ov）。"""
    ep = hf_endpoint()
    env = dict(os.environ)
    if "hf-mirror" in ep:
        env["HF_ENDPOINT"] = ep
        # 镜像只代理经典 HTTPS，不代理 Xet 协议：hf-xet 会绕过 HF_ENDPOINT
        # 直连 cas-server.xethub.hf.co 取分块（匿名 401），大文件必败。
        # 官方端点保持 Xet 不动（保留其速度优势），仅镜像分支强制经典下载。
        env["HF_HUB_DISABLE_XET"] = "1"
    print(tr("[setup] 使用 HF 端点: %s") % ep)
    if kind == "ct2":
        repo, dst = "Systran/faster-whisper-large-v3", CT2_DIR
    else:
        repo, dst = "OpenVINO/whisper-large-v3-int8-ov", OV_DIR
    print(tr("[setup] 下载 %s -> models/%s（约 1.6~3GB）...")
          % (repo, os.path.basename(dst)))
    code = ("from huggingface_hub import snapshot_download;"
            "snapshot_download(%r, local_dir=%r)" % (repo, dst))
    rc = subprocess.call([VENV_PY, "-c", code], cwd=SCRIPT_DIR, env=env)
    if rc != 0:
        print(tr("[ERROR] 模型下载失败。"))
        return False
    print(tr("[setup] 下载完成。"))
    return True


# ================================================================ 就绪度
# OpenVINO 编译缓存按设备分目录：拷贝/上传/下载都以目录为单位，
# 且互不干扰（此前混在一个目录时，核显的 blob 会让 NPU 误报"已就绪"）。
# 注意：OpenVINO 设备名 "GPU" = Intel 核显（igpu）；NVIDIA 独显走
# CUDA/CT2 路径，不经过 OpenVINO，无编译缓存。cpu 同理（CT2 无缓存）。
CACHE_ROOT = os.path.join(SCRIPT_DIR, "cache")
OV_CACHE_DIR = {"NPU": os.path.join(CACHE_ROOT, "npu"),
                "GPU": os.path.join(CACHE_ROOT, "igpu")}

# CT2 轮子按 CUDA 12 构建（要 cublas64_12.dll，CUDA 13 工具包不兼容）；
# cuDNN 不随 CUDA 工具包分发。缺库时用 pip 把运行库装进 venv 的
# nvidia/* 布局，再由 _relocate_nvidia_dlls 归位到 ctranslate2 包目录，
# 无需系统安装。
CUDA_RUNTIME_PKGS = ["nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12",
                     "nvidia-cudnn-cu12"]


def _relocate_nvidia_dlls():
    """pip 装的 nvidia-* 轮子把 DLL 放在 nvidia/*/bin/，但 CT2 扩展
    只从自身目录加载依赖 —— 把它们移到 ctranslate2 包目录（同盘
    rename，瞬间完成；wheel 元数据不受影响）。"""
    import ctranslate2
    dst = os.path.dirname(os.path.abspath(ctranslate2.__file__))
    moved = 0
    for p in sys.path:
        if not p.lower().endswith(("site-packages", "dist-packages")):
            continue
        for bin_dir in glob.glob(os.path.join(p, "nvidia", "*", "bin")):
            for dll in glob.glob(os.path.join(bin_dir, "*.dll")):
                target = os.path.join(dst, os.path.basename(dll))
                if os.path.exists(target):
                    continue
                try:
                    shutil.move(dll, target)
                    moved += 1
                except Exception:
                    pass
    if moved:
        # 归位后必须立刻把 DLL 预加载进本进程：ctranslate2 多半在本函数之前
        # 就被 import 过（probe 探测时），其 __init__ 里"CDLL 包目录下全部
        # DLL"的预加载循环已在归位前跑完，不会再跑；不补载则本进程内首次
        # CUDA 转写报 "cublas64_12.dll is not found" 而退回 CPU，重启才恢复。
        # 这里按 CT2 __init__ 的同款方式补载（add_dll_directory + 逐个 CDLL）。
        import ctypes
        try:
            os.add_dll_directory(dst)
        except Exception:
            pass
        for dll in glob.glob(os.path.join(dst, "*.dll")):
            try:
                ctypes.CDLL(dll)
            except OSError:
                pass


def ensure_cuda_libs():
    """确保 cublas/cudnn 运行库可用（缺则 pip 装并归位），返回是否就绪。"""
    if probe()["cuda_libs"]:
        return True
    print(tr("[setup] 安装 CUDA 12 运行库（约 1.3GB）..."))
    if _pip_install_into(VENV_PY, *CUDA_RUNTIME_PKGS) != 0:
        print(tr("[setup] 运行库自动安装失败。"
                "手动方案：把 cudart64_12/cublas64_12/cublasLt64_12/"
                "cudnn64_9 四个 DLL 拷进 venv 的 "
                "site-packages\\ctranslate2\\。"))
        return False
    _relocate_nvidia_dlls()
    if not probe(refresh=True)["cuda_libs"]:
        print(tr("[setup] 安装完成但未检测到运行库，cuda 将退回 CPU。"))
        return False
    print(tr("[setup] CUDA 运行库就绪。"))
    return True


def runtime_ready(rt, p=None):
    """检测 runtime 所需的 模型/预编译缓存 是否已就绪。
    返回 (ok, need_compile, note)。ok=False 即需先下载模型；
    预编译缓存按内容哈希自动匹配：有且匹配→秒载；无或不匹配→需编译。"""
    if p is None:
        p = probe()
    model = RUNTIME_MODEL[rt]
    have_model = model in models_present()
    need_compile = False
    note = ""
    if rt in ("npu", "igpu") and have_model:
        # OpenVINO: 模型必须编译成设备 blob 才能跑。
        # CACHE_DIR 里有 blob → 秒载；否则首次需编译。
        if rt == "npu":
            # blob 与 NPU 硬件代际绑定；不匹配时 OV 会自动重编译并
            # 重建缓存，不会出错，只是慢一次。
            need_compile = not glob.glob(
                os.path.join(OV_CACHE_DIR["NPU"], "*.blob"))
            note = (tr("首次需编译（约 5-30 分钟）")
                    if need_compile else tr("预编译缓存已就绪"))
        else:  # igpu：GPU 内核编译秒级~十几秒，无需提示编译
            need_compile = False
    if rt == "cuda" and p["nvidia"] and not p["cuda_libs"]:
        note = tr("CUDA 12 运行库缺失（选中后自动安装，约 1.3GB）")
    ok = have_model
    return (ok, need_compile, note)


# ================================================================ 菜单
def _rt_enabled(p):
    return {"cuda": p["nvidia"], "npu": p["npu"],
            "igpu": p["igpu"], "cpu": True}


def _rt_disabled_reason(rt):
    return {"cuda": tr("未检测到 NVIDIA 显卡"), "npu": tr("未检测到 NPU"),
            "igpu": tr("未检测到 Intel 核显")}.get(rt, "")


def _build_rt_menu(p, have, rec, cur=None, order=None):
    """构建 runtime 菜单。返回 (items, rts)，下标一一对应，
    避免 idx→runtime 的平行数组错位。"""
    items, rts = [], []
    for rt in (order or PRIORITY):
        label = _rt_label(rt)
        if not _rt_enabled(p)[rt]:
            items.append((label, _rt_disabled_reason(rt), False))
        else:
            ok, need_compile, note = runtime_ready(rt, p)
            sub = _rt_action(rt, ok, need_compile, note)
            if rt == cur:
                sub += tr("　[当前]")
            if rt == rec:
                sub += tr("　[Recommended]")
            items.append((label, sub, True))
        rts.append(rt)
    return items, rts
def _arrow_menu(items, title):
    """方向键菜单。items: [(label, sub, enabled)]。返回下标或 None(Esc)。
    只在交互式终端可用，非交互返回 None。"""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return None
    try:
        import msvcrt
    except ImportError:
        return None
    os.system("")   # 启用 ANSI
    GREY, REV, END = "\033[90m", "\033[7m", "\033[0m"
    n = len(items)
    sel = next((i for i, it in enumerate(items) if it[2]), 0)
    while True:
        os.system("cls")
        print(title)
        print("")
        for i, it in enumerate(items):
            label, sub, enabled = it
            line = "%s  -  %s" % (label, sub) if sub else label
            if not enabled:
                print("    x %s%s%s" % (GREY, line, END))
            elif i == sel:
                print("  > %s%s%s" % (REV, line, END))
            else:
                print("    " + line)
        print("")
        print(tr("←/→ 或 ↑/↓ 移动   Enter 确认   Esc 取消"))
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            k = msvcrt.getwch()
            if k in ("K", "H"):      # 左 / 上
                while True:
                    sel = (sel - 1) % n
                    if items[sel][2]:
                        break
            elif k in ("M", "P"):    # 右 / 下
                while True:
                    sel = (sel + 1) % n
                    if items[sel][2]:
                        break
        elif ch == "\r":
            os.system("cls")
            return sel
        elif ch == "\x1b":
            os.system("cls")
            return None


# ================================================================ 首次配置
def _rt_label(rt):
    return {"cuda": tr("CUDA（NVIDIA 显卡）"), "npu": "Intel NPU",
            "igpu": tr("Intel 核显"), "cpu": tr("仅 CPU")}[rt]


def _rt_action(rt, have_model, need_compile, note):
    """该项还需要做什么（下载/编译提示）。"""
    acts = []
    if not have_model:
        acts.append(tr("需下载模型(%.1fGB)") % (2.9 if RUNTIME_MODEL[rt] == "ct2"
                                              else 1.6))
    if need_compile:
        acts.append(tr("首次需编译（约 5-30 分钟）"))
    if note:
        acts.append(note)
    if not acts:
        return tr("就绪，可直接使用")
    return tr("；").join(acts)


def _prepare_runtime(rt, p, have):
    """选定 runtime 后的备齐工作：CUDA 缺运行库 → pip 自动补齐；
    缺模型 → 自动下载。返回 (rc, have)：rc=1 即下载失败；have 为
    刷新后的本地模型清单（供写状态用）。"""
    if rt == "cuda" and not p["cuda_libs"]:
        ensure_cuda_libs()
    if RUNTIME_MODEL[rt] not in have:
        if not download_model(RUNTIME_MODEL[rt]):
            return 1, have
        have = models_present()
    return 0, have


def first_run_setup(state):
    """无配置或机器码变化时的引导：探测硬件 → 按 PRIORITY 列出可用
    方案（标注 Recommended 与 下载/编译 需求）→ 选择 → 准备 → 写状态。"""
    p = probe()
    rec = best_runtime(p)
    have = models_present()
    if p["nvidia"] and not p["cuda_libs"]:
        print(tr("[setup] 检测到 NVIDIA 显卡但缺 CUDA 12 运行库，"
                "选择 CUDA 方案时会自动安装。"))
    print("")

    order = [rec] + [r for r in PRIORITY if r != rec]  # 推荐项排最前
    items, rts = _build_rt_menu(p, have, rec, order=order)

    idx = _arrow_menu(items, tr("首次使用 - 选择方案（速度优先级: "
                               "cuda > npu > igpu > cpu）"))
    if idx is None:
        print(tr("默认使用推荐方案 ..."))
        idx = 0
    rt = rts[idx]

    rc, have = _prepare_runtime(rt, p, have)
    if rc:
        return 1

    state.update({"models": have, "runtime": rt, "first_run_done": True})
    save_state(state)
    print("")
    print(tr("[setup] 完成。当前 runtime: %s") % rt)
    print(tr("        以后双击 bat 直接粘贴文件即可；/s 可随时调整 runtime。"))
    return 0


# ================================================================ /s 菜单
def runtime_menu(state):
    p = probe()
    have = models_present()
    cur = state.get("runtime", "cpu")
    rec = best_runtime(p)

    items, rts = _build_rt_menu(p, have, rec, cur=cur)

    idx = _arrow_menu(items, tr("调整 Runtime"))
    if idx is None:
        print(tr("未更改。"))
        return 0
    rt = rts[idx]

    rc, have = _prepare_runtime(rt, p, have)
    if rc:
        return 1

    state.update({"models": have, "runtime": rt})
    save_state(state)
    _MODELS.clear()   # 切换 runtime：旧实例的显存/内存随引用释放
    print(tr("[setup] Runtime 已设为 %s") % rt)
    return 0


# ================================================================ 转写
def _write_text_robust(path, desired_out, text):
    """源目录不可写（如微信目录）时退回 程序目录/transcripts/。"""
    try:
        with open(desired_out, "w", encoding="utf-8") as f:
            f.write(text)
        return desired_out
    except OSError as e:
        os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
        fb = os.path.join(TRANSCRIPTS_DIR,
                          os.path.splitext(os.path.basename(path))[0] + ".txt")
        with open(fb, "w", encoding="utf-8") as f:
            f.write(text)
        print(tr("[提示] 源目录不可写入(%s)，已改存到: %s")
              % (type(e).__name__, fb))
        return fb


# 已加载的模型实例（ct2/ov × 设备 缓存）：交互会话连转多个文件、bat
# 批量直转时，免去每个文件 ~5-10s 的重复加载。/s 切换 runtime 时清空，
# 旧实例占用的显存/内存随引用释放。
_MODELS = {}


def _run_ct2(device, ctype, path):
    from faster_whisper import WhisperModel
    model = _MODELS.get(("ct2", device, ctype))
    if model is None:
        print(tr("加载模型 ..."))
        t0 = time.time()
        model = WhisperModel(CT2_DIR, device=device, compute_type=ctype)
        _MODELS[("ct2", device, ctype)] = model
        print(tr("模型加载 %.1fs") % (time.time() - t0))

    from faster_whisper.audio import decode_audio
    audio = decode_audio(path, sampling_rate=16000)
    print(tr("音频时长 %.1f 秒") % (audio.shape[0] / 16000.0))

    print(tr("转写中 ... 实时输出："))
    t0 = time.time()
    segments, info = model.transcribe(audio, language=None, **CT2_DECODE)
    texts = []
    for seg in segments:
        txt = seg.text.strip()
        if txt:
            texts.append(txt)
            print("  %s" % txt, flush=True)
    print(tr("转写完成，共 %d 段") % len(texts))
    print(tr("识别语言: %s (置信度 %.3f)")
          % (info.language, info.language_probability))
    print(tr("转写用时 %.1fs") % (time.time() - t0))
    return " ".join(texts)


def _run_ov(ov_device, path, compiler=None):
    import openvino_genai
    from faster_whisper.audio import decode_audio
    # 编译缓存按设备分目录（cache/npu、cache/igpu）。首次编译后存入，
    # 之后秒级导入；blob 与硬件绑定，不匹配时 OV 自动重编译一次。
    cache_dir = OV_CACHE_DIR[ov_device]
    if compiler:
        # 换编译器重试必须换缓存目录：同一目录里那份被设备拒收的 blob
        # 会被命中，重试等于没试。
        cache_dir = cache_dir + "-" + compiler.lower()
    os.makedirs(cache_dir, exist_ok=True)
    if not glob.glob(os.path.join(cache_dir, "*.blob")):
        if ov_device == "NPU":
            # 实测 int8 本地编译（约 5 分钟）通常快于下载 4.3GB，
            # 故不做网络下载，直接本地编译。
            print(tr("（首次使用需编译内核，约 5-30 分钟；之后秒级加载）"))
        else:
            print(tr("（首次使用需编译内核，几十秒；结果会缓存）"))
    pipe = _MODELS.get(("ov", ov_device, compiler))
    if pipe is None:
        print(tr("加载模型（%s, int8）...") % ov_device)
        t0 = time.time()
        cfg = {"CACHE_DIR": cache_dir}
        if compiler:
            # 编译器由驱动侧提供还是插件自带，决定产出的 blob 驱动认不
            # 认；属性 + 环境变量都设（插件在进程内可能已初始化过，属性
            # 这条更可靠）。
            cfg["NPU_COMPILER_TYPE"] = compiler
            os.environ["NPU_COMPILER_TYPE"] = compiler
        pipe = openvino_genai.WhisperPipeline(
            OV_DIR, device=ov_device, config=cfg)
        _MODELS[("ov", ov_device, compiler)] = pipe
        print(tr("模型加载 %.1fs") % (time.time() - t0))

    audio = decode_audio(path, sampling_rate=16000)
    print(tr("音频时长 %.1f 秒") % (audio.shape[0] / 16000.0))

    print(tr("转写中（%s）...") % ov_device)
    t0 = time.time()
    # 注：openvino_genai 2026.3.1 传 WhisperGenerationConfig 对象会抛
    # "ValueError: vector too long"（绑定层问题，位置/关键字传法都炸），
    # 故只能用 kwargs 字段形式下发解码参数。
    res = pipe.generate(audio, **OV_DECODE)
    texts = [t.strip() for t in list(res.texts) if t.strip()]
    print(tr("转写完成"))
    print(tr("转写用时 %.1fs") % (time.time() - t0))
    return " ".join(texts)


def _fallback_order(rt):
    """失败后的退回顺序：同一份模型的另一套设备优先（npu 与 igpu 共用
    ov 模型，退回去不用重新下载模型），最后才是 CPU（ct2 模型可能没下
    过，得先下）。cpu 没有再可退的。"""
    if rt == "npu":
        return ["igpu", "cpu"]
    if rt in ("cuda", "igpu"):
        return ["cpu"]
    return []


def transcribe(path):
    if not os.path.exists(path):
        print(tr("[错误] 文件不存在: %s") % path)
        return 1
    stem, _ = os.path.splitext(path)
    out_txt = stem + ".txt"
    print(tr("输入: %s") % path)
    print(tr("输出: %s") % out_txt)

    state = load_state()
    rt = state.get("runtime", "cpu")
    print("Runtime: %s" % rt)

    def _once():
        if rt == "cuda" and probe()["cuda_libs"]:
            return _run_ct2("cuda", "float16", path)
        if rt == "npu":
            return _run_ov("NPU", path)
        if rt == "igpu":
            return _run_ov("GPU", path)
        return _run_ct2("cpu", "int8", path)

    text = None
    try:
        text = _once()
    except Exception as e:
        # 模型加载/运行失败。先区分"依赖坏了"（venv 被外来工具重盖、
        # 安装中断等）：import 检查只在这里才做，坏了就强制重装并
        # 原路重试一次；否则按设备问题依次退回（见 _fallback_order）。
        print("[ERROR] %s" % e)
        if not _import_ok(_imports_for(rt)):
            print(tr("[setup] 依赖缺失/损坏，重新安装 ..."))
            if _pip_install_into(VENV_PY, *DEP_PKGS, reinstall=True) == 0:
                try:
                    text = _once()
                except Exception as e2:
                    print(tr("[ERROR] 重试仍失败: %s") % e2)
        # NPU 特有：编译/执行被驱动拒（blob 不认）时，换另一套 NPU 编译器
        # 重编一次——插件编译器和驱动编译器产出的 blob，驱动认哪个不一样。
        if text is None and rt == "npu":
            print(tr("换 NPU 编译器（%s）重新编译并重试 ..."
                     ) % NPU_RETRY_COMPILER)
            try:
                text = _run_ov("NPU", path, compiler=NPU_RETRY_COMPILER)
            except Exception as e2:
                print(tr("[ERROR] 换编译器后仍失败: %s") % e2)

        if text is None:
            for alt in _fallback_order(rt):
                kind = RUNTIME_MODEL[alt]
                if alt in ("npu", "igpu") and not probe()[alt]:
                    continue
                if kind not in models_present():
                    print(tr("退回 %s：先下载它用的模型 ...") % alt)
                    if not download_model(kind):
                        print(tr("[ERROR] 模型下载失败，%s 不可用。") % alt)
                        continue
                print(tr("退回 %s 重试...") % alt)
                try:
                    text = (_run_ov("GPU", path) if alt == "igpu"
                            else _run_ct2("cpu", "int8", path))
                    break
                except Exception as e2:
                    print(tr("[ERROR] %s 也失败: %s") % (alt, e2))
            if text is None:
                return 1

    out_real = _write_text_robust(path, out_txt, text)
    print(tr("文本已保存: %s") % out_real)
    return 0


# ================================================================ 入口
def norm_path(raw):
    p = raw.strip()
    if len(p) >= 2:
        # 成对剥离首尾引号：直引号首尾是同一字符；中文弯引号则是
        # 左右各半（“ ” / ‘ ’，输入法或聊天软件复制的路径常见），
        # 必须按“对”匹配——按同字符匹配永远凑不齐，等于没剥。
        if p[0] == p[-1] and p[0] in ('"', "'"):
            p = p[1:-1]
        elif p[:1] + p[-1:] in ("\u201c\u201d", "\u2018\u2019"):
            p = p[1:-1]
    return p.strip()


def _bg_dep_check(state):
    """后台依赖校验（interactive_loop 启动时开线程跑）：按配置的
    runtime 只 import 对应引擎（1-3 秒）。正常则无声；失败打印提示，
    由用户决定输 /fix 立即修复，或无视（转写失败时也会自动自愈）。
    不在本线程读输入/写状态，避免与主循环抢键盘、抢配置文件。"""
    rt = state.get("runtime", "cpu")
    if _import_ok(_imports_for(rt)):
        return
    print(tr("\n[提示] 后台依赖检查失败：{} 方案所需的库缺失或损坏。"
            "输入 /fix 立即修复（重装）；直接转写也会自动修复。").format(rt))
    print("> ", end="", flush=True)


def fix_deps():
    """手动修复依赖（/fix）：按当前 runtime 校验 → 强制重装 → 复检。"""
    rt = load_state().get("runtime", "cpu")
    print(tr("[setup] 检查 {} 方案依赖 ...").format(rt))
    if _import_ok(_imports_for(rt)):
        print(tr("[setup] 依赖正常，无需修复。"))
        return 0
    print(tr("[setup] 重装依赖 ..."))
    if _pip_install_into(VENV_PY, *DEP_PKGS, reinstall=True) != 0:
        print(tr("[ERROR] 依赖修复失败，请检查网络后重试。"))
        return 1
    if _import_ok(_imports_for(rt)):
        print(tr("[setup] 依赖修复完成。"))
        return 0
    print(tr("[ERROR] 重装后仍异常，请尝试删除 .venv 后重新运行。"))
    return 1


def interactive_loop(state):
    """省去回车那一步：直接粘贴路径 / 输入 /s。"""
    threading.Thread(target=_bg_dep_check, args=(state,),
                     daemon=True).start()
    print("=" * 56)
    print(tr("  粘贴音频/视频文件（支持拖拽）→ 转写"))
    print(tr("  /s → 调整 Runtime      /x → 退出"))
    print(tr("  当前 Runtime: %s") % state.get("runtime", "cpu"))
    print("=" * 56)
    while True:
        try:
            raw = input("> ").strip()
        except EOFError:
            return 0
        if not raw:
            continue
        if raw[:1] == "/":
            cmd = raw[1:].lower()
            if cmd == "s":
                if runtime_menu(state) != 0:
                    return 1
                print(tr("当前 Runtime: %s") % load_state().get("runtime"))
                continue
            elif cmd == "fix":
                fix_deps()
                continue
            elif cmd == "x":
                return 0
            else:
                print(tr("无效命令: %s（可用: /s 调整Runtime, /fix 修复依赖, "
                        "/x 退出）") % raw)
                continue
        rc = transcribe(norm_path(raw))
        if rc != 0:
            print(tr("（转写失败，可重试其它文件或 /s 调整 runtime）"))


def main():
    # 0) python / venv 前提
    py = ensure_venv()
    in_venv = (os.path.abspath(sys.executable).lower()
               == os.path.abspath(py).lower())
    if not in_venv:
        rc = subprocess.call([py, os.path.abspath(__file__)] + sys.argv[1:])
        return rc

    state = load_state()

    # 1) 无配置文件 / 机器码不对 / 没有任何模型 → 都按"首次运行"处理
    sig = _machine_sig()
    first = (not state.get("first_run_done")
             or state.get("host_sig") != sig
             or not models_present())
    if state.get("host_sig") != sig and state.get("first_run_done"):
        print(tr("[setup] 检测到配置来自其它机器，重新进入配置..."))
    if first:
        rc = first_run_setup(state)
        if rc != 0:
            return rc
        state = load_state()
        state["host_sig"] = sig   # first_run_setup 未写指纹，归属在此落账
        save_state(state)

    # 2) 命令行给定了文件 → 逐个转写后退出（bat "file.mp3" 直转模式）
    if sys.argv[1:]:
        rc = 0
        for a in sys.argv[1:]:
            if transcribe(norm_path(a)) != 0:
                rc = 1
        return rc

    return interactive_loop(state)


if __name__ == "__main__":
    sys.exit(main())
