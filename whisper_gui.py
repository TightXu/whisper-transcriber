# -*- coding: utf-8 -*-
"""
whisper_gui.py — Whisper Transcriber 图形界面（pywebview + WebView2）

与 CLI（whisper_transcribe.py，同目录）的关系：
  - 本文件不改核心：直接 import 同目录的 whisper_transcribe，核心逻辑全部复用
    （转写/退回阶梯 core.transcribe()、硬件探测 core.probe()、
    就绪度 core.runtime_ready()、状态 core.load_state()/save_state()、
    模型会话缓存 core._MODELS（窗口常驻 = 进程常驻，连转免重载）。
  - 本文件不改核心代码；进度通过"stdout 桥"拿：进程内 sys.stdout/stderr
    换成 Tee，core 的全部 print() 进日志台；行首两空格的段文本同时视为
    实时识别输出（对应 _run_ct2 的逐段打印）。
  - 唯一进程内替代 core 子进程调用的地方：模型下载（core.download_model
    走 subprocess 无进度；GUI 用同款端点/Xet 逻辑进程内跑 + tqdm 捕获，
    出真进度条）。
  - models / cache / .venv 就在同目录，两种模式共用同一份：状态文件也只有
    一份（.venv/runtimes.json），所以 GUI 与命令行看到的配置始终一致。

用法：
  whisper_transcribe.bat            → 本 GUI（bat 引导环境后拉起；窗口加载
                                      期间按任意键可取消，改走命令行模式）
  whisper_transcribe.bat --cli      → 命令行交互（no-gui 同义）
  whisper_transcribe.bat "a.mp3"    → GUI 打开并把文件入队
  whisper_gui.py --selftest [文件]  → 无窗口自检（打印 status JSON）
"""
import os
import sys
import json
import time
import queue
import shutil
import subprocess
import threading
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import whisper_transcribe as core   # noqa: E402  (同目录副本，字节级同源)


# ================================================================ i18n
# GUI 自有文案：键 → (中文, 英文)。核心代码的文案仍走 core.tr，
# 经 stdout 桥进日志台，天然跟随同一语言。
_I18N = {
    "app.title":        ("Whisper 转写", "Whisper Transcriber"),
    "app.subtitle":     ("本地语音转文字，文件不出本机", "Local speech-to-text, files never leave this machine"),
    "badge.rt":         ("当前方案", "Runtime"),
    "badge.busy":       ("运行中", "busy"),
    "badge.idle":       ("就绪", "idle"),
    "drop.title":       ("拖入音频 / 视频文件", "Drop audio / video files here"),
    "drop.hint":        ("也可以拖入文件，或点“浏览”从资源管理器选择",
                         "You can also drop files, or click Browse to pick from Explorer"),
    "drop.browse":      ("浏览", "Browse"),
    "drop.path.ph":     ("粘贴文件路径后回车", "Paste a file path and press Enter"),
    "queue.title":      ("队列", "Queue"),
    "queue.empty":      ("还没有文件。拖入或选择要转写的文件。", "Nothing yet. Drop or pick files to transcribe."),
    "st.queued":        ("排队中", "queued"),
    "st.active":        ("转写中", "transcribing"),
    "st.done":          ("完成", "done"),
    "st.failed":        ("失败", "failed"),
    "act.open":         ("打开文件夹", "Show in folder"),
    "act.opentxt":      ("打开文本", "Open transcript"),
    "act.saveas":       ("另存为", "Save as"),
    "act.retry":        ("重试", "Retry"),
    "act.remove":       ("移除", "Remove"),
    "live.title":       ("实时识别", "Live transcript"),
    "live.empty":       ("转写开始后，识别文字逐段出现在这里（OpenVINO 方案整段完成后一次给出）。",
                        "Recognized text appears here segment by segment (OpenVINO runtimes return it all at once)."),
    "live.clear":       ("清空", "Clear"),
    "log.title":        ("日志", "Log"),
    "settings.title":   ("设置", "Settings"),
    "settings.rt.h":    ("转写方案（Runtime）", "Transcription runtime"),
    "settings.outdir.h": ("转写默认保存位置", "Default transcript location"),
    "settings.outdir.def": ("源文件同目录（默认）", "Next to the source file (default)"),
    "settings.outdir.change": ("更改", "Change"),
    "settings.outdir.reset": ("恢复默认", "Reset"),
    "settings.fix":     ("修复依赖", "Repair dependencies"),
    "settings.fix.hint": ("转写总失败时点这里：校验并重装当前方案的依赖。",
                         "Use this if transcription keeps failing: verify and reinstall the current runtime's dependencies."),
    "settings.about":   ("图形界面与命令行共用同一份模型、缓存与配置。",
                         "The GUI and the CLI share the same models, caches and configuration."),
    "cur.rt":           ("当前", "current"),
    "rec.rt":           ("推荐", "recommended"),
    "rt.ready":         ("就绪，可直接使用", "Ready"),
    "rt.switch":        ("切换", "Switch"),
    "rt.need_dl":       ("需下载模型(%.1fGB)", "Needs model download (%.1fGB)"),
    "wiz.title":        ("首次使用", "First run"),
    "wiz.detecting":    ("正在检测硬件 ...", "Detecting hardware ..."),
    "wiz.pick":         ("选择一个转写方案（按速度排序，推荐项排最前）",
                        "Pick a runtime (ordered by speed, recommended first)"),
    "wiz.preparing":    ("正在准备所选方案（可能需要下载模型）...",
                        "Preparing the selected runtime (model download may take a while) ..."),
    "wiz.done":         ("准备完成，进入主界面。", "All set. Opening the main window."),
    "wiz.cuda.libs":    ("安装 CUDA 运行库（约 1.3GB）...", "Installing CUDA runtime libraries (~1.3GB) ..."),
    "dl.model":         ("下载模型", "Downloading model"),
    "dl.unknown":       ("下载中 ...", "Downloading ..."),
    "busy.transcribe":  ("转写中", "Transcribing"),
    "busy.switch":      ("切换方案", "Switching runtime"),
    "busy.fix":         ("修复依赖", "Repairing dependencies"),
    "busy.wizard":      ("准备方案", "Preparing runtime"),
    "confirm.switch":   ("切换方案可能需要下载模型或重新编译，期间不能转写。继续？",
                        "Switching may download a model or recompile. Transcription is paused meanwhile. Continue?"),
    "toast.noop":       ("没有可添加的文件。", "Nothing to add."),
    "toast.badpath":    ("路径不存在：", "Path does not exist: "),
    "toast.added":      ("已入队：", "Queued: "),
    "toast.saved":      ("已保存到", "Saved to"),
    "toast.busy":       ("有任务进行中，请稍候。", "A task is running, please wait."),
    "toast.copied":     ("已复制", "Copied"),
    "err.gui":          ("界面出错了，详细信息见 logs/gui-error.log",
                        "UI error, see logs/gui-error.log for details"),
    "open.nofile":      ("没找到输出文本。", "No transcript file found."),
    "selftest.ok":      ("自检通过", "selftest OK"),
    # GUI 层日志文案（与 core 文案重合的走 core.tr，不在此重复）
    "log.prepfailed":   ("[setup] 方案准备失败，未切换。",
                         "[setup] Runtime preparation failed; not switched."),
    "log.firstwiz":     ("[setup] 首次使用，进入配置向导 ...",
                         "[setup] First run, entering setup ..."),
    "log.guiloading":   ("[setup] 正在加载窗口界面 ... 按任意键取消（改用命令行模式）",
                         "[setup] Loading the window interface ... press any key to cancel and use the console"),
    "log.guicancel":    ("[setup] 已取消，改用命令行模式。",
                         "[setup] Cancelled; using the console instead."),
    "log.outdir.moved": ("[gui] 已按默认保存位置存放: %s",
                         "[gui] Saved to the default location: %s"),
    "log.outdir.movefail": ("[gui] 移动到默认保存位置失败，文本留在原位:",
                            "[gui] Could not move to the default location; "
                            "transcript left in place:"),
    "log.savefail":     ("[gui] 另存失败: %s",
                         "[gui] Save-as failed: %s"),
}


def g(key):
    zh, en = _I18N[key]
    return zh if core.LANG == "zh" else en


def i18n_pack():
    return {k: (v[0] if core.LANG == "zh" else v[1]) for k, v in _I18N.items()}


# ================================================================ stdout 桥
class _Tee:
    """替换 sys.stdout / sys.stderr：所有 print()（core 的进度、pip 输出、
    异常 traceback）进 GUI 日志台；有真控制台时同时回显。
    行首两个空格 = CT2 逐段识别文本（_run_ct2 的 "  %s"），标记为段。"""

    def __init__(self, real, tag):
        self._real = real          # 可能为 None（pythonw 无控制台）
        self._tag = tag            # 'out' | 'err'
        self._buf = ""

    def write(self, s):
        try:
            log_write(s, stream=self._tag)
        except Exception:
            pass
        if self._real is not None:
            try:
                self._real.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self):
        if self._real is not None:
            try:
                self._real.flush()
            except Exception:
                pass

    def isatty(self):
        return False

    # tqdm /个别库会探测这些属性
    @property
    def encoding(self):
        return "utf-8"

    def fileno(self):
        raise OSError("no fileno")


_LOG_LOCK = threading.Lock()
_LOG = []            # [{"seq","t","s","seg","err"}]
_LOG_SEQ = [0]
_LOG_CAP = 4000


def log_write(s, stream="out"):
    """写入日志台。按行拆分（print 分多次 write：文本 + 换行），
    攒够一行或收到换行才落账；段判定在行级做。"""
    if not s:
        return
    with _LOG_LOCK:
        parts = s.split("\n")
        buf = _LOG_BUF[0] + parts[0]
        if len(parts) == 1:
            _LOG_BUF[0] = buf
            return                      # 没有换行，继续攒
        _LOG_BUF[0] = parts[-1]
        _emit_lines([buf] + parts[1:-1], stream)


_LOG_BUF = [""]


def _emit_lines(lines, stream):
    for line in lines:
        _LOG_SEQ[0] += 1
        _LOG.append({
            "seq": _LOG_SEQ[0],
            "t": round(time.time(), 2),
            "s": line,
            "seg": line.startswith("  ") and bool(line.strip()),
            "err": stream == "err",
        })
    if len(_LOG) > _LOG_CAP:
        del _LOG[: len(_LOG) - _LOG_CAP]


def logs_after(seq):
    with _LOG_LOCK:
        return [l for l in _LOG if l["seq"] > seq]


def install_tee():
    if isinstance(sys.stdout, _Tee):     # 幂等：重复安装不二次包装
        return
    out, err = sys.stdout, sys.stderr
    sys.stdout = _Tee(out, "out")
    sys.stderr = _Tee(err, "err")


# ================================================================ 界面状态
_UI_LOCK = threading.Lock()
UI = {
    "view": "main",          # wizard | main
    "probe_ready": False,
    "runtime": None,
    "models": [],
    "busy": None,            # {"kind","detail"}
    "download": None,        # {"label","done","total","frac"}
    "queue": [],             # 每项 {"path","name","state","error","out","start_ts","end_ts","dur"}
    "runtimes": None,        # 探测完成后填充
    "out_dir": None,         # 转写默认保存位置（缓存自 runtimes.json，
                             # 设置变更时同步；轮询快照不读盘）
}
_pending_files = []          # 启动参数带来的文件（向导完成后入队）


def ui_snapshot():
    with _UI_LOCK:
        snap = dict(UI)
        snap["queue"] = [dict(q) for q in UI["queue"]]
        snap["lang"] = core.LANG
        # 注：不塞 log_seq —— 前端日志游标走自己的 logSeq 计数，
        # 而它每条日志都变，会把前端的"状态未变不重渲染"守卫打穿。
        snap["out_dir"] = UI.get("out_dir")
        return snap


def ui_set(**kw):
    with _UI_LOCK:
        UI.update(kw)


def queue_find(path):
    with _UI_LOCK:
        for q in UI["queue"]:
            if q["path"].lower() == path.lower():
                return q
    return None


def queue_update(path, **kw):
    with _UI_LOCK:
        for q in UI["queue"]:
            if q["path"].lower() == path.lower():
                q.update(kw)
                return q
    return None


def refresh_runtime_info(background=True):
    """重算 runtime 就绪度表（probe 有进程级缓存，首次约 1 秒）。"""
    def _calc():
        try:
            p = core.probe()
            rec = core.best_runtime(p)
            state = core.load_state()
            cur = state.get("runtime", "cpu")
            rts = []
            for rt in core.PRIORITY:
                ok, need_compile, note = core.runtime_ready(rt, p)
                rts.append({
                    "id": rt,
                    "label": core._rt_label(rt),
                    "available": bool(core._rt_enabled(p)[rt]),
                    "reason": core._rt_disabled_reason(rt) if not core._rt_enabled(p)[rt] else "",
                    "have_model": ok,
                    "need_compile": need_compile,
                    "note": note,
                    # 提示用的估算值，须与 core._rt_action 里的数字保持
                    # 一致（core 冻结，无法以常量共享）
                    "model_gb": 2.9 if core.RUNTIME_MODEL[rt] == "ct2" else 1.6,
                    "cur": rt == cur,
                    "rec": rt == rec,
                })
            ui_set(runtimes=rts, runtime=cur,
                   models=core.models_present(), probe_ready=True)
        except Exception:
            log_write("[gui] refresh_runtime_info failed:\n%s"
                      % traceback.format_exc(), stream="err")
    if background:
        threading.Thread(target=_calc, daemon=True).start()
    else:
        _calc()


# ================================================================ 下载（进程内，带进度）
_DL_LOCK = threading.Lock()
_DL = {"done": 0, "total": 0}     # 字节（部分 bar 无 total，退化为 None）


def _download_progress(frac, done, total):
    ui_set(download={"label": g("dl.model"), "done": done,
                     "total": total, "frac": frac})


def _make_capture_bar():
    """tqdm 子类：构造时即禁用渲染（进度条字符流只会污染日志台），
    把各文件 bar 的增量汇总成总进度。hub 的 snapshot_download 给每个
    文件建一个 bar（unit=B），也有个别计数型 bar —— 一并累加，
    总量未知时置 None（界面转圈）。"""
    from tqdm import tqdm

    class _Cap(tqdm):
        def __init__(self, *a, **k):
            try:
                total = int(k.get("total") or 0)
            except Exception:
                total = 0
            k["disable"] = True          # 必须在构造前禁用，bar 一笔都不画
            super().__init__(*a, **k)
            with _DL_LOCK:
                _DL["total"] += total

        def update(self, n=1):
            super().update(n)
            with _DL_LOCK:
                _DL["done"] += int(n or 0)
                done, total = _DL["done"], _DL["total"]
            frac = (min(done / total, 1.0) if total > 0 else None)
            _download_progress(frac, done, total if total > 0 else None)

    return _Cap


def download_model_inproc(kind, log=print):
    """GUI 版模型下载：与 core.download_model 同款端点/Xet 策略，
    进程内执行并出进度。返回是否成功。"""
    ep = core.hf_endpoint()
    mirror = "hf-mirror" in ep
    # HF_ENDPOINT / HF_HUB_DISABLE_XET 必须在 huggingface_hub 首次 import
    # 前进环境；同进程此前没 import 过（core 走子进程），这里设了就生效。
    os.environ["HF_ENDPOINT"] = ep
    if mirror:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    with _DL_LOCK:
        _DL.update(done=0, total=0)
    _download_progress(None, 0, None)     # 先把进度卡片立起来（转圈态）
    if kind == "ct2":
        repo, dst = "Systran/faster-whisper-large-v3", core.CT2_DIR
    else:
        repo, dst = "OpenVINO/whisper-large-v3-int8-ov", core.OV_DIR
    log(core.tr("[setup] 使用 HF 端点: %s") % ep)
    log(core.tr("[setup] 下载 %s -> models/%s（约 1.6~3GB）...")
        % (repo, os.path.basename(dst)))
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, local_dir=dst, tqdm_class=_make_capture_bar())
    except Exception:
        log("%s\n%s" % (core.tr("[ERROR] 模型下载失败。"),
                        traceback.format_exc()))
        return False
    log(core.tr("[setup] 下载完成。"))
    return True


# ================================================================ 后台工作线程
_OPS = queue.Queue()
_OP_PEND = {"drain": False}


def post_op(kind, **kw):
    _OPS.put({"kind": kind, **kw})


def post_drain():
    """把"清空转写队列"压入操作通道（去重：同一时刻只需一个 drain）。"""
    with _UI_LOCK:
        if _OP_PEND["drain"]:
            return
        _OP_PEND["drain"] = True
    post_op("drain")


def _find_output_txt(path):
    """转写成功后定位输出文本：源同目录同名 .txt；源目录不可写时
    core 会落到 transcripts/。都找不到返回 None。"""
    stem, _ = os.path.splitext(path)
    cand = stem + ".txt"
    if os.path.exists(cand):
        return cand
    fb = os.path.join(core.TRANSCRIPTS_DIR,
                      os.path.splitext(os.path.basename(path))[0] + ".txt")
    return fb if os.path.exists(fb) else None


def _relocate_output(path):
    """设置了"转写默认保存位置"时，把 core 落盘的 txt 挪过去
    （key: txt_out_dir，存 runtimes.json，core 不认识也不受影响）。
    挪不动（目录删了/被占用）就留在原位并记日志，不视为转写失败。"""
    out_dir = core.load_state().get("txt_out_dir")
    src = _find_output_txt(path)
    if not out_dir or not src:
        return src
    try:
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, os.path.basename(src))
        shutil.move(src, dst)
        log_write(g("log.outdir.moved") % dst + "\n")
        return dst
    except Exception:
        log_write(g("log.outdir.movefail") + "\n%s\n"
                  % traceback.format_exc(), stream="err")
        return src


def _prepare_runtime(rt, log=print):
    """对应 core._prepare_runtime，下载走 GUI 进程内版（带进度）。
    busy 标签由调用方（worker）设置，这里不覆盖。"""
    p = core.probe()
    if rt == "cuda" and not p["cuda_libs"]:
        log(g("wiz.cuda.libs"))
        if not core.ensure_cuda_libs():
            return False
    if core.RUNTIME_MODEL[rt] not in core.models_present():
        if not download_model_inproc(core.RUNTIME_MODEL[rt], log=log):
            return False
    return True


def _op_choose_runtime(op):
    rt = op["rt"]
    wizard = op.get("wizard")
    log = lambda s: log_write(s + "\n")
    if not _prepare_runtime(rt, log=log):
        ui_set(busy=None, download=None)
        log(g("log.prepfailed"))
        return
    state = core.load_state()
    state.update({"models": core.models_present(), "runtime": rt})
    if wizard:
        state["first_run_done"] = True
        state["host_sig"] = core._machine_sig()
    core.save_state(state)
    core._MODELS.clear()          # 与 core.runtime_menu 同语义：即时释放旧实例
    ui_set(busy=None, download=None, runtime=rt, view="main")
    log(core.tr("[setup] Runtime 已设为 %s") % rt)
    refresh_runtime_info(background=True)
    _flush_pending_files()


def _op_fix(op):
    log_write("\n")
    core.fix_deps()
    ui_set(busy=None)
    refresh_runtime_info(background=True)


def _op_drain(op):
    while True:
        with _UI_LOCK:
            nxt = next((q for q in UI["queue"]
                        if q["state"] == "queued"), None)
            if nxt is None:
                # 去重标志与"查无待转"同临界区复位：此刻新入队的文件
                # 要么已被上面的 next() 看到，要么 post_drain 会看到
                # False 而补发 op —— 两条路都不丢文件（原先复位在
                # 循环外单独加锁，间隙里入队的文件会被去重掉且无兜底）。
                _OP_PEND["drain"] = False
                break
        path = nxt["path"]
        queue_update(path, state="active", start_ts=time.time(),
                     end_ts=None, error=None, out=None)
        ui_set(busy={"kind": "transcribe", "detail": os.path.basename(path)})
        t0 = time.time()
        log_write("\n" + "=" * 52 + "\n")
        try:
            rc = core.transcribe(path)
        except Exception:
            log_write(traceback.format_exc(), stream="err")
            rc = 1
        dur = round(time.time() - t0, 1)
        if rc == 0:
            queue_update(path, state="done", end_ts=time.time(), dur=dur,
                         out=_relocate_output(path))
        else:
            queue_update(path, state="failed", end_ts=time.time(), dur=dur,
                         error="see log")
        ui_set(busy=None)


def worker():
    while True:
        op = _OPS.get()
        try:
            if op["kind"] == "choose":
                ui_set(busy={"kind": "switch" if UI["view"] == "main" else "wizard",
                             "detail": op["rt"]})
                _op_choose_runtime(op)
            elif op["kind"] == "fix":
                ui_set(busy={"kind": "fix", "detail": ""})
                _op_fix(op)
            elif op["kind"] == "drain":
                _op_drain(op)
        except Exception:
            log_write("[gui] worker op failed:\n%s" % traceback.format_exc(),
                      stream="err")
            ui_set(busy=None, download=None)
            with _UI_LOCK:
                _OP_PEND["drain"] = False


def add_paths(paths):
    """入队（去重、验存在）。返回 (added_count, skipped_list)。"""
    added, skipped = 0, []
    for raw in paths:
        if not raw:
            continue
        p = core.norm_path(raw)
        if not p:
            continue
        if not os.path.exists(p):
            skipped.append(p)
            continue
        if queue_find(p):
            continue
        with _UI_LOCK:
            UI["queue"].append({
                "path": p, "name": os.path.basename(p),
                "state": "queued", "error": None, "out": None,
                "start_ts": None, "end_ts": None, "dur": None,
            })
        added += 1
    if added and UI["view"] == "main":
        post_drain()
    return added, skipped


def _flush_pending_files():
    """向导完成/主界面就绪后，把启动参数带来的文件压进队列。"""
    global _pending_files
    if _pending_files and UI["view"] == "main":
        files, _pending_files = _pending_files, []
        add_paths(files)


# ================================================================ js_api
class Api:
    """pywebview 暴露给前端的方法（前端经 pywebview.api.xxx 调用）。"""

    def boot(self):
        return {
            "i18n": i18n_pack(),
            "lang": core.LANG,
            "status": ui_snapshot(),
            "logs": logs_after(0)[:400],
        }

    def status(self):
        return ui_snapshot()

    def logs(self, since):
        return logs_after(int(since or 0))

    def pick_files(self):
        import webview
        paths = window.create_file_dialog(
            webview.FileDialog.OPEN, allow_multiple=True,
            file_types=(
                "Media files (*.mp3;*.wav;*.m4a;*.flac;*.ogg;*.opus;*.aac;"
                "*.mp4;*.mkv;*.mov;*.avi;*.webm;*.wmv;*.ts;*.flv)",
                "All files (*.*)",
            ))
        return list(paths or [])

    def bind_drop(self):
        """前端 renderMain() 重建拖放区后调用（见 _bind_drop）。"""
        _bind_drop()

    def add_paths(self, paths):
        if UI["view"] != "main":
            return {"added": 0, "skipped": []}
        added, skipped = add_paths(list(paths or []))
        return {"added": added, "skipped": skipped}

    def remove(self, path):
        with _UI_LOCK:
            UI["queue"] = [q for q in UI["queue"]
                           if not (q["path"].lower() == path.lower()
                                   and q["state"] == "queued")]

    def retry(self, path):
        if queue_find(path):
            queue_update(path, state="queued", error=None)
            post_drain()

    def choose_runtime(self, rt):
        if rt not in core.PRIORITY:
            return
        post_op("choose", rt=rt, wizard=(UI["view"] == "wizard"))

    def refresh_runtimes(self):
        """同步重算就绪度并返回最新快照，前端拿到后可立即刷新弹层。"""
        refresh_runtime_info(background=False)
        return ui_snapshot()

    def fix(self):
        if UI.get("busy"):
            return
        post_op("fix")

    def open_folder(self, path):
        try:
            subprocess.Popen(["explorer", "/select,", path])
        except Exception:
            log_write("[gui] open_folder failed: %s\n" % path, stream="err")

    def open_txt(self, path):
        out = _find_output_txt(path)
        if out:
            subprocess.Popen(["explorer", "/select,", out])
        return out

    def choose_outdir(self):
        """设置-转写默认保存位置：选目录并立刻持久化。取消返回 None。"""
        import webview
        r = window.create_file_dialog(webview.FileDialog.FOLDER)
        if not r:
            return None
        d = r[0] if isinstance(r, (list, tuple)) else r
        state = core.load_state()
        state["txt_out_dir"] = d
        core.save_state(state)
        ui_set(out_dir=d)
        return d

    def reset_outdir(self):
        state = core.load_state()
        state.pop("txt_out_dir", None)
        core.save_state(state)
        ui_set(out_dir=None)

    def save_transcript_as(self, path):
        """队列"另存为"：保存对话框选目标，把已有 txt 复制过去。
        取消返回 None；成功返回新路径。"""
        import webview
        src = _find_output_txt(path)
        if not src:
            return None
        r = window.create_file_dialog(
            webview.FileDialog.SAVE,
            directory=os.path.dirname(src),
            save_filename=os.path.basename(src))
        if not r:
            return None
        dst = r[0] if isinstance(r, (list, tuple)) else r
        if not str(dst).lower().endswith(".txt"):
            dst += ".txt"
        try:
            shutil.copyfile(src, dst)
        except Exception:
            log_write("%s\n%s" % (g("log.savefail") % dst,
                                  traceback.format_exc()), stream="err")
            return None
        return dst


window = None       # webview.create_window 的实例（start 前赋值）

_READY = {"ok": False}          # 页面加载完成（window.events.loaded）
_CANCELLED = {"yes": False}     # 用户在显示窗口前按了键
_CANCEL_MAX_WAIT = 60.0         # 等加载的上限；超时就按"不取消"处理
_CANCEL_GRACE = 1.0             # 就绪后再等这么久，给用户决定


def _key_pending():
    """控制台里是否已有按键待读。没有控制台（pythonw 拉起）时恒 False。"""
    try:
        import msvcrt
        return bool(msvcrt.kbhit())
    except Exception:
        return False


def _watch_cancel(win):
    """显示窗口前给一次反悔机会：从加载开始到就绪后 _CANCEL_GRACE 秒内
    按任意键即取消——不显示窗口，转命令行模式（窗口是隐藏创建
    的，所以取消时不会先闪一下）。
    键盘检测轮询 50ms；无控制台时 _key_pending 恒 False，等于不提供
    该功能，不会误取消。"""
    end = time.time() + _CANCEL_MAX_WAIT
    pressed = False
    while not _READY["ok"] and time.time() < end:
        pressed = _key_pending() or pressed
        time.sleep(0.05)
    end = time.time() + _CANCEL_GRACE
    while time.time() < end:
        pressed = _key_pending() or pressed
        time.sleep(0.05)
    if pressed:
        _CANCELLED["yes"] = True
        try:
            win.destroy()
        except Exception:
            pass
        return
    try:
        win.show()
    except Exception:
        log_write("[gui] window.show failed:\n%s"
                  % traceback.format_exc(), stream="err")


def _on_drop(e):
    try:
        files = e["dataTransfer"].get("files", [])
        paths = [f["pywebviewFullPath"] for f in files
                 if "pywebviewFullPath" in f]
        if paths:
            add_paths(paths)   # Python 侧直接入队
    except Exception:
        log_write("[gui] drop handler failed:\n%s"
                  % traceback.format_exc(), stream="err")


def _bind_drop():
    """把拖放处理绑到当前 #drop 元素。由前端 renderMain() 每次重建
    拖放区后经 api.bind_drop() 触发 —— 元素每次都是新建的，不会叠加
    重复监听；也覆盖了"首跑向导 → 主界面"重建 DOM 后旧绑定失效的
    场景（早期版本在 loaded 里只绑一次，首跑后拖放就是死的）。
    在 js_api 线程里执行；loaded 回调线程里禁止碰 DOM —— 渲染器
    未就绪时会死锁 UI 线程（见 README 技术要点）。"""
    if window is None:
        return
    try:
        el = window.dom.get_element("#drop")
        if el is not None:
            el.on("drop", _on_drop)
    except Exception:
        log_write("[gui] drag-drop setup skipped:\n%s"
                  % traceback.format_exc(), stream="err")


# ================================================================ 前端页面
HTML = r"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
:root{
  --bg:#101418; --panel:#171d24; --panel2:#1d252e; --line:#2a333d;
  --fg:#e8edf2; --dim:#8b98a5; --accent:#4f8cff; --ok:#3fb96f;
  --warn:#e0a93e; --err:#e05d5d; --mono:Consolas,"Cascadia Mono",monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{
  background:var(--bg);color:var(--fg);overflow:hidden;
  font:14px/1.5 "Segoe UI","Microsoft YaHei",system-ui,sans-serif;
  display:flex;flex-direction:column;
}
header{
  display:flex;align-items:center;gap:12px;padding:12px 18px;
  border-bottom:1px solid var(--line);flex:none;
}
header .logo{width:26px;height:26px;border-radius:7px;flex:none;
  background:linear-gradient(135deg,#4f8cff,#7a5cff);
  display:flex;align-items:center;justify-content:center;font-size:15px}
header h1{font-size:16px;font-weight:600}
header .sub{color:var(--dim);font-size:12px;margin-left:2px}
header .grow{flex:1}
.badge{display:flex;align-items:center;gap:7px;background:var(--panel);
  border:1px solid var(--line);border-radius:999px;padding:5px 12px;
  font-size:12.5px;cursor:pointer;user-select:none}
.badge:hover{border-color:var(--accent)}
.badge .dot{width:8px;height:8px;border-radius:50%;background:var(--ok)}
.badge.busy .dot{background:var(--warn);animation:pulse 1s infinite}
@keyframes pulse{50%{opacity:.35}}
.iconbtn{background:none;border:none;color:var(--dim);cursor:pointer;
  font-size:17px;padding:6px 8px;border-radius:8px}
.iconbtn:hover{background:var(--panel);color:var(--fg)}
main{flex:1;overflow-y:auto;padding:16px 18px 10px}
.wrap{max-width:880px;margin:0 auto;display:flex;flex-direction:column;gap:14px}

/* ---- 拖放区 ---- */
.drop{
  border:2px dashed var(--line);border-radius:14px;padding:26px 16px;
  text-align:center;transition:border-color .15s,background .15s;cursor:pointer}
.drop:hover,.drop.over{border-color:var(--accent);background:rgba(79,140,255,.06)}
.drop .t{font-size:15.5px;font-weight:600;margin-bottom:4px}
.drop .h{color:var(--dim);font-size:12.5px}
.pathrow{display:flex;gap:8px;margin-top:10px}
.pathrow input{
  flex:1;background:var(--panel);border:1px solid var(--line);color:var(--fg);
  border-radius:9px;padding:9px 12px;font-size:13px;outline:none}
.pathrow input:focus{border-color:var(--accent)}
.btn{
  background:var(--panel2);color:var(--fg);border:1px solid var(--line);
  border-radius:9px;padding:9px 14px;font-size:13px;cursor:pointer;white-space:nowrap}
.btn:hover{border-color:var(--accent)}
.btn.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.btn.primary:hover{filter:brightness(1.1)}
.btn:disabled{opacity:.45;cursor:default}
.btn.mini{padding:4px 10px;font-size:12px;border-radius:7px}

/* ---- 队列 ---- */
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  overflow:hidden}
.card .hd{display:flex;align-items:center;padding:10px 14px;gap:10px;
  border-bottom:1px solid var(--line);font-weight:600;font-size:13.5px}
.card .hd .grow{flex:1}
.card .bd{padding:6px 8px}
.empty{color:var(--dim);text-align:center;padding:16px 6px;font-size:13px}
.qrow{display:flex;align-items:center;gap:10px;padding:9px 8px;border-radius:9px}
.qrow:hover{background:var(--panel2)}
.qrow .st{width:10px;height:10px;border-radius:50%;flex:none;background:var(--dim)}
.qrow.active .st{background:var(--warn);animation:pulse 1s infinite}
.qrow.done .st{background:var(--ok)}
.qrow.failed .st{background:var(--err)}
.qrow .nm{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;font-size:13.5px}
.qrow .meta{color:var(--dim);font-size:12px;flex:none}
.qrow .acts{display:flex;gap:6px;flex:none}
.spin{display:inline-block;width:11px;height:11px;border-radius:50%;
  border:2px solid var(--warn);border-top-color:transparent;
  animation:rot .8s linear infinite;flex:none}
@keyframes rot{to{transform:rotate(360deg)}}

/* ---- 实时文本 / 日志 ---- */
.live,.logbox{
  font-family:var(--mono);font-size:12.5px;white-space:pre-wrap;
  word-break:break-all;padding:10px 14px;max-height:220px;overflow-y:auto}
.logbox{max-height:260px;color:var(--dim);background:#0b0e12}
.logbox .err{color:var(--err)}
.collapsible .hd{cursor:pointer;user-select:none}
.collapsible .hd .chev{transition:transform .15s;color:var(--dim)}
.collapsible.open .hd .chev{transform:rotate(90deg)}
.collapsible .bd{display:none}
.collapsible.open .bd{display:block}

/* ---- 下载进度 / 忙碌 ---- */
.progress{display:flex;align-items:center;gap:10px;padding:10px 14px}
.progress .lbl{font-size:12.5px;color:var(--fg);flex:none}
.progress .bar{flex:1;height:8px;border-radius:99px;background:var(--panel2);
  overflow:hidden}
.progress .bar i{display:block;height:100%;background:var(--accent);
  width:0%;transition:width .3s}
.progress .pct{font-size:12px;color:var(--dim);flex:none;min-width:80px;
  text-align:right;font-family:var(--mono)}
.busybar{display:flex;align-items:center;gap:10px;padding:9px 14px;
  font-size:13px;color:var(--warn)}
.stage{color:var(--dim);font-size:12px;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;flex:1}

/* ---- runtime 卡片（设置/向导共用） ---- */
.rts{display:flex;flex-direction:column;gap:8px;padding:10px}
.rt{display:flex;align-items:center;gap:12px;background:var(--panel2);
  border:1px solid var(--line);border-radius:11px;padding:12px 14px;cursor:pointer}
.rt:hover{border-color:var(--accent)}
.rt.off{opacity:.45;cursor:default}
.rt.off:hover{border-color:var(--line)}
.rt .rname{font-weight:600;font-size:14px}
.rt .rtags{display:flex;gap:6px}
.tag{font-size:11px;padding:2px 8px;border-radius:99px}
.tag.cur{background:rgba(79,140,255,.18);color:#8db6ff}
.tag.rec{background:rgba(63,185,111,.16);color:#6fd598}
.rt .rsub{color:var(--dim);font-size:12px;margin-top:2px}
.rt .grow{flex:1}
.rt .sel{width:16px;height:16px;border-radius:50%;border:2px solid var(--line);flex:none}
.rt.on .sel{border-color:var(--accent);
  background:radial-gradient(circle,var(--accent) 45%,transparent 50%)}

/* ---- 向导 ---- */
.wizhead{padding:34px 0 6px;text-align:center}
.wizhead .t{font-size:21px;font-weight:700}
.wizhead .s{color:var(--dim);margin-top:6px;font-size:13.5px}
.steps{display:flex;gap:8px;justify-content:center;margin:14px 0 4px}
.steps i{width:26px;height:4px;border-radius:99px;background:var(--line)}
.steps i.on{background:var(--accent)}
.detect{display:flex;justify-content:center;padding:44px 0;color:var(--dim);
  gap:10px;align-items:center}

/* ---- 弹层 / 提示 ---- */
.overlay{position:fixed;inset:0;background:rgba(6,9,12,.62);display:none;
  align-items:center;justify-content:center;z-index:50}
.overlay.show{display:flex}
.modal{background:var(--panel);border:1px solid var(--line);border-radius:14px;
  width:min(620px,92vw);max-height:86vh;display:flex;flex-direction:column}
.modal .mhd{display:flex;align-items:center;padding:14px 18px;
  border-bottom:1px solid var(--line);font-weight:600}
.modal .mhd .grow{flex:1}
.modal .mbd{overflow-y:auto;padding:6px 0 12px}
.toast{
  position:fixed;left:50%;bottom:26px;transform:translateX(-50%) translateY(80px);
  background:#26303b;border:1px solid var(--line);color:var(--fg);
  padding:10px 18px;border-radius:10px;font-size:13px;transition:transform .25s;
  z-index:60;max-width:80vw;white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis}
.toast.show{transform:translateX(-50%) translateY(0)}
::-webkit-scrollbar{width:9px;height:9px}
::-webkit-scrollbar-thumb{background:var(--line);border-radius:99px}
::-webkit-scrollbar-thumb:hover{background:#3a4652}
</style></head>
<body>

<header>
  <div class="logo">♪</div>
  <div>
    <h1 id="hTitle"></h1>
    <div class="sub" id="hSub"></div>
  </div>
  <div class="grow"></div>
  <div class="badge" id="rtBadge" onclick="openSettings()">
    <span class="dot"></span><span id="rtBadgeTxt">…</span>
  </div>
  <button class="iconbtn" id="gearBtn" onclick="openSettings()" title="">⚙</button>
</header>

<main><div class="wrap" id="mainWrap"></div></main>

<div class="overlay" id="overlay">
  <div class="modal">
    <div class="mhd"><span id="mTitle"></span><div class="grow"></div>
      <button class="iconbtn" onclick="closeModal()">✕</button></div>
    <div class="mbd" id="mBody"></div>
  </div>
</div>
<div class="toast" id="toast"></div>

<script>
"use strict";
var T={}, LANG='zh', S=null, logSeq=0, liveLines=0, lastStage='';
var $=function(id){return document.getElementById(id)};
function h(s){return String(s).replace(/[&<>"]/g,function(c){
  return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
function esc(s){return String(s==null?'':s);}
function fmtMB(b){if(!b&&b!==0)return '';
  return b>=1073741824?(b/1073741824).toFixed(2)+' GB'
  :b>=1048576?(b/1048576).toFixed(1)+' MB':Math.round(b/1024)+' KB';}
function fmtDur(s){if(s==null)return '';s=Math.round(s);
  return s>=60?Math.floor(s/60)+'m'+('0'+s%60).slice(-2)+'s':s+'s';}
function basename(p){p=String(p||'');var i=Math.max(p.lastIndexOf('/'),p.lastIndexOf('\\'));
  return i<0?p:p.slice(i+1);}
function toast(msg){var t=$('toast');t.textContent=msg;t.classList.add('show');
  clearTimeout(t._tm);t._tm=setTimeout(function(){t.classList.remove('show')},2600);}

/* ---------- 视图 ---------- */
function renderMain(){
  var w=$('mainWrap');
  w.innerHTML =
  '<div class="drop" id="drop">'+
    '<div class="t">'+T['drop.title']+'</div>'+
    '<div class="h">'+T['drop.hint']+'</div>'+
    '<div class="pathrow" onclick="event.stopPropagation()">'+
      '<input id="pathIn" placeholder="'+h(T['drop.path.ph'])+
        '" onkeydown="if(event.key===\'Enter\')addFromInput()">'+
      '<button class="btn" onclick="pickFiles()">'+T['drop.browse']+'</button>'+
    '</div>'+
  '</div>'+
  '<div id="busySlot"></div>'+
  '<div class="card"><div class="hd"><span>'+T['queue.title']+'</span>'+
    '<span class="grow"></span></div><div class="bd" id="queueBox"></div></div>'+
  '<div class="card collapsible open" id="liveCard">'+
    '<div class="hd" onclick="toggleColl(this)"><span class="chev">▶</span>'+
      '<span>'+T['live.title']+'</span><span class="grow"></span>'+
      '<button class="btn mini" onclick="event.stopPropagation();clearLive()">'+
        T['live.clear']+'</button></div>'+
    '<div class="bd"><div class="live" id="liveBox"></div></div></div>'+
  '<div class="card collapsible" id="logCard">'+
    '<div class="hd" onclick="toggleColl(this)"><span class="chev">▶</span>'+
      '<span>'+T['log.title']+'</span></div>'+
    '<div class="bd"><div class="logbox" id="logBox"></div></div></div>';
  $('drop').onclick=function(){pickFiles();};
  pywebview.api.bind_drop().catch(function(){});  /* 新建的 #drop 重绑拖放（幂等） */
  ['dragenter','dragover'].forEach(function(ev){
    $('drop').addEventListener(ev,function(e){e.preventDefault();
      this.classList.add('over');});});
  ['dragleave','drop'].forEach(function(ev){
    $('drop').addEventListener(ev,function(e){e.preventDefault();
      this.classList.remove('over');});});
  document.addEventListener('dragover',function(e){e.preventDefault();});
  document.addEventListener('drop',function(e){e.preventDefault();});
  renderQueue(); renderBusy();
}
function toggleColl(hd){hd.parentElement.classList.toggle('open');}

function renderQueue(){
  var box=$('queueBox'); if(!box)return;
  var q=(S&&S.queue)||[];
  if(!q.length){box.innerHTML='<div class="empty">'+T['queue.empty']+'</div>';return;}
  var now=Date.now()/1000, out='';
  q.forEach(function(it){
    var st=it.state, meta='';
    if(st==='queued')meta=T['st.queued'];
    else if(st==='active'){meta=T['st.active']+' '+fmtDur(now-it.start_ts);}
    else if(st==='done')meta=T['st.done']+(it.dur?' · '+fmtDur(it.dur):'');
    else meta=T['st.failed']+(it.dur?' · '+fmtDur(it.dur):'');
    var acts='';
    if(st==='done'){
      acts+='<button class="btn mini" onclick="saveAs(\''+h(enc(it.path))+'\')">'+T['act.saveas']+'</button>';
      acts+='<button class="btn mini" onclick="openTxt(\''+h(enc(it.path))+'\')">'+T['act.opentxt']+'</button>';
      acts+='<button class="btn mini" onclick="openFolder(\''+h(enc(it.path))+'\')">'+T['act.open']+'</button>';
    }else if(st==='failed'){
      acts+='<button class="btn mini" onclick="retry(\''+h(enc(it.path))+'\')">'+T['act.retry']+'</button>';
    }else if(st==='queued'){
      acts+='<button class="btn mini" onclick="removeQ(\''+h(enc(it.path))+'\')">'+T['act.remove']+'</button>';
    }
    out+='<div class="qrow '+st+'"><span class="st"></span>'+
      '<span class="nm" title="'+h(it.path)+'">'+h(it.name)+'</span>'+
      '<span class="meta">'+meta+'</span><span class="acts">'+acts+'</span></div>';
  });
  box.innerHTML=out;
}
function enc(p){return String(p).replace(/\\/g,'\\\\').replace(/'/g,"\\'");}

function renderBusy(){
  var slot=$('busySlot'); if(!slot)return;
  var b=S&&S.busy, dl=S&&S.download, out='';
  if(dl){
    var pct=dl.frac!=null?Math.round(dl.frac*100)+'%':'…';
    var bytes=(dl.total?fmtMB(dl.done)+' / '+fmtMB(dl.total):fmtMB(dl.done))||'';
    out+='<div class="card"><div class="progress"><span class="lbl">'+h(dl.label)+
      '</span><div class="bar"><i style="width:'+
      (dl.frac!=null?Math.round(dl.frac*100):0)+'%"></i></div>'+
      '<span class="pct">'+pct+(bytes?' · '+bytes:'')+'</span></div></div>';
  }
  if(b){
    var label=T['busy.'+b.kind]||b.kind;
    var stage=lastStage?('· '+h(lastStage)):'';
    out+='<div class="card"><div class="busybar"><span class="spin"></span>'+
      '<span>'+h(label)+(b.detail?(' — '+h(b.detail)):'')+'</span>'+
      '<span class="stage">'+stage+'</span></div></div>';
  }
  slot.innerHTML=out;
  var badge=$('rtBadge');
  badge.classList.toggle('busy', !!(b||dl));
}

/* ---------- runtime 弹层 / 向导 ---------- */
function rtRows(rts, mode){
  var out='<div class="rts">';
  rts.forEach(function(r){
    var subs=[];
    if(!r.available)subs.push(h(r.reason));
    else{
      if(!r.have_model)subs.push(T['rt.need_dl'].replace('%.1f', r.model_gb.toFixed(1)));
      else if(r.need_compile)subs.push(h(r.note));
      else if(r.note)subs.push(h(r.note));
      if(!subs.length)subs.push(T['rt.ready']);
    }
    var tags='';
    if(r.cur)tags+='<span class="tag cur">'+T['cur.rt']+'</span>';
    if(r.rec)tags+='<span class="tag rec">'+T['rec.rt']+'</span>';
    var on=r.available?(mode==='settings'?r.cur:false):false;
    var click=(r.available&&!(mode==='settings'&&r.cur))?
      ' onclick="chooseRt(\''+r.id+'\')"':'';
    out+='<div class="rt'+(r.available?'':' off')+(on?' on':'')+'"'+click+'>'+
      '<span class="sel"></span>'+
      '<div style="flex:1"><div class="rname">'+h(r.label)+' '+tags+'</div>'+
      '<div class="rsub">'+subs.join(' · ')+'</div></div>'+
      (mode==='settings'&&!r.cur&&r.available?
        '<button class="btn mini">'+T['rt.switch']+'</button>':'')+
      '</div>';
  });
  return out+'</div>';
}
function openSettings(fromRefresh){
  if(S&&S.busy){toast(T['toast.busy']);return;}
  if(!fromRefresh&&(!S||!S.runtimes)){
    /* 就绪度还没算好：先渲染转圈态，算完拿到快照、弹层仍开着就重建。
       fromRefresh 标记防止探测持续失败时重入成刷新循环。 */
    pywebview.api.refresh_runtimes().then(function(st){
      S=st;
      if($('overlay').classList.contains('show'))openSettings(true);
    }).catch(function(){});
  }
  $('mTitle').textContent=T['settings.title'];
  var body=$('mBody');
  body.innerHTML='<div style="padding:4px 18px 0;font-weight:600">'+
    T['settings.rt.h']+'</div>'+
    (S&&S.runtimes?rtRows(S.runtimes,'settings')+
      '<div style="padding:16px 18px 4px;font-weight:600">'+
        T['settings.outdir.h']+'</div>'+
      '<div style="padding:6px 18px 2px;display:flex;gap:8px;align-items:center">'+
        '<span style="flex:1;color:var(--dim);font-size:12.5px;overflow:hidden;'+
          'text-overflow:ellipsis;white-space:nowrap" title="'+
          h(S.out_dir||'')+'">'+
          (S&&S.out_dir?h(S.out_dir):T['settings.outdir.def'])+'</span>'+
        '<button class="btn mini" onclick="changeOutdir()">'+
          T['settings.outdir.change']+'</button>'+
        (S&&S.out_dir?'<button class="btn mini" onclick="resetOutdir()">'+
          T['settings.outdir.reset']+'</button>':'')+
      '</div>'+
      '<div style="padding:14px 18px 4px;display:flex;gap:10px;align-items:center">'+
      '<button class="btn" onclick="doFix()">'+T['settings.fix']+'</button>'+
      '<span style="color:var(--dim);font-size:12px">'+T['settings.fix.hint']+'</span>'+
      '</div><div style="padding:4px 18px 12px;color:var(--dim);font-size:12px">'+
      T['settings.about']+'</div>'
      :'<div class="detect"><span class="spin"></span>'+T['wiz.detecting']+'</div>');
  $('overlay').classList.add('show');
}
function closeModal(){$('overlay').classList.remove('show');}
function chooseRt(id){
  if(S&&S.busy){toast(T['toast.busy']);return;}
  if(S&&S.view==='main'&&!confirm(T['confirm.switch']))return;
  closeModal();
  pywebview.api.choose_runtime(id);
}
function doFix(){closeModal();pywebview.api.fix();}

function renderWizard(){
  var w=$('mainWrap');
  var step=S&&S.runtimes?2:1;
  w.innerHTML='<div class="wizhead"><div class="t">'+T['wiz.title']+'</div>'+
    '<div class="s">'+T['app.subtitle']+'</div></div>'+
    '<div class="steps"><i class="'+(step>=1?'on':'')+'"></i>'+
    '<i class="'+(step>=2?'on':'')+'"></i><i class="'+(step>=3?'on':'')+'"></i></div>'+
    (S&&S.runtimes
      ?'<div style="text-align:center;color:var(--dim);font-size:13px">'+
        T['wiz.pick']+'</div>'+
        '<div class="card"><div class="bd">'+rtRows(S.runtimes,'wizard')+'</div></div>'
      :'<div class="detect"><span class="spin"></span>'+T['wiz.detecting']+'</div>')+
    '<div id="busySlot"></div>';
  renderBusy();
}

/* ---------- 日志 / 实时文本 ---------- */
function addLogLines(lines){
  var lb=$('logBox'), lv=$('liveBox');
  if(!lb)return;
  var atBottom=lb.scrollHeight-lb.scrollTop-lb.clientHeight<40;
  var frag='';
  lines.forEach(function(l){
    frag+='<div'+(l.err?' class="err"':'')+'>'+h(l.s)+'</div>';
    if(l.seg&&lv){
      liveLines++;
      var keep='<div>'+h(l.s.trim())+'</div>';
      lv.insertAdjacentHTML('beforeend',keep);
      if(liveLines>400)lv.removeChild(lv.firstChild);
      lv.scrollTop=lv.scrollHeight;
    }
    if(!l.seg&&l.s&&l.s.trim())lastStage=l.s.trim();
  });
  lb.insertAdjacentHTML('beforeend',frag);
  while(lb.childNodes.length>1500)lb.removeChild(lb.firstChild);
  if(atBottom)lb.scrollTop=lb.scrollHeight;
  renderBusy();
}
function clearLive(){var lv=$('liveBox');if(lv)lv.innerHTML='';liveLines=0;}

/* ---------- 动作 ---------- */
function pickFiles(){
  pywebview.api.pick_files().then(function(ps){
    if(ps&&ps.length)doAdd(ps);});
}
function doAdd(ps){
  pywebview.api.add_paths(ps).then(function(r){
    if(!r)return;
    if(r.added)toast(T['toast.added']+r.added);
    if(r.skipped&&r.skipped.length)toast(T['toast.badpath']+basename(r.skipped[0]));
  });
}
function addFromInput(){
  var inp=$('pathIn');if(!inp||!inp.value.trim())return;
  doAdd([inp.value.trim()]);inp.value='';
}
function openFolder(p){pywebview.api.open_folder(p);}
function openTxt(p){pywebview.api.open_txt(p).then(function(out){
  if(!out)toast(T['open.nofile']);});}
function saveAs(p){
  pywebview.api.save_transcript_as(p).then(function(r){
    if(r)toast(T['toast.saved']+' '+basename(r));
  });
}
function changeOutdir(){
  pywebview.api.choose_outdir().then(function(p){
    if(p){S.out_dir=p;openSettings();}
  });
}
function resetOutdir(){
  pywebview.api.reset_outdir().then(function(){S.out_dir=null;openSettings();});
}
function retry(p){pywebview.api.retry(p);}
function removeQ(p){pywebview.api.remove(p);}

/* ---------- 轮询 ---------- */
var _lastStat='';
function applyStatus(st){
  var j=JSON.stringify(st),tick=false;
  (st.queue||[]).forEach(function(q){if(q.state==='active')tick=true;});
  /* 状态没变就不重渲染（active 项例外：耗时计时每拍都要走）。
     避免长队列下按钮被整块重建吞掉点击、向导整页闪刷。 */
  if(j===_lastStat&&!tick)return;
  _lastStat=j;
  var prevView=S?S.view:null;
  S=st;
  if(st.view!==prevView){
    if(st.view==='wizard')renderWizard();else renderMain();
    logSeq=0;var lb=$('logBox');if(lb)lb.innerHTML='';
  }
  if(st.view==='main'){
    renderQueue();renderBusy();
    var bd=$('rtBadgeTxt');
    var rtName=(st.runtimes&&st.runtime)?
      (st.runtimes.find(function(r){return r.id===st.runtime;})||{}).label:st.runtime;
    bd.textContent=rtName||'…';
    $('gearBtn').title=T['settings.title'];
  }else if(st.view==='wizard'){
    renderWizard();
  }
}
function poll(){
  pywebview.api.status().then(applyStatus).catch(function(){});
  pywebview.api.logs(logSeq).then(function(ls){
    if(ls&&ls.length){logSeq=ls[ls.length-1].seq;addLogLines(ls);}
  }).catch(function(){});
}

/* ---------- 启动 ----------
   必须等 pywebviewready：页面脚本执行时桥多半还没注入，
   直接调 pywebview.api.* 会全部失败（曾表现为"整个界面点了没反应"）。 */
var _booted=false;
function boot(){
  if(_booted)return; _booted=true;
  pywebview.api.boot().then(function(b){
    T=b.i18n;LANG=b.lang;S=b.status;
    $('hTitle').textContent=T['app.title'];
    $('hSub').textContent=T['app.subtitle'];
    if(S.view==='wizard'){renderWizard();
      if(!S.runtimes)pywebview.api.refresh_runtimes();}
    else renderMain();
    if(b.logs&&b.logs.length){logSeq=b.logs[b.logs.length-1].seq;addLogLines(b.logs);}
    setInterval(poll,700);
  }).catch(function(e){document.body.textContent='boot failed: '+e;});
}
if(window.pywebview&&window.pywebview.api){boot();}
else{
  window.addEventListener('pywebviewready',boot);
  setTimeout(function(){if(!_booted)document.body.textContent=
    'pywebview bridge not ready (10s)';},10000);
}
</script>
</body></html>"""


# ================================================================ 入口
_REEXEC_FLAG = "WT_GUI_REEXEC"


def _in_venv():
    """当前解释器是否已是 venv 的解释器。
    core.VENV_PY 固定是 python.exe，而 GUI 由 bat 用 pythonw.exe 拉起 ——
    必须按"所在目录 + 主名"比较；整串比较会让 pythonw 永远被判"不在
    venv"而反复重启自己（2026-09-16 曾因此产生数千 pythonw 进程的事故，
    此函数就是那条防线，改动前先想清楚）。"""
    exe = os.path.abspath(sys.executable)
    vdir = os.path.dirname(os.path.abspath(core.VENV_PY)).lower()
    stem = os.path.splitext(os.path.basename(exe))[0].lower()
    return os.path.dirname(exe).lower() == vdir and stem in ("python",
                                                             "pythonw")


def _reexec_into_venv(argv):
    """对齐 core.main() 的引导：不在 venv 里就换 venv 解释器重启自己。
    三重防循环：
      ① _in_venv 按"目录+主名"判定，pythonw 也算在 venv 内（正常路径
         由此直接 return，根本不会中转）；
      ② WT_GUI_REEXEC 只允许中转一跳 —— 子进程见到它绝不再中转，
         无论判定逻辑将来怎么写坏，进程数的数学上限是 2；
      ③ os.execv 原地替换进程，不留下排队等待的父进程。"""
    if _in_venv():
        return
    if os.environ.get(_REEXEC_FLAG):
        # 已经中转过一跳还走到这里 = 判定出了问题。绝不再起进程，
        # 留在当前解释器继续跑（后续 import 失败会进 gui-error.log，
        # 而不是复制自己）。
        print("[gui] re-exec guard hit; continuing with current interpreter")
        return
    exe = core.VENV_PY
    if sys.executable.lower().endswith("pythonw.exe"):
        cand = os.path.join(os.path.dirname(core.VENV_PY), "pythonw.exe")
        if os.path.exists(cand):
            exe = cand
    try:
        os.environ[_REEXEC_FLAG] = "1"    # execv 继承本进程环境块
        os.execv(exe, [exe, os.path.abspath(__file__)] + list(argv))
    except OSError:
        # execv 失败（罕见）：退回子进程方式，环境里同样带一跳守卫
        env = dict(os.environ)
        env[_REEXEC_FLAG] = "1"
        rc = subprocess.call([exe, os.path.abspath(__file__)] + list(argv),
                             env=env)
        sys.exit(rc)


_CREATE_NO_WINDOW = 0x08000000   # subprocess.CREATE_NO_WINDOW（win 专用）


def _harden_subprocess_if_windowless():
    """pythonw（无控制台）模式下的两件事，都在这一层做，core 保持原样：
    ① 控制台子进程（reg/curl/pip ...）从 GUI 进程拉起时会各自弹出一个
      控制台窗口 —— 屏幕上周期性闪黑框的来源。重定向 std 挡不住它
      （窗口分配与句柄无关），必须 CREATE_NO_WINDOW。
    ② 无效标准句柄的继承会偶发写失败：未显式指定 std 的一律 DEVNULL。
    core 的成败判定全靠返回码，自己的阶段输出走 stdout 桥，不损失信息。
    有真控制台（调试）时不启用。覆盖 call/run —— check_call/check_output
    在 stdlib 内部转调这两个入口，连带生效。"""
    windowless = (sys.executable.lower().endswith("pythonw.exe")
                  or sys.stdout is None or sys.stderr is None)
    if not windowless:
        return
    _orig_call, _orig_run = subprocess.call, subprocess.run

    def _call(*a, **k):
        k["creationflags"] = (k.get("creationflags") or 0) | _CREATE_NO_WINDOW
        if k.get("stdout") is None:
            k["stdout"] = subprocess.DEVNULL
        if k.get("stderr") is None:
            k["stderr"] = subprocess.DEVNULL
        return _orig_call(*a, **k)

    def _run(*a, **k):
        k["creationflags"] = (k.get("creationflags") or 0) | _CREATE_NO_WINDOW
        if k.get("stdout") is None and not k.get("capture_output"):
            k["stdout"] = subprocess.DEVNULL
        if k.get("stderr") is None and not k.get("capture_output"):
            k["stderr"] = subprocess.DEVNULL
        return _orig_run(*a, **k)

    subprocess.call = _call
    subprocess.run = _run


def _strip_gui_args(argv):
    files, flags = [], set()
    for a in argv:
        if a.startswith("--"):
            flags.add(a)
        else:
            files.append(a)
    return files, flags


def _selftest(files):
    """无窗口自检：状态快照 + 指定文件走真实转写链路。"""
    install_tee()
    print("[selftest] lang=%s view=%s" % (core.LANG, UI["view"]))
    refresh_runtime_info(background=False)
    snap = ui_snapshot()
    print("[selftest] status=%s" % json.dumps(
        {k: snap[k] for k in ("view", "runtime", "models", "probe_ready",
                              "busy", "runtimes")}, ensure_ascii=False, indent=1))
    ok = True
    if files:
        add_paths(files)
        _op_drain({"kind": "drain"})
        for q in UI["queue"]:
            print("[selftest] %s -> %s (%s)" % (q["name"], q["state"], q["out"]))
            ok = ok and q["state"] == "done"
    print("[selftest] %s" % g("selftest.ok"))
    return 0 if ok else 1


def gui_main():
    argv = sys.argv[1:]
    if "--cli" in argv or "no-gui" in argv:
        # core 不认识这些开关（会当文件名转写报错），在这里剥掉再交接。
        # bat 的控制台分支也依赖这里完成剥离。
        sys.argv = [sys.argv[0]] + [a for a in argv
                                    if a not in ("--cli", "no-gui")]
        return core.main()

    _reexec_into_venv(argv)
    files, flags = _strip_gui_args(argv)

    global _pending_files
    _pending_files = files

    _harden_subprocess_if_windowless()   # 先于 Tee：靠 sys.stdout is None 判无控制台
    install_tee()

    state = core.load_state()
    sig = core._machine_sig()
    need_wizard = (not state.get("first_run_done")
                   or state.get("host_sig") != sig
                   or not core.models_present())
    ui_set(view="wizard" if need_wizard else "main",
           runtime=state.get("runtime", "cpu"), models=core.models_present(),
           out_dir=state.get("txt_out_dir"))
    if need_wizard:
        banner = (core.tr("[setup] 检测到配置来自其它机器，重新进入配置...")
                  if (state.get("host_sig") != sig
                      and state.get("first_run_done"))
                  else g("log.firstwiz"))
        log_write(banner + "\n")

    if "--selftest" in flags:
        return _selftest(files)

    threading.Thread(target=worker, daemon=True).start()
    refresh_runtime_info(background=True)

    import webview
    api = Api()
    global window
    print(g("log.guiloading"), flush=True)   # print 才经 Tee 回显到控制台（log_write 只进日志面板）
    window = webview.create_window(
        g("app.title"), html=HTML, js_api=api, hidden=True,
        width=940, height=700, min_size=(780, 560))

    def on_loaded():
        # 拖放绑定不在这里做：renderMain() 重建拖放区后由前端调
        # api.bind_drop()（见 _bind_drop），首跑向导完成后的新 #drop
        # 也能绑上。loaded 回调里只做非 DOM 工作。
        _READY["ok"] = True
        if UI["view"] == "main":
            _flush_pending_files()

    window.events.loaded += on_loaded
    threading.Thread(target=_watch_cancel, args=(window,),
                     daemon=True).start()

    webview.start()
    if _CANCELLED["yes"]:      # 用户按了键：这里才转命令行模式
        _CANCELLED["yes"] = False
        print(g("log.guicancel"), flush=True)
        return core.main()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(gui_main())
    except SystemExit:
        raise
    except Exception:
        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(os.path.join(LOGS_DIR, "gui-error.log"), "w",
                  encoding="utf-8") as f:
            f.write(traceback.format_exc())
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0, traceback.format_exc()[-800:], "whisper_gui", 0x10)
        except Exception:
            pass
        sys.exit(1)
