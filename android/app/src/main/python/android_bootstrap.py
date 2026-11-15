"""android_bootstrap — loader baked into the APK.

The MiniCoder brain itself is NOT in the APK: the app downloads the repo
zip straight from GitHub at first launch, extracts it to the app's private
storage, and this module then imports it from there and starts the
dashboard server on localhost.  After that download, no network is needed.
"""
import sys
import threading
import traceback

_started = False
_error = ""


def start(src_dir: str, data_dir: str, port: int = 8000) -> str:
    """Start MiniCoder's dashboard server in a background thread.

    src_dir  – directory containing the downloaded `minicoder/` package
    data_dir – writable app dir for the brain's SQLite memory
    """
    global _started, _error
    if _started:
        return "already running"
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    def _run():
        global _error
        try:
            from minicoder.brain import Brain
            from minicoder.config import BrainConfig
            from minicoder.dashboard.server import serve
            brain = Brain(BrainConfig(base_dir=data_dir))
            serve(brain, host="127.0.0.1", port=port)
        except Exception:                                    # noqa: BLE001
            _error = traceback.format_exc()
            print(_error, file=sys.stderr)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    _started = True
    return "starting"


def status() -> str:
    return _error or ("running" if _started else "not started")
