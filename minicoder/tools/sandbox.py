"""Sandboxed tool executor: Python, Node, shell, filesystem, git.

Safety model (explicit permission boundaries):
* every child process is confined to ``root`` (cwd), env scrubbed, limited to
  ``cpu_seconds`` CPU, ``mem_mb`` address space and ``timeout_s`` wall clock;
* shell commands must start with an allow-listed binary and must not match a
  dangerous-pattern list;
* filesystem helpers refuse paths escaping ``root``.

This is a pragmatic sandbox for a local assistant — NOT a hardened security
boundary; stronger isolation (containers, seccomp) is documented in README.
"""
from __future__ import annotations

import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import time
from typing import Optional


class SandboxError(Exception):
    pass


ALLOWED_BINARIES = {
    "python3", "python", "node", "git", "ls", "cat", "echo", "grep", "head",
    "tail", "wc", "find", "true", "false", "printf", "sed", "awk", "sort",
    "uniq", "diff", "mkdir", "cp", "mv", "rm", "touch",
}

_DANGEROUS = [
    re.compile(r, re.IGNORECASE) for r in [
        r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f?|-[a-zA-Z]*f[a-zA-Z]*r)\b",   # rm -rf
        r"\bmkfs\b", r"\bdd\b\s+.*of=/dev", r":\(\)\s*{",             # fork bomb
        r">\s*/dev/sd", r"\bshutdown\b", r"\breboot\b", r"\bkill\s+-9\s+1\b",
        r"git\s+push\b.*--force", r"git\s+reset\s+--hard\s+origin/",
        r"\bcurl\b|\bwget\b",          # no network from sandbox
        r"\bnc\b|\bncat\b|\bsocat\b",
    ]
]


def _limits(cpu_seconds: int, mem_mb: int):
    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        mem_bytes = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        # group-kill friendly
        os.setsid()
    return apply


class Sandbox:
    def __init__(self, root: str, timeout_s: float = 5.0,
                 cpu_seconds: int = 5, mem_mb: int = 256):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.timeout_s = timeout_s
        self.cpu_seconds = cpu_seconds
        self.mem_mb = mem_mb
        self.calls = 0

    # ------------------------------------------------------------ execution
    def _scrub_env(self) -> dict:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": self.root,
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "GIT_CONFIG_GLOBAL": os.path.join(self.root, ".gitconfig"),
        }
        return env

    def _exec(self, argv: list[str], timeout: Optional[float] = None,
              mem_mb: Optional[int] = None) -> dict:
        timeout = timeout or self.timeout_s
        mem = mem_mb or self.mem_mb
        self.calls += 1
        started = time.time()
        try:
            proc = subprocess.run(
                argv, cwd=self.root, env=self._scrub_env(), capture_output=True,
                text=True, timeout=timeout,
                preexec_fn=_limits(self.cpu_seconds, mem),
            )
            dur = time.time() - started
            return {
                "ok": proc.returncode == 0,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
                "returncode": proc.returncode,
                "error": None if proc.returncode == 0
                else (proc.stderr.strip().splitlines() or [f"exit {proc.returncode}"])[-1],
                "duration_s": round(dur, 3),
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "stderr": "", "returncode": -9,
                    "error": f"timeout after {timeout}s (sandbox limit)", "duration_s": timeout}

    # --------------------------------------------------------------- python
    def run_python(self, code: str, tests: Optional[str] = None,
                   timeout: Optional[float] = None) -> dict:
        """Run Python code; optionally append test asserts.  Returns
        ``tests_passed`` when tests were given."""
        full = code
        if tests:
            full = (code.rstrip() + "\n\n# --- verification tests ---\n"
                    + tests.strip() + "\n")
        path = os.path.join(self.root, "_snippet.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(full)
        res = self._exec([sys.executable, "-I", "_snippet.py"], timeout=timeout)
        if tests is not None:
            res["tests_passed"] = bool(res["ok"])
        _safe_unlink(path)
        return res

    def run_node(self, code: str, timeout: Optional[float] = None) -> dict:
        if shutil.which("node") is None:
            return {"ok": False, "error": "node not available", "stdout": "", "stderr": ""}
        path = os.path.join(self.root, "_snippet.js")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(code)
        # V8 reserves large virtual address space (CodeRange) — allow it more
        # virtual memory while CPU/time limits stay strict.
        res = self._exec(["node", "--max-old-space-size=192", "_snippet.js"],
                         timeout=timeout, mem_mb=max(self.mem_mb, 3072))
        _safe_unlink(path)
        return res

    # ---------------------------------------------------------------- shell
    def run_shell(self, cmd: str, timeout: Optional[float] = None) -> dict:
        cmd = cmd.strip()
        if not cmd:
            return {"ok": False, "error": "empty command", "stdout": "", "stderr": ""}
        for pat in _DANGEROUS:
            if pat.search(cmd):
                return {"ok": False, "error": f"blocked by sandbox policy: {cmd}",
                        "stdout": "", "stderr": ""}
        binary = cmd.split()[0]
        if binary not in ALLOWED_BINARIES:
            return {"ok": False, "error": f"binary not permitted: {binary}",
                    "stdout": "", "stderr": ""}
        return self._exec(["/bin/sh", "-c", cmd], timeout=timeout)

    # ---------------------------------------------------------------- files
    def _confined(self, path: str) -> str:
        full = os.path.abspath(os.path.join(self.root, path))
        if not (full == self.root or full.startswith(self.root + os.sep)):
            raise SandboxError(f"path escapes sandbox: {path}")
        return full

    def write_file(self, path: str, content: str) -> dict:
        try:
            full = self._confined(path)
            os.makedirs(os.path.dirname(full) or self.root, exist_ok=True)
            with open(full, "w", encoding="utf-8") as fh:
                fh.write(content)
            return {"ok": True, "stdout": path}
        except SandboxError as e:
            return {"ok": False, "error": str(e), "stdout": ""}

    def read_file(self, path: str) -> dict:
        try:
            full = self._confined(path)
            with open(full, "r", encoding="utf-8") as fh:
                return {"ok": True, "stdout": fh.read()}
        except (SandboxError, OSError) as e:
            return {"ok": False, "error": str(e), "stdout": ""}

    # ------------------------------------------------------------------ git
    def git(self, *args: str, cwd: Optional[str] = None,
            timeout: Optional[float] = None) -> dict:
        """Git is allowed read/inspect + local commits, never force-push."""
        blocked = {"push --force", "push -f", "clean -fd", "reset --hard origin"}
        joined = " ".join(args)
        if any(b in joined for b in blocked):
            return {"ok": False, "error": f"git operation blocked: {joined}",
                    "stdout": "", "stderr": ""}
        argv = ["git", "-C", os.path.abspath(cwd) if cwd else self.root, *args]
        return self._exec(argv, timeout=timeout)

    # ---------------------------------------------------------------- stats
    def stats(self) -> dict:
        return {"calls": self.calls, "root": self.root,
                "timeout_s": self.timeout_s, "mem_mb": self.mem_mb}


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
