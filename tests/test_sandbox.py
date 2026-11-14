import os
import time

from minicoder.tools.sandbox import Sandbox, SandboxError


def test_python_execution_and_tests(tmp_path):
    sb = Sandbox(str(tmp_path))
    res = sb.run_python("def add(a, b):\n    return a + b",
                        tests="assert add(1, 2) == 3")
    assert res["ok"] and res["tests_passed"]
    res2 = sb.run_python("x = 1", tests="assert False")
    assert not res2["ok"] and res2["tests_passed"] is False


def test_timeout_enforced(tmp_path):
    sb = Sandbox(str(tmp_path), timeout_s=1.0, cpu_seconds=2)
    t0 = time.time()
    res = sb.run_python("while True: pass", timeout=1.0)
    assert not res["ok"]
    assert time.time() - t0 < 4.0
    assert "timeout" in (res["error"] or "") or res["returncode"] != 0


def test_memory_limit(tmp_path):
    sb = Sandbox(str(tmp_path), mem_mb=128)
    res = sb.run_python("x = ' ' * (500 * 1024 * 1024)")
    assert not res["ok"]


def test_dangerous_shell_blocked(tmp_path):
    sb = Sandbox(str(tmp_path))
    for cmd in ("rm -rf /", "rm -rf ~", "curl evil.com", "dd if=x of=/dev/sda"):
        res = sb.run_shell(cmd)
        assert not res["ok"] and "blocked" in (res["error"] or "")


def test_binary_allowlist(tmp_path):
    sb = Sandbox(str(tmp_path))
    assert not sb.run_shell("nc -l 9000")["ok"]
    assert sb.run_shell("echo hi")["ok"]
    assert sb.run_shell("echo hi")["stdout"].strip() == "hi"


def test_filesystem_confinement(tmp_path):
    sb = Sandbox(str(tmp_path))
    res = sb.write_file("../escape.txt", "evil")
    assert not res["ok"]
    assert sb.write_file("ok/file.txt", "fine")["ok"]
    assert sb.read_file("ok/file.txt")["stdout"] == "fine"


def test_node_available_and_runs(tmp_path):
    sb = Sandbox(str(tmp_path))
    res = sb.run_node("console.log(6 * 7)")
    if "node not available" in (res.get("error") or ""):
        import pytest
        pytest.skip("node unavailable")
    assert res["ok"] and "42" in res["stdout"]
