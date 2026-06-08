from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 8.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.1)
    return False


@unittest.skipIf(
    os.name != "nt" or not hasattr(signal, "CTRL_BREAK_EVENT"),
    "Windows console control event test",
)
class MonitorShutdownTests(unittest.TestCase):
    def test_control_event_stops_monitor(self) -> None:
        port = _free_port()
        proc = subprocess.Popen(
            [
                sys.executable,
                "-X",
                "utf8",
                "-u",
                "monitor",
                "--provider",
                "deepseek",
                "--model",
                "deepseek-v4-pro",
                "--api-key",
                "dummy-key-for-shutdown-test",
                "--port",
                str(port),
            ],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        try:
            self.assertTrue(_wait_for_port(port), "monitor did not listen in time")
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            stdout, stderr = proc.communicate(timeout=8)
            self.assertEqual(proc.returncode, 0, stdout + stderr)
            self.assertIn("Token Monitor 已启动", stdout)
            self.assertIn("正在关闭", stdout)
        finally:
            if proc.poll() is None:
                proc.kill()
