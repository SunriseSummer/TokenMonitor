from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monitor.config import MonitorConfig, PROVIDERS


class MonitorConfigTests(unittest.TestCase):
    def test_minimax_provider_resolves_openai_compatible_endpoint(self) -> None:
        config = MonitorConfig(upstream_provider="minimax")

        self.assertIn("minimax", PROVIDERS)
        self.assertEqual(
            config.resolve_endpoint(),
            "https://api.minimaxi.com/v1/chat/completions",
        )


if __name__ == "__main__":
    unittest.main()
