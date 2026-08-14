import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import main


REPO_ROOT = Path(__file__).resolve().parents[1]


class MainEntrypointTests(unittest.TestCase):
    def test_bootstrap_calls_freeze_support_before_application_runner(self):
        calls = []

        with patch(
                "main.multiprocessing.freeze_support",
                side_effect=lambda: calls.append("freeze_support"),
        ):
            result = main.bootstrap(lambda: calls.append("application") or 17)

        self.assertEqual(calls, ["freeze_support", "application"])
        self.assertEqual(result, 17)

    def test_importing_main_does_not_import_the_gui(self):
        script = (
            "import sys; import main; "
            "raise SystemExit(1 if 'window' in sys.modules or 'PySide6' in sys.modules else 0)"
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            check=False,
        )

        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
