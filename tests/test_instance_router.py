import json
import subprocess
import sys
import time
import unittest
import uuid

from PySide6.QtCore import QEventLoop
from PySide6.QtWidgets import QApplication

from instance_router import (
    LocalApplicationRouter,
    MAX_REQUEST_BYTES,
    decode_request,
    encode_request,
)


_APP = QApplication.instance() or QApplication([])


class InstanceRouterTests(unittest.TestCase):
    def test_request_encoding_round_trips_paths_with_spaces(self):
        request = {
            "action": "open",
            "path": "C:/Data/Run 2/specimen video.tif",
        }

        encoded = encode_request(request)

        self.assertTrue(encoded.endswith(b"\n"))
        self.assertEqual(decode_request(encoded.rstrip(b"\n")), request)

    def test_non_object_request_is_rejected(self):
        with self.assertRaises(ValueError):
            decode_request(json.dumps(["new_window"]).encode("utf-8"))

    def test_oversized_request_is_rejected(self):
        with self.assertRaises(ValueError):
            encode_request({"action": "open", "path": "x" * MAX_REQUEST_BYTES})

    def test_request_is_delivered_over_local_application_server(self):
        server_name = f"TykockiLab.BRAID.Test.{uuid.uuid4().hex}"
        router = LocalApplicationRouter(server_name)
        received = []
        router.request_received.connect(received.append)
        self.addCleanup(router.close)
        self.assertTrue(router.listen())

        request = {"action": "open", "path": "C:/Data/Run 3/video.tif"}
        sender_script = (
            "import json, sys; "
            "from PySide6.QtCore import QCoreApplication; "
            "from instance_router import LocalApplicationRouter; "
            "app = QCoreApplication(sys.argv); "
            "sent = LocalApplicationRouter.forward_request("
            "sys.argv[1], json.loads(sys.argv[2])); "
            "raise SystemExit(0 if sent else 1)"
        )
        sender = subprocess.Popen(
            [sys.executable, "-c", sender_script, server_name, json.dumps(request)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        deadline = time.monotonic() + 5
        while (sender.poll() is None or not received) and time.monotonic() < deadline:
            _APP.processEvents(QEventLoop.AllEvents, 50)

        stdout, stderr = sender.communicate(timeout=1)
        self.assertEqual(sender.returncode, 0, stdout + stderr)
        self.assertEqual(received, [request])


if __name__ == "__main__":
    unittest.main()
