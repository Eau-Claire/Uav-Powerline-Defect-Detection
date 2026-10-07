"""Check orchestration without starting real Modal or local server processes."""
import io
import unittest
from unittest.mock import MagicMock, patch

import run_demo


class RunnerTests(unittest.TestCase):
    def test_dev_url_forwarded_and_children_stopped(self):
        modal = MagicMock()
        modal.stdout = io.StringIO("Created web https://example--drone-yolo11s-baseline-v5-web-dev.modal.run\n")
        modal.poll.return_value = None
        web = MagicMock()
        web.wait.return_value = 0
        web.poll.return_value = 0
        with patch("run_demo.sys.argv", ["run_demo.py"]), patch("run_demo.socket.socket"), \
                patch("run_demo.subprocess.Popen", side_effect=[modal, web]) as spawn:
            self.assertEqual(run_demo.main(), 0)
        self.assertEqual(spawn.call_args_list[1].kwargs["env"]["MODAL_DETECT_URL"],
                         "https://example--drone-yolo11s-baseline-v5-web-dev.modal.run/detect")
        self.assertIn("serve", spawn.call_args_list[0].args[0])
        modal.send_signal.assert_called_once()

    def test_modal_failure_does_not_start_web(self):
        modal = MagicMock()
        modal.stdout = io.StringIO("Authentication failed\n")
        modal.poll.return_value = None
        with patch("run_demo.sys.argv", ["run_demo.py"]), patch("run_demo.socket.socket"), \
                patch("run_demo.subprocess.Popen", return_value=modal) as spawn:
            self.assertEqual(run_demo.main(), 1)
        self.assertEqual(spawn.call_count, 1)


if __name__ == "__main__":
    unittest.main()
