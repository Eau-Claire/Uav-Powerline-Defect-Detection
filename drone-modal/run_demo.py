"""Run Modal dev and local camera web together. Ctrl+C stops both."""
import argparse
import os
from pathlib import Path
import queue
import re
import signal
import socket
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parent
URL_PATTERN = re.compile(r"https://[a-zA-Z0-9-]+\.modal\.run")
ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def main():
    parser = argparse.ArgumentParser(description="Start Modal CPU dev + local web; Ctrl+C stops both. Cloud CPU usage may be billed.")
    parser.parse_args()
    # Fail before starting cloud dev if the fixed local UI port is already used.
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", 8000))
        except OSError:
            print("Port 8000 đang được dùng. Dừng web local cũ rồi chạy lại.", file=sys.stderr)
            return 1
    children = []
    events = queue.Queue()
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "COLUMNS": "240", "TERM": "dumb", "NO_COLOR": "1"}

    def logs(process):
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                clean = ANSI_PATTERN.sub("", line)
                for url in URL_PATTERN.findall(clean):
                    if "-web" in url:
                        events.put(url)
        finally:
            events.put(None)

    try:
        print("Khởi động Modal dev CPU. Lần build đầu có thể mất vài phút. Ctrl+C dừng cả hai dịch vụ.", flush=True)
        modal = subprocess.Popen([sys.executable, "-m", "modal", "serve", "app.py"],
            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        children.append(modal)
        threading.Thread(target=logs, args=(modal,), daemon=True).start()
        url = None
        while url is None:
            try:
                url = events.get(timeout=0.5)
                if url is None:
                    print("Modal kết thúc trước khi tạo URL. Kiểm tra modal setup, Volume và lỗi phía trên.", file=sys.stderr)
                    return 1
            except queue.Empty:
                if modal.poll() is not None:
                    return 1
        detect_url = url + "/detect"
        print(f"\nDetect API: {detect_url}\nWeb: http://127.0.0.1:8000\n", flush=True)
        web = subprocess.Popen([sys.executable, "local_web.py"], cwd=ROOT,
            env={**env, "MODAL_DETECT_URL": detect_url})
        children.append(web)
        while True:
            try:
                return web.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                if modal.poll() is not None:
                    print("Modal dev đã dừng; đóng web local.", file=sys.stderr)
                    return modal.returncode or 1
    except KeyboardInterrupt:
        print("\nĐang dừng web local và Modal dev…", flush=True)
        return 0
    finally:
        for process in reversed(children):
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for process in reversed(children):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
