#!/usr/bin/env python
"""Upload the portable targeted-audit bundle to Camber Stash."""
from __future__ import annotations

import os
import argparse
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "artifacts/finding_nemo_targeted_audit_nms50.tar"
DEST = "stash://bunpmc/projects/finding_nemo/audits/finding_nemo_targeted_audit_nms50.tar"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--destination", default=DEST)
    args = parser.parse_args()
    source = args.source.resolve()
    for line in (REPO / ".env").read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "CAMBER_API_KEY":
            os.environ["CAMBER_API_KEY"] = value.strip().strip("'\"")
            break
    else:
        raise RuntimeError("CAMBER_API_KEY missing")
    entries = [item for item in os.environ.get("WSLENV", "").split(":") if item]
    if "CAMBER_API_KEY" not in entries:
        entries.append("CAMBER_API_KEY")
    os.environ["WSLENV"] = ":".join(entries)
    src = f"/mnt/{source.drive[0].lower()}{source.as_posix().split(':', 1)[1]}"
    subprocess.run(
        ["wsl.exe", "-d", "Ubuntu", "--", "/home/intern/.camber/bin/camber", "stash", "cp", src, args.destination],
        check=True,
    )
    print(args.destination)


if __name__ == "__main__":
    main()
