#!/bin/sh
cd "$(dirname "$0")"
setsid nohup python run.py > log.txt 2>&1 < /dev/null &
