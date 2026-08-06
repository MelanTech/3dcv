#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec python3 main.py --round round2 --config config/config.yaml
