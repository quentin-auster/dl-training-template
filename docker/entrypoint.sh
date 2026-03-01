#!/usr/bin/env bash
set -euo pipefail

# Decode rclone config from env var (set RCLONE_CONF_B64 to base64 of rclone.conf)
if [ -n "${RCLONE_CONF_B64:-}" ]; then
    mkdir -p ~/.config/rclone
    echo "$RCLONE_CONF_B64" | base64 -d > ~/.config/rclone/rclone.conf
    chmod 600 ~/.config/rclone/rclone.conf
    echo "rclone config loaded from RCLONE_CONF_B64"
fi

exec "$@"
