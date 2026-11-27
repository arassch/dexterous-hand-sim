#!/usr/bin/env bash
# Fetch the LEAP hand from MuJoCo Menagerie into third_party/ (sparse checkout, small download).
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=third_party/mujoco_menagerie
if [ ! -d "$DEST" ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie "$DEST"
fi
git -C "$DEST" sparse-checkout set leap_hand
echo "LEAP hand ready at $DEST/leap_hand"
