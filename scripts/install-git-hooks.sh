#!/bin/bash
# Install the tracked NOUMENON hardening git hooks into .git/hooks.
# Idempotent. Run from anywhere inside the repo.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
src="$root/scripts/git-hooks"
dst="$root/.git/hooks"

for hook in pre-commit pre-push; do
  install -m 0755 "$src/$hook" "$dst/$hook"
  echo "installed $dst/$hook"
done
echo "git hooks installed. Bypass once with: git <cmd> --no-verify"
