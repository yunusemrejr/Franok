#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $(git branch --show-current) == main && -z $(git status --porcelain) ]] || { echo "Commit release changes on main first." >&2; exit 1; }
rev=$(git rev-parse HEAD)
git push origin main
git push namecheap main
[[ $(curl --fail --silent --show-error "https://franok.yunusemrevurgun.com/revision.txt?rev=$rev") == "$rev" ]]
echo "Verified $rev"
