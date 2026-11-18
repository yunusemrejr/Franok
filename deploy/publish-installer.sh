#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
deb=${1:?Pass the path to the built amd64 .deb}
deb=$(realpath "$deb")
name=$(basename "$deb")
[[ $name =~ ^franok_[0-9]+\.[0-9]+\.[0-9]+_amd64\.deb$ ]] || { echo 'Unexpected installer filename' >&2; exit 1; }
hash=$(sha256sum "$deb" | cut -d' ' -f1)
python3 - "$name" "$hash" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path('web/installer.json').read_text())
if (r['file'],r['sha256'])!=tuple(sys.argv[1:]):raise SystemExit('Installer does not match the web receipt')
PY
remote='franok.yunusemrevurgun.com/downloads'
pending=".$name.$$.part"
ssh namecheap-git "mkdir -p '$remote'"
scp -q "$deb" "namecheap-git:$remote/$pending"
ssh namecheap-git "cd '$remote' && printf '%s  %s\n' '$hash' '$pending' | sha256sum -c - && chmod 644 '$pending' && mv -- '$pending' '$name' && printf '%s  %s\n' '$hash' '$name' > '$name.sha256'"
echo "https://franok.yunusemrevurgun.com/downloads/$name"
