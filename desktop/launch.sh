#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
electron="$root/node_modules/electron/dist/electron"
if [[ ! -x "$electron" ]]; then
  echo 'Run npm ci and npx install-electron in the Franok checkout.' >&2
  exit 1
fi
if [[ -z ${CHROME_DEVEL_SANDBOX:-} ]]; then
  for helper in /opt/google/chrome/chrome-sandbox /usr/lib/chromium/chrome-sandbox; do
    if [[ -u "$helper" && $(stat -c %u "$helper") == 0 ]]; then
      export CHROME_DEVEL_SANDBOX="$helper"
      break
    fi
  done
fi
exec "$electron" --class=franok "$root" "$@"
