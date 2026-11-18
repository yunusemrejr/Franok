#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
dist="$root/node_modules/electron/dist"
electron="$dist/electron"
if [[ ! -x "$electron" ]]; then
  echo 'Run npm ci and npx install-electron in the Franok checkout.' >&2
  exit 1
fi
valid_helper() {
  [[ -f "$1" && -x "$1" && -u "$1" && $(stat -Lc %u "$1") == 0 ]]
}
if ! valid_helper "$dist/chrome-sandbox"; then
  for helper in "${CHROME_DEVEL_SANDBOX:-}" /opt/google/chrome/chrome-sandbox /usr/lib/chromium/chrome-sandbox; do
    valid_helper "$helper" || continue
    # Chromium searches next to its actual executable before reading CHROME_DEVEL_SANDBOX.
    # A hard link gives the source launcher its own executable directory without copying Electron.
    cache="$root/node_modules/.franok-runtime-$(stat -Lc %i "$electron")-$(stat -Lc %i "$helper")"
    if [[ ! -x "$cache/electron" ]]; then
      stage=$(mktemp -d "$root/node_modules/.franok-build-XXXXXXXX")
      trap 'rm -rf -- "$stage"' EXIT
      for asset in "$dist"/*; do
        case "${asset##*/}" in
          chrome-sandbox) ln -s -- "$helper" "$stage/chrome-sandbox" ;;
          electron) ln -- "$electron" "$stage/electron" ;;
          *) ln -s -- "$asset" "$stage/${asset##*/}" ;;
        esac
      done
      [[ -e "$stage/chrome-sandbox" ]] || ln -s -- "$helper" "$stage/chrome-sandbox"
      if ! mv -T -- "$stage" "$cache" 2>/dev/null; then
        [[ -x "$cache/electron" ]] || exit 1
        rm -rf -- "$stage"
      fi
      trap - EXIT
    fi
    electron="$cache/electron"
    break
  done
fi
exec "$electron" --class=franok "$root" "$@"
