#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
data="${XDG_DATA_HOME:-$HOME/.local/share}"
mkdir -p "$data/applications" "$data/icons/hicolor/scalable/apps" "$HOME/.local/bin"
install -m644 "$root/web/cup.svg" "$data/icons/hicolor/scalable/apps/franok.svg"
ln -sfn "$root/desktop/launch.sh" "$HOME/.local/bin/franok"
cat > "$data/applications/franok.desktop" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=Franok
Comment=Chat locally in English and Turkish
Exec="$HOME/.local/bin/franok"
Icon=franok
Terminal=false
Categories=Utility;
StartupNotify=true
StartupWMClass=franok
EOF
desktop-file-validate "$data/applications/franok.desktop"
if command -v update-desktop-database >/dev/null; then update-desktop-database "$data/applications"; fi
if [[ ${1:-} == --pin ]]; then
  python3 - <<'PY'
import ast,subprocess
key=['org.gnome.shell','favorite-apps']
old=subprocess.check_output(['gsettings','get',*key],text=True).strip().removeprefix('@as ')
items=ast.literal_eval(old)
if 'franok.desktop' not in items:
    subprocess.run(['gsettings','set',*key,repr(items+['franok.desktop'])],check=True)
PY
fi
