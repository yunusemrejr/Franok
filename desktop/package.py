"""Build a standalone Debian package from the pinned Electron Linux distribution."""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY = """[Desktop Entry]
Type=Application
Version=1.0
Name=Franok
Comment=Chat locally in English and Turkish
Exec=/usr/bin/franok
Icon=franok
Terminal=false
Categories=Utility;
StartupNotify=true
StartupWMClass=franok
"""


def write(path, text, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "work/packages")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    version = json.loads((ROOT / "package.json").read_text())["version"]
    dist = ROOT / "node_modules/electron/dist"
    if (
        subprocess.check_output(["dpkg", "--print-architecture"], text=True).strip()
        != "amd64"
        or not (dist / "electron").is_file()
    ):
        raise SystemExit("Build on amd64 Linux after npm ci and npx install-electron")
    with tempfile.TemporaryDirectory(prefix="franok-deb-", dir=args.out) as tmp:
        tree = Path(tmp) / "root"
        opt = tree / "opt/franok"
        app = opt / "resources/app"
        shutil.copytree(dist, opt, ignore=shutil.ignore_patterns("default_app.asar"))
        (opt / "electron").rename(opt / "franok")
        (opt / "chrome-sandbox").chmod(0o4755)
        app.mkdir()
        (app / "desktop").mkdir()
        for name in [
            "package.json",
            "LICENSE",
            "desktop/main.cjs",
            "desktop/franok.png",
        ]:
            shutil.copy2(ROOT / name, app / name)
        shutil.copytree(
            ROOT / "web", app / "web", ignore=shutil.ignore_patterns("installer.json")
        )
        # Smaller Debian package with the app's English and Turkish Electron locales.
        for file in (opt / "locales").glob("*.pak"):
            if file.name not in ["en-US.pak", "en-GB.pak", "tr.pak"]:
                file.unlink()
        icon = tree / "usr/share/icons/hicolor/256x256/apps/franok.png"
        icon.parent.mkdir(parents=True)
        shutil.copy2(ROOT / "desktop/franok.png", icon)
        write(tree / "usr/share/applications/franok.desktop", ENTRY)
        write(
            tree / "usr/bin/franok",
            '#!/bin/sh\nexec /opt/franok/franok --class=franok "$@"\n',
            0o755,
        )
        doc = tree / "usr/share/doc/franok"
        doc.mkdir(parents=True)
        shutil.copy2(ROOT / "LICENSE", doc / "copyright")
        size = sum(p.stat().st_size for p in tree.rglob("*") if p.is_file()) // 1024
        write(
            tree / "DEBIAN/control",
            f"""Package: franok
Version: {version}
Section: utils
Priority: optional
Architecture: amd64
Maintainer: Yunus Emre Vurgun <yunusemrejr@users.noreply.github.com>
Installed-Size: {size}
Depends: libc6, libasound2t64 | libasound2, libatk-bridge2.0-0t64 | libatk-bridge2.0-0, libatk1.0-0t64 | libatk1.0-0, libatspi2.0-0t64 | libatspi2.0-0, libcairo2, libcups2t64 | libcups2, libdbus-1-3, libdrm2, libexpat1, libgbm1, libglib2.0-0t64 | libglib2.0-0, libgtk-3-0t64 | libgtk-3-0, libnspr4, libnss3, libpango-1.0-0, libx11-6, libxcb1, libxcomposite1, libxdamage1, libxext6, libxfixes3, libxkbcommon0, libxrandr2
Homepage: https://franok.yunusemrevurgun.com
Description: Local English and Turkish language model workspace
 Chat with Franok on Ubuntu, save conversations, and explore its architecture.
 The Q8 model downloads on first use and runs locally.
""",
        )
        subprocess.run(
            ["desktop-file-validate", tree / "usr/share/applications/franok.desktop"],
            check=True,
        )
        dest = args.out / f"franok_{version}_amd64.deb"
        subprocess.run(
            [
                "dpkg-deb",
                "--root-owner-group",
                "--threads-max=4",
                "-Zxz",
                "-z5",
                "--build",
                tree,
                dest,
            ],
            check=True,
        )
        print(dest)


if __name__ == "__main__":
    main()
