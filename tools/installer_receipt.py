"""Generate web metadata and a sha256sum-compatible file for a completed installer."""

import argparse, hashlib, json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("deb", type=Path)
args = p.parse_args()
release = json.loads(Path("web/release.json").read_text())
if args.deb.name != release["installer"]["file"]:
    raise SystemExit("Wrong package version")
with args.deb.open("rb") as f:
    sha = hashlib.file_digest(f, "sha256").hexdigest()
Path("web/installer.json").write_text(
    json.dumps(
        dict(file=args.deb.name, bytes=args.deb.stat().st_size, sha256=sha), indent=2
    )
    + "\n"
)
(args.deb.parent / "SHA256SUMS").write_text(sha + "  " + args.deb.name + "\n")
print(sha)
