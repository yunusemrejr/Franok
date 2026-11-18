# Deploying Franok

The public repository holds the model, trainer, browser app, Ubuntu app and deployment hook. The website checkout uses the same history. Configure the `origin` remote for GitHub and the `namecheap` remote for the bare repository on your own cPanel account. Authentication belongs in your SSH configuration.

```sh
git remote add namecheap namecheap-git:/home/ACCOUNT/repositories/franok.yunusemrevurgun.com.git
git push origin main
git push namecheap main
```

Install `deploy/post-receive` as the executable post-receive hook of the dedicated bare repository. The Namecheap virtual host serves the subdomain's own document root. Its proxied Cloudflare A record points to the Namecheap origin. The hook extracts just the web assets, checks the model manifests, gives each bundle its own revision URL, then atomically replaces the homepage. Browser sessions share the hostname across revisions. The host keeps previous releases and its existing cPanel files.

Run `deploy/publish.sh` to push a reviewed clean main branch and verify the live revision. To undo a release, revert the release changes on main and publish that commit. The production model URL in `web/release.json` must be pinned to a Hugging Face commit. The client checks the SHA-256 in `web/model.json`. The desktop defaults to Q8 and checks the SHA-256 from `web/model-q8.json`. A source installation also checks `work/v2/export/model.q8` before downloading.

## Ubuntu

```sh
npm ci
npx install-electron
npm run install:ubuntu -- --pin
gtk-launch franok
```

Omit `--pin` to register the app without adding a dock favorite. The launcher reuses an installed, root-owned Chromium sandbox helper where one is available. The renderer runs with sandboxing and context isolation. Move the checkout only after quitting Franok, then run the installer again to update the launcher path.

## Ubuntu and media releases

The project checkout builds the Electron installer with `npm ci`, `npx install-electron`, and `npm run package:ubuntu`. The package carries the UI, WASM runtime, the film and both subtitle tracks. It downloads the pinned Q8 file on first use, verifies its SHA-256 and stores it in the user's application data. The package registers `franok.desktop` with the application menu and Ubuntu dock. Source installations use `desktop/install.sh --pin`.

Produce `web/installer.json` with `python3 tools/installer_receipt.py work/packages/franok_0.2.0_amd64.deb`. Publish the checksum-matched binary to the stable hosting path with `deploy/publish-installer.sh work/packages/franok_0.2.0_amd64.deb`. The hook keeps that downloads directory across code releases. Attach the identical file and its `SHA256SUMS` to the matching GitHub release. The web workspace uses its small same-origin receipt to display the download size and digest.

The narrated MP4, bilingual subtitles and cover frame are included in `web/media/`. These assets are versioned with the chat app. The native desktop player can seek offline. The YouTube link opens the independently published film.
