# Deploying Franok

The public repository holds the model, trainer, browser app, Ubuntu app and deployment hook. The website checkout uses the same history. Configure the `origin` remote for GitHub and the `namecheap` remote for the bare repository on your own cPanel account. Authentication belongs in your SSH configuration.

```sh
git remote add namecheap namecheap-git:/home/ACCOUNT/repositories/franok.yunusemrevurgun.com.git
git push origin main
git push namecheap main
```

Install `deploy/post-receive` as the executable post-receive hook of the dedicated bare repository. The Namecheap virtual host serves the subdomain's own document root. Its proxied Cloudflare A record points to the Namecheap origin. The hook extracts just the web assets, checks the model manifests, gives each bundle its own revision URL, then atomically replaces the homepage. Browser sessions share the hostname across revisions. The host keeps previous releases and its existing cPanel files.

Run `deploy/publish.sh` to push a reviewed clean main branch and verify the live revision. To undo a release, revert the release changes on main and publish that commit. The production model URL in `web/release.json` must be pinned to a Hugging Face commit. The client checks the SHA-256 in `web/model.json`. The desktop defaults to Q8 and checks the SHA-256 from `web/model-q8.json`. A source installation checks local release export manifests under `work/v*/export/` and verifies matching weight bytes before downloading. A packaged installation keeps the verified Q8 file in its user data directory.

## Ubuntu

```sh
npm ci
npx install-electron
npm run install:ubuntu -- --pin
gtk-launch franok
```

Omit `--pin` to register the app without adding a dock favorite. The launcher reuses an installed, root-owned Chromium sandbox helper where one is available. The renderer runs with sandboxing and context isolation. Move the checkout only after quitting Franok, then run the installer again to update the launcher path.

On Ubuntu systems that restrict user namespaces, Electron checks the helper beside its executable first. The source launcher hard-links Electron into a private runtime directory alongside a link to the existing system helper. The npm distribution stays intact. A system-wide Debian installation uses the root-owned helper included in the package.

## Ubuntu and media releases

The project checkout builds the Electron installer with `npm ci`, `npx install-electron`, and `npm run package:ubuntu`. The package carries the UI, WASM runtime, the film and both subtitle tracks. It downloads the pinned Q8 file on first use, verifies its SHA-256 and stores it in the user's application data. The package registers `franok.desktop` with the application menu and Ubuntu dock. Source installations use `desktop/install.sh --pin`.

Produce `web/installer.json` with `python3 tools/installer_receipt.py work/packages/franok_0.3.0_amd64.deb`. Publish the checksum-matched binary to the stable hosting path with `deploy/publish-installer.sh work/packages/franok_0.3.0_amd64.deb`. The hook keeps that downloads directory across code releases. After the mirror is online, create an annotated version tag on the reviewed release commit and push that tag to GitHub. The [release workflow](../.github/workflows/ubuntu-release.yml) verifies the size, SHA-256 and Debian version against the tagged receipts, then publishes the identical installer and `SHA256SUMS` on GitHub. The web workspace uses its small same-origin receipt to display the download size and digest.

The narrated MP4, bilingual subtitles and cover frame are included in `web/media/`. These assets are versioned with the chat app. The native desktop player can seek offline. The YouTube link opens the independently published film.

After the GitHub assets are verified, the public versioned download URL can redirect to the identical GitHub release asset. Large transfers from the shared hosting account can be interrupted. Append `?mirror=origin` to download the verified backup directly from Namecheap. Both locations use the checksum published in `web/installer.json`.

The portfolio links to the stable [Ubuntu download](https://franok.yunusemrevurgun.com/downloads/ubuntu). Update its temporary redirect in `deploy/htaccess` after the versioned installer is available. The versioned URL also accepts `?mirror=origin` for the checksum-matched Namecheap copy.
