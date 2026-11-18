# Deploying Franok

The public repository holds the model, trainer, browser app, Ubuntu app and deployment hook. The website checkout uses the same history. Configure the `origin` remote for GitHub and the `namecheap` remote for the bare repository on your own cPanel account. Authentication belongs in your SSH configuration.

```sh
git remote add namecheap namecheap-git:/home/ACCOUNT/repositories/franok.yunusemrevurgun.com.git
git push origin main
git push namecheap main
```

Install `deploy/post-receive` as the executable post-receive hook of the dedicated bare repository. The Namecheap virtual host serves the subdomain's own document root. Its proxied Cloudflare A record points to the Namecheap origin. The hook extracts just the web assets, checks the model manifests, gives each bundle its own revision URL, then atomically replaces the homepage. Browser sessions share the hostname across revisions. The host keeps previous releases and its existing cPanel files.

Run `deploy/publish.sh` to push a reviewed clean main branch and verify the live revision. To undo a release, revert the release changes on main and publish that commit. The production model URL in `web/release.json` must be pinned to a Hugging Face commit. The client checks the SHA-256 in `web/model.json`. The desktop additionally reads `work/export/model.q4` when bundled locally.

## Ubuntu

```sh
npm ci
npx install-electron
npm run install:ubuntu -- --pin
gtk-launch franok
```

Omit `--pin` to register the app without adding a dock favorite. The launcher reuses an installed, root-owned Chromium sandbox helper where one is available. The renderer runs with sandboxing and context isolation. Move the checkout only after quitting Franok, then run the installer again to update the launcher path.
