const { app, BrowserWindow, protocol, net, shell, session } = require('electron');
const path = require('node:path'),
  fs = require('node:fs'),
  fsp = fs.promises;
const { createHash, randomUUID } = require('node:crypto');
const { Readable } = require('node:stream');
const { pathToFileURL } = require('node:url');
const root = path.resolve(__dirname, '..'),
  web = path.join(root, 'web');
app.setName('Franok');
app.setDesktopName('franok.desktop');
app.enableSandbox();
protocol.registerSchemesAsPrivileged([
  {
    scheme: 'franok',
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      corsEnabled: true,
      stream: true,
      codeCache: true,
    },
  },
]);
const external = new Set([
  'github.com',
  'huggingface.co',
  'franok.yunusemrevurgun.com',
  'www.youtube.com',
  'youtu.be',
]);
const types = {
  '.mjs': 'text/javascript',
  '.css': 'text/css',
  '.html': 'text/html',
  '.webp': 'image/webp',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.wasm': 'application/wasm',
  '.bin': 'application/octet-stream',
  '.safetensors': 'application/octet-stream',
  '.json': 'application/json',
  '.woff2': 'font/woff2',
  '.vtt': 'text/vtt',
  '.mp4': 'video/mp4',
};
const csp =
  "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self'; worker-src 'self'; connect-src 'self' https://huggingface.co https://*.hf.co; media-src 'self' https://franok.yunusemrevurgun.com; img-src 'self' https://franok.yunusemrevurgun.com; font-src 'self'; object-src 'none'; base-uri 'none'";
let win;
const release = () => JSON.parse(fs.readFileSync(path.join(web, 'release.json'), 'utf8'));
function externalUrl(href) {
  try {
    const u = new URL(href);
    return u.protocol === 'https:' && external.has(u.hostname);
  } catch {
    return false;
  }
}
async function videoFile(file, range) {
  const { size } = await fsp.stat(file);
  const headers = { 'content-type': 'video/mp4', 'accept-ranges': 'bytes' };
  let start = 0,
    end = size - 1;
  if (range) {
    const match = /^bytes=(\d*)-(\d*)$/.exec(range);
    if (!match || !(match[1] || match[2]))
      return new Response(null, { status: 416, headers: { 'content-range': 'bytes */' + size } });
    if (match[1]) {
      start = Number(match[1]);
      if (match[2]) end = Math.min(end, Number(match[2]));
    } else start = Math.max(0, size - Number(match[2]));
    if (start > end || start >= size)
      return new Response(null, { status: 416, headers: { 'content-range': 'bytes */' + size } });
    headers['content-range'] = 'bytes ' + start + '-' + end + '/' + size;
  }
  headers['content-length'] = String(end - start + 1);
  return new Response(Readable.toWeb(fs.createReadStream(file, { start, end })), {
    status: range ? 206 : 200,
    headers,
  });
}
function show() {
  if (win) {
    if (win.isMinimized()) win.restore();
    win.show();
    win.focus();
    return;
  }
  win = new BrowserWindow({
    title: 'Franok',
    width: 1200,
    height: 830,
    minWidth: 360,
    minHeight: 420,
    backgroundColor: '#f4f2ec',
    icon: path.join(root, 'desktop', 'franok.png'),
    autoHideMenuBar: true,
    webPreferences: { sandbox: true, contextIsolation: true, nodeIntegration: false },
  });
  win.setMenu(null);
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (externalUrl(url)) shell.openExternal(url);
    return { action: 'deny' };
  });
  win.webContents.on('will-navigate', (e, url) => {
    if (new URL(url).origin === new URL('franok://app/').origin && url.startsWith('franok://app/')) return;
    e.preventDefault();
    if (externalUrl(url)) shell.openExternal(url);
  });
  win.on('closed', () => (win = null));
  win.loadURL('franok://app/index.html');
}
async function weights(format) {
  const build = release(),
    meta = build.formats?.[format];
  if (!meta) return new Response('Unknown model', { status: 404 });
  const manifest = JSON.parse(await fsp.readFile(path.join(web, path.basename(meta.manifest)), 'utf8'));
  const dir = path.join(app.getPath('userData'), 'weights'),
    dest = path.join(dir, manifest.sha256 + '.' + format);
  await fsp.mkdir(dir, { recursive: true });
  const localFiles = [dest];
  if (!app.isPackaged) {
    const work = path.join(root, 'work');
    const runs = await fsp.readdir(work, { withFileTypes: true }).catch(() => []);
    for (const entry of runs) {
      if (!entry.isDirectory() || !/^v[0-9]+$/.test(entry.name)) continue;
      const exported = path.join(work, entry.name, 'export');
      const index = await fsp.readFile(path.join(exported, path.basename(meta.manifest)), 'utf8')
        .then(JSON.parse).catch(() => null);
      if (index?.sha256 === manifest.sha256) localFiles.push(path.join(exported, 'model.' + format));
    }
    localFiles.push(path.join(work, 'export', 'model.' + format));
  }
  for (const local of localFiles) {
    const st = await fsp.stat(local).catch(() => null);
    if (!st) continue;
    if (st.size === manifest.bytes) {
      const hash = createHash('sha256');
      for await (const chunk of fs.createReadStream(local)) hash.update(chunk);
      if (hash.digest('hex') === manifest.sha256) return net.fetch(pathToFileURL(local).href);
    }
    if (local === dest) await fsp.rm(dest, { force: true });
  }
  const url = new URL(meta.url);
  if (
    url.protocol !== 'https:' ||
    url.hostname !== 'huggingface.co' ||
    !url.pathname.startsWith('/yunusemrejr/Franok/resolve/')
  )
    throw Error('Invalid release URL');
  const remote = await net.fetch(url.href);
  if (!remote.ok) return new Response('Weight service unavailable', { status: 502 });
  const tmp = dest + '.' + randomUUID() + '.part';
  async function* save() {
    let file;
    const hash = createHash('sha256');
    let n = 0;
    try {
      file = await fsp.open(tmp, 'wx', 0o600);
      for await (const chunk of remote.body) {
        n += chunk.length;
        if (n > manifest.bytes) throw Error('Oversized model transfer');
        hash.update(chunk);
        await file.writeFile(chunk);
        yield chunk;
      }
      await file.close();
      file = null;
      if (n !== manifest.bytes || hash.digest('hex') !== manifest.sha256) throw Error('Invalid model digest');
      await fsp.rename(tmp, dest);
    } finally {
      if (file) await file.close();
      await fsp.rm(tmp, { force: true });
    }
  }
  return new Response(Readable.toWeb(Readable.from(save())), {
    headers: { 'content-type': 'application/octet-stream', 'content-length': String(manifest.bytes) },
  });
}
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', show);
  app.whenReady().then(() => {
    session.defaultSession.setPermissionRequestHandler((_w, _p, cb) => cb(false));
    protocol.handle('franok', async (req) => {
      try {
        const u = new URL(req.url),
          name = decodeURIComponent(u.pathname.slice(1));
        if (u.hostname !== 'app' || req.method !== 'GET') return new Response('Missing', { status: 404 });
        if (name === 'release.json') {
          const build = release();
          build.native = true;
          build.default = build.formats?.q8 ? 'q8' : 'q4';
          for (const fmt of ['q4', 'q8'])
            if (build.formats?.[fmt]) {
              const a = build.formats[fmt];
              a.publicUrl = a.url;
              a.url = './weights.' + fmt;
            }
          return Response.json(build);
        }
        if (/^weights\.q[48]$/.test(name)) return weights(name.slice(8));
        const file = path.resolve(web, name),
          mime = types[path.extname(file)];
        if (
          !file.startsWith(web + path.sep) ||
          !mime ||
          !fs.existsSync(file) ||
          !fs.realpathSync(file).startsWith(web + path.sep)
        )
          return new Response('Missing', { status: 404 });
        if (mime === 'video/mp4') return videoFile(file, req.headers.get('range'));
        return new Response(await fsp.readFile(file), {
          headers: { 'content-type': mime, 'content-security-policy': csp },
        });
      } catch (e) {
        console.error('Franok resource:', e.message);
        return new Response('Unable to load resource', { status: 500 });
      }
    });
    show();
  });
  app.on('window-all-closed', () => app.quit());
}
