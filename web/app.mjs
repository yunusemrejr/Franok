import { unsupportedOutput, languageNotice } from './language.mjs';
import { calculate } from './calculator.mjs';
import { conversationAction } from './conversation.mjs';
import { diagram } from './diagram.mjs';
import { retrieve, recallMessage } from './session.mjs';
import { defaults, settings } from './config.mjs';
import { loadSessions, saveSession, saveDraft, removeSession, activeId, setActive } from './store.mjs';

const $ = (id) => document.getElementById(id);
const dateLabel = (ms) =>
  ms
    ? new Intl.DateTimeFormat(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      }).format(ms)
    : '';
const make = () => ({
  id: crypto.randomUUID(),
  lang: 'en',
  turns: [],
  draft: '',
  options: {},
  created: Date.now(),
  updated: Date.now(),
});
const title = (s) =>
  s.name || s.turns.find((t) => t.role === 'user')?.text.slice(0, 60) || 'New conversation';
const sessionList = await loadSessions();
let current = sessionList.find((s) => s.id === activeId()) || sessionList[0] || make();
let view = 'chat',
  worker,
  answer = null,
  busy = false,
  ready = false;
const native = location.protocol === 'franok:';
const release = await fetch('./release.json')
  .then((r) => r.json())
  .catch(() => ({}));
const cfgWindow = release.window || 256;
const platform = native ? 'Ubuntu desktop' : 'Browser runtime';
const format = release.default || (native ? 'q8' : 'q4');
const size = release.formats?.[format]?.bytes || 196331520;
const initialStatus = 'First model download · ' + Math.round(size / 1e6) + ' MB';
$('status').textContent = initialStatus;
$('platform-name').textContent = platform;
$('platform-detail').textContent = format.toUpperCase() + ' · WebAssembly SIMD';
const modelVersion = release.version || '0.1.0';
$('precision').textContent = 'v' + modelVersion + ' · ' + format.toUpperCase() + ' / WASM';
$('memory-copy').textContent =
  'The neural model reads up to ' +
  cfgWindow.toLocaleString() +
  ' tokens at once. The workspace saves complete conversations on your device. Search for an old turn and attach it once, or pin up to three turns to keep a passage in future prompts.';
if (release.video) {
  $('film').hidden = false;
  $('explainer').src = release.video.url;
  if (release.video.poster) $('explainer').poster = release.video.poster;
  for (const [lang, url] of Object.entries(release.video.captions || {})) {
    const track = document.createElement('track');
    track.kind = 'captions';
    track.srclang = lang;
    track.label = lang === 'tr' ? 'Türkçe' : 'English';
    track.src = url;
    $('explainer').append(track);
  }
  if (release.video.youtube) {
    $('youtube').href = release.video.youtube;
    $('youtube').hidden = false;
  }
}
if (release.installer) {
  const pkg = { ...release.installer };
  if (!native) {
    const proof = await fetch('./installer.json')
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null);
    if (proof?.file === pkg.file) Object.assign(pkg, proof);
  }
  $('downloaddeb').href = pkg.url;
  $('packageinfo').textContent =
    'Ubuntu · amd64 · ' + pkg.version + (pkg.bytes ? ' · ' + Math.round(pkg.bytes / 1e6) + ' MB' : '');
  $('installhelp').textContent =
    'Open the .deb in Ubuntu App Center, or install it with sudo apt install ./' +
    pkg.file +
    '. The desktop app downloads the Q8 weights on first use.';
  $('debsha').textContent = pkg.sha256 || '';
  $('debsha').parentElement.hidden = !pkg.sha256;
} else {
  $('downloaddeb').textContent = 'Install from source ↗';
  $('downloaddeb').href = 'https://github.com/yunusemrejr/Franok/blob/main/docs/deployment.md#ubuntu';
  $('installhelp').textContent = 'The source installation instructions are available on GitHub.';
  $('debsha').parentElement.hidden = true;
}
for (const [name, desc] of [
  ['q4', 'Browser default'],
  ['q8', 'Ubuntu default'],
  ['fp16', 'Native checkpoint'],
  ['fp32', 'Unrounded final merge'],
]) {
  const asset = release.formats?.[name];
  if (!asset) continue;
  const a = document.createElement('a');
  a.href = asset.publicUrl || asset.url;
  const tag = document.createElement('strong'),
    label = document.createElement('span'),
    meta = document.createElement('small');
  tag.textContent = name.toUpperCase();
  label.textContent = desc;
  meta.textContent = (asset.bytes / 1e6).toFixed(1) + ' MB ↗';
  a.append(tag, label, meta);
  $('weightfiles').append(a);
}
function save() {
  if (!sessionList.includes(current)) sessionList.unshift(current);
  const target = current;
  setActive(target.id);
  return saveSession(target)
    .then(() => {
      if (target === current) setActive(target.id);
      return true;
    })
    .catch(() => {
      $('status').textContent = 'Storage is full or unavailable. Export the transcript to keep a copy.';
      return false;
    });
}
function history() {
  const list = $('sessions');
  list.replaceChildren();
  const filter = $('filter').value.trim().toLocaleLowerCase();
  for (const s of sessionList
    .filter((s) => title(s).toLocaleLowerCase().includes(filter))
    .sort((a, b) => b.updated - a.updated)) {
    const b = document.createElement('button'),
      span = document.createElement('span');
    span.textContent = title(s);
    b.title = title(s);
    b.append(span);
    b.setAttribute('aria-current', String(s === current));
    b.onclick = () => {
      if (busy) stop();
      current = s;
      setActive(s.id);
      changeView('chat');
      paint();
    };
    list.append(b);
  }
  if (!list.children.length) {
    const p = document.createElement('p');
    p.className = 'nohistory';
    p.textContent = filter ? 'No matching sessions.' : 'Your saved chats will appear here.';
    list.append(p);
  }
}
function menu(open) {
  document.body.classList.toggle('menu', open);
  $('menu').setAttribute('aria-expanded', String(open));
  $('shade').hidden = !open;
  if (open) $('filter').focus();
}
function changeView(name, replace = false) {
  view = ['chat', 'about', 'downloads'].includes(name) ? name : 'chat';
  for (const v of ['chat', 'about', 'downloads']) $(v).hidden = v !== view;
  document.querySelectorAll('.tabs [data-view]').forEach((b) => {
    if (b.dataset.view === view) b.setAttribute('aria-current', 'page');
    else b.removeAttribute('aria-current');
  });
  $('viewname').textContent =
    view === 'chat' ? title(current) : view === 'about' ? 'Model documentation' : 'Distribution';
  $('find').hidden = $('actions').hidden = view !== 'chat';
  menu(false);
  if (view === 'about') diagram();
  else if (!$('explainer').paused) $('explainer').pause();
  if (location.hash !== '#' + view)
    window.history[replace ? 'replaceState' : 'pushState'](
      null,
      '',
      location.pathname + location.search + '#' + view,
    );
}
function writeBody(body, text) {
  body.replaceChildren();
  const chunks = text.split(/\x60\x60\x60[^\n]*\n?|\x60\x60\x60/g);
  chunks.forEach((part, i) => {
    if (!part) return;
    const el = document.createElement(i % 2 ? 'pre' : 'p');
    el.textContent = part;
    body.append(el);
  });
}
const authorLabels = {
  calculator: ['Calculator', 'Hesap makinesi'],
  language: ['Language setting', 'Dil ayarı'],
  recall: ['Session search', 'Konuşma araması'],
  social: ['Conversation helper', 'Sohbet yardımı'],
  user: ['You', 'Siz'],
  assistant: ['Franok', 'Franok'],
};
function authorName(turn, lang) {
  const label = Object.hasOwn(authorLabels, turn.tool)
    ? turn.tool
    : turn.role === 'user'
      ? 'user'
      : 'assistant';
  return authorLabels[label][lang === 'tr' ? 1 : 0];
}
function message(t, index) {
  const item = document.createElement('article');
  item.className = 'message ' + t.role;
  item.id = 'turn-' + index;
  const h = document.createElement('header');
  h.textContent = authorName(t, current.lang);
  if (t.at) {
    const time = document.createElement('small');
    time.textContent = dateLabel(t.at);
    h.append(time);
  }
  if (t.text) {
    const pin = document.createElement('button'),
      pinned = (current.pins || []).includes(index);
    pin.type = 'button';
    pin.className = 'pin';
    pin.textContent = pinned ? 'Pinned' : 'Pin';
    pin.setAttribute('aria-pressed', String(pinned));
    pin.setAttribute('aria-label', (pinned ? 'Unpin' : 'Pin') + ' turn ' + (index + 1));
    pin.title = pinned ? 'Remove from persistent context' : 'Keep this turn in future prompts';
    pin.onclick = () => {
      const pins = current.pins || [];
      if (!pinned && pins.length >= 3) {
        $('status').textContent = 'Three turns are pinned. Unpin one to add another.';
        return;
      }
      current.pins = pinned ? pins.filter((i) => i !== index) : [...pins, index];
      save();
      const scroll = $('stream'),
        top = scroll.scrollTop;
      paint();
      scroll.scrollTop = top;
      $('turn-' + index)
        ?.querySelector('.pin')
        ?.focus({ preventScroll: true });
    };
    h.append(pin);
  }
  const body = document.createElement('div');
  body.className = 'body';
  writeBody(body, t.text);
  item.append(h, body);
  if (Number.isInteger(t.quote) && current.turns[t.quote]) {
    const a = document.createElement('button');
    a.className = 'source';
    a.textContent = 'Attached turn ' + (t.quote + 1);
    a.onclick = () => highlight(t.quote);
    item.append(a);
  }
  if (Number.isInteger(t.revisit) && current.turns[t.revisit]) {
    const link = document.createElement('button');
    link.className = 'source';
    link.textContent =
      (current.lang === 'tr' ? 'Yeniden yanıtlanan mesaj ' : 'Revisited question · turn ') + (t.revisit + 1);
    link.onclick = () => highlight(t.revisit);
    item.append(link);
  }
  if (t.tool === 'recall') {
    for (const source of t.sources || []) {
      const link = document.createElement('button');
      link.className = 'source';
      link.textContent = (current.lang === 'tr' ? 'Mesaja git ' : 'Go to turn ') + (source + 1);
      link.onclick = () => highlight(source);
      item.append(link);
    }
  }
  if (t.role === 'assistant' && t.usage?.prompt && Array.isArray(t.usage.used)) {
    const details = document.createElement('details'),
      summary = document.createElement('summary'),
      description = document.createElement('span');
    details.className = 'contextinfo';
    summary.textContent = 'Context · ' + t.usage.prompt + ' tokens';
    description.textContent =
      (t.usage.systemTokens || 0) +
      ' system tokens. ' +
      (t.usage.model ? t.usage.model + '. ' : '') +
      (t.canonicalGreeting ? 'Standalone greeting normalized to “' + t.canonicalGreeting + '”. ' : '') +
      (t.usage.truncated ? 'The current message was shortened. ' : '') +
      (t.usage.used?.length ? 'Earlier turns included:' : 'No earlier turns recorded.');
    details.append(summary, description);
    for (const source of t.usage.used || []) {
      const btn = document.createElement('button');
      btn.textContent = 'Turn ' + (source + 1);
      btn.onclick = () => highlight(source);
      details.append(btn);
    }
    item.append(details);
  }
  return item;
}
function highlight(i) {
  const el = $('turn-' + i);
  if (!el) return;
  el.scrollIntoView({ block: 'center' });
  el.classList.add('hit');
  setTimeout(() => el.classList.remove('hit'), 1700);
}
function paint() {
  const empty = !current.turns.length;
  $('chat').classList.toggle('empty', empty);
  $('emptylinks').hidden = $('welcome').hidden = !empty;
  $('quotechip').hidden = !Number.isInteger(current.quote);
  $('quotechip').querySelector('span').textContent = 'Attached turn ' + (current.quote + 1);
  $('lang').value = current.lang;
  $('draft').value = current.draft || '';
  $('draft').placeholder = current.lang === 'tr' ? 'Franok’a mesaj yaz…' : 'Message Franok…';
  $('messages').replaceChildren(...current.turns.map(message));
  $('stream').scrollTop = $('stream').scrollHeight;
  $('find').disabled = empty;
  $('actions').disabled = busy || !sessionList.includes(current);
  if (view === 'chat') $('viewname').textContent = title(current);
  history();
}
function flags(on) {
  busy = on;
  $('send').hidden = on;
  $('send').disabled = on;
  $('stop').hidden = !on;
  $('lang').disabled = on;
  $('controls').disabled = on;
  $('actions').disabled = on;
}
function stop() {
  if (answer && !answer.text && current.turns.at(-1) === answer) current.turns.pop();
  worker?.terminate();
  worker = null;
  ready = false;
  flags(false);
  answer = null;
  save();
  paint();
  $('status').textContent = 'Generation stopped.';
}
function startWorker() {
  if (worker) return;
  const activeWorker = (worker = new Worker('./worker.mjs', { type: 'module' }));
  function fail(msg) {
    if (answer && !answer.text) current.turns.pop();
    answer = null;
    worker?.terminate();
    worker = null;
    ready = false;
    flags(false);
    save();
    paint();
    $('status').textContent = msg;
  }
  worker.onerror = (e) => {
    if (worker === activeWorker) fail(e.message || 'The local runtime could not start.');
  };
  worker.onmessage = async ({ data: d }) => {
    if (activeWorker !== worker) return;
    if (d.kind === 'download')
      $('status').textContent = 'Downloading ' + format.toUpperCase() + ' weights · ' + d.pct + '%';
    if (d.kind === 'ready') {
      ready = true;
      $('status').textContent = 'Weights verified.';
    }
    if (d.kind === 'prefill')
      $('status').textContent =
        'Reading ' + d.count + ' prompt tokens' + (d.truncated ? ' · current message shortened…' : '…');
    if (d.kind === 'part' && answer) {
      answer.text = d.text;
      const b = $('messages').lastElementChild?.querySelector('.body');
      const stream = $('stream'),
        nearEnd = stream.scrollHeight - stream.scrollTop - stream.clientHeight < 100;
      if (b) {
        let p = b.firstChild;
        if (!p) {
          p = document.createElement('p');
          b.append(p);
        }
        p.textContent = d.text;
      }
      if (nearEnd) stream.scrollTop = stream.scrollHeight;
      $('status').textContent = 'Generating on this device…';
    }
    if (d.kind === 'done' && answer) {
      answer.text = d.text;
      answer.usage = {
        model: 'Franok v' + modelVersion + ' ' + format.toUpperCase(),
        revision: release.revision || null,
        tokens: d.tokens,
        seconds: d.seconds,
        prompt: d.prompt,
        used: d.used,
        systemTokens: d.systemTokens,
        truncated: d.truncated,
      };
      if (!answer.text) current.turns.pop();
      const finishedSession = current;
      answer = null;
      flags(false);
      await save();
      if (activeWorker !== worker || current !== finishedSession || busy) return;
      paint();
      $('status').textContent = d.tokens
        ? 'Generated locally · ' +
          d.tokens +
          ' tokens · ' +
          (d.tokens / Math.max(0.01, d.seconds)).toFixed(1) +
          ' tokens/s'
        : 'Franok ended its reply without generating text.';
    }
    if (d.kind === 'error') fail(d.message);
  };
}
$('composer').onsubmit = (e) => {
  e.preventDefault();
  const text = $('draft').value.trim();
  if (!text || busy) return;
  const config = settings(current.options, cfgWindow);
  const social =
    config.conversationHelp && !config.system.trim() && !Number.isInteger(current.quote)
      ? conversationAction(text, current.turns, current.lang)
      : null;
  const canonicalGreeting = release.systemConditioned && social?.canonical;
  const recall = recallMessage(text, current.turns, current.lang);
  current.turns.push({
    role: 'user',
    text,
    at: Date.now(),
    ...(Number.isInteger(current.quote) ? { quote: current.quote } : {}),
    ...(Number.isInteger(social?.source) ? { revisit: social.source } : {}),
  });
  delete current.quote;
  current.draft = '';
  if (recall || (social?.text && !canonicalGreeting)) {
    current.turns.push({
      role: 'assistant',
      tool: recall ? 'recall' : 'social',
      ...(recall || social),
      at: Date.now(),
    });
    save();
    paint();
    $('status').textContent = recall
      ? current.lang === 'tr'
        ? 'Cihazdaki konuşmada arandı.'
        : 'Searched this conversation on your device.'
      : current.lang === 'tr'
        ? 'Yerel sohbet yardımı.'
        : 'Local conversation helper.';
    return;
  }
  const result = config.calculator ? calculate(text, current.lang) : null;
  if (result || unsupportedOutput(text)) {
    const tool = result ? 'calculator' : 'language',
      reply = result?.text || languageNotice[current.lang];
    current.turns.push({ role: 'assistant', text: reply, tool, at: Date.now() });
    save();
    paint();
    $('status').textContent = result
      ? current.lang === 'tr'
        ? 'Cihazda hesaplandı.'
        : 'Calculated on this device.'
      : current.lang === 'tr'
        ? 'Dil ayarı uygulandı.'
        : 'Workspace language setting.';
    return;
  }
  const context = canonicalGreeting
    ? [{ role: 'user', text: canonicalGreeting }]
    : Number.isInteger(social?.source)
      ? [
          ...current.turns.slice(0, social.source),
          {
            ...current.turns[social.source],
            text: current.turns[social.source].text + '\n' + social.instruction,
          },
        ]
      : [...current.turns];
  answer = {
    role: 'assistant',
    text: '',
    at: Date.now(),
    ...(canonicalGreeting ? { canonicalGreeting } : {}),
  };
  current.turns.push(answer);
  save();
  paint();
  flags(true);
  $('status').textContent = ready ? 'Preparing prompt…' : 'Opening model weights…';
  startWorker();
  worker.postMessage({
    type: 'chat',
    lang: current.lang,
    turns: context,
    pins: current.pins || [],
    options: canonicalGreeting ? { ...config, temperature: 0, maxNew: Math.min(24, config.maxNew) } : config,
  });
};
$('draft').oninput = () => {
  current.draft = $('draft').value;
  if (!sessionList.includes(current)) {
    save();
    history();
    return;
  }
  saveDraft(current).catch(() => {
    $('status').textContent = 'Draft could not be saved on this device.';
  });
};
$('draft').onkeydown = (e) => {
  if (
    e.key === 'Enter' &&
    !e.shiftKey &&
    !e.isComposing &&
    (settings(current.options).enterSends || e.ctrlKey || e.metaKey)
  ) {
    e.preventDefault();
    $('composer').requestSubmit();
  }
};
$('stop').onclick = stop;
$('new').onclick = () => {
  if (busy) stop();
  current = make();
  save();
  changeView('chat');
  paint();
  $('draft').focus();
};
$('lang').onchange = () => {
  current.lang = $('lang').value;
  save();
  paint();
};
$('menu').onclick = () => menu(!document.body.classList.contains('menu'));
$('closemenu').onclick = $('shade').onclick = () => {
  menu(false);
  $('menu').focus();
};
document.querySelectorAll('[data-view]').forEach(
  (b) =>
    (b.onclick = (e) => {
      e.preventDefault();
      changeView(b.dataset.view);
    }),
);
document.querySelector('.brand').onclick = (e) => {
  e.preventDefault();
  changeView('chat');
};
for (const id of ['slot', 'anchor', 'layer']) $(id).oninput = diagram;
$('filter').oninput = history;
function search() {
  const q = $('query').value.trim(),
    hits = q ? retrieve(current.turns, q, 20) : [];
  const list = $('matches');
  list.replaceChildren();
  if (q && !hits.length) list.textContent = 'No matching turns.';
  for (const h of hits) {
    const row = document.createElement('div');
    row.className = 'matchrow';
    const jump = document.createElement('button'),
      attach = document.createElement('button');
    jump.textContent = 'Turn ' + (h.index + 1) + ' · ' + h.turn.text.slice(0, 150);
    jump.onclick = () => {
      $('search').close();
      highlight(h.index);
    };
    attach.textContent = 'Attach';
    attach.className = 'attach';
    attach.setAttribute('aria-label', 'Attach turn ' + (h.index + 1) + ' to next message');
    attach.onclick = () => {
      current.quote = h.index;
      save();
      $('search').close();
      paint();
      $('draft').focus();
    };
    row.append(jump, attach);
    list.append(row);
  }
}
$('find').onclick = () => {
  $('query').value = '';
  search();
  $('search').showModal();
  $('query').focus();
};
$('query').oninput = search;
$('quotechip').querySelector('button').onclick = () => {
  delete current.quote;
  save();
  paint();
};
function systemHint() {
  const remaining = cfgWindow - (+$('tokens').value || 80),
    cap = Math.min(180, Math.floor(remaining * 0.22));
  $('systemhint').textContent =
    'The first ' + cap + ' tokens are sent before each turn. A small model can miss instructions.';
}
$('tokens').oninput = systemHint;
function fillPrefs(o) {
  $('sysprompt').value = o.system || defaults[current.lang];
  $('temp').value = o.temperature;
  $('topp').value = o.topP;
  $('tempout').value = Number(o.temperature).toFixed(2);
  $('pout').value = Number(o.topP).toFixed(2);
  $('tokens').max = Math.min(256, cfgWindow / 2);
  $('tokens').value = o.maxNew;
  systemHint();
  $('calc').checked = o.calculator;
  $('casual').checked = o.conversationHelp;
  $('recall').checked = o.recall;
  $('enter').checked = o.enterSends;
}
$('controls').onclick = () => {
  fillPrefs(settings(current.options, cfgWindow));
  $('prefs').showModal();
};
$('temp').oninput = () => ($('tempout').value = Number($('temp').value).toFixed(2));
$('topp').oninput = () => ($('pout').value = Number($('topp').value).toFixed(2));
$('resetprefs').onclick = () => fillPrefs(settings({}, cfgWindow));
$('prefsform').onsubmit = async (e) => {
  e.preventDefault();
  const sys = $('sysprompt').value.trim();
  current.options = settings(
    {
      system: sys === defaults[current.lang] ? '' : sys,
      temperature: $('temp').value,
      topP: $('topp').value,
      maxNew: $('tokens').value,
      calculator: $('calc').checked,
      conversationHelp: $('casual').checked,
      recall: $('recall').checked,
      enterSends: $('enter').checked,
    },
    cfgWindow,
  );
  if (await save()) {
    $('prefs').close();
    $('status').textContent = 'Session controls saved.';
  }
};
$('actions').onclick = () => {
  $('rename').value = title(current);
  $('manage').showModal();
};
$('renameform').onsubmit = (e) => {
  e.preventDefault();
  current.name = $('rename').value.trim();
  save();
  paint();
  $('manage').close();
};
$('exportchat').onclick = () => {
  const text = current.turns.map((t) => authorName(t, current.lang) + '\n' + t.text).join('\n\n');
  const a = document.createElement('a');
  const url = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }));
  a.href = url;
  a.download = 'franok-' + current.id + '.txt';
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  $('manage').close();
};
$('deletechat').onclick = () => {
  $('manage').close();
  $('confirmdelete').showModal();
};
$('reallydelete').onclick = async () => {
  await removeSession(current.id);
  sessionList.splice(sessionList.indexOf(current), 1);
  current = sessionList[0] || make();
  setActive(current.id);
  $('confirmdelete').close();
  paint();
};
document.querySelectorAll('[data-close]').forEach((b) => (b.onclick = () => b.closest('dialog').close()));
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && document.body.classList.contains('menu')) menu(false);
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k' && !document.querySelector('dialog[open]')) {
    e.preventDefault();
    if (innerWidth <= 650) menu(true);
    $('filter').focus();
  }
});
const theme = () =>
  localStorage.getItem('franok.theme') ||
  (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
document.documentElement.dataset.theme = theme();
$('theme').onclick = () => {
  const v = theme() === 'dark' ? 'light' : 'dark';
  localStorage.setItem('franok.theme', v);
  document.documentElement.dataset.theme = v;
};
window.addEventListener('popstate', () => changeView(location.hash.slice(1) || 'chat', true));
changeView(location.hash.slice(1) || 'chat', true);
paint();
