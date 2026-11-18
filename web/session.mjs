import { defaults, settings } from './config.mjs';
export const terms = (text) =>
  text
    .normalize('NFKC')
    .toLocaleLowerCase('en')
    .replace(/i\u0307/g, 'i')
    .match(/[\p{L}\p{N}]+/gu) || [];
export function retrieve(turns, query, count = 3) {
  const needles = [...new Set(terms(query))];
  const docs = turns.map((turn, index) => ({ turn, index, words: new Set(terms(turn.text)) }));
  const weights = new Map(
    needles.map((k) => [
      k,
      Math.log(1 + (docs.length + 0.5) / (0.5 + docs.filter((d) => d.words.has(k)).length)),
    ]),
  );
  return docs
    .map((d) => ({ ...d, score: needles.reduce((s, k) => s + (d.words.has(k) ? weights.get(k) : 0), 0) }))
    .filter((d) => d.score > 0)
    .sort((a, b) => b.score - a.score || b.index - a.index)
    .slice(0, count);
}
const recallCommand = /^\/(?:recall|hatırla|hatirla)(?:\s+([\s\S]+))?$/iu;
export function recallMessage(text, turns, lang) {
  const command = recallCommand.exec(text.trim());
  if (!command) return null;
  const query = command[1]?.trim();
  if (!query)
    return {
      text:
        lang === 'tr'
          ? 'Önceki mesajlarında aramak için /hatırla ve bir kelime yaz.'
          : 'Type /recall followed by a word from an earlier message.',
      sources: [],
    };
  const messages = turns
    .map((turn, id) => ({ ...turn, id }))
    .filter((turn) => turn.role === 'user' && !recallCommand.test(turn.text.trim()));
  const hits = retrieve(messages, query, 3);
  const needles = new Set(terms(query));
  const quotes = hits.map(({ turn }) => {
    const source = turn.text;
    const match = Array.from(source.matchAll(/[\p{L}\p{M}\p{N}]+/gu)).find((m) =>
      needles.has(terms(m[0])[0]),
    );
    const start = Math.max(0, (match?.index || 0) - 60);
    const end = Math.min(source.length, start + 240);
    return (
      (lang === 'tr' ? 'Mesaj ' : 'Turn ') +
      (turn.id + 1) +
      ' · ' +
      (start ? '…' : '') +
      source.slice(start, end) +
      (end < source.length ? '…' : '')
    );
  });
  return {
    text:
      quotes.join('\n\n') ||
      (lang === 'tr'
        ? 'Önceki mesajlarında eşleşen kelime bulunamadı.'
        : 'No matching words in your earlier messages.'),
    sources: hits.map(({ turn }) => turn.id),
  };
}
export function buildPrompt(tok, turns, lang, limit, raw = {}, systemConditioned = true, pinned = []) {
  const config = settings(raw),
    current = turns.at(-1),
    prefix = [1, lang === 'tr' ? 6 : 5];
  if (systemConditioned) {
    const sys = tok
      .encode(config.system.trim() || defaults[lang] || defaults.en)
      .slice(0, Math.min(180, Math.floor(limit * 0.22)));
    prefix.push(...sys, 2);
  }
  const attached = Number.isInteger(current.quote) && current.quote >= 0 && current.quote < turns.length - 1;
  const pins = [...new Set(pinned)]
    .filter((i) => Number.isInteger(i) && i >= 0 && i < turns.length - 1)
    .slice(-3);
  const pinnedBudget = pins.length ? Math.min(150, Math.floor(limit * 0.17)) : 0;
  const ids = tok.encode(current.text);
  const cap =
    limit - prefix.length - 2 - pinnedBudget - (attached ? Math.min(110, Math.floor(limit / 3)) : 0);
  const tail = [3, ...ids.slice(-Math.max(1, cap)), 4];
  let space = limit - prefix.length - tail.length;
  const snippets = [];
  function take(index, max) {
    if (space < 5 || snippets.some((s) => s.index === index)) return;
    const t = turns[index],
      role = t.role === 'user' ? 3 : 4;
    const span = tok.encode(t.text).slice(0, Math.min(max, space - 2));
    const seq = [role, ...span, ...(role === 4 ? [2] : [])];
    snippets.push({ index, seq });
    space -= seq.length;
  }
  if (attached) take(current.quote, Math.min(240, Math.ceil(limit / 3)));
  for (const index of pins) take(index, Math.floor(pinnedBudget / pins.length) - 2);
  const hist = turns.slice(0, -1),
    n = Math.min(hist.length, limit > 400 ? 6 : 2);
  for (let i = hist.length - 1; i >= hist.length - n; i--) take(i, limit > 400 ? 112 : 44);
  if (config.recall && space >= 5)
    for (const h of retrieve(hist.slice(0, hist.length - n), current.text, 4)) take(h.index, 96);
  const ordered = snippets.sort((a, b) => a.index - b.index);
  return {
    ids: [...prefix, ...ordered.flatMap((s) => s.seq), ...tail],
    used: ordered.map((s) => s.index),
    systemTokens: Math.max(0, prefix.length - 3),
    truncated: ids.length > cap,
  };
}
export function promptTokens(tok, turns, lang, limit, raw) {
  return buildPrompt(tok, turns, lang, limit, raw).ids;
}
