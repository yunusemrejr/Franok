import { defaults, settings } from './config.mjs';
const wordPattern = /[\p{L}\p{M}\p{N}]+/gu;
const normalized = (s, lang) =>
  s
    .normalize('NFKC')
    .toLocaleLowerCase(lang === 'tr' ? 'tr' : 'en')
    .replace(/i\u0307/g, 'i');
export const terms = (text, lang = 'en') => normalized(text, lang).match(wordPattern) || [];
const lexicalCache = new WeakMap();
function indexedWords(turn, lang) {
  const cached = lexicalCache.get(turn);
  if (cached?.text === turn.text && cached.lang === lang) return cached;
  const tokens = terms(turn.text, lang),
    frequencies = new Map();
  for (const t of tokens) frequencies.set(t, (frequencies.get(t) || 0) + 1);
  const index = { text: turn.text, lang, frequencies, length: tokens.length };
  lexicalCache.set(turn, index);
  return index;
}
const locate = (text, term, lang) => {
  if (!term) return -1;
  for (const m of text.matchAll(wordPattern)) if (normalized(m[0], lang) === term) return m.index;
  return -1;
};
export function excerpt(text, term, lang, width = 240) {
  if (text.length <= width) return text;
  const at = locate(text, term, lang);
  const start = Math.max(0, Math.min(text.length - width, at - Math.floor(width / 4)));
  return (start ? '…' : '') + text.slice(start, start + width) + (start + width < text.length ? '…' : '');
}
export function retrieve(turns, query, count = 3, lang = 'en', encoder = null) {
  const needles = [...new Set(terms(query, lang))];
  if (!turns.length || !needles.length) return [];
  const docs = turns.map((turn, index) => ({ turn, index, words: indexedWords(turn, lang) }));
  const mean = docs.reduce((n, d) => n + d.words.length, 0) / docs.length || 1;
  const weighted = needles
    .map((term) => {
      const df = docs.reduce((n, d) => n + Number(d.words.frequencies.has(term)), 0);
      return { term, df, weight: Math.log1p((docs.length - df + 0.5) / (df + 0.5)) };
    })
    .sort((a, b) => b.weight - a.weight);
  const total = weighted.reduce((n, v) => n + v.weight, 0);
  const vector = encoder?.encode(query);
  const hasIdentifier = needles.some((t) => /\d/u.test(t));
  return docs
    .map(({ turn, index, words }) => {
      const matches = weighted.filter((v) => words.frequencies.has(v.term));
      const lengthPenalty = 1.2 * (0.25 + (0.75 * words.length) / mean);
      let lexical = 0;
      for (const v of matches) {
        const tf = words.frequencies.get(v.term);
        lexical += (v.weight * tf * 2.2) / (tf + lengthPenalty);
      }
      const semantic = vector ? encoder.score(vector, turn) : 0;
      // Preserve precise lookup for a short literal name or an alphanumeric identifier.
      const exact = matches.length === needles.length && (needles.length <= 2 || hasIdentifier);
      const score = vector ? semantic + (0.6 * lexical) / total + (exact ? 1 : 0) : lexical;
      return { turn, index, focus: matches[0]?.term, score, semantic, lexical };
    })
    .filter((v) => v.lexical > 0 || (vector && v.semantic >= 0.3))
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
  const sourceIds = [];
  const messages = turns.filter((turn, index) => {
    if (turn.role !== 'user' || recallCommand.test(turn.text.trim())) return false;
    sourceIds.push(index);
    return true;
  });
  const hits = retrieve(messages, query, 3, lang);
  const quotes = hits.map(
    ({ turn, focus, index }) =>
      (lang === 'tr' ? 'Mesaj ' : 'Turn ') + (sourceIds[index] + 1) + ' · ' + excerpt(turn.text, focus, lang),
  );
  return {
    text:
      quotes.join('\n\n') ||
      (lang === 'tr'
        ? 'Önceki mesajlarında eşleşen kelime bulunamadı.'
        : 'No matching words in your earlier messages.'),
    sources: hits.map(({ index }) => sourceIds[index]),
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
  function take(index, max, focus) {
    if (space < 4 || snippets.some((s) => s.index === index)) return;
    const t = turns[index],
      role = t.role === 'user' ? 3 : 4;
    const budget = Math.min(Math.floor(max), space) - (role === 4 ? 2 : 1);
    if (budget < 1) return;
    const all = tok.encode(t.text);
    let span = all;
    if (all.length > budget) {
      const at = locate(t.text, focus, lang);
      if (at >= 0) {
        const left = Math.floor(budget / 4);
        const before = left ? tok.encode(t.text.slice(0, at)).slice(-left) : [];
        span = [...before, ...tok.encode(t.text.slice(at)).slice(0, budget - before.length)];
      } else {
        const gap = tok.encode(' … ');
        if (budget < gap.length + 6) span = all.slice(-budget);
        else {
          const head = Math.floor((budget - gap.length) / 3);
          span = [...all.slice(0, head), ...gap, ...all.slice(-(budget - head - gap.length))];
        }
      }
    }
    const seq = [role, ...span, ...(role === 4 ? [2] : [])];
    snippets.push({ index, seq, partial: span.length < all.length });
    space -= seq.length;
  }
  if (attached) take(current.quote, Math.min(300, Math.ceil(limit / 3)), current.quoteFocus);
  for (const index of pins) take(index, Math.floor(pinnedBudget / pins.length), undefined);
  const hist = turns.slice(0, -1);
  const recentStart = Math.max(0, hist.length - 4);
  const stop = new Set([
    'the',
    'that',
    'this',
    'and',
    'what',
    'was',
    'are',
    'bir',
    'biraz',
    'için',
    'ile',
    'mı',
    'mi',
    'ne',
  ]);
  const search = terms(current.text, lang)
    .filter((word) => word.length >= 3 && !stop.has(word))
    .join(' ');
  if (config.recall && search && recentStart && space > 48) {
    const reserve = Math.min(164, Math.floor(space * 0.19));
    const hits = retrieve(hist.slice(0, recentStart), search, 2, lang);
    for (const hit of hits) take(hit.index, reserve / hits.length, hit.focus);
  }
  for (let i = hist.length - 1; i >= Math.max(0, hist.length - 24) && space >= 4; i--) {
    // Preserve room for the question preceding a long assistant reply.
    const keep =
      i > 0 && hist[i].role === 'assistant' && i >= recentStart ? Math.min(160, Math.floor(space * 0.28)) : 0;
    take(i, space - keep, undefined);
  }
  const ordered = snippets.sort((a, b) => a.index - b.index);
  return {
    ids: [...prefix, ...ordered.flatMap((s) => s.seq), ...tail],
    used: ordered.map((s) => s.index),
    shortened: ordered.filter((s) => s.partial).map((s) => s.index),
    systemTokens: Math.max(0, prefix.length - 3),
    truncated: ids.length > cap,
  };
}
export function promptTokens(tok, turns, lang, limit, raw) {
  return buildPrompt(tok, turns, lang, limit, raw).ids;
}
