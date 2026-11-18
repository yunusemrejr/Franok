import fs from 'node:fs';
import { retrieve } from '../../web/session.mjs';
import { Tokenizer } from '../../web/tokenizer.mjs';
import { RecallEncoder } from '../../web/semantic.mjs';
const root = (process.argv[2] || 'work/v3/semantic') + '/';
const model = new RecallEncoder(
  new Tokenizer(JSON.parse(fs.readFileSync('web/tokenizer.json'))),
  JSON.parse(fs.readFileSync(root + 'recall.json')),
  fs.readFileSync(root + 'recall.bin'),
);
const sets = JSON.parse(fs.readFileSync('work/v3/semantic/evaluation/sets.json')),
  result = {};
for (const [name, set] of Object.entries(sets)) {
  const turns = set.docs.map((text) => ({ text }));
  const rows = [];
  for (const [query, gold, lang] of set.queries) {
    const x = model.encode(query),
      n = turns.length;
    const lex = retrieve(turns, query, n, lang);
    const hybrid = retrieve(turns, query, n, lang, model);
    const values = turns.map((t, i) => ({ i, semantic: model.score(x, t) }));
    const rank = () =>
      values.toSorted((a, b) => b.semantic - a.semantic || b.i - a.i).findIndex((r) => r.i === gold) + 1;
    rows.push({
      query,
      gold,
      lexical: lex.findIndex((r) => r.index === gold) + 1 || n + 1,
      semantic: rank(),
      hybrid: hybrid.findIndex((r) => r.index === gold) + 1 || n + 1,
    });
  }
  const summary = {};
  for (const k of ['lexical', 'semantic', 'hybrid'])
    summary[k] = {
      top1: rows.filter((r) => r[k] === 1).length,
      top3: rows.filter((r) => r[k] <= 3).length,
      mrr: rows.reduce((s, r) => s + 1 / r[k], 0) / rows.length,
    };
  result[name] = { n: turns.length, queries: rows.length, summary, rows };
  console.log(name, rows.length, JSON.stringify(summary));
}
fs.mkdirSync(root + 'evaluation', { recursive: true });
fs.writeFileSync(root + 'evaluation/result.json', JSON.stringify(result, null, 2) + '\n');
