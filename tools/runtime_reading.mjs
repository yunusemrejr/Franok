// Score the public transfer items using the same quantized WASM kernels as the app.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { createEngine } from '../web/engine.mjs';
import { Tokenizer } from '../web/tokenizer.mjs';
import { createReadout, readPassage } from '../web/reader.mjs';

const [dir, format, input, suite, result, readerDir] = process.argv.slice(2);
if (!result || !['q4', 'q8'].includes(format) || !['belebele', 'xquad', 'reader'].includes(suite))
  throw Error('Pass EXPORT q4|q8 CASES belebele|xquad|reader RESULT.jsonl [READER_DIR]');
const json = (file) => JSON.parse(fs.readFileSync(file));
const wasm = fs.readFileSync('web/franok.wasm');
const manifest = json(path.join(dir, format === 'q4' ? 'model.json' : 'model-q8.json'));
const bytes = fs.readFileSync(path.join(dir, 'model.' + format));
const sha = crypto.createHash('sha256').update(bytes).digest('hex');
if (sha !== manifest.sha256) throw Error('Quantized weight checksum mismatch');
const model = await createEngine(wasm, manifest, bytes);
const tokenizer = new Tokenizer(json(path.join(dir, 'tokenizer.json')));
const { system, suites } = json(input);
const cases = suites[suite === 'reader' ? 'xquad' : suite];
const head =
  suite === 'reader'
    ? await createReadout(
        wasm,
        fs.readFileSync(path.join(readerDir, 'head.safetensors')),
        json(path.join(readerDir, 'config.json')),
      )
    : null;
const previous = fs.existsSync(result)
  ? fs.readFileSync(result, 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse)
  : [];

function start(ids) {
  const reused = model.reusePrefix(ids, ids.length - 1);
  for (let i = reused; i < ids.length - 1; ++i) model.step(ids[i], false);
  return model.step(ids.at(-1));
}
function score(logits, token) {
  let peak = -Infinity,
    total = 0;
  for (const v of logits) peak = Math.max(peak, v);
  for (const v of logits) total += Math.exp(v - peak);
  return logits[token] - peak - Math.log(total);
}
for (const [i, item] of cases.entries()) {
  if (i < previous.length) {
    const r = previous[i];
    if (
      r.sha256 !== sha ||
      r.format !== format ||
      r.suite !== suite ||
      r.lang !== item.lang ||
      JSON.stringify(r.key) !== JSON.stringify(item.key)
    )
      throw Error('The partial result belongs to another checkpoint or case.');
    continue;
  }
  const ids = [
    1,
    item.lang === 'tr' ? 6 : 5,
    ...tokenizer.encode(system[item.lang]),
    2,
    3,
    ...tokenizer.encode(item.prompt),
    4,
  ];
  const row = { suite, format, sha256: sha, lang: item.lang, key: item.key };
  if (suite === 'belebele') {
    row.scores = item.choices.map((text) => {
      const answer = tokenizer.encode(text);
      let logits = start(ids),
        sum = 0;
      for (const [j, t] of answer.entries()) {
        sum += score(logits, t);
        if (j + 1 < answer.length) logits = model.step(t);
      }
      return sum / Array.from(text).length;
    });
    row.prediction = row.scores.indexOf(Math.max(...row.scores));
  } else if (head) {
    const prediction = readPassage(model, tokenizer, head, item.question, item.passage, item.lang);
    row.answer = prediction.quote;
    row.accepted = prediction.accepted;
    row.margin = prediction.margin;
  } else {
    let logits = start(ids);
    const reply = [];
    while (reply.length < 24) {
      let pick = 2;
      for (let j = 7; j < logits.length; j++) if (logits[j] > logits[pick]) pick = j;
      if (pick === 2) break;
      reply.push(pick);
      if (reply.length < 24) logits = model.step(pick);
    }
    row.answer = tokenizer.decode(reply).trim();
  }
  fs.appendFileSync(result, JSON.stringify(row) + '\n');
  if ((i + 1) % 40 === 0) console.log(suite, format, i + 1);
}
