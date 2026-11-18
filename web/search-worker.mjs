import { RecallEncoder } from './semantic.mjs';
import { retrieve } from './session.mjs';
import { Tokenizer } from './tokenizer.mjs';

let turns, spec, encoder;
async function load() {
  const [weights, tokenizer] = await Promise.all([fetch(spec.url), fetch('./tokenizer.json')]);
  if (!weights.ok || !tokenizer.ok) throw Error('Search files unavailable.');
  const bytes = await weights.arrayBuffer();
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  if (Array.from(digest, b => b.toString(16).padStart(2, '0')).join('') !== spec.sha256)
    throw Error('Search weights failed verification.');
  return new RecallEncoder(new Tokenizer(await tokenizer.json()), spec, bytes);
}

onmessage = async ({data}) => {
  if ('turns' in data) {
    turns = data.turns;
    spec = data.spec;
    return;
  }
  const { sequence, query, lang } = data;
  try {
    encoder ??= load();
    const model = await encoder;
    const matches = retrieve(turns, query, 20, lang, model).map(({index, focus, lexical}) => ({index, focus, lexical}));
    postMessage({sequence, matches});
  } catch {
    encoder = null;
    postMessage({sequence, error: true});
  }
};
