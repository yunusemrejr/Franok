import { loadRecallEncoder } from './semantic.mjs';
import { retrieve } from './session.mjs';
import { Tokenizer } from './tokenizer.mjs';

let turns, spec, encoder;
async function load() {
  const response = await fetch('./tokenizer.json');
  if (!response.ok) throw Error('Tokenizer unavailable.');
  return loadRecallEncoder(new Tokenizer(await response.json()), spec);
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
