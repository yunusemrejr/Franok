import assert from 'node:assert/strict';
import { retrieve, promptTokens, buildPrompt, recallMessage } from '../web/session.mjs';
const turns = Array.from({ length: 1000 }, (_, i) => ({
  role: i % 2 ? 'assistant' : 'user',
  text: 'Regular entry ' + i,
}));
turns[7].text = 'Kahve kutusunun kodu: ZÜMRÜT-719.';
const hits = retrieve(turns, 'ZÜMRÜT-719');
assert.equal(hits[0].index, 7);
const tok = { encode: (text) => Array.from(new TextEncoder().encode(text)) };
turns.push({ role: 'user', text: 'What was the code?', quote: 7 });
const prompt = promptTokens(tok, turns, 'en', 184);
assert(prompt.length <= 184);
const bytes = Array.from(tok.encode('ZÜMRÜT-719'));
assert(prompt.some((_, i) => bytes.every((b, j) => prompt[i + j] === b)));
console.log('Retrieved and attached turn 8 after 1000 saved turns.');

import { calculate } from '../web/calculator.mjs';
assert.equal(calculate('1/3 + 1/6', 'en').text, '0.5');
assert.equal(calculate('hesapla 0,1 + 0,2', 'tr').text, '0,3');
assert.equal(calculate('/calc -(4 + 7)/3', 'en').text, '-11/3');
assert.equal(calculate('1/0', 'en').text, 'Division by zero is undefined.');
for (const q of ['five plus one', '1 2 + 1', 'What is 2+2 for my plan?', '2..2+3'])
  assert.equal(calculate(q, 'en'), null);

import { unsupportedOutput } from '../web/language.mjs';
for (const msg of ['Please respond in Spanish.', 'Write a poem in Japanese', 'Bunu Fransızcaya çevir.'])
  assert.equal(unsupportedOutput(msg), true);
for (const msg of [
  'Tell me about coffee made in French presses.',
  'Translate French into Turkish.',
  'What does the French word café mean?',
  'Please respond in English.',
  'Almanca konuşma.',
])
  assert.equal(unsupportedOutput(msg), false);

const retained = buildPrompt(
  tok,
  [...turns.slice(0, 1000), { role: 'user', text: 'Different topic' }],
  'en',
  220,
  { recall: false },
  true,
  [7],
);
assert(retained.used.includes(7));
assert(retained.ids.length <= 220);
const archive = Array.from({ length: 1000 }, () => ({ role: 'user', text: 'An ordinary note.' }));
archive[6] = { role: 'user', text: 'Çay kutusunun etiketi ZÜMRÜT-719.' };
archive[997] = { role: 'assistant', text: 'ZÜMRÜT-333' };
archive[998] = { role: 'user', text: '/recall ZÜMRÜT' };
const quoted = recallMessage('/hatırla ZÜMRÜT', archive, 'tr');
assert.deepEqual(quoted.sources, [6]);
assert.equal(quoted.text, 'Mesaj 7 · ' + archive[6].text);
assert.equal(recallMessage('recall something', archive, 'en'), null);
assert.match(recallMessage('/recall missing', archive, 'en').text, /No matching words/);

import { conversationAction } from '../web/conversation.mjs';
assert.match(conversationAction('Hi!', [], 'en').text, /talk about/);
assert.match(conversationAction('selam!', [], 'tr').text, /Merhaba/);
assert.equal(conversationAction('Hello, why is the sky blue?', [], 'en'), null);
const greeting = [
  { role: 'user', text: 'Hi there' },
  { role: 'assistant', text: 'Hi!', tool: 'social' },
];
assert.match(conversationAction('wdym?', greeting, 'en').text, /saying hello/);
const factual = [
  { role: 'user', text: 'Why do rainbows form?' },
  { role: 'assistant', text: 'Unclear answer.' },
];
assert.equal(conversationAction('I don’t understand.', factual, 'en').source, 0);
assert.match(
  conversationAction(
    'Ne demek istedin?',
    [...factual, { role: 'user', text: 'I don’t understand.', revisit: 0 }],
    'tr',
  ).text,
  /açık bir yanıt/,
);
assert.match(conversationAction('wdym', [
  { role: 'user', text: 'hi' },
  { role: 'assistant', text: 'An unrelated answer from an earlier version.' },
], 'en').text, /You said hello/);

const sentence = 'The last word is coriander.';
const previous = [{role:'user',text:'Tell me the ingredient.'},{role:'assistant',text:'First, check the recipe. '.repeat(15)+sentence}];
const preserved=buildPrompt(tok,[...previous,{role:'user',text:'Which was the last word?'}],'en',900,{recall:false});
const has = (ids,text) => ids.some((_,i) => tok.encode(text).every((b,j) => ids[i+j]===b));
assert(has(preserved.ids,sentence));
assert.deepEqual(preserved.shortened,[]);
const longReply=buildPrompt(tok,[{role:'assistant',text:'First caveat. '+'Example '.repeat(500)+sentence},{role:'user',text:'Last word?'}],'en',400);
assert(has(longReply.ids,sentence) && has(longReply.ids,'First caveat.'));
assert.deepEqual(longReply.shortened,[0]);
const searchText='Genel notlar. '.repeat(260)+'IŞIK numarası K99-71. Gün tamamlandı.';
const late=[{role:'user',text:searchText},...Array.from({length:50},()=>({role:'user',text:'Genel notlar.'}))];
const cited=recallMessage('/hatırla genel ışık',late,'tr');
assert(cited.text.startsWith('Mesaj 1 · ') && cited.text.includes('IŞIK numarası K99-71'));
const attached=buildPrompt(tok,[...late,{role:'user',text:'Kod neydi?',quote:0,quoteFocus:'ışık'}],'tr',400);
assert(has(attached.ids,'IŞIK numarası K99-71') && attached.ids.length<=400);
const short=Array.from({length:12},(_,i)=>({role:i%2?'assistant':'user',text:String(i)}));
assert.equal(buildPrompt(tok,[...short,{role:'user',text:'Continue.'}],'en',400,{recall:false}).used.length,12);
console.log('Preserved recent answers and found a late Turkish citation in a long session.');
