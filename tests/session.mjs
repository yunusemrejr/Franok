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
assert.equal(
  conversationAction(
    'Ne demek istedin?',
    [...factual, { role: 'user', text: 'I don’t understand.', revisit: 0 }],
    'tr',
  ).source,
  0,
);
