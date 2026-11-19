import assert from 'node:assert/strict';
import { detectLanguage } from '../web/language.mjs';
const cases = [
  ['hey how are you', 'en'],
  ['can you help me with math?', 'en'],
  ['write me a poem', 'en'],
  ['who r u', 'en'],
  ['I am tired', 'en'],
  ['what is 12 plus 30', 'en'],
  ['selam naber', 'tr'],
  ['sen kimsin', 'tr'],
  ['bana bir şiir yaz', 'tr'],
  ['cayim hep aci oluyor, neyi yanlis yapiyorum?', 'tr'],
  ['Bir haftada kaç gün var?', 'tr'],
  ['bugün ayın kaçı?', 'tr'],
  ['yumurta nasıl haşlanır', 'tr'],
  ['matematikte bana yardım eder misin?', 'tr'],
  ['calculus', null],
  ['42', null],
  ['ok', 'en'],
  ['kesirler', null],
  ['python', null],
];
for (const [text, want] of cases) assert.equal(detectLanguage(text), want, text);
console.log('ok: language detection,', cases.length, 'cases');
