import assert from 'node:assert/strict';
import { weightFormat } from '../web/config.mjs';
const web = { default: 'q4', formats: { q4: {}, q8: {} } };
const desktop = (memory, cores = 4) => ({
  userAgent: 'Mozilla/5.0 (X11; Linux x86_64) Chrome/130',
  deviceMemory: memory,
  hardwareConcurrency: cores,
});
const cases = [
  ['native release', { ...web, native: true, default: 'q8' }, desktop(8), 'q8'],
  ['native release without default', { ...web, native: true }, desktop(8), 'q4'],
  ['no q8 in formats', { default: 'q4', formats: { q4: {} } }, desktop(8), 'q4'],
  ['empty release', {}, desktop(8), 'q4'],
  ['phone', web, { ...desktop(8, 8), userAgent: 'Mozilla/5.0 (Linux; Android 14) Mobile' }, 'q4'],
  ['iPad', web, { ...desktop(8, 8), userAgent: 'Mozilla/5.0 (iPad; CPU OS 17_0)' }, 'q4'],
  ['8 GB desktop', web, desktop(8), 'q8'],
  ['4 GB desktop', web, desktop(4, 16), 'q4'],
  ['no deviceMemory, 4 cores', web, desktop(undefined, 4), 'q4'],
  ['no deviceMemory, 12 cores', web, desktop(undefined, 12), 'q8'],
  ['no deviceMemory, no cores', web, { userAgent: 'x' }, 'q4'],
  ['missing nav', web, undefined, 'q4'],
];
for (const [name, release, nav, want] of cases) {
  // An explicit undefined nav would pick up the global default, so pass null for "missing".
  assert.equal(weightFormat(release, nav === undefined ? null : nav), want, name);
}
console.log('config ok (' + cases.length + ' cases)');
