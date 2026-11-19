export const defaults = {
  en: 'You are Franok. Reply in English. Answer the latest message directly and use earlier turns when relevant.',
  tr: "Sen Franok'sun. Türkçe yanıt ver. Son mesaja doğrudan cevap ver ve gerektiğinde önceki konuşmayı kullan.",
};
const clamp = (value, lo, hi, otherwise) =>
  Number.isFinite(+value) ? Math.max(lo, Math.min(hi, +value)) : otherwise;
export function settings(raw = {}, windowSize = 1024) {
  return {
    system: typeof raw.system === 'string' ? raw.system.slice(0, 1800) : '',
    temperature: clamp(raw.temperature, 0, 1.5, 0),
    topP: clamp(raw.topP, 0.15, 1, 0.89),
    maxNew: Math.round(clamp(raw.maxNew, 8, Math.min(256, Math.floor(windowSize / 2)), 80)),
    calculator: raw.calculator !== false,
    conversationHelp: raw.conversationHelp !== false,
    recall: raw.recall !== false,
    semanticRecall: raw.semanticRecall === true,
    enterSends: raw.enterSends !== false,
  };
}
// Q8 is closer to the trained weights; phones and small machines keep the smaller Q4 file.
export function weightFormat(release = {}, nav = globalThis.navigator) {
  const fallback = release.default || 'q4';
  if (release.native || !release.formats?.q8 || !nav) return fallback;
  const phone = /Android|iPhone|iPad|iPod|Mobile/i.test(nav.userAgent || '');
  const memory = nav.deviceMemory;
  const roomy = memory === undefined ? nav.hardwareConcurrency >= 8 : memory >= 8;
  return !phone && roomy ? 'q8' : fallback;
}
