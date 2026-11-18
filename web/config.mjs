export const defaults = {
  en: 'You are Franok. Reply in English. Answer the latest message directly and use earlier turns when relevant.',
  tr: "Sen Franok'sun. Türkçe yanıt ver. Son mesaja doğrudan cevap ver ve gerektiğinde önceki konuşmayı kullan.",
};
const clamp = (value, lo, hi, otherwise) =>
  Number.isFinite(+value) ? Math.max(lo, Math.min(hi, +value)) : otherwise;
export function settings(raw = {}, windowSize = 1024) {
  return {
    system: typeof raw.system === 'string' ? raw.system.slice(0, 1800) : '',
    temperature: clamp(raw.temperature, 0, 1.5, 0.58),
    topP: clamp(raw.topP, 0.15, 1, 0.89),
    maxNew: Math.round(clamp(raw.maxNew, 8, Math.min(256, Math.floor(windowSize / 2)), 80)),
    calculator: raw.calculator !== false,
    conversationHelp: raw.conversationHelp !== false,
    recall: raw.recall !== false,
    enterSends: raw.enterSends !== false,
  };
}
