// Intercept explicit requests for an unsupported output language, including common Turkish forms.
const languages =
  'french|spanish|german|italian|russian|arabic|japanese|chinese|korean|portuguese|dutch|greek|swedish|hindi|polish|persian|hebrew|latin';
const others =
  'fransızca|ispanyolca|almanca|italyanca|rusça|arapça|japonca|çince|korece|portekizce|flemenkçe|hollandaca|yunanca|isveççe|hintçe|lehçe|farsça|ibranice|latince';
const enRequest = new RegExp(
  '(?:^|\\b(?:please|could you|can you|you must) )(?:(?:reply|respond|answer|speak|talk|write|say|translate|tell me)\\b[^.!?\\n]{0,90}?)\\b(?:in|into|to)\\s+(?:the\\s+)?(?:' +
    languages +
    ')\\b(?=\\s*(?:[.!?,:]|$|please\\b|for me\\b|about\\b))',
  'iu',
);
const trRequest = new RegExp(
  '(?:^|\\s)(?:' +
    others +
    ')(?:ya|ye|ya\\x27|ye\\x27)?\\s+(?:yanıt|yanit|cevap|konuş|konus|yaz|söyle|anlat|çevir)',
  'iu',
);
export function unsupportedOutput(text) {
  const s = text.toLocaleLowerCase('tr').replaceAll('’', "'");
  if (/\b(?:don't|do not|never) (?:speak|reply|answer|write|translate|respond)/iu.test(text)) return false;
  if (/(?:konuşma|yazma|çevirme|yanıt verme|cevap verme)\s*[.!?]?$/iu.test(s)) return false;
  return enRequest.test(text) || trRequest.test(s.replaceAll("'", ''));
}
export const languageNotice = {
  en: 'Franok supports English and Turkish. Choose either language beside the message box.',
  tr: 'Franok Türkçe ve İngilizceyi destekler. Mesaj kutusunun yanından bu iki dilden birini seçebilirsin.',
};
