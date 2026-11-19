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

// Which of the two supported languages a message is written in; null when the words give no clear sign.
const trWords = new Set(
  'merhaba selam selamlar naber nasılsın nasilsin nasıl nasil neden niye nedir kim kimsin kaç kac hangi nerede evet hayır hayir tamam lütfen lutfen teşekkürler tesekkurler teşekkür tesekkur sağol sagol için icin değil degil var yok ben sen biz siz bana sana beni seni benim senin bir bu şu su ve veya ama çünkü cunku çok cok iyi kötü kotu güzel guzel şey sey yardım yardim yaz yazar söyle soyle anlat misin mısın musun müsün miyim mıyım mi mı mu mü ne gibi daha kadar bugün bugun yarın yarin şimdi simdi günaydın gunaydin adın adin hakkında hakkinda istiyorum lazım lazim olur olsun peki yani abi kanka ya'.split(
    ' ',
  ),
);
const enWords = new Set(
  "the a an is are am was were be you your i my me we it this that these what who why how when where which can could would should will do does did have has not no yes and or but if so to of in on at for with about from hello hi hey thanks thank please help write tell give make show explain know want need like good bad ok okay there here any some one two more much many im i'm don't dont what's whats it's".split(
    ' ',
  ),
);
const trEnding = /(?:[iu]yor(?:um|sun|uz|lar)?|[ae]c[ae][kg]\w{0,4}|m[ae]l[iı]y\w+|[iu]yorsun|l[ae]r[iı]n?[iı]?|[dt][iu][gk]\w{1,4})$/u;
export function detectLanguage(text) {
  const lower = text.replaceAll('I', 'ı').replaceAll('İ', 'i').toLowerCase().replaceAll('’', "'");
  const words = lower.match(/[\p{L}']+/gu) || [];
  let tr = 0;
  let en = 0;
  for (const word of words) {
    const ascii = word.replaceAll('ı', 'i');
    if (/[çğışö]/u.test(word) && !enWords.has(ascii)) tr += 2;
    else if (enWords.has(ascii) && !trWords.has(word)) en += 1;
    else if (trWords.has(word) || trWords.has(ascii)) tr += 1;
    else if (/ü/u.test(word) || trEnding.test(ascii)) tr += 1;
  }
  if (tr > en) return 'tr';
  if (en > tr) return 'en';
  return null;
}
