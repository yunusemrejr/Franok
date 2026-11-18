// Only standalone social turns use this layer. Substantive questions always go to the neural model.
const greetings = new Set([
  'hi',
  'hello',
  'hey',
  'hiya',
  'hey there',
  'hi there',
  'hello there',
  'good morning',
  'good afternoon',
  'good evening',
  'selam',
  'merhaba',
  'selamlar',
  'günaydın',
  'iyi akşamlar',
]);
const repairs = new Set([
  'wdym',
  'what do you mean',
  'what did you mean',
  'what does that mean',
  'can you explain that',
  'could you explain that',
  'please clarify',
  'could you clarify',
  'i do not follow',
  "i don't follow",
  "i don't understand",
  'i do not understand',
  'you make no sense',
  'that makes no sense',
  "that doesn't make sense",
  'that does not make sense',
  'ne demek',
  'ne demek istedin',
  'ne demek istiyorsun',
  'anlamadım',
  'bir daha açıklar mısın',
  'daha açık söyler misin',
  'söylediğini anlamadım',
  'bu hiç mantıklı değil',
  'söylediğin mantıklı değil',
]);
function normalized(text) {
  return text
    .normalize('NFKC')
    .toLowerCase()
    .replaceAll('i\u0307', 'i')
    .replaceAll('’', "'")
    .trim()
    .replace(/[.!?…]+$/u, '')
    .trim()
    .replace(/\s+/gu, ' ');
}
export function conversationAction(text, history, lang) {
  const key = normalized(text);
  if (greetings.has(key))
    return {
      canonical: lang === 'tr' ? 'Merhaba!' : 'Hello!',
      text:
        lang === 'tr' ? 'Merhaba! Ne hakkında konuşmak istersin?' : 'Hi! What would you like to talk about?',
    };
  if (!repairs.has(key)) return null;
  const lastUser = history.findLastIndex((t) => t.role === 'user');
  if (lastUser >= 0) {
    const original = history[lastUser];
    const previous = normalized(original.text);
    if (Number.isInteger(original.revisit)) {
      return {
        text: lang === 'tr'
          ? 'Soruna açık bir yanıt veremedim. Hangi kısmı birlikte ele alalım?'
          : 'I haven’t given you a clear answer. Which part of your question should we focus on?',
      };
    }
    if (!greetings.has(previous) && !repairs.has(previous) && !original.text.startsWith('/')) {
      // Re-answer the user's actual question. Exclude the answer they already found confusing.
      return {
        source: lastUser,
        instruction: lang === 'tr' ? 'Kısa ve sade bir cevap ver.' : 'Give a short answer in plain language.',
      };
    }
  }
  if (lastUser >= 0 && greetings.has(normalized(history[lastUser].text))) {
    const reply = history[lastUser + 1];
    if (!reply?.canonicalGreeting && reply?.tool !== 'social')
      return {
        text: lang === 'tr'
          ? 'Selam vermiştin. Oradan başlayalım. Ne hakkında konuşmak istersin?'
          : 'You said hello. Let’s start there. What would you like to talk about?',
      };
    return {
      text:
        lang === 'tr'
          ? 'Sadece selamlaşıyordum. Aklında bir soru var mı?'
          : 'I was just saying hello. Is there something on your mind?',
    };
  }
  return {
    text:
      lang === 'tr'
        ? 'Baştan başlayalım. Hangi soruyu yanıtlamamı istersin?'
        : 'Let’s start again. What question would you like me to answer?',
  };
}
