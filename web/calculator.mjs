// Exact rational arithmetic for standalone expressions. Natural-language questions go to Franok.
const gcd = (a, b) => {
  a = a < 0n ? -a : a;
  b = b < 0n ? -b : b;
  while (b) [a, b] = [b, a % b];
  return a || 1n;
};
function reduced(n, d) {
  if (d === 0n) throw Error('zero');
  if (d < 0n) {
    n = -n;
    d = -d;
  }
  const g = gcd(n, d);
  return [n / g, d / g];
}
function decimal(n, d, lang) {
  const sign = n < 0n ? '-' : '';
  if (n < 0n) n = -n;
  const whole = n / d;
  let rem = n % d,
    digits = '';
  if (!rem) return sign + whole;
  while (rem && digits.length < 12) {
    rem *= 10n;
    digits += rem / d;
    rem %= d;
  }
  if (rem) return sign + n + '/' + d;
  return sign + whole + (lang === 'tr' ? ',' : '.') + digits;
}
export function calculate(message, lang) {
  let expr = message.trim();
  const explicit = /^(?:\/calc\s+|calculate\s+|hesapla\s+)/i.test(expr);
  expr = expr
    .replace(/^(?:\/calc\s+|calculate\s+|hesapla\s+)/i, '')
    .replace(/[=?]\s*$/, '')
    .trim();
  if (!expr || expr.length > 120 || (!explicit && !/[+*/×÷−-]/.test(expr))) return null;
  if (!/^[\d\s.,+*/×÷−()-]+$/u.test(expr) || (lang !== 'tr' && expr.includes(','))) return null;
  const compact = expr
    .replace(/\s+/g, '')
    .replaceAll('×', '*')
    .replaceAll('÷', '/')
    .replaceAll('−', '-')
    .replaceAll(',', '.');
  // Reject two adjacent numbers separated by whitespace instead of joining them.
  if (/\d\s+\d/.test(expr)) return null;
  const tokens = compact.match(/\d+(?:\.\d+)?|[()+*/-]/g) || [];
  if (tokens.join('') !== compact || tokens.length > 70 || tokens.some((t) => t.length > 21)) return null;
  let at = 0;
  function atom() {
    const t = tokens[at++];
    if (t === '+') return atom();
    if (t === '-') {
      const [n, d] = atom();
      return [-n, d];
    }
    if (t === '(') {
      const v = sum();
      if (tokens[at++] !== ')') throw Error('syntax');
      return v;
    }
    if (!t || !/^\d/.test(t)) throw Error('syntax');
    const [w, f = ''] = t.split('.');
    return reduced(BigInt(w + f), 10n ** BigInt(f.length));
  }
  function product() {
    let [n, d] = atom();
    while (tokens[at] === '*' || tokens[at] === '/') {
      const op = tokens[at++],
        [a, b] = atom();
      [n, d] = op === '*' ? reduced(n * a, d * b) : reduced(n * b, d * a);
    }
    return [n, d];
  }
  function sum() {
    let [n, d] = product();
    while (tokens[at] === '+' || tokens[at] === '-') {
      const op = tokens[at++],
        [a, b] = product();
      [n, d] = reduced(n * b + (op === '+' ? a : -a) * d, d * b);
    }
    return [n, d];
  }
  try {
    const [n, d] = sum();
    if (at !== tokens.length) return null;
    return { expression: expr, text: decimal(n, d, lang) };
  } catch (e) {
    return e.message === 'zero'
      ? {
          expression: expr,
          text: lang === 'tr' ? 'Sıfıra bölme tanımsızdır.' : 'Division by zero is undefined.',
        }
      : null;
  }
}
