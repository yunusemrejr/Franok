// Pure sampling function. Temperature zero is deterministic and the end token remains available.
export function sampleToken(logits, generated, options, random = Math.random) {
  const recent = new Set(generated.slice(-32)),
    banned = new Set();
  if (generated.length >= 3) {
    const [a, b, c] = generated.slice(-3);
    for (let i = 3; i < generated.length; i++)
      if (generated[i - 3] === a && generated[i - 2] === b && generated[i - 1] === c)
        banned.add(generated[i]);
  }
  const best = [];
  for (let i = 2; i < logits.length; i++) {
    if ((i < 7 && i !== 2) || banned.has(i)) continue;
    const value = logits[i] - (recent.has(i) ? 0.55 : 0);
    if (best.length === 48 && value <= best[47][1]) continue;
    let j = best.length;
    while (j > 0 && best[j - 1][1] < value) j--;
    best.splice(j, 0, [i, value]);
    if (best.length > 48) best.pop();
  }
  if (options.temperature <= 0.001) return best[0][0];
  const weights = best.map(([, v]) => Math.exp((v - best[0][1]) / options.temperature));
  const cutoff = weights.reduce((a, b) => a + b, 0) * options.topP;
  let total = 0,
    n = 0;
  while (n < weights.length && total < cutoff) total += weights[n++];
  let draw = random() * total;
  for (let j = 0; j < n; j++) if ((draw -= weights[j]) <= 0) return best[j][0];
  return best[0][0];
}
