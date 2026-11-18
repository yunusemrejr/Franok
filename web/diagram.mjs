const $ = (id) => document.getElementById(id);
export function diagram() {
  const slot = +$('slot').value,
    anchor = +$('anchor').value,
    layer = +$('layer').value;
  for (const [id, n] of Object.entries({ slot, anchor, layer })) $(id + 'out').value = n;
  const rotation = ((anchor >> layer % 4) + layer) % 8,
    bank = (slot + rotation) % 8;
  const x = (i) => 96 + 67 * i;
  let wires = '',
    nodes = '',
    active = '';
  for (let i = 0; i < 8; i++) {
    const b = (i + rotation) % 8;
    const d = 'M' + x(i) + ' 62 C' + x(i) + ' 109 ' + x(b) + ' 112 ' + x(b) + ' 158';
    if (i === slot) active = '<path d="' + d + '" class="litwire"/>';
    else wires += '<path d="' + d + '" class="wire"/>';
    nodes +=
      '<g class="' +
      (i === slot ? 'litnode' : 'node') +
      '"><rect x="' +
      (x(i) - 19) +
      '" y="29" width="38" height="32" rx="3"/><text x="' +
      x(i) +
      '" y="50">' +
      i +
      '</text></g>';
    nodes +=
      '<g class="' +
      (i === bank ? 'litnode' : 'node') +
      '"><rect x="' +
      (x(i) - 23) +
      '" y="158" width="46" height="35" rx="3"/><text x="' +
      x(i) +
      '" y="181">' +
      i +
      '</text></g>';
  }
  $('dial').innerHTML =
    '<g fill="none">' +
    wires +
    active +
    '</g>' +
    nodes +
    '<g class="diagramlabel"><text x="12" y="50">SLOT</text><text x="12" y="181">BANK</text></g>';
  $('route').textContent = 'Token slot ' + slot + ' → bank ' + bank + ' of 8 · layer ' + layer;
}

const examples = {
  en: 'I made coffee before the morning train.',
  tr: 'Sabah treninden önce kahve yaptım.',
};
let traceLoading,
  traceIds,
  selected = -1;
const decodePiece = new TextDecoder('utf-8', { fatal: true });
function pieceLabel(tokenizer, id) {
  const control = { 1: 'Start', 2: 'End', 5: 'EN', 6: 'TR' }[id];
  if (control) return control;
  try {
    return decodePiece.decode(tokenizer.pieces[id]).replaceAll(' ', '▁');
  } catch {
    return Array.from(tokenizer.pieces[id], (b) => b.toString(16).padStart(2, '0')).join(' ');
  }
}
function selectPiece(i) {
  selected = i;
  const group = Math.floor(i / 8);
  $('slot').value = i % 8;
  $('anchor').value = group ? traceIds[group * 8 - 1] : 0;
  for (const [index, button] of Array.from($('tracePieces').children).entries()) {
    button.setAttribute('aria-pressed', String(index === i));
    button.tabIndex = index === i ? 0 : -1;
  }
  $('traceInfo').textContent =
    'Token ID ' +
    traceIds[i] +
    ' · group ' +
    (group + 1) +
    ' · ' +
    (group ? 'anchor ID ' + $('anchor').value : 'first group, anchor 0');
  diagram();
}
async function initTrace() {
  const [{ Tokenizer }, response] = await Promise.all([import('./tokenizer.mjs'), fetch('./tokenizer.json')]);
  if (!response.ok) throw Error('Tokenizer unavailable');
  const tokenizer = new Tokenizer(await response.json());
  const lang = $('traceLanguage'),
    field = $('traceText'),
    pieces = $('tracePieces');
  function render() {
    traceIds = [1, lang.value === 'tr' ? 6 : 5, ...tokenizer.encode(field.value), 2];
    const buttons = traceIds.map((id, i) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = pieceLabel(tokenizer, id);
      b.className = id < 7 ? 'trace-control' : 'trace-piece';
      b.setAttribute('aria-label', b.textContent + ', token ID ' + id + ', position ' + (i + 1));
      b.onclick = () => selectPiece(i);
      return b;
    });
    pieces.replaceChildren(...buttons);
    selectPiece(Math.min(2, traceIds.length - 1));
  }
  lang.onchange = () => {
    if (!field.value || Object.values(examples).includes(field.value)) field.value = examples[lang.value];
    render();
  };
  field.oninput = render;
  pieces.onkeydown = (e) => {
    const step = { ArrowLeft: -1, ArrowRight: 1 }[e.key];
    if (!step) return;
    e.preventDefault();
    const i = Math.max(0, Math.min(traceIds.length - 1, selected + step));
    selectPiece(i);
    pieces.children[i].focus();
  };
  for (const id of ['slot', 'anchor'])
    $(id).addEventListener('input', () => {
      if (selected >= 0) pieces.children[selected]?.setAttribute('aria-pressed', 'false');
      $('traceInfo').textContent = 'Manual routing example';
    });
  field.value = examples[lang.value];
  render();
  $('tokenExplorer').hidden = false;
}
export function openTrace() {
  traceLoading ??= initTrace().catch(() => {
    traceLoading = undefined;
  });
}
