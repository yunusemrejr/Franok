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
