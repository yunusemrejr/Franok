const $=id=>document.getElementById(id);
const ring=(i,r,depth=0)=>{const a=(i/8)*Math.PI*2-Math.PI/2;return [210+Math.cos(a)*r,118+Math.sin(a)*r*.62+depth];};
export function diagram(){
  const slot=+$('slot').value,anchor=+$('anchor').value,layer=+$('layer').value;
  for(const [key,n] of Object.entries({slot,anchor,layer}))$(key+'out').value=n;
  const bank=(slot+(anchor>>(layer%4))+layer)%8;
  const stations=Array.from({length:8},(_,i)=>{
    const [x,y]=ring(i,92),active=i===bank;
    return '<g transform="translate('+x+','+y+')"><ellipse rx="22" ry="13" cy="4" fill="#616f60"/><ellipse rx="22" ry="13" fill="'+(active?'#eeb97e':'#e4e5d3')+'" stroke="#43574a" stroke-width="2"/><text y="4" text-anchor="middle" font-family="monospace" font-size="12">'+i+'</text></g>';
  }).join('');
  const screws=[[-143,-41],[-143,52],[143,-41],[143,52]].map(([x,y])=>'<g transform="translate('+(210+x)+','+(118+y)+')"><ellipse rx="5" ry="3" fill="#a0a792" stroke="#697764"/><path d="M-3 0h6" stroke="#65705d"/></g>').join('');
  const [endX,endY]=ring(bank,68);
  $('dial').innerHTML='<ellipse cx="210" cy="164" rx="161" ry="33" fill="#dfd8c4"/><rect x="54" y="66" width="312" height="118" rx="21" fill="#777e69" stroke="#586451"/><rect x="54" y="58" width="312" height="118" rx="21" fill="#d7d9c5" stroke="#78816b"/>'+screws+'<ellipse cx="210" cy="123" rx="116" ry="76" fill="#818c79"/><ellipse cx="210" cy="118" rx="116" ry="75" fill="#c0c9b4" stroke="#6b7b67" stroke-width="2"/><ellipse cx="210" cy="118" rx="102" ry="65" fill="none" stroke="#95a48f"/>'+stations+'<path d="M210 118 L'+endX+' '+endY+'" stroke="#344d3d" stroke-width="9" stroke-linecap="round"/><ellipse cx="210" cy="118" rx="18" ry="12" fill="#8e6044"/><ellipse cx="210" cy="114" rx="18" ry="12" fill="#cc956c" stroke="#845f49"/><path d="M203 114h14" stroke="#7c533e" stroke-width="2"/>';
  $('route').textContent='Token slot '+slot+' → bank '+bank+' of 8 · layer '+layer;
}
