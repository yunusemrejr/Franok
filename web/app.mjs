import {diagram} from './diagram.mjs';
import {retrieve} from './session.mjs';
const $=id=>document.getElementById(id);
const storage='franok.sessions.v1',activeKey='franok.active.v1';
let sessions=[],current,worker,busy=false,answer=null,loaded=false;
try{sessions=JSON.parse(localStorage.getItem(storage)||'[]');if(!Array.isArray(sessions))sessions=[];}catch{sessions=[];}
const make=()=>({id:crypto.randomUUID(),lang:'en',turns:[],draft:''});
current=sessions.find(s=>s.id===localStorage.getItem(activeKey))||sessions[0]||make();
function keep(){
  if(!sessions.includes(current))sessions.unshift(current);
  try{localStorage.setItem(storage,JSON.stringify(sessions,(_key,value)=>Array.isArray(value)&&value.some(t=>t?.role==='assistant'&&!t.text)?value.filter(t=>t.role!=='assistant'||t.text):value));localStorage.setItem(activeKey,current.id);}
  catch{$('status').textContent='Device storage is full. Your latest changes could not be saved.';}
}
function menu(open){document.body.classList.toggle('menu',open);$('menu').setAttribute('aria-expanded',String(open));}
function showAbout(on){$('about').hidden=!on;$('chat').hidden=on;$('viewname').textContent=on?'Architecture':'Chat';$('find').hidden=on;$('lang').hidden=on;menu(false);if(on)diagram();}
function history(){
  const nav=$('sessions');nav.replaceChildren();
  for(const s of sessions){
    const b=document.createElement('button');b.textContent=s.turns.find(t=>t.role==='user')?.text.slice(0,54)||'New chat';
    b.setAttribute('aria-current',String(s===current));b.onclick=()=>{if(busy)stop();current=s;keep();showAbout(false);paint();};nav.append(b);
  }
}
function message(t,index) {
  const item=document.createElement('article');item.className='message '+t.role;item.id='turn-'+index;
  const h=document.createElement('header'),p=document.createElement('p');
  h.textContent=t.role==='user'?(current.lang==='tr'?'Siz':'You'):'Franok';
  p.textContent=t.text;item.append(h,p);
  if(Number.isInteger(t.quote)&&current.turns[t.quote]){
    const link=document.createElement('a');link.href='#turn-'+t.quote;link.className='source';link.textContent='Attached turn '+(t.quote+1);item.append(link);
  }
  return item;
}
function paint() {
  $('quotechip').hidden=!Number.isInteger(current.quote);
  $('quotechip').querySelector('span').textContent='Attached turn '+(current.quote+1);
  $('lang').value=current.lang;$('draft').value=current.draft||'';$('draft').placeholder=current.lang==='tr'?'Bir mesaj yazın…':'Write a message…';
  $('welcome').hidden=current.turns.length>0;$('messages').replaceChildren(...current.turns.map(message));
  $('stream').scrollTop=$('stream').scrollHeight;history();$('find').disabled=!current.turns.length;
}
function flags(on){busy=on;$('send').disabled=on;$('stop').hidden=!on;$('lang').disabled=on;}
function stop(){
  if(answer&&!answer.text&&current.turns.at(-1)===answer)current.turns.pop();
  worker?.terminate();worker=null;loaded=false;flags(false);answer=null;keep();$('status').textContent='Stopped';paint();
}
function startWorker(){
  if(worker)return;
  worker=new Worker('./worker.mjs',{type:'module'});
  const fail=msg=>{if(answer&&!answer.text)current.turns.pop();answer=null;worker?.terminate();worker=null;loaded=false;flags(false);keep();paint();$('status').textContent=msg;};
  worker.onerror=e=>fail(e.message||'The runtime could not start.');
  worker.onmessage=({data:d})=>{
    if(d.kind==='download')$('status').textContent='Downloading model · '+d.pct+'%';
    if(d.kind==='ready'){loaded=true;$('status').textContent='Model ready';}
    if(d.kind==='prefill')$('status').textContent='Reading your message…';
    if(d.kind==='part'){
      answer.text=d.text;const text=$('messages').lastElementChild?.querySelector('p');if(text)text.textContent=d.text;
      $('stream').scrollTop=$('stream').scrollHeight;$('status').textContent='Generating on this device…';
    }
    if(d.kind==='done'){
      answer.text=d.text; if(!answer.text)current.turns.pop();answer=null;flags(false);keep();paint();
      $('status').textContent=d.tokens?'Generated locally · '+d.tokens+' tokens · '+(d.tokens/Math.max(.01,d.seconds)).toFixed(1)+' tokens/s':'Franok ended its reply without generating text.';
    }
    if(d.kind==='error')fail(d.message);
  };
}
$('composer').onsubmit=e=>{
  e.preventDefault();const text=$('draft').value.trim();if(!text||busy)return;
  current.turns.push({role:'user',text,...(Number.isInteger(current.quote)?{quote:current.quote}:{})});delete current.quote;current.draft='';const context=current.turns.map(t=>({...t}));
  answer={role:'assistant',text:''};current.turns.push(answer);keep();paint();flags(true);
  $('status').textContent=loaded?'Preparing…':'Loading Franok · first download is about 196 MB';
  startWorker();worker.postMessage({type:'chat',lang:current.lang,turns:context});
};
$('draft').oninput=()=>{current.draft=$('draft').value;keep();};
$('draft').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();$('composer').requestSubmit();}};
$('stop').onclick=stop;
$('new').onclick=()=>{if(busy)stop();current=make();keep();showAbout(false);paint();$('draft').focus();};
$('lang').onchange=()=>{current.lang=$('lang').value;keep();paint();};
$('menu').onclick=()=>menu(!document.body.classList.contains('menu'));
$('architecture').onclick=()=>showAbout(true);$('back').onclick=()=>showAbout(false);
for(const id of ['slot','anchor','layer'])$(id).oninput=diagram;
function search(){
  const value=$('query').value.trim();
  const hits=value?retrieve(current.turns,value,20):[];
  const list=$('matches');list.replaceChildren();
  if(value&&!hits.length)list.textContent='No matching turns.';
  for(const hit of hits){
    const row=document.createElement('div');row.className='matchrow';
    const b=document.createElement('button');b.textContent='Turn '+(hit.index+1)+' · '+hit.turn.text.slice(0,160);
    b.onclick=()=>{$('search').close();const el=$('turn-'+hit.index);el.scrollIntoView({block:'center'});el.classList.add('hit');setTimeout(()=>el.classList.remove('hit'),1600);};
    const attach=document.createElement('button');attach.className='attach';attach.textContent='Attach';
    attach.setAttribute('aria-label','Attach turn '+(hit.index+1)+' to next message');
    attach.onclick=()=>{current.quote=hit.index;keep();$('search').close();paint();$('draft').focus();};
    row.append(b,attach);list.append(row);
  }
}
$('quotechip').querySelector('button').onclick=()=>{delete current.quote;keep();paint();};
$('find').onclick=()=>{$('query').value='';search();$('search').showModal();$('query').focus();};$('query').oninput=search;
paint();
