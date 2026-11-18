const {app,BrowserWindow,protocol,net,shell,session} = require('electron');
const path=require('node:path');
const fs=require('node:fs');
const {pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'..');
app.setName('Franok');
app.setDesktopName('franok.desktop');
if (typeof app.setAppUserModelId==='function') app.setAppUserModelId('franok');
app.enableSandbox();
protocol.registerSchemesAsPrivileged([{scheme:'franok',privileges:{standard:true,secure:true,supportFetchAPI:true,corsEnabled:true,stream:true,codeCache:true}}]);
const allowedSites=new Set(['github.com','huggingface.co','franok.yunusemrevurgun.com']);
const publicFiles=new Set(['index.html','cup.svg','style.css','app.mjs','diagram.mjs','worker.mjs','engine.mjs','session.mjs','tokenizer.mjs','franok.wasm','tokenizer.json','model.json','release.json']);
let win;
function show() {
  if(win){if(win.isMinimized())win.restore();win.show();win.focus();return;}
  win=new BrowserWindow({title:'Franok',width:1100,height:780,minWidth:360,minHeight:380,backgroundColor:'#faf7f0',icon:path.join(root,'desktop','franok.png'),autoHideMenuBar:true,webPreferences:{sandbox:true,contextIsolation:true,nodeIntegration:false}});
  win.setMenu(null);
  win.webContents.setWindowOpenHandler(()=>({action:'deny'}));
  win.webContents.on('will-navigate',(event,url)=>{
    if(url==='franok://app/index.html')return;
    event.preventDefault();const dest=new URL(url);if(dest.protocol==='https:'&&allowedSites.has(dest.hostname))shell.openExternal(dest.href);
  });
  win.on('closed',()=>{win=null;});
  win.loadURL('franok://app/index.html');
}
if(!app.requestSingleInstanceLock()){app.quit();}
else{
  app.on('second-instance',show);
  app.whenReady().then(()=>{
    session.defaultSession.setPermissionRequestHandler((_wc,_perm,answer)=>answer(false));
    protocol.handle('franok',async req=>{
      const url=new URL(req.url), file=url.pathname.slice(1);
      if(url.hostname!=='app'||req.method!=='GET')return new Response('Missing',{status:404});
      const bundle=path.join(root,'work','export','model.q4');
      if(file==='release.json'&&fs.existsSync(bundle))return Response.json({weights:'./model.q4'});
      if(file==='model.q4'&&fs.existsSync(bundle))return net.fetch(pathToFileURL(bundle).href);
      if(!publicFiles.has(file))return new Response('Missing',{status:404});
      const local=path.join(root,'web',file);
      if(!fs.existsSync(local))return new Response('Missing',{status:404});
      const types={'.mjs':'text/javascript','.css':'text/css','.html':'text/html','.svg':'image/svg+xml','.wasm':'application/wasm','.json':'application/json'};
      return new Response(fs.readFileSync(local),{headers:{'content-type':types[path.extname(file)],'Content-Security-Policy':"default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self'; worker-src 'self'; connect-src 'self' https:; object-src 'none'; base-uri 'none'"}});
    });
    show();
  });
  app.on('window-all-closed',()=>app.quit());
}
