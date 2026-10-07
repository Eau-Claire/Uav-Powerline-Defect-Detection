const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
const path=require('node:path');

test('single live canvas receives AI, survives tab switch, and reconnects after stop',async()=>{
  const elements=new Map(), handlers={}, sockets=[];
  let rectangles=0;
  function element(id){
    if(!elements.has(id))elements.set(id,{
      value:id==='camera-url'?'http://192.168.1.20/video':'1',
      classList:{add(){},remove(){},toggle(){}},setAttribute(){},replaceChildren(){},append(){},
      getContext(){return {drawImage(){},setLineDash(){},strokeRect(){rectangles++;},fillRect(){},fillText(){},measureText(){return {width:30};}}}
    });return elements.get(id);
  }
  class Socket{
    constructor(){sockets.push(this);this.closed=false;}
    send(data){this.config=JSON.parse(data);}
    close(){this.closed=true;this.onclose?.();}
  }
  const document={hidden:false,getElementById:element,createElement:()=>({append(){}}),addEventListener:(name,fn)=>handlers[name]=fn};
  const context=vm.createContext({document,window:{addEventListener(){}},location:{host:'127.0.0.1:8000',protocol:'http:'},
    WebSocket:Socket,ArrayBuffer,File,Blob,FormData,AbortSignal,performance,setInterval(){},requestAnimationFrame:fn=>fn(),
    createImageBitmap:async()=>({width:32,height:24,close(){}}),
    fetch:async(url)=>{assert.equal(url,'/ui-config');return {ok:true,json:async()=>({camera_bridge:true,token_managed:true})};}
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname,'static/app.js'),'utf8'),context);
  const flush=()=>new Promise(resolve=>setImmediate(resolve));await flush();
  element('stream-start').onclick();sockets[0].onopen();
  assert.equal(sockets[0].config.url,'http://192.168.1.20/video');
  element('stream-start').onclick();assert.equal(sockets.length,1);
  sockets[0].onmessage({data:new ArrayBuffer(20)});await flush();
  sockets[0].onmessage({data:JSON.stringify({type:'result',count:1,inference_ms:10,total_ms:50,width:32,height:24,captured_at:Date.now(),detections:[{class_id:5,class_name:'dx_jypps',confidence:.8,bbox:[1,2,20,22]}]})});
  assert.equal(element('count').textContent,1);assert.ok(rectangles>0,'AI boxes must be drawn on the live canvas');
  document.hidden=true;handlers.visibilitychange();assert.equal(sockets[0].closed,false);
  document.hidden=false;handlers.visibilitychange();assert.equal(sockets[0].closed,false);
  sockets[0].onmessage({data:JSON.stringify({type:'ai_error',message:'checkpoint missing'})});
  assert.match(element('ai-status').textContent,/checkpoint missing/);assert.equal(sockets[0].closed,false);
  element('stream-stop').onclick();assert.equal(sockets[0].closed,true);assert.equal(element('stream-start').disabled,false);
  element('stream-start').onclick();assert.equal(sockets.length,2);
});
