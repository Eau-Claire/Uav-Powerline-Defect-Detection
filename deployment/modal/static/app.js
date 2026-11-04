"use strict";
const $ = id => document.getElementById(id);
const state = {file:null, bitmap:null, busy:false, loading:false, bridge:false, socket:null,
  result:null, decoding:false, pending:null, processing:false, processingAt:0,
  drawQueued:false, frames:0, rateAt:performance.now(), lastResultAt:0, lastPainted:null};
const colors = ["#eab308","#06b6d4","#a78bfa","#fb7185","#fb923c","#4ade80","#38bdf8"];
function status(message,error=false) { $("status").textContent=message; $("status").classList.toggle("error",error); }
function controls() {
  const locked=state.busy||state.loading||!!state.socket;
  $("detect").disabled=locked||!state.file;
  $("capture").disabled=locked||!state.bridge;
  $("file").disabled=locked;
  $("stream-start").disabled=locked||!state.bridge;
  $("stream-stop").disabled=!state.socket;
  $("camera-url").disabled=$("stream-delay").disabled=!!state.socket;
  $("upload-tab").disabled=$("camera-tab").disabled=state.busy||state.loading;
}
function clearResults() {
  state.result=null;
  for(const id of ["count","inference","latency"]) $(id).textContent="—";
  $("detections").replaceChildren(); $("results-table").hidden=true;
  $("result-placeholder").hidden=false; $("result-placeholder").textContent="Chưa có kết quả phân tích.";
}
function draw() {
  if(state.drawQueued)return;
  state.drawQueued=true;
  requestAnimationFrame(()=>{state.drawQueued=false;paint();});
}
function paint() {
  const bitmap=state.bitmap; if(!bitmap) return;
  const canvas=$("canvas"),ctx=canvas.getContext("2d");
  const scale=Math.min(1,1600/Math.max(bitmap.width,bitmap.height));
  const width=Math.round(bitmap.width*scale),height=Math.round(bitmap.height*scale);
  if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}
  ctx.drawImage(bitmap,0,0,width,height);
  if(state.socket&&state.lastPainted!==bitmap)state.frames++;
  state.lastPainted=bitmap;
  const result=state.result;
  const stale=!!state.socket&&result&&(Date.now()-result.captured_at)>2000;
  const font=Math.max(13,width/65);ctx.font=`600 ${font}px system-ui`;
  ctx.setLineDash(stale?[8,5]:[]);
  if(result&&(!result.width||(result.width===bitmap.width&&result.height===bitmap.height))) {
    for(const item of result.detections) {
      const [x1,y1,x2,y2]=item.bbox.map(v=>v*scale), color=colors[item.class_id%colors.length];
      ctx.strokeStyle=color;ctx.lineWidth=Math.max(2,width/500);ctx.strokeRect(x1,y1,x2-x1,y2-y1);
      const label=`${item.class_name} ${(item.confidence*100).toFixed(1)}%`;
      const w=ctx.measureText(label).width+12,h=font+10;
      const x=Math.max(0,Math.min(x1,width-w)),y=Math.max(0,y1-h);
      ctx.fillStyle=color;ctx.fillRect(x,y,w,h);ctx.fillStyle="#15201b";ctx.fillText(label,x+6,y+font+1);
    }
  }
  ctx.setLineDash([]);canvas.hidden=false;$("empty").hidden=true;
}
function showResult(result,totalMs) {
  state.result=result;
  $("count").textContent=result.count;$("inference").textContent=`${result.inference_ms.toFixed(0)} ms`;
  $("latency").textContent=`${(totalMs/1000).toFixed(2)} s`;
  $("detections").replaceChildren();$("results-table").hidden=result.count===0;
  $("result-placeholder").hidden=result.count>0;
  $("result-placeholder").textContent="AI đã phân tích: không thấy đối tượng ở confidence 0.10.";
  for(const item of result.detections) {
    const row=document.createElement("tr");
    for(const value of [`${item.class_id} · ${item.class_name}`,`${(item.confidence*100).toFixed(1)}%`,item.bbox.map(v=>v.toFixed(1)).join(", ")]) {
      const cell=document.createElement("td");cell.textContent=value;row.append(cell);
    }
    $("detections").append(row);
  }
  draw();
}
async function selectFile(file,source="UPLOAD") {
  if(state.busy||state.loading||state.socket||!file)return;
  state.loading=true;controls();
  try {
    if(!["image/jpeg","image/png"].includes(file.type))throw new Error("Chỉ hỗ trợ JPEG/PNG. Chuyển ảnh HEIC sang JPEG trước.");
    if(file.size>10*1024*1024)throw new Error("Ảnh vượt quá 10 MiB.");
    const bitmap=await createImageBitmap(file,{imageOrientation:"none"});
    if(bitmap.width*bitmap.height>20_000_000){bitmap.close();throw new Error("Ảnh vượt quá 20 megapixel.");}
    state.bitmap?.close();state.bitmap=bitmap;state.file=file;
    $("file-name").textContent=`${file.name} · ${bitmap.width} × ${bitmap.height}`;
    $("source-label").textContent=source;clearResults();draw();status("Ảnh sẵn sàng. Bấm phân tích để nhận diện.");
  }catch(error){status(error.message,true);}finally{state.loading=false;controls();}
}
function stopStream(message="Đã dừng camera và ngừng gửi frame mới.") {
  const socket=state.socket;state.socket=null;state.pending=null;state.processing=false;
  if(socket)socket.close();
  $("camera-status").textContent="Camera đã dừng";
  $("stream-start").textContent="Kết nối lại + nhận diện";
  status(message);controls();
}
function tab(mode){
  if(state.busy||state.loading)return;
  if(mode==="upload"){if(state.socket)stopStream();$("live-panel").hidden=true;}
  for(const name of ["upload","camera"]){$(`${name}-tab`).setAttribute("aria-selected",String(mode===name));$(`${name}-panel`).hidden=mode!==name;}
}
$("upload-tab").onclick=()=>tab("upload");$("camera-tab").onclick=()=>tab("camera");
$("file").onchange=e=>selectFile(e.target.files[0]);
$("dropzone").ondragover=e=>{e.preventDefault();$("dropzone").classList.add("drag");};
$("dropzone").ondragleave=()=>$("dropzone").classList.remove("drag");
$("dropzone").ondrop=e=>{e.preventDefault();$("dropzone").classList.remove("drag");selectFile(e.dataTransfer.files[0]);};
async function responseError(response){
  const body=await response.json().catch(()=>({}));
  if(response.status===401)return new Error(state.bridge?"Đặt DETECT_API_TOKEN trong terminal chạy web local.":"API token thiếu hoặc không đúng.");
  return new Error(typeof body.detail==="string"?body.detail:`HTTP ${response.status}`);
}
$("detect").onclick=async()=>{
  if(!state.file||state.busy||state.loading||state.socket)return;
  state.busy=true;controls();clearResults();draw();status("AI đang phân tích… Lần đầu có thể cần khởi động model.");
  const start=performance.now(),body=new FormData();body.append("file",state.file);
  const token=$("token").value.trim(),headers=token?{Authorization:`Bearer ${token}`} : {};
  try{
    const response=await fetch("/detect",{method:"POST",headers,body,signal:AbortSignal.timeout(180000)});
    if(!response.ok)throw await responseError(response);
    const result=await response.json();showResult(result,performance.now()-start);
    status(`AI hoàn tất: ${result.count} đối tượng.`);
  }catch(error){status(error.message,true);}finally{state.busy=false;controls();}
};
$("capture").onclick=async()=>{
  if(!state.bridge||state.busy||state.loading||state.socket)return;
  state.busy=true;controls();status("Đang lấy ảnh camera…");
  try{
    const response=await fetch("/camera/frame",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({url:$("camera-url").value.trim()}),signal:AbortSignal.timeout(15000)});
    if(!response.ok)throw await responseError(response);
    const blob=await response.blob();state.busy=false;
    await selectFile(new File([blob],"camera.jpg",{type:blob.type}),"CAMERA");
  }catch(error){status(error.message,true);}finally{state.busy=false;controls();}
};
async function renderPending(socket){
  if(state.decoding)return;
  state.decoding=true;
  try{
    while(state.pending&&state.socket===socket){
      const data=state.pending;state.pending=null;
      const bitmap=await createImageBitmap(new Blob([data.slice(8)],{type:"image/jpeg"}),{imageOrientation:"none"});
      if(state.socket!==socket){bitmap.close();break;}
      state.bitmap?.close();state.bitmap=bitmap;draw();
      $("camera-status").textContent=`● LIVE · ${bitmap.width} × ${bitmap.height}`;
    }
  }catch(error){if(state.socket===socket)stopStream("Không đọc được frame camera. Bấm Kết nối lại.");}
  finally{state.decoding=false;}
}
$("stream-stop").onclick=()=>stopStream();
$("stream-start").onclick=()=>{
  if(!state.bridge||state.socket||state.busy||state.loading)return;
  const url=$("camera-url").value.trim();if(!url){status("Nhập URL MJPEG camera trước.",true);return;}
  const socket=new WebSocket(`${location.protocol==="https:"?"wss:":"ws:"}//${location.host}/camera/live`);
  state.socket=socket;state.result=null;state.file=null;
  state.frames=0;state.rateAt=performance.now();state.lastResultAt=0;
  $("video-fps").textContent="Video: — FPS";$("ai-fps").textContent="AI: chờ kết quả";
  socket.binaryType="arraybuffer";$("live-panel").hidden=false;clearResults();
  $("camera-status").textContent="Đang kết nối camera…";$("ai-status").textContent="AI chờ khung hình đầu tiên…";
  $("ai-status").classList.remove("error");$("source-label").textContent="LIVE + AI";
  status("Camera và AI sẽ chạy tới khi bấm Dừng. Chuyển tab không tự dừng.");controls();
  socket.onopen=()=>{if(state.socket===socket)socket.send(JSON.stringify({url,delay:Number($("stream-delay").value)||0}));};
  socket.onmessage=event=>{
    if(state.socket!==socket)return;
    if(event.data instanceof ArrayBuffer){state.pending=event.data;renderPending(socket);return;}
    const message=JSON.parse(event.data);
    if(message.type==="ai_status"){
      state.processing=true;state.processingAt=Date.now();$("ai-status").textContent=message.message;
    }else if(message.type==="result"){
      state.processing=false;showResult(message,message.total_ms);
      const now=performance.now();
      $("ai-fps").textContent=state.lastResultAt?`AI: ${(1000/(now-state.lastResultAt)).toFixed(1)} FPS`:"AI: đã có kết quả đầu";
      state.lastResultAt=now;
      $("ai-status").textContent=`AI đã phân tích · ${message.count} đối tượng · ${message.inference_ms.toFixed(0)} ms`;
    }else if(message.type==="ai_error"){
      state.processing=false;$("ai-status").textContent=`AI lỗi: ${message.message}`;
      $("ai-status").classList.add("error");status("Video vẫn chạy. Dừng rồi Kết nối lại sau khi xử lý lỗi AI.",true);
    }else if(message.type==="camera_error"){
      stopStream(message.message);status(message.message,true);
    }
  };
  socket.onclose=()=>{if(state.socket===socket)stopStream("Kết nối bị ngắt. Bấm Kết nối lại để tiếp tục.");};
  socket.onerror=()=>{if(state.socket===socket)stopStream("Không kết nối được web local. Khởi động lại run_demo.py rồi bấm Kết nối lại.");};
};
setInterval(()=>{
  if(!state.socket)return;
  const now=performance.now(),elapsed=(now-state.rateAt)/1000;
  $("video-fps").textContent=`Video: ${(state.frames/Math.max(elapsed,0.001)).toFixed(1)} FPS`;
  state.frames=0;state.rateAt=now;
  if(state.processing)$("ai-status").textContent=`AI đang xử lý · ${Math.floor((Date.now()-state.processingAt)/1000)} giây (lần đầu có thể lâu hơn)`;
  if(state.result){
    const age=Math.max(0,(Date.now()-state.result.captured_at)/1000);
    $("box-age").textContent=`Bbox từ ${age.toFixed(1)} giây trước${age>2?" · nét đứt = kết quả có độ trễ":""}.`;
  }
},1000);
document.addEventListener("visibilitychange",()=>{if(!document.hidden&&state.socket){draw();renderPending(state.socket);}});
window.addEventListener("pagehide",()=>{if(state.socket)stopStream();});
fetch("/ui-config").then(r=>{if(!r.ok)throw new Error();return r.json();}).then(config=>{
  state.bridge=config.camera_bridge;$("token-section").hidden=config.token_managed;
  $("camera-help").textContent=state.bridge?"Sẵn sàng: cùng Wi-Fi với iPhone, dán URL MJPEG rồi Bắt đầu stream + nhận diện.":"Camera LAN cần web local: python run_demo.py, rồi mở http://127.0.0.1:8000.";controls();
}).catch(()=>{$("camera-help").textContent="Không đọc được cấu hình. Tải lại trang.";});
