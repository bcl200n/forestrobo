/* Offline WebGL assembly viewer: real meshes, depth testing and local data. */
'use strict';
const $=id=>document.getElementById(id), canvas=$('viewport');
if(new URLSearchParams(location.search).get('embed')==='1')document.body.classList.add('embedded');
const gl=canvas.getContext('webgl',{antialias:true,alpha:true,preserveDrawingBuffer:true});
let data,parts=[],active=[],yaw=-.7,pitch=.38,zoom=1,pan=[0,0],drag=null;
let center=[0,0,0],radius=1,selected='',exploded=0,playing=false,previous=0,direction=1;
let program,loc={},labels=[];
const vs=`attribute vec3 aPos,aDelta,aNormal,aColor;
uniform vec3 uCenter;uniform vec2 uScale,uPan;
uniform float uYaw,uPitch,uExplode,uRadius;
varying vec3 vNormal,vColor;
void main(){vec3 p=aPos+aDelta*uExplode-uCenter;
float x=p.x*cos(uYaw)-p.y*sin(uYaw),d=p.x*sin(uYaw)+p.y*cos(uYaw);
float y=p.z*cos(uPitch)-d*sin(uPitch),z=d*cos(uPitch)+p.z*sin(uPitch);
gl_Position=vec4(vec2(x,y)*uScale+uPan,-z/(uRadius*5.0),1.0);
vNormal=aNormal;vColor=aColor;}`;
const fs=`precision highp float;varying vec3 vNormal,vColor;
uniform vec3 uView;uniform float uSelected;
void main(){vec3 n=normalize(vNormal)*(gl_FrontFacing?1.0:-1.0);
vec3 key=normalize(vec3(-.3,-.6,1.0));
float diffuse=max(0.0,dot(n,key)),fill=max(0.0,dot(n,normalize(vec3(.6,.5,.3))));
float spec=pow(max(dot(n,normalize(key+uView)),0.0),40.0)*.23;
vec3 col=mix(vColor,vec3(1.0,.42,.08),uSelected*.7);
gl_FragColor=vec4(pow(max(col*(.33+.64*diffuse+.22*fill)+vec3(spec),vec3(0.0)),vec3(1.0/2.2)),1.0);}`;
function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;}
function setup(){
 if(!gl)throw Error('浏览器未启用 WebGL，请启用硬件加速后重试。');
 program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vs));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fs));gl.linkProgram(program);
 if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));
 gl.useProgram(program);gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);
 for(const n of ['aPos','aDelta','aNormal','aColor'])loc[n]=gl.getAttribLocation(program,n);
 for(const n of ['uCenter','uScale','uPan','uYaw','uPitch','uExplode','uRadius','uView','uSelected'])loc[n]=gl.getUniformLocation(program,n);
}
function group(p){return p.group||'躯干';}
function buffer(p){
 const arr=[];
 p.triangles.forEach(([ids,mi],face)=>{
  const a=p.vertices[ids[0]],b=p.vertices[ids[1]],c=p.vertices[ids[2]],ab=b.map((v,i)=>v-a[i]),ac=c.map((v,i)=>v-a[i]);
  const normal=[ab[1]*ac[2]-ab[2]*ac[1],ab[2]*ac[0]-ab[0]*ac[2],ab[0]*ac[1]-ab[1]*ac[0]],size=Math.hypot(...normal)||1;
  ids.forEach((id,corner)=>arr.push(...p.vertices[id],...p.exploded[id].map((v,i)=>v-p.vertices[id][i]),...(p.normals?.[face]?.[corner]||normal.map(v=>v/size)),...(p.colors[mi]||[.2,.3,.3])));
 });
 p.buffer=gl.createBuffer();p.count=arr.length/12;gl.bindBuffer(gl.ARRAY_BUFFER,p.buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(arr),gl.STATIC_DRAW);
}
function position(p,id){return p.vertices[id].map((v,i)=>v+(p.exploded[id][i]-v)*exploded);}
function bounds(list){
 const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
 for(const p of list)p.vertices.forEach((_,id)=>position(p,id).forEach((v,i)=>{lo[i]=Math.min(lo[i],v);hi[i]=Math.max(hi[i],v);}));
 return list.length?{center:lo.map((v,i)=>(v+hi[i])/2),radius:Math.max(.12,Math.hypot(...hi.map((v,i)=>(v-lo[i])/2)))}:{center:[0,0,0],radius:1};
}
function fit(list=active){const b=bounds(list);center=b.center;radius=b.radius;zoom=1;pan=[0,0];}
function visible(p){const filter=$('group').value;return (!p.variants||p.variants.includes($('variant').value))&&(!p.tool||p.tool===$('tool').value)&&(!filter||(filter==='机械臂工具'?['机械臂','工具'].includes(group(p)):group(p)===filter))&&!($('cutaway').checked&&/主躯干|上甲板|侧护板|电池舱/.test(p.name));}
function update(refit=true){
 active=parts.filter(visible);$('count').textContent=active.length;
 $('part').replaceChildren(new Option('点击模型或选择零件',''));for(const p of active)$('part').add(new Option(p.label||p.name,p.name));
 if(!active.some(p=>p.name===selected))select('');else $('part').value=selected;
 if(refit)fit();rebuildLabels();render();
}
function variant(){
 const v=data.catalog.variants[$('variant').value];$('tool').replaceChildren(...v.tools.map(k=>new Option(data.catalog.tools[k].name,k)));
 $('mission').textContent=v.tasks;$('limitations').textContent=v.limit;$('variantTitle').textContent=v.name;
 const arm=v.arm?data.catalog.arms[v.arm]:null;
 $('armSpec').textContent=arm?`${arm.dof} 轴 · 理论臂展 ${(arm.links_m.reduce((a,b)=>a+b,0)*1000).toFixed(0)} mm`:'无机械臂 · 保留导航载荷';
 selected='';$('group').value='';update();
}
function select(name){
 selected=name;$('part').value=name;const p=parts.find(p=>p.name===name);
 $('detailName').textContent=p?.label||p?.name||'选择一个零件';$('detailSpec').textContent=p?.spec||'点击模型或使用列表查看规格。';
 $('detailNote').textContent=p?.note||'用于结构布局讨论；不代表已完成选型或制造。';
 $('detailSize').textContent=p?.dimensions_mm?`装配包围尺寸：${p.dimensions_mm.map(Math.round).join(' × ')} mm`:'尺寸来自概念模型，不等于加工公差。';
 rebuildLabels();render();
}
function project(v){
 const p=v.map((v,i)=>v-center[i]),d=p[0]*Math.sin(yaw)+p[1]*Math.cos(yaw);
 const x=p[0]*Math.cos(yaw)-p[1]*Math.sin(yaw),y=p[2]*Math.cos(pitch)-d*Math.sin(pitch),z=d*Math.cos(pitch)+p[2]*Math.sin(pitch);
 const r=canvas.getBoundingClientRect(),s=Math.min(r.width,r.height)*.4/radius*zoom;
 return [r.width/2+x*s+pan[0]*r.width/2,r.height/2-y*s-pan[1]*r.height/2,z];
}
function rebuildLabels(){
 $('labels').replaceChildren();labels=[];let targets=active.filter(p=>p.name===selected);
 if($('labelsToggle').checked){
  const keys=[/地震·雷达微动/,/地震·热像可见光/,/地震·连续履带/,/城市·测绘扫描仪主机/,/城市·近距成像镜头/,/主躯干/,/前脸传感器盒/,/^雷达转鼓$/, /IMU隔振/, /后视双目/, /J1回转/,/J2肩驱动/,/J3肘驱动/,/J4腕俯仰/, /腕回转模组/,/六维力传感/,/电动快换/,/400mm测径/,/200mm锯片/,/夹爪电机/,/左前.*轮毂/,/左前.*髋电机/];
  targets=keys.map(k=>active.find(p=>k.test(p.name))).filter(Boolean);
  if(selected&&!targets.some(p=>p.name===selected))targets.push(...active.filter(p=>p.name===selected));
 }
 for(const p of targets){const node=document.createElement('button');node.className='part-label';node.textContent=p.label||p.name;node.title=p.spec;node.onclick=()=>select(p.name);$('labels').append(node);labels.push([p,node]);}
}
function labelPositions(){
 const occupied=[],r=canvas.getBoundingClientRect();
 for(const [p,node] of labels){const v=project(bounds([p]).center);let y=v[1],tries=0;
  while(occupied.some(a=>Math.abs(a[0]-v[0])<170&&Math.abs(a[1]-y)<29)&&tries++<7)y+=29;
  const x=Math.max(6,Math.min(r.width-190,v[0]+12));y=Math.max(7,Math.min(r.height-30,y));
  node.style.transform=`translate(${x}px,${y}px)`;node.classList.toggle('selected',p.name===selected);occupied.push([x,y]);
 }
}
function render(){
 if(!gl||!program)return;const rect=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2),w=Math.max(1,Math.round(rect.width*dpr)),h=Math.max(1,Math.round(rect.height*dpr));
 if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}gl.viewport(0,0,w,h);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.useProgram(program);
 const s=Math.min(w,h)*.4/radius*zoom;gl.uniform2f(loc.uScale,2*s/w,2*s/h);gl.uniform2f(loc.uPan,...pan);gl.uniform3fv(loc.uCenter,center);gl.uniform1f(loc.uRadius,radius);gl.uniform1f(loc.uYaw,yaw);gl.uniform1f(loc.uPitch,pitch);gl.uniform1f(loc.uExplode,exploded);
 gl.uniform3f(loc.uView,Math.sin(yaw)*Math.cos(pitch),Math.cos(yaw)*Math.cos(pitch),Math.sin(pitch));
 for(const p of active){gl.bindBuffer(gl.ARRAY_BUFFER,p.buffer);['aPos','aDelta','aNormal','aColor'].forEach((n,i)=>{gl.enableVertexAttribArray(loc[n]);gl.vertexAttribPointer(loc[n],3,gl.FLOAT,false,48,i*12);});gl.uniform1f(loc.uSelected,p.name===selected?1:0);gl.drawArrays(gl.TRIANGLES,0,p.count);}
 $('value').textContent=Math.round(exploded*100)+'%';labelPositions();
}
function pick(x,y){
 let best=null,depth=-Infinity;
 for(const p of active){const vertices=p.vertices.map((_,id)=>project(position(p,id)));
  for(const [ids] of p.triangles){const a=vertices[ids[0]],b=vertices[ids[1]],c=vertices[ids[2]],det=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1]);if(Math.abs(det)<1e-5)continue;
   const u=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/det,v=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/det,w=1-u-v;
   if(u>=0&&v>=0&&w>=0){const z=u*a[2]+v*b[2]+w*c[2];if(z>depth){best=p;depth=z;}}
  }
 }select(best?.name||'');
}
function pause(){playing=false;$('play').textContent='自动分解';}
function explosion(value){exploded=value/100;const b=bounds(active);center=b.center;radius=b.radius;render();}
canvas.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,pan:e.shiftKey||e.button===2};canvas.setPointerCapture(e.pointerId);};
canvas.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(drag.pan){const r=canvas.getBoundingClientRect();pan[0]+=dx/r.width*2;pan[1]-=dy/r.height*2;}else{yaw+=dx*.007;pitch=Math.max(-1.45,Math.min(1.45,pitch+dy*.007));}drag.x=e.clientX;drag.y=e.clientY;render();};
canvas.onpointerup=e=>{if(drag&&Math.hypot(e.clientX-drag.startX,e.clientY-drag.startY)<5){const r=canvas.getBoundingClientRect();pick(e.clientX-r.left,e.clientY-r.top);}drag=null;};canvas.onpointercancel=()=>drag=null;canvas.oncontextmenu=e=>e.preventDefault();
canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(.25,Math.min(7,zoom*Math.exp(-e.deltaY*.001)));render();};
$('explode').oninput=()=>{pause();explosion(+$('explode').value);};$('variant').onchange=variant;$('tool').onchange=()=>update();$('group').onchange=()=>update();$('cutaway').onchange=()=>update(false);$('part').onchange=()=>select($('part').value);$('labelsToggle').onchange=()=>{rebuildLabels();render();};
$('reset').onclick=()=>{pause();yaw=-.7;pitch=.38;exploded=0;$('explode').value=0;fit();render();};$('focus').onclick=()=>{fit(selected?active.filter(p=>p.name===selected):active);render();};
for(const b of document.querySelectorAll('[data-view]'))b.onclick=()=>{[yaw,pitch]=({iso:[-.7,.38],side:[0,0],front:[Math.PI/2,0],top:[0,Math.PI/2]})[b.dataset.view];fit();render();};
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'暂停分解':'自动分解';previous=0;if(playing)requestAnimationFrame(animate);};
function animate(now){if(!playing)return;if(!previous)previous=now;const dt=Math.min(now-previous,80)/1000;previous=now;exploded+=dt*.22*direction;if(exploded>=1||exploded<=0){exploded=Math.max(0,Math.min(1,exploded));direction*=-1;}$('explode').value=exploded*100;explosion(exploded*100);requestAnimationFrame(animate);}
window.addEventListener('resize',render);canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();pause();$('status').textContent='图形上下文中断，请刷新页面。';});
Promise.resolve(globalThis.FOREST_ROBOT).then(model=>{
 if(!model)throw Error('未找到本地模型数据，请确认 robot-data.js 在同一目录。');data=model;parts=model.parts;setup();parts.forEach(buffer);
 for(const [id,v] of Object.entries(data.catalog.variants))$('variant').add(new Option(v.name,id));
 const query=new URLSearchParams(location.search),initialVariant=query.get('variant');
 $('variant').value=Object.hasOwn(data.catalog.variants,initialVariant)?initialVariant:'work';variant();
 const initialTool=query.get('tool'),initialGroup=query.get('group');
 if([...$('tool').options].some(o=>o.value===initialTool))$('tool').value=initialTool;
 if([...$('group').options].some(o=>o.value===initialGroup))$('group').value=initialGroup;
 if(query.get('embed')==='1'){$('variant').disabled=true;$('tool').disabled=true;}
 update();
 $('source').textContent=`${model.version} · ${parts.length} 个模块零件`;$('status').textContent='离线模型已就绪 · 点击零件查看规格';
 window.forestViewer={getState:()=>({variant:$('variant').value,visibleParts:active.length,selected,exploded,webgl:!!gl}),render};
}).catch(e=>{$('status').textContent=e.message;$('status').classList.add('error');console.error(e);});
