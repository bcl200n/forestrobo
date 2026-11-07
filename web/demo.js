'use strict';
const demo=globalThis.DEMO_DATA,el=id=>document.getElementById(id);
let page='structure',routeTime=0,routePlaying=false,routeStart=0;
for(const button of document.querySelectorAll('[data-page]'))button.onclick=()=>{
 page=button.dataset.page;document.querySelectorAll('.page').forEach(p=>p.classList.toggle('active',p.id===page));
 document.querySelectorAll('[data-page]').forEach(b=>b.classList.toggle('active',b===button));
 document.querySelectorAll('video').forEach(v=>v.pause());
 if(page!=='navigation'){routePlaying=false;el('routePlay').textContent='播放路径示意';}
 redraw();
};
function context(id){const c=el(id),r=c.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2);c.width=r.width*d;c.height=r.height*d;const ctx=c.getContext('2d');ctx.scale(d,d);return [ctx,r.width,r.height];}
function plot(){
 const [c,w,h]=context('velocityPlot');c.clearRect(0,0,w,h);const left=48,right=w-15,top=15,bottom=h-30;
 const all=demo.replays.flatMap(r=>r.frames.map(f=>f.vx));const low=Math.min(-1,...all),high=Math.max(.6,...all);
 const x=t=>left+t/8*(right-left),y=v=>bottom-(v-low)/(high-low)*(bottom-top);
 c.font='11px system-ui';c.strokeStyle='#3a504a';c.fillStyle='#a2b8ac';
 for(const v of [-1,-.5,0,.3,.6]){c.beginPath();c.moveTo(left,y(v));c.lineTo(right,y(v));c.stroke();c.fillText(v.toFixed(1),5,y(v)+4);}
 for(let t=0;t<=8;t+=2)c.fillText(t+'s',x(t)-5,h-7);
 c.setLineDash([5,4]);c.strokeStyle='#dce8d4';c.beginPath();c.moveTo(left,y(.3));c.lineTo(right,y(.3));c.stroke();c.setLineDash([]);
 demo.replays.forEach((r,i)=>{c.strokeStyle=i?'#8ed0ad':'#efb374';c.lineWidth=1.7;c.beginPath();r.frames.forEach((f,j)=>j?c.lineTo(x(f.t),y(f.vx)):c.moveTo(x(f.t),y(f.vx)));c.stroke();});
}
el('syncPlay').onclick=async()=>{const videos=[el('legacyVideo'),el('correctedVideo')];videos.forEach(v=>v.currentTime=0);const r=await Promise.allSettled(videos.map(v=>v.play()));if(r.some(x=>x.status==='rejected'))el('syncPlay').textContent='请点击视频播放按钮';};
for(const [i,prefix] of [[0,'old'],[1,'new']]){el(prefix+'Mean').textContent=demo.replays[i].mean_vx.toFixed(3);el(prefix+'Error').textContent=demo.replays[i].vx_rmse.toFixed(3);}
el('teacherStatus').textContent=`速度教师验收：基元 ${demo.teacher.baseline}/${demo.teacher.cases}，学习策略 ${demo.teacher.learned}/${demo.teacher.cases}；候选教师未晋级，尚未启动学生蒸馏。`;
function map(){
 const m=demo.maps[+el('mapCase').value],[c,w,h]=context('mapCanvas'),s=Math.min((w-20)/m.cols,(h-20)/m.rows),ox=(w-m.cols*s)/2,oy=(h-m.rows*s)/2;
 c.clearRect(0,0,w,h);
 for(let r=0;r<m.rows;r++)for(let col=0;col<m.cols;col++){c.fillStyle=m.occupied[r][col]?'#526c62':m.uncertain[r][col]?'#b69055':'#1d312d';c.fillRect(ox+col*s,oy+r*s,s-1,s-1);}
 const point=p=>[ox+(p[1]+.5)*s,oy+(p[0]+.5)*s];
 const path=m.result.path;c.strokeStyle='#8ed0ad';c.lineWidth=3;c.beginPath();path.forEach((p,i)=>i?c.lineTo(...point(p)):c.moveTo(...point(p)));c.stroke();
 for(const [p,color,label] of [[m.start,'#8ed0ad','起点'],[m.goal,'#efb374','目标']]){const q=point(p);c.fillStyle=color;c.beginPath();c.arc(...q,5,0,Math.PI*2);c.fill();c.font='12px system-ui';c.fillText(label,q[0]+8,q[1]-5);}
 if(path.length){const p=path[Math.min(path.length-1,Math.floor(routeTime*(path.length-1)))];c.fillStyle='#fff';c.beginPath();c.arc(...point(p),5,0,Math.PI*2);c.fill();}
 el('routeResult').textContent=m.result.found?`已找到可行栅格路线 · ${path.length} 个路径点 · 代价 ${m.result.cost_m.toFixed(2)} m`:'当前条件无可行路线：'+m.result.reason;
}
el('mapCase').onchange=()=>{routeTime=0;routePlaying=false;el('routePlay').textContent='播放路径示意';map();};
el('routePlay').onclick=()=>{routePlaying=!routePlaying;el('routePlay').textContent=routePlaying?'暂停':'播放路径示意';routeStart=performance.now()-routeTime*6000;if(routePlaying)requestAnimationFrame(tick);};
function tick(t){if(!routePlaying||page!=='navigation')return;routeTime=Math.min(1,(t-routeStart)/6000);map();if(routeTime===1){routePlaying=false;el('routePlay').textContent='重新播放';routeTime=0;}else requestAnimationFrame(tick);}
function arm(){
 const a=demo.arms.find(a=>a.arm===el('armType').value&&a.pose===el('armPose').value),f=+el('force').value;
 el('forceValue').textContent=f+' N';el('armBars').replaceChildren();let max=0;
 a.rated.forEach((rating,i)=>{const torque=a.zero.torque_nm[i]+f/500*(a.force500.torque_nm[i]-a.zero.torque_nm[i]),ratio=Math.abs(torque)/rating;max=Math.max(max,ratio);
  const row=document.createElement('div');row.className='barrow';const title=document.createElement('span');title.textContent='J'+(i+1);const track=document.createElement('div');track.className='track';const fill=document.createElement('div');fill.className='fill';fill.style.width=Math.min(100,ratio*100)+'%';fill.style.background=ratio>1?'#ef9274':'#8ed0ad';track.append(fill);const value=document.createElement('span');value.textContent=Math.round(ratio*100)+'% · '+torque.toFixed(1)+' Nm';row.append(title,track,value);el('armBars').append(row);
 });
 el('armResult').textContent=max>1?'超出额定力矩目标':'静态力矩在额定目标内';el('armResult').style.color=max>1?'#ef9274':'#8ed0ad';
 el('heightResult').textContent=`假设基座高 0.65 m，理论最高 ${(a.zero.theoretical_max_height_m).toFixed(2)} m。未计关节限位、碰撞和姿态约束。`;
}
for(const id of ['armType','armPose'])el(id).onchange=arm;el('force').oninput=arm;
function redraw(){if(page==='motion')plot();if(page==='navigation')map();if(page==='arm')arm();}
window.addEventListener('resize',redraw);arm();

if(globalThis.FOREST_EVIDENCE){const f=globalThis.FOREST_EVIDENCE;el('forestEvidence').textContent=`${f.successes}/${f.total} 到达目标；${f.collision_count} 次碰撞、${f.fall_count} 次跌倒。`+f.groups.map(g=>`${g.name}：${g.success}/${g.count}`).join('；');}
