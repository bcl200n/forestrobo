'use strict';
const $=id=>document.getElementById(id),bundle=globalThis.RESCUE_CATALOG;
const apiBase=location.protocol==='file:'?'http://127.0.0.1:8766':location.port==='8766'?'':'http://127.0.0.1:8766';
let online=false,scenario='void',armResult=null,mission=null,replay=null,frameIndex=0,playing=false,playTimer=null;
let training=globalThis.RESCUE_TRAINING||null,trainingReport=null;
const tabs=['selection','arm','cooperation','training'];
const names={idle:'任务未启动',autonomy:'自主搜索',review:'等待人工复核',paused:'已暂停',manual:'人工接管',estop:'急停锁存'};
const eventNames={start:'启动仿真任务',candidate:'记录疑似线索',review:'人工复核',pause:'暂停',manual:'人工接管',resume:'恢复自主搜索',disconnect:'通信中断并锁存',estop:'急停并锁存',rearm:'人工复位，保持暂停',tool_check:'软件使能检查'};
function node(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;}
function row(parent,label,value){const e=node('div',undefined,'spec-row');e.append(node('span',label),node('b',String(value)));parent.append(e);}
function clear(id){$(id).replaceChildren();return $(id);}
function number(value,digits=2){return Number.isFinite(value)?value.toFixed(digits):'—';}
async function api(path,payload){
 const response=await fetch(apiBase+'/api/rescue/'+path,{method:payload?'POST':'GET',cache:'no-store',headers:payload?{'Content-Type':'application/json'}:{},body:payload?JSON.stringify(payload):undefined,signal:AbortSignal.timeout(8000)});
 const data=await response.json();if(!response.ok)throw Error(data.error||'本地服务请求失败');return data;
}
function showTab(id){
 if(!tabs.includes(id))return;
 for(const t of tabs)$(t).classList.toggle('active',t===id);
 document.querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('active',b.dataset.tab===id));
 if(id==='arm'){drawArm();if(!armResult&&online)calculateArm();}
 if(id==='training'){drawSearch();if(!replay&&online)loadReplay();}
 if(id!=='training')stopReplay();
}
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>showTab(b.dataset.tab));
for(const [id,s] of Object.entries(bundle.scenarios))$('scenario').add(new Option(s.name,id));
$('scenario').onchange=()=>{scenario=$('scenario').value;updateSelection();};
function updateSelection(){
 const s=bundle.scenarios[scenario],c=bundle.catalog,v=s.variant?c.variants[s.variant]:null;
 $('scenarioTask').textContent=s.task;
 $('conditions').replaceChildren(...s.conditions.map(x=>node('li',x)));
 $('matchState').textContent=s.status==='unsupported'?'当前平台不匹配':s.status==='unvalidated_mobility'?'通过性未验证，暂不能作为任务准入':'概念配置匹配，仍需工程验证';
 $('matchState').className='callout';
 const config=clear('configuration');
 if(v){
  const arm=c.arms[v.arm],tool=c.tools[s.tool];
  row(config,'机型',v.name);row(config,'机械臂',arm.name);row(config,'末端工具',tool.name);
  row(config,'臂展目标',Math.round(arm.links_m.reduce((a,b)=>a+b,0)*1000)+' mm');
  $('modelTitle').textContent=v.name;
  $('robotFrame').hidden=false;
  $('robotFrame').src='index.html?'+new URLSearchParams({variant:s.variant,tool:s.tool,embed:'1'});
  $('armFrame').src='index.html?'+new URLSearchParams({variant:s.variant,tool:s.tool,group:'机械臂工具',embed:'1'});
  $('armName').textContent=arm.name;
  const specs=clear('armSpecs');row(specs,'当前任务',s.name);row(specs,'自由度',arm.dof+' 轴');
  row(specs,'工具自重',tool.mass_kg+' kg');row(specs,'标称末端载荷目标',arm.payload_kg+' kg（含工具）');
  row(specs,'关节额定力矩目标',arm.rated_torque_nm.join(' / ')+' N·m');
  row(specs,'参数状态',arm.source);
 }else{
  row(config,'原因',s.conditions[0]);row(config,'机型推荐','无');
  $('modelTitle').textContent='当前没有匹配机型';$('robotFrame').hidden=true;
  $('armName').textContent='当前场景没有可用机械臂配置';clear('armSpecs');
 }
 $('toArm').disabled=!v;$('calculateArm').disabled=!v||!online;$('unreachablePreset').disabled=!v;
 $('armFrame').hidden=!v;armResult=null;clear('torqueBars');$('armSummary').textContent='设置目标位置后计算。';$('payloadResult').textContent='';drawArm();
}
$('toArm').onclick=()=>showTab('arm');$('wholeRobot').onclick=()=>showTab('selection');
$('reaction').oninput=()=>{$('reactionValue').textContent=$('reaction').value;};
$('unreachablePreset').onclick=()=>{$('targetX').value='1.5';$('targetY').value='0';$('targetZ').value='0';calculateArm();};
$('calculateArm').onclick=calculateArm;
async function calculateArm(){
 if(!online){$('armSummary').textContent='请启动本地服务后计算。结构模型仍可离线查看。';return;}
 try{
  const target=[$('targetX'),$('targetY'),$('targetZ')].map(e=>+e.value);
  const mass=+$('objectMass').value;
  if(!target.every(Number.isFinite)||!Number.isFinite(mass)||mass<0)throw Error('请输入有效目标与质量。');
  armResult=await api('arm',{scenario,target_m:target,object_mass_kg:mass,reaction_n:[0,0,+$('reaction').value]});
  const {ik,static:s,selection:m}=armResult;
  $('armSummary').textContent=(ik.reachable?'位置目标可达':'位置目标不可达')+' · 残差 '+number(ik.residual_m*1000,1)+' mm · '+(s.within_rated?'静态力矩未超方案目标':'静态力矩超出方案目标');
  $('armSummary').className='callout'+(ik.reachable&&s.within_rated&&m.within_nominal_payload?' ok':'');
  $('payloadResult').textContent='工具 + 物体 = '+number(m.combined_payload_kg)+' kg；标称载荷余量 '+number(m.payload_margin_kg)+' kg。'+(m.within_nominal_payload?'':'已超过方案载荷目标。');
  const bars=clear('torqueBars');
  s.utilization.forEach((u,i)=>{const line=node('div',undefined,'torque-row'+(u>1?' over':'')),track=node('div',undefined,'bar'),fill=node('i');fill.style.width=Math.min(u*100,100)+'%';track.append(fill);line.append(node('span','J'+(i+1)),track,node('span',Math.round(u*100)+'% / '+number(s.torque_nm[i],1)));bars.append(line);});
  $('ikFlags').textContent='关节角：'+ik.joints_rad.map(x=>number(x*180/Math.PI,1)+'°').join(' / ')+'。只求位置；未核验工具朝向、碰撞、接触力控或底盘稳定性，不能直接执行。';
  drawArm();
 }catch(e){$('armSummary').textContent=e.message;$('armSummary').className='callout';}
}
function canvas(id){const c=$(id),r=c.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2);c.width=Math.max(r.width,1)*d;c.height=Math.max(r.height,1)*d;const ctx=c.getContext('2d');ctx.scale(d,d);return [ctx,r.width,r.height];}
function drawArm(){
 const [c,w,h]=canvas('armCanvas');if(!w)return;c.clearRect(0,0,w,h);c.font='11px system-ui';c.fillStyle='#9bb1b8';
 if(!armResult){c.fillText('运动学线框 · 计算后显示关节与目标位置',20,30);return;}
 const target=[$('targetX'),$('targetY'),$('targetZ')].map(e=>+e.value),points=[...armResult.joint_origins_m,armResult.static.tcp_m];
 const raw=p=>[p[0]*.9-p[1]*.65,-p[2]*.95-(p[0]+p[1])*.25];
 const all=[...points,target,[0,0,0]],xy=all.map(raw),lo=[Math.min(...xy.map(p=>p[0])),Math.min(...xy.map(p=>p[1]))],hi=[Math.max(...xy.map(p=>p[0])),Math.max(...xy.map(p=>p[1]))];
 const scale=Math.min((w-110)/Math.max(.3,hi[0]-lo[0]),(h-85)/Math.max(.3,hi[1]-lo[1]));
 const project=p=>{const q=raw(p);return [w/2+(q[0]-(lo[0]+hi[0])/2)*scale,h/2+(q[1]-(lo[1]+hi[1])/2)*scale];};
 const zero=project([0,0,0]);
 for(const [axis,color,label] of [[[.2,0,0],'#b7746d','X'],[[0,.2,0],'#6fbc9c','Y'],[[0,0,.2],'#7aa7cd','Z']]){const p=project(axis);c.strokeStyle=color;c.lineWidth=1;c.beginPath();c.moveTo(...zero);c.lineTo(...p);c.stroke();c.fillStyle=color;c.fillText(label,p[0]+4,p[1]);}
 c.strokeStyle='#90d6ba';c.lineWidth=10;c.lineCap='round';c.beginPath();points.forEach((p,i)=>i?c.lineTo(...project(p)):c.moveTo(...project(p)));c.stroke();
 points.forEach((p,i)=>{const q=project(p);c.fillStyle='#e8eff0';c.beginPath();c.arc(...q,5,0,Math.PI*2);c.fill();if(i>0){c.fillStyle='#a6c8ca';c.fillText(i===points.length-1?'TCP':'J'+(i+1),q[0]+8,q[1]-10);}});
 const q=project(target);c.strokeStyle='#ffbc72';c.lineWidth=2;c.beginPath();c.arc(...q,9,0,Math.PI*2);c.moveTo(q[0]-14,q[1]);c.lineTo(q[0]+14,q[1]);c.moveTo(q[0],q[1]-14);c.lineTo(q[0],q[1]+14);c.stroke();c.fillStyle='#ffbc72';c.fillText('目标',q[0]+16,q[1]+4);
 c.fillStyle='#9bb1b8';c.fillText('实际 IK 计算结果 / 关节原点与连杆示意',18,h-16);
}
const checks=[['linkOk','通信心跳有效',true],['gasFresh','气体测量未过期',true],['gasClear','气体条件已确认',false],['stationary','底盘已停驻',false],['support_confirmed','支撑状态已确认',false],['tool_locked','工具锁紧反馈有效',false],['tool_id_confirmed','工具 ID 已确认',false],['human_zone_clear','人员区域已确认',false],['structural_clearance','现场人员已确认结构与对象',false],['operator_hold','操作员持续使能（模拟）',false]];
for(const [id,text,checked] of checks){const l=node('label',undefined,'check'),input=node('input');input.type='checkbox';input.id=id;input.checked=checked;l.append(input,node('span',text));$('interlocks').append(l);}
function toolState(){const s={heartbeat_age_s:$('linkOk').checked ? .1 : 2,gas_age_s:$('gasFresh').checked ? .1 : 2,gas_clear:$('gasClear').checked};for(const [key] of checks.slice(3))s[key]=$(key).checked;return s;}
async function command(action,data={},actor=$('operatorName').value.trim()){
 if(!online){$('missionError').textContent='协同状态机需要本地服务。请打开 http://127.0.0.1:8766/rescue.html';return;}
 try{mission=await api('mission',{action,actor,...data});$('missionError').textContent=mission.imported===undefined?'':'已导入 '+mission.imported+' 条回放线索。';renderMission();if(mission.tool_result){const r=mission.tool_result;$('toolResult').textContent=r.cut_enabled?'软件条件满足。本次只是权限检查，没有启动任何真实工具。':'不允许使能：'+r.blocked_by.join(' / ');$('toolResult').className='callout'+(r.cut_enabled?' ok':'');}}
 catch(e){$('missionError').textContent=e.message;}
}
for(const [id,action] of [['startMission','start'],['pauseMission','pause'],['manualMission','manual'],['resumeMission','resume'],['disconnectMission','disconnect'],['estopMission','estop']])$(id).onclick=()=>command(action);
$('addCandidate').onclick=()=>command('candidate',{id:'SIM-'+Date.now().toString().slice(-7),confidence:.62,source:'网页注入的合成热源线索 / 非真实传感器'});
$('importEvidence').onclick=()=>command('import_replay',{controller:'validation'});
$('rearmMission').onclick=()=>command('rearm',{confirmed:$('rearmConfirm').checked,state:toolState()});
$('checkTool').onclick=()=>{const candidate=mission?.candidates.find(c=>c.status==='reviewed_cue');if(!candidate){$('missionError').textContent='先采集并人工采纳一条线索。';return;}command('tool_check',{id:candidate.id,state:toolState()});};
function renderMission(){
 if(!mission)return;
 $('missionMode').textContent=(names[mission.mode]||mission.mode)+(mission.latched?' · 已锁存':'');$('missionMode').className='pill'+(mission.latched||mission.mode==='review'?' warn':'');
 $('startMission').disabled=mission.mode!=='idle';$('addCandidate').disabled=!['autonomy','manual','review'].includes(mission.mode);
 $('importEvidence').disabled=!['autonomy','manual','review'].includes(mission.mode);
 const container=clear('candidates');
 if(!mission.candidates.length)container.append(node('div','暂无线索。先启动任务，再注入模拟线索。','empty'));
 for(const candidate of mission.candidates){const e=node('div',undefined,'candidate');e.append(node('b',candidate.id),node('div',({pending:'待人工复核',reviewed_cue:'人工采纳线索，仍需现场复查',rejected:'人工排除'})[candidate.status]),node('div','合成线索分数 '+number(candidate.synthetic_score)+' · 非生命概率'),node('small','证据摘要 '+candidate.sha256.slice(0,16)));
 try{const source=JSON.parse(candidate.source);if(source.event)e.append(node('div','地图 '+source.map_seed+' · 第 '+source.event.step+' 步 · 网格 '+source.event.grid.join(', ')));}catch{}
 if(candidate.status==='pending'){const actions=node('div',undefined,'inline');for(const [decision,text] of [['accept_cue','采纳线索'],['recheck','复查'],['reject','排除']]){const button=node('button',text);button.onclick=()=>command('review',{id:candidate.id,decision},$('reviewerName').value.trim());actions.append(button);}e.append(actions);}container.append(e);}
 const list=clear('eventLog');for(const event of [...mission.events].reverse()){const li=node('li');li.append(node('small',new Date(event.time).toLocaleTimeString('zh-CN',{hour12:false})),node('b',eventNames[event.action]||event.action),node('span',event.actor+' · #'+event.sequence));if(event.details?.decision)li.append(node('div',event.details.decision));list.append(li);}
 if(!mission.tool_result){$('toolResult').textContent=mission.latched?'已锁存，恢复通信不会自动重新使能。':'等待本次人工复核与软件条件检查。';$('toolResult').className='callout';}
}
$('downloadAudit').onclick=()=>{if(!mission)return;const blob=new Blob([JSON.stringify(mission,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=node('a');a.href=url;a.download='rescue-mission-audit.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
function renderTraining(){
 const t=training;if(!t){$('trainState').textContent='暂无状态';return;}
 const age=(Date.now()-new Date(t.updated_at).getTime())/1000,running=['training','baseline_evaluation','test_evaluation'].includes(t.status);
 const state=({training:'训练进行中',complete:'实验已完成',failed:'实验失败',stopped:'实验已停止',baseline_evaluation:'基线评测中',test_evaluation:'最终测试中'})[t.status]||t.status;
 $('trainState').textContent=running&&age>90?'状态心跳过期，需核对':state;$('trainState').className='pill'+(t.status==='failed'||running&&age>90?' warn':'');
 $('actualSteps').textContent=(t.actual_steps||0).toLocaleString('zh-CN');$('targetSteps').textContent='目标 '+(t.target_steps||2000000).toLocaleString('zh-CN')+' 步';$('worldCount').textContent=t.worlds||'—';$('trainFps').textContent=Number.isFinite(t.fps)?Math.round(t.fps).toLocaleString():'—';$('heartbeat').textContent=t.updated_at?new Date(t.updated_at).toLocaleTimeString('zh-CN',{hour12:false}):'—';$('checkpoint').textContent=t.last_checkpoint||'尚无周期检查点';$('progressBar').style.width=Math.min(100,(t.actual_steps||0)/(t.target_steps||2000000)*100)+'%';
 $('trainSnapshot').textContent=(online?'本地服务每 5 秒读取状态。':'当前显示文件快照；启用本地服务后可实时读取。')+'训练集近期成功率 '+(t.recent_success_rate===null||t.recent_success_rate===undefined?'尚无统计':Math.round(t.recent_success_rate*100)+'%')+'，不能用它替代独立评测。';
 const p=trainingReport?.protocol,info=clear('protocolInfo');row(info,'并行场景','通道 / 碎块障碍 / 模拟掉线');row(info,'训练地图',p?p.training_map_seeds.length+' 个种子':'128 个种子（配置值）');row(info,'阶段验证种子',(p?.validation_map_seeds||[5000,5001]).join(' / '));row(info,'最终测试种子',(p?.test_map_seeds||[10000,10001,10002,10003]).join(' / '));row(info,'输出位置',trainingReport?.run||'rl/models/rescue_search_20261106_v1');
 const b=trainingReport?.baseline,v=trainingReport?.validation,e=trainingReport?.evaluation;
 let text=b?'前沿基线：'+b.successes+'/'+b.total+' 完成搜索。':'基线结果待读取。';
 text+=v?' 阶段验证：'+v.successes+'/'+v.total+'。':' 阶段验证未产出。';
 text+=e?' 最终测试：'+e.successes+'/'+e.total+'；发生碰撞的回合 '+e.collision_episodes+'。':' 最终独立测试尚未完成。';
 if(t.status==='complete')text+=' '+(t.promoted?'通过本轮与基线的实验比较；仍不可部署。':'未通过基线晋级条件。');
 if(t.error)text+=' 错误：'+t.error;$('evaluationResult').textContent=text;
}
async function refresh(){
 try{trainingReport=await api('status');training=trainingReport.training;const was=online;online=true;$('connection').textContent='本地服务在线 · 无硬件';$('connection').className='pill';if(!was){$('calculateArm').disabled=!bundle.scenarios[scenario].variant;mission=await api('mission');renderMission();}renderTraining();}
 catch(e){online=false;$('connection').textContent='离线预览 · 协同需本地服务';$('connection').className='pill warn';$('calculateArm').disabled=true;renderTraining();}
}
async function loadReplay(){stopReplay();try{replay=await api('replay?controller='+$('replayController').value);frameIndex=0;$('replayTime').max=replay.frames.length-1;$('replayTime').value=0;drawSearch();}catch(e){replay=null;$('replayText').textContent='所选控制器尚无回放。'+e.message;drawSearch();}}
$('replayController').onchange=loadReplay;
function stopReplay(){playing=false;if(playTimer)clearInterval(playTimer);playTimer=null;$('playReplay').textContent='播放';}
$('playReplay').onclick=()=>{if(!replay){loadReplay();return;}if(playing){stopReplay();return;}if(frameIndex>=replay.frames.length-1)frameIndex=0;playing=true;$('playReplay').textContent='暂停';playTimer=setInterval(()=>{frameIndex=Math.min(frameIndex+1,replay.frames.length-1);$('replayTime').value=frameIndex;drawSearch();if(frameIndex>=replay.frames.length-1)stopReplay();},140);};
$('replayTime').oninput=()=>{stopReplay();frameIndex=+$('replayTime').value;drawSearch();};
function drawSearch(){
 const [c,w,h]=canvas('searchCanvas');if(!w)return;c.clearRect(0,0,w,h);c.font='12px system-ui';c.fillStyle='#9bb1b8';
 if(!replay){c.fillText('载入真实环境记录后显示；没有预设成功动画。',20,30);return;}
 const f=replay.frames[frameIndex],n=f.known.length,s=Math.min((w-40)/n,(h-35)/n),ox=(w-n*s)/2,oy=(h-n*s)/2;
 const point=p=>[ox+(p[0]+.5)*s,oy+(n-1-p[1]+.5)*s];
 for(let x=0;x<n;x++)for(let y=0;y<n;y++){c.fillStyle=({'0':'#16242b','1':'#31504f','-1':'#8b9fa2'})[f.known[x][y]];c.fillRect(ox+x*s,oy+(n-1-y)*s,s-1,s-1);}
 c.strokeStyle='#90d6ba';c.lineWidth=2;c.beginPath();replay.frames.slice(0,frameIndex+1).forEach((v,i)=>i?c.lineTo(...point(v.position)):c.moveTo(...point(v.position)));c.stroke();
 for(const e of f.events){const p=point(e.grid);c.strokeStyle='#ffbc72';c.lineWidth=2;c.beginPath();c.arc(...p,s*.5,0,Math.PI*2);c.stroke();}
 const p=point(f.position);c.fillStyle=f.link_valid?'#90d6ba':'#ff9690';c.beginPath();c.arc(...p,s*.36,0,Math.PI*2);c.fill();
 $('replayText').textContent=`${replay.controller} · ${replay.scenario} · 地图种子 ${replay.map_seed} · 第 ${f.step} 步 · 地图观测覆盖 ${Math.round(f.coverage*100)}% · 已记录线索 ${f.candidates} · ${f.link_valid?'通信有效':'失联保持'}。只画已观测地图。`;
}
window.addEventListener('resize',()=>{drawArm();drawSearch();});
updateSelection();renderTraining();refresh();setInterval(refresh,5000);
