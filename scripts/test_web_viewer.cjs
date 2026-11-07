/* Logic regression with local data; does not substitute for browser visual QA. */
const fs=require('fs'),vm=require('vm'),assert=require('assert');
let drawCalls=0,bufferBytes=0;
const gl=new Proxy({}, {get:(_,name)=>{
 if(name==='getShaderParameter'||name==='getProgramParameter')return ()=>true;
 if(name.startsWith('create'))return ()=>({});
 if(name==='getAttribLocation')return (_,n)=>['aPos','aDelta','aNormal','aColor'].indexOf(n);
 if(name==='getUniformLocation')return (_,n)=>n;
 if(name==='bufferData')return (_,array)=>{assert(array.every(Number.isFinite));bufferBytes+=array.byteLength;};
 if(name==='drawArrays')return (_,start,count)=>{assert(count>0&&count%3===0);drawCalls++;};
 if(name.startsWith('uniform'))return (_, ...values)=>{for(const v of values.flatMap(v=>Array.isArray(v)?v:[v]))assert(Number.isFinite(v));};
 return name.toUpperCase()===name?1:()=>{};
}});
const nodes=new Map();
function element(id){
 if(nodes.has(id))return nodes.get(id);
 const n={value:'',textContent:'',checked:id==='labelsToggle',children:[],style:{},
  classList:{add(){},toggle(){}},append(child){this.children.push(child);},
  add(child){this.append(child);if(!this.value)this.value=child.value;},
  replaceChildren(...children){this.children=children;this.value=children[0]?.value||'';},
  addEventListener(){}};
 if(id==='viewport')Object.assign(n,{getContext:()=>gl,getBoundingClientRect:()=>({width:1200,height:850,left:0,top:0})});
 if(id==='explode')n.value='0';nodes.set(id,n);return n;
}
const sandbox={console,Promise,Math,Float32Array,devicePixelRatio:1,requestAnimationFrame(){},
 Option:function(text,value){this.text=text;this.value=value;},
 document:{getElementById:element,querySelectorAll:()=>[],createElement:()=>element(Symbol())},
 addEventListener(){},fetch(){throw Error('Network must not be required');}};
sandbox.window=sandbox;vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('web/robot-data.js','utf8'),sandbox);
vm.runInContext(fs.readFileSync('web/viewer.js','utf8'),sandbox);
setImmediate(()=>{
 assert(sandbox.forestViewer,'initialization failed');assert(drawCalls>0);
 assert.equal(element('variant').value,'work');
 const work=sandbox.forestViewer.getState().visibleParts;
 element('variant').value='patrol';element('variant').onchange();
 const patrol=sandbox.forestViewer.getState().visibleParts;assert(patrol<work);
 assert(element('part').children.every(p=>!p.value.includes('work5')&&!p.value.includes('light4')&&!p.value.includes('主动轮胎')));
 element('variant').value='survey';element('variant').onchange();
 assert.equal(element('tool').value,'caliper');
 assert(element('part').children.some(p=>p.value.includes('light4')));
 assert(!element('part').children.some(p=>p.value.includes('work5')));
 element('tool').value='sampler';element('tool').onchange();
 assert(!element('part').children.some(p=>p.value.startsWith('T3')));
 element('explode').value='100';element('explode').oninput();
 assert.equal(sandbox.forestViewer.getState().exploded,1);
 element('group').value='导航';element('group').onchange();
 assert(sandbox.forestViewer.getState().visibleParts>0);
 const choice=element('part').children[1].value;
 element('part').value=choice;element('part').onchange();
 const selected=sandbox.FOREST_ROBOT.parts.find(p=>p.name===choice);
 assert.equal(element('detailName').textContent,selected.label||choice);element('focus').onclick();
 element('group').value='工具';element('variant').value='patrol';element('variant').onchange();
 element('group').value='工具';element('group').onchange();
 assert.equal(sandbox.forestViewer.getState().visibleParts,0);
 for(const id of ['rescue_scout','rescue_clear','urban_scan','urban_inspect']){
  element('variant').value=id;element('variant').onchange();
  assert(sandbox.forestViewer.getState().visibleParts>0,id);
  assert(element('tool').children.length>0);
  if(id==='rescue_clear')assert(element('part').children.some(p=>p.value.includes('连续履带')));
  if(id==='urban_scan')assert(element('part').children.some(p=>p.value.includes('测绘扫描仪')));
 }
 console.log(`PASS: offline initialization, 7 variants, tools, explosion, labels, selection, focus, empty group; ${bufferBytes} buffer bytes`);
});
