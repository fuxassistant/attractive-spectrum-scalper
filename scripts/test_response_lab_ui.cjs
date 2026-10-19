// Execute the standalone inspector with a deterministic DOM/canvas double.
// This checks interactions/data plumbing, not rendered visual appearance.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const file=process.argv[2]||'reports/response_lab_v1/run_001/index.html';
const html=fs.readFileSync(file,'utf8');const source=html.match(/<script>([\s\S]*)<\/script>/)[1];
let strokes=0;const ctx=new Proxy({}, {get:(o,k)=>o[k]||((...args)=>{if(k==='stroke')strokes++;for(const v of args)if(typeof v==='number')assert(Number.isFinite(v),`Nonfinite canvas ${k}`)}),set:(o,k,v)=>(o[k]=v,true)});
const nodes=new Map();function node(id){if(!nodes.has(id))nodes.set(id,{id,value:id==='span'?'120':id==='center'?'50':'0',checked:false,clientWidth:1100,clientHeight:150,children:[],appendChild(x){this.children.push(x)},getContext(){return ctx}});return nodes.get(id)}
const sandbox={document:{getElementById:node,createElement:()=>({})},window:{devicePixelRatio:1,addEventListener(){}},console,Date,Math,Number};
vm.createContext(sandbox);vm.runInContext(source,sandbox);
assert(node('scene').children.length>=12);assert(strokes>20);assert(node('comparison').innerHTML.includes('vs B'));
const first=node('sceneInfo').textContent;node('next').onclick();assert.notEqual(node('sceneInfo').textContent,first);node('prev').onclick();assert.equal(node('sceneInfo').textContent,first);
node('span').value='30';node('span').oninput();node('center').value='80';node('center').oninput();node('future').checked=true;node('future').oninput();
node('price').onmousemove({offsetX:600});assert(node('readout').textContent.includes('UTC'));node('price').onmouseleave();
node('scene').value=String(node('scene').children.length-1);node('scene').onchange({target:node('scene')});
console.log(JSON.stringify({status:'PASS',scenes:node('scene').children.length,strokes,checks:['render finite','scene navigation','zoom','pan','future toggle','cursor','C versus B table']}));
