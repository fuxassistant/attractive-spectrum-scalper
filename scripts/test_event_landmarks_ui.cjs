// DOM/canvas double: causal display boundaries and controls, not visual QA.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(process.argv[2]||'reports/event_landmarks_v1/run_003/index.html','utf8');
const source=html.match(/<script>([\s\S]*)<\/script>/)[1];let strokes=0;
const ctx=new Proxy({}, {get:(o,k)=>o[k]||((...args)=>{if(k==='stroke')strokes++;for(const x of args)if(typeof x==='number')assert(Number.isFinite(x),k)}),set:(o,k,v)=>(o[k]=v,true)});
const nodes=new Map();function node(id){if(!nodes.has(id))nodes.set(id,{value:'0',checked:false,children:[],events:{},appendChild(n){this.children.push(n)},addEventListener(k,f){this.events[k]=f},getBoundingClientRect(){return {width:1000,height:520}},getContext(){return ctx}});return nodes.get(id)}
const sandbox={document:{getElementById:node,createElement:()=>({})},window:{devicePixelRatio:1,addEventListener(){}},console};
vm.createContext(sandbox);vm.runInContext(source,sandbox);
assert.equal(node('scene').children.length,12);assert(strokes>10);assert(node('futureFacts').hidden);
assert(!node('sceneInfo').textContent.includes('no repeat'));assert(!node('sceneInfo').textContent.includes('later repeat'));
const first=node('sceneInfo').textContent;node('next').onclick();assert.notEqual(node('sceneInfo').textContent,first);node('prev').onclick();assert.equal(node('sceneInfo').textContent,first);
node('clock').value='-180';node('clock').events.input();assert(node('facts').innerHTML.includes('ещё не произошло'));
node('reset').onclick();assert(node('facts').innerHTML.includes('Известно при распознавании'));node('step').onclick();assert.equal(+node('clock').value,30);
node('future').checked=true;node('future').events.change();assert(!node('futureFacts').hidden);assert(node('futureFacts').textContent.includes('РЕТРОСПЕКТИВА'));
node('future').checked=false;node('future').events.change();assert(node('futureFacts').hidden);
for(let i=0;i<12;i++){node('scene').value=String(i);node('scene').events.change();assert(!node('sceneInfo').textContent.includes('no repeat'));assert(!node('sceneInfo').textContent.includes('later repeat'))}
assert(node('table').innerHTML.includes('Первый всплеск'));
console.log(JSON.stringify({status:'PASS',scenes:12,strokes,checks:['finite render','scene navigation','before-recognition features hidden','hindsight outcome and selection hidden','time step/reset','future toggle','summary table']}));
