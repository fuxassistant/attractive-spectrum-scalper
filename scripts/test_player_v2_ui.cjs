// Unit test the exported local UI with a small DOM/canvas test double.
// No browser process, computer interaction or network requests. Not a screenshot.
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert');
const root=path.resolve(process.argv[2]||'reports/player_v2/run_001');
const nodes=new Map();let context;let draws=0;const callbacks=new Map();let nextTimer=1;let commands=[];
const ctx=new Proxy({}, {get:(o,k)=>k in o?o[k]:(...args)=>{draws++;commands.push([k,...args]);for(const x of args)if(typeof x==='number')assert(Number.isFinite(x),`nonfinite canvas ${k}`);},set:(o,k,v)=>(commands.push(['set',k,v]),o[k]=v,true)});
class Element{
 constructor(id){this.id=id;this.value='';this._html='';this.checked=false;this.children=[];this.clientWidth=1080;this.width=0;this.height=0;}
 replaceChildren(...items){this.children=items;this.value=items[0]?.value??'';}
 set innerHTML(v){this._html=v;}get innerHTML(){return this._html;}
 querySelectorAll(){return [];}getContext(){return ctx;}getBoundingClientRect(){return{left:0};}
 remove(){}addEventListener(){}
}
function get(id){if(!nodes.has(id))nodes.set(id,new Element(id));return nodes.get(id);}
Object.assign(get('mode'),{value:'selected_test'});Object.assign(get('span'),{value:'180'});Object.assign(get('cursor'),{value:'539'});get('outcomes').checked=true;
const document={getElementById:get,createElement:()=>new Element('script'),body:{appendChild(el){if(el.src){const filename=path.resolve(root,el.src);assert(filename.startsWith(root+path.sep));vm.runInContext(fs.readFileSync(filename,'utf8'),context);}}}};
context=vm.createContext({document,console,devicePixelRatio:1,Option:function(t,v){this.text=t;this.value=v;},setInterval:fn=>{const id=nextTimer++;callbacks.set(id,fn);return id;},clearInterval:id=>callbacks.delete(id)});
context.window=context;context.addEventListener=()=>{};
vm.runInContext(fs.readFileSync(path.join(root,'manifest.js'),'utf8'),context);
const html=fs.readFileSync(path.join(root,'index.html'),'utf8');const js=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(x=>x[1]).join('\n');
vm.runInContext(js,context);
assert(get('state').innerHTML.includes('Намерение участия'));assert(get('status').textContent.includes('1379'));
get('seed').value='61017';get('seed').onchange();assert(get('dayMetrics').textContent.includes('сделок'));
get('nextTrade').onclick();assert(get('state').innerHTML.includes('LONG')||get('state').innerHTML.includes('SHORT'));
get('span').value='60';get('span').onchange();get('peak').onclick();
get('gatezoom').checked=true;get('gatezoom').onchange();get('gatezoom').checked=false;get('maskedplans').checked=true;get('maskedplans').onchange();get('maskedplans').checked=false;
get('attribution').value=get('attribution').children[1].value;get('attribution').onchange();assert(get('why').innerHTML.includes('log-odds'));
get('mode').value='validation';get('mode').onchange();get('epoch').value='validation_epoch_060';get('epoch').onchange();assert(get('status').textContent.includes('epoch_060'));
get('play').onclick();assert(callbacks.size===1);const tick=callbacks.values().next().value;const before=+get('cursor').value;tick();assert(+get('cursor').value===before+1);get('play').onclick();assert(callbacks.size===0);
// Every exported dataset must render without nonfinite canvas coordinates.
let count=0;for(const ds of context.AUDIT.datasets){commands=[];vm.runInContext(`wanted=${JSON.stringify(ds.key)};`,context);vm.runInContext(fs.readFileSync(path.join(root,ds.file),'utf8'),context);count++;}
get('future').checked=false;get('cursor').value='539';commands=[];vm.runInContext('render()',context);const prefix=JSON.stringify(commands);commands=[];
vm.runInContext('for(let j=540;j<D.t.length;j++){D.candles[j]=[999999,999999,999999,999999];D.intent[j]=.999;D.equity[j]=999999;}render();',context);
assert.strictEqual(JSON.stringify(commands),prefix,'Future price/signal/equity must not affect prefix canvas or axes');
get('future').checked=true;commands=[];vm.runInContext('render()',context);assert.notStrictEqual(JSON.stringify(commands),prefix);
const receipt={kind:'JS unit harness with DOM and canvas test doubles; not browser QA',datasets:count,draw_calls:draws,controls:['seed','mode','date loading','epoch60','span','next trade','peak','sensitivity point','play/pause','gate zoom','raw vs masked plans','future hiding and prefix-axis invariance'],passed:true,browser_status:'file URL browser automation blocked; not bypassed'};
fs.writeFileSync(path.join(root,'ui_unit_receipt.json'),JSON.stringify(receipt,null,2));console.log(receipt);
