/* Dependency-free 3D scene: world-space cuboids projected into accessible SVG. */
(() => {
'use strict';
const $=id=>document.getElementById(id), T=String.raw;
const palette={mint:'#9ae4cb',blue:'#58c4dd',purple:'#b79ae9',gold:'#f4ce72',gray:'#8792a6'};
const config={name:'Pythia-1B',blocks:16,heads:8,d:2048,ff:8192,vocab:50304,rope:64};
let level='model',block=0,selected='embedding',camera={yaw:-.42,pitch:-.20,zoom:1},nodes=[],edges=[],sequence=[],completed=new Set(),raf=0,flight=null,drag=null,suppressClick=false,parallel=false;
const narrow=matchMedia('(max-width: 700px)');
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const math=t=>`<span class="math" data-tex="${esc(t)}"></span>`;
const N=(id,label,x,y,z,color='mint',w=150,h=38,d=40)=>({id,label,x,y,z,color,w,h,d});
const edge=(from,to,via=[])=>({from,to,via});
function build(){
 const p=config;nodes=[];edges=[];
 if(level==='model'){
  nodes.push(N('tokens','Tokens',0,-285,0,'gray',260,14,120),N('embedding','Embedding',0,-240,0,'blue',260,18,120));
  for(let i=0;i<p.blocks;i++){const y=-195+i*25;nodes.push(N(`block-${i}`,`Bloco ${String(i+1).padStart(2,'0')}`,0,y,0,'mint',260,10,120));}
  nodes.push(N('finalnorm','LayerNorm final',0,235,0,'gold',260,16,120),N('logits','Projeção de saída',0,290,0,'purple',260,16,120));
  nodes.forEach((n,i)=>{if(i)edges.push(edge(nodes[i-1].id,n.id));});
 }else if(level==='block'){
  nodes=[N('input','Entrada',0,-220,0,'gray',170),N('mean','LayerNorm',-195,-115,30,'blue'),N('mlpnorm','LayerNorm',195,-115,-30,'purple'),N('qkv','Atenção',-195,45,30,'blue',190,55,65),N('expand','MLP',195,45,-30,'purple',190,55,65),N('residual','Soma residual',0,220,0,'mint',220,42,55)];
  edges=[edge('input','mean'),edge('input','mlpnorm'),edge('mean','qkv'),edge('mlpnorm','expand'),edge('qkv','residual'),edge('expand','residual'),edge('input','residual',[[-345,-220,0],[-345,220,0]])];
 }else if(level==='attention'){
  nodes=[N('qkv','Query · Key · Value',-285,-120,0,'blue',166),N('position','Posição',-95,-120,0,'blue',140),N('scores','Produto interno',95,-120,0,'gold',160),N('mask','Máscara causal',285,-120,0,'gold',166),N('weights','Softmax',285,100,0,'gold',166),N('weighted','Somar Values',95,100,0,'blue',166),N('concat','Concatenar',-95,100,0,'blue',166),N('project','Projeção',-285,100,0,'mint',166)];
  nodes.forEach((n,i)=>{if(i)edges.push(edge(nodes[i-1].id,n.id));});
 }else if(level==='mlp'){
  nodes=[N('mlpnorm','LayerNorm',-285,0,0,'purple',160),N('expand','Expandir',-95,0,0,'purple',160,64,65),N('activation','GELU',95,0,0,'purple',150),N('contract','Projetar',285,0,0,'mint',160)];
  nodes.forEach((n,i)=>{if(i)edges.push(edge(nodes[i-1].id,n.id));});
 }
 if(level==='output'){
  nodes=[N('finalnorm','LayerNorm final',-260,-110,0,'gold',200),N('logits','Logits',0,-110,0,'purple',180),N('temperature','Temperatura',260,-110,0,'gold',200),N('probabilities','Softmax',260,110,0,'gold',200),N('choose','Próximo token',0,110,0,'mint',200)];
  nodes.forEach((n,i)=>{if(i)edges.push(edge(nodes[i-1].id,n.id));});
 }
 if(narrow.matches){
  if(level==='block'){nodes.forEach(n=>{if(n.x)n.x=Math.sign(n.x)*115;n.w=Math.min(n.w,170);n.d=30;n.z=0;});edges.find(e=>e.via.length).via=[[-210,-220,0],[-210,220,0]];}
  if(level==='attention'){const coords=[[-110,-225],[110,-225],[110,-75],[-110,-75],[-110,75],[110,75],[110,225],[-110,225]];nodes.forEach((n,i)=>{[n.x,n.y]=coords[i];n.w=190;n.d=30;});}
  if(level==='mlp'){const coords=[[-110,-130],[110,-130],[110,130],[-110,130]];nodes.forEach((n,i)=>{[n.x,n.y]=coords[i];n.w=180;n.d=30;});}
  if(level==='output'){const coords=[[-110,-210],[110,-210],[110,0],[-110,0],[-110,210]];nodes.forEach((n,i)=>{[n.x,n.y]=coords[i];n.w=190;n.d=30;});}
 }
 sequence=level==='block'?[['input'],['mean','mlpnorm'],['qkv','expand'],['residual']]:nodes.map(n=>[n.id]);
 if(!nodes.some(n=>n.id===selected))selected=nodes[0].id;
}
// Camera-space +z points toward the viewer. Screen +y points down.
function rotate([x,y,z]){
 const a=camera.yaw,b=camera.pitch,rx=x*Math.cos(a)+z*Math.sin(a),rz=-x*Math.sin(a)+z*Math.cos(a);
 return {x:rx,y:y*Math.cos(b)-rz*Math.sin(b),z:y*Math.sin(b)+rz*Math.cos(b)};
}
function project(point){
 const r=rotate(point),mobile=narrow.matches,fit=level==='model'?(mobile?.78:.83):(mobile?.93:1),scale=fit*camera.zoom;
 return {x:(mobile?(level==='model'?140:240):500)+r.x*scale,y:(mobile?300:275)+r.y*scale,z:r.z};
}
// Front surfaces follow the gesture; pointer and keyboard use the same convention.
function orbit(yaw,pitch){
 camera.yaw=Math.max(-.8,Math.min(.8,yaw));camera.pitch=Math.max(-.48,Math.min(.48,pitch));
}
function tint(color,amount,base=[16,17,23]){
 const rgb=color.slice(1).match(/../g).map(v=>parseInt(v,16));
 return `rgb(${rgb.map((v,i)=>Math.round(base[i]+(v-base[i])*amount)).join(',')})`;
}
const faces=[
 {id:'back',ids:[0,1,2,3],normal:[0,0,-1]},
 {id:'front',ids:[4,7,6,5],normal:[0,0,1]},
 {id:'top',ids:[0,4,5,1],normal:[0,-1,0]},
 {id:'bottom',ids:[3,2,6,7],normal:[0,1,0]},
 {id:'left',ids:[0,3,7,4],normal:[-1,0,0]},
 {id:'right',ids:[1,5,6,2],normal:[1,0,0]}
];
function vertices(n){return [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]].map(([x,y,z])=>project([n.x+x*n.w/2,n.y+y*n.h/2,n.z+z*n.d/2]));}
function pointsString(points){return points.map(p=>`${p.x.toFixed(2)},${p.y.toFixed(2)}`).join(' ');}
function edgeWorld(e){const a=nodes.find(n=>n.id===e.from),b=nodes.find(n=>n.id===e.to);return [[a.x,a.y,a.z],...e.via,[b.x,b.y,b.z]];}
function pathFor(points){return points.map((p,i)=>`${i?'L':'M'}${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(' ');}
function cube(n){
 const v=vertices(n),color=palette[n.color],active=n.id===selected||(parallel&&(sequence.find(g=>g.includes(selected))||[selected]).includes(n.id)),done=completed.has(n.id);
 // Only outward faces pointing toward the viewer are drawn. Opaque surfaces
 // hide rear edges, which otherwise make the rotation perceptually reversible.
 const visible=faces.map(f=>({...f,normal:rotate(f.normal)})).filter(f=>f.normal.z>1e-6);
 const polys=visible.map(({ids,id,normal})=>{
  // A fixed light above/left of the viewer; shading changes continuously on orbit.
  const light=Math.max(0,-.45*normal.x-.65*normal.y+.61*normal.z);
  const shade=(active?.23:done?.17:.12)+.30*light;
  return `<polygon data-face="${id}" points="${pointsString(ids.map(j=>v[j]))}" fill="${tint(color,shade)}" style="--print-fill:${tint(color,.10+.14*light,[255,255,255])}" stroke="${color}" stroke-opacity="${active?.95:done?.65:.45}" stroke-width="${active?1.8:.8}"/>`;
 }).join('');
 const center=project([n.x,n.y,n.z+n.d/2]);
 let decoration='';
 if(level==='block'&&n.id==='qkv'){
  const count=config.heads;decoration=Array.from({length:count},(_,i)=>{const q=project([n.x-n.w*.35+i*n.w*.7/(count-1),n.y-9,n.z+n.d/2+1]);return `<circle cx="${q.x}" cy="${q.y}" r="3" fill="${color}" opacity=".8"/>`;}).join('');
 }
 const showLabel=level!=='model';
 return `<g class="architecture-node ${active?'selected':''}" data-node="${n.id}" role="button" tabindex="0" aria-label="${esc(n.label)}" aria-pressed="${n.id===selected}"><title>${esc(n.label)}</title><g class="block-surfaces">${polys}</g>${decoration}${showLabel?`<text x="${center.x}" y="${center.y+6+(decoration?9:0)}" text-anchor="middle" class="node-label">${esc(n.label)}</text>`:''}</g>`;
}
function draw(){
 const focusedNode=document.activeElement?.dataset.node;
 const svg=$('architectureCanvas');svg.setAttribute('viewBox',narrow.matches?'0 0 480 640':'0 0 1000 580');
 const activePaths=flight?.paths||[];
 const links=edges.map(e=>{const world=edgeWorld(e);if(world.length===2)world.splice(1,0,world[0].map((v,j)=>(v+world[1][j])/2));const path=pathFor(world.map(project)),lit=activePaths.includes(e)||completed.has(e.to);return `<path class="architecture-edge" marker-mid="url(#flow-arrow)" d="${path}" fill="none" stroke="${e.via.length?palette.gray:palette.mint}" stroke-width="${lit?2.4:1.2}" stroke-opacity="${lit?.8:.28}" ${e.via.length?'stroke-dasharray="5 5"':''}/>`;}).join('');
 const sorted=nodes.slice().sort((a,b)=>project([a.x,a.y,a.z]).z-project([b.x,b.y,b.z]).z);
 let labels='';
 if(level==='model'){
  const important=['tokens','embedding',selected,'finalnorm','logits'].filter((v,i,a)=>a.indexOf(v)===i);
  labels=important.map(id=>{const n=nodes.find(n=>n.id===id);if(!n)return '';const p=project([n.x+n.w/2,n.y,0]),y=p.y,offset=narrow.matches?30:57;return `<path d="M${p.x+8} ${y}h${offset-20}" stroke="${palette[n.color]}" stroke-opacity=".5"/><text x="${p.x+offset}" y="${y+6}" class="model-label" fill="${palette[n.color]}">${esc(n.label)}</text>`;}).join('');
  const a=project([-190,0,0]);if(!narrow.matches)labels+=`<text x="${a.x-45}" y="${a.y}" text-anchor="end" class="model-count">${config.blocks}</text><text x="${a.x-45}" y="${a.y+26}" text-anchor="end" class="model-label">blocos</text>`;
 }
 if(level==='block'){
  const a=project([narrow.matches?-210:-345,0,0]);labels+=`<text x="${a.x+(narrow.matches?20:-16)}" y="${a.y}" text-anchor="${narrow.matches?'start':'end'}" class="route-label">Residual</text>`;
  const a2=project([0,-170,0]);labels+=`<text x="${a2.x}" y="${a2.y+12}" text-anchor="middle" class="route-label">mesma entrada</text>`;
 }
 let particles='';
 if(flight){const u=Math.min(1,(performance.now()-flight.start)/flight.duration),eased=u*u*(3-2*u);for(const path of flight.paths){const pts=edgeWorld(path),lengths=pts.slice(1).map((p,i)=>Math.hypot(...p.map((v,j)=>v-pts[i][j]))),total=lengths.reduce((a,b)=>a+b,0);let distance=eased*total;let i=0;while(i<lengths.length-1&&distance>lengths[i])distance-=lengths[i++];const t=lengths[i]?distance/lengths[i]:1,p=project(pts[i].map((v,j)=>v+(pts[i+1][j]-v)*t));particles+=`<circle class="signal-halo" cx="${p.x}" cy="${p.y}" r="9" fill="${palette.mint}" opacity=".15"/><circle cx="${p.x}" cy="${p.y}" r="3.5" fill="${palette.mint}"/>`;}}
 svg.innerHTML=`<defs><filter id="block-shadow" x="-25%" y="-70%" width="150%" height="260%" color-interpolation-filters="sRGB"><feDropShadow dx="2" dy="4" stdDeviation="2.2" flood-color="#000000" flood-opacity=".38"/></filter><marker id="flow-arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M2 2L8 5L2 8" fill="none" stroke="${palette.mint}" stroke-width="1.2"/></marker></defs><title>${esc(config.name)} · ${level==='model'?'modelo completo':level==='output'?'saída':`bloco ${block+1}`}</title><desc>O fluxo segue a ordem dos componentes. Em cada bloco, atenção e MLP recebem a mesma entrada por duas LayerNorms independentes; a soma reúne os dois resultados e o residual.</desc>${links}${sorted.map(cube).join('')}${labels}<g aria-hidden="true">${particles}</g>`;
 if(focusedNode)svg.querySelector(`[data-node="${focusedNode}"]`)?.focus({preventScroll:true});
 $('cameraOut').disabled=camera.zoom<=.7;$('cameraIn').disabled=camera.zoom>=1.35;
}
function info(id){
 // Equations use the terms shown in the slides; labels explain each operation directly.
 if(id.startsWith('block-'))return {title:`Bloco ${String(Number(id.slice(6))+1).padStart(2,'0')}`,description:'A atenção combina informações entre tokens. A MLP transforma cada token. Os dois ramos recebem a mesma entrada, e seus resultados são somados a ela.',equation:T`\text{saída}=\text{entrada}+\text{atenção}+\text{MLP}`,open:'Abrir este bloco',nextLevel:'block',slide:3};
 const entries={
 tokens:['Texto em tokens','O texto é dividido em tokens, partes do texto identificadas por números. A sequência de entrada pode ter até 2.048 tokens.',T`\text{texto}\longrightarrow\text{tokens}\longrightarrow\text{identificadores}`,2],
 embedding:['De um token a um vetor','Cada identificador seleciona uma linha da tabela de embedding. Essa linha fornece as 2.048 coordenadas do vetor do token.',T`\underbrace{50\,304}_{\text{linhas}}\times\underbrace{2\,048}_{\text{coordenadas}}`,2],
 input:['Uma entrada, dois ramos','O vetor de cada token segue para a atenção e para a MLP. Uma cópia também percorre o caminho residual, até a soma.',T`\text{vetor de entrada}:\quad 2\,048\text{ coordenadas}`,3],
 mean:['LayerNorm da atenção','LayerNorm ajusta a média e a escala das coordenadas de cada vetor antes da atenção. A quantidade de coordenadas permanece igual.',T`2\,048\xrightarrow{\text{LayerNorm}}2\,048`,18],
 mlpnorm:['LayerNorm da MLP','Esta normalização recebe a entrada original do bloco e ajusta a média e a escala de cada vetor. Seus parâmetros são próprios, separados dos usados na atenção.',T`2\,048\xrightarrow{\text{LayerNorm}}2\,048`,18],
 qkv:level==='block'?['Atenção em oito cabeças','Cada cabeça compara uma Query com as Keys permitidas e combina os Values. As 2.048 coordenadas são divididas igualmente entre oito cabeças.',T`\underbrace{2\,048}_{\text{coordenadas}}\div\underbrace{8}_{\text{cabeças}}=\underbrace{256}_{\text{por cabeça}}`,10,'Abrir a atenção','attention']:['Query, Key e Value','Três matrizes aprendidas transformam o vetor normalizado. Cada transformação também soma um vetor aprendido. Os resultados são separados em oito cabeças, com 256 coordenadas cada.',T`\begin{gathered}2\,048\xrightarrow{W_Q,\;W_K,\;W_V}\text{Query, Key, Value}\\2\,048=8\times256\end{gathered}`,5],
 position:['Considerar a posição','A ordem dos tokens também entra no cálculo: pares de coordenadas de Query e Key giram conforme a posição do token. Por cabeça, isso altera 64 das 256 coordenadas; Value permanece igual.',T`\underbrace{64}_{\text{com posição}}+\underbrace{192}_{\text{sem rotação}}=256`,18],
 scores:['Score/peso · comparar','O produto interno multiplica as coordenadas correspondentes de Query e Key e soma os resultados. Dividir por 16 ajusta a escala. O resultado ainda é um score, antes da softmax.',T`\text{score}=\frac{\text{Query}\cdot\text{Key}}{\sqrt{256}}=\frac{\text{Query}\cdot\text{Key}}{16}`,6],
 mask:['Score/peso · ocultar o futuro','Uma posição pode consultar a si mesma e as anteriores. Os scores das posições futuras recebem menos infinito; depois da softmax, seus pesos serão zero.',T`\text{posição futura}:\quad -\infty\xrightarrow{\text{softmax}}0`,7],
 weights:['Score/peso · obter os pesos','A softmax exponencia os scores e divide cada resultado pela soma. Para cada Query, os pesos de atenção somam um.',T`\text{peso}=\frac{e^{\text{score}}}{\text{soma dos exponenciais}}`,7],
 weighted:['Combinar os Values','Cada peso multiplica todas as coordenadas de um Value. A soma dessas contribuições produz o resultado da cabeça, com 256 coordenadas.',T`\text{resultado}=\text{peso}_1\,\text{Value}_1+\text{peso}_2\,\text{Value}_2+\cdots`,8],
 concat:['Juntar as oito cabeças','Concatenar é colocar os oito resultados lado a lado, preservando suas coordenadas. Ainda não há uma soma entre as cabeças.',T`8\times256=2\,048`,10],
 project:['Projeção da atenção','A matriz de saída da atenção mistura as 2.048 coordenadas reunidas das cabeças. Depois, soma um vetor aprendido. O resultado volta para a soma residual.',T`2\,048\xrightarrow{\;W_O:\;2\,048\times2\,048\;}2\,048`,10],
 expand:level==='block'?['MLP por token','A MLP expande o vetor, aplica GELU e projeta de volta. Ela usa a mesma transformação em cada token, separadamente.',T`2\,048\xrightarrow{\text{expansão}}8\,192\xrightarrow{\text{GELU}}8\,192\xrightarrow{\text{projeção}}2\,048`,11,'Abrir a MLP','mlp']:['Expandir o vetor','A matriz de expansão transforma 2.048 coordenadas em 8.192, e soma um vetor aprendido. A dimensão intermediária é quatro vezes maior.',T`2\,048\xrightarrow{\;W_{\uparrow}:\;2\,048\times8\,192\;}8\,192`,11],
 activation:['GELU em cada coordenada','GELU transforma cada valor suavemente: atenua entradas negativas em vez de cortá-las de uma vez. Aplica a mesma regra a cada uma das 8.192 coordenadas.',T`8\,192\xrightarrow{\text{GELU}}8\,192`,13],
 contract:['Projetar de volta','A matriz de projeção combina as 8.192 coordenadas e soma um vetor aprendido. A saída tem 2.048 coordenadas, para poder ser somada à entrada do bloco.',T`8\,192\xrightarrow{\;W_{\downarrow}:\;8\,192\times2\,048\;}2\,048`,11],
 residual:['Somar os três caminhos','A soma é coordenada por coordenada: entrada original, resultado da atenção e resultado da MLP. O vetor resultante segue para o próximo bloco; depois do bloco 16, para a LayerNorm final.',T`\begin{gathered}\text{saída}=\text{entrada}+\text{atenção}+\text{MLP}\\2\,048\text{ coordenadas em cada vetor}\end{gathered}`,3],
 temperature:['Ajustar a temperatura','Antes da softmax, cada logit é dividido pela temperatura. Valores positivos menores que um concentram as probabilidades; maiores que um as distribuem mais.',T`\text{logit ajustado}=\frac{\text{logit}}{\text{temperatura}}`,16],
 probabilities:['Dos logits às probabilidades','A softmax exponencia os logits ajustados e divide cada resultado pela soma de todos eles. Há 50.304 probabilidades, uma por saída do modelo; juntas, somam um.',T`\text{probabilidade}=\frac{e^{\text{logit ajustado}}}{\text{soma dos exponenciais}}`,15],
 choose:['Acrescentar o próximo token','A distribuição permite escolher o próximo token. Ele é acrescentado ao texto, e o processo se repete até encerrar a geração.',T`\text{sequência}\longrightarrow\text{próximo token}\longrightarrow\text{sequência ampliada}`,1,'Voltar à entrada','model'],
 finalnorm:['LayerNorm final','Após os 16 blocos, LayerNorm ajusta a média e a escala de cada vetor. Para prever o próximo token, usamos o vetor da última posição.',T`\underbrace{2\,048}_{\text{última posição}}\xrightarrow{\text{LayerNorm}}2\,048`,18],
 logits:['Projetar para o vocabulário','A matriz de saída transforma o último vetor em 50.304 logits, os valores que antecedem as probabilidades. Essa matriz tem pesos próprios, diferentes dos do embedding.',T`2\,048\xrightarrow{\;W_{\mathrm{out}}:\;2\,048\times50\,304\;}50\,304`,14]
 };
 const [title,description,equation,slide,open,nextLevel]=entries[id]||entries.input;
 return {title,description,equation,slide,open:level==='model'&&['logits','finalnorm'].includes(id)?'Abrir a saída':open,nextLevel:level==='model'&&['logits','finalnorm'].includes(id)?'output':nextLevel};
}
function typeset(root){root.querySelectorAll('[data-tex]').forEach(el=>katex.render(el.dataset.tex,el,{throwOnError:true,strict:false}));}
function renderInfo(){
 const p=config,i=info(selected),position=sequence.findIndex(group=>group.includes(selected));
 if(parallel&&level==='block'){
  if(selected==='mean'){i.title='Normalizar os dois ramos';i.description='As duas LayerNorms ajustam a média e a escala da mesma entrada. Cada ramo tem seus próprios parâmetros; ambos preservam as 2.048 coordenadas.';i.equation=T`2\,048\xrightarrow{\text{cada LayerNorm}}2\,048`;}
  if(selected==='qkv'){i.title='Atenção e MLP em paralelo';i.description='A atenção combina informações entre tokens; a MLP transforma cada token separadamente. Um ramo não depende do resultado do outro.';i.equation=T`\text{saída}=\text{entrada}+\text{atenção}+\text{MLP}`;}
 }
 $('architectureSpecs').innerHTML=`<span>16 blocos</span><span>8 cabeças por bloco</span><span>${math(T`2\,048`)} coordenadas por vetor</span>`;typeset($('architectureSpecs'));
 $('architectureBreadcrumb').innerHTML=`<button data-level="model" ${level==='model'?'aria-current="location"':''}>Modelo completo</button>${!['model','output'].includes(level)?`<span>/</span><button data-level="block" ${level==='block'?'aria-current="location"':''}>Bloco ${String(block+1).padStart(2,'0')}</button>`:''}${['attention','mlp','output'].includes(level)?`<span>/</span><span aria-current="location">${level==='attention'?'Atenção':level==='mlp'?'MLP':'Saída'}</span>`:''}`;
 $('componentTitle').textContent=i.title;$('componentDescription').textContent=i.description;
 $('componentEquation').innerHTML=math(i.equation);typeset($('componentEquation'));
 $('architecturePosition').textContent=`${String(position+1).padStart(2,'0')} / ${String(sequence.length).padStart(2,'0')}`;
 $('openComponent').hidden=!i.open;$('openComponent').textContent=i.open||'';
 $('reviewSlide').href=`../index.html#slide-${i.slide}`;$('reviewSlide').textContent=`Rever slide ${String(i.slide+1).padStart(2,'0')}`;
 $('architectureBack').disabled=position===0;const atEnd=position===sequence.length-1;
 $('architectureNext').disabled=atEnd&&level==='output';
 $('architectureNext').textContent=!atEnd?'Seguir o fluxo':level==='attention'?'Explorar a MLP':level==='mlp'?'Ver a soma':level==='block'?(block<p.blocks-1?'Próximo bloco':'Ver a saída'):level==='model'?'Explorar a saída':'Seguir o fluxo';
 $('architectureStatus').textContent=`Etapa ${position+1} de ${sequence.length}: ${i.title}`;
 PythiaDetails.update({level,selected,block});
}
function stop(){if(raf)cancelAnimationFrame(raf);raf=0;flight=null;}
function selectNode(id,animate=false,guided=false){
 const focusId=document.activeElement?.dataset.node;
 stop();selected=id;const index=sequence.findIndex(g=>g.includes(id));completed=new Set(sequence.slice(0,index).flat());const group=guided?(sequence[index]||[id]):[id];parallel=guided&&group.length>1;
 if(animate&&!reduced.matches&&!document.hidden){const paths=edges.filter(e=>group.includes(e.to));flight={targets:group,paths,start:performance.now(),duration:1000/1.25};draw();
  const tick=()=>{draw();if(flight&&performance.now()-flight.start<flight.duration){raf=requestAnimationFrame(tick);}else{group.forEach(v=>completed.add(v));stop();draw();}};raf=requestAnimationFrame(tick);
 }else{group.forEach(v=>completed.add(v));draw();}
 renderInfo();if(focusId)$('architectureCanvas').querySelector(`[data-node="${id}"]`)?.focus({preventScroll:true});
}
function openLevel(next,id){stop();level=next;selected=id||'';completed.clear();camera={yaw:level==='model'?-.42:-.18,pitch:level==='model'?-.2:-.3,zoom:1};build();selectNode(selected);}
function openSelected(){const i=info(selected);if(!i.nextLevel)return;if(selected.startsWith('block-'))block=Number(selected.slice(6));openLevel(i.nextLevel);}
function init(){
 PythiaDetails.init(({target,id,targetBlock})=>{if(targetBlock!==undefined)block=targetBlock;openLevel(target,id);});
 $('openComponent').onclick=openSelected;
 $('architectureNext').onclick=()=>{const i=sequence.findIndex(g=>g.includes(selected));if(i<sequence.length-1){selectNode(sequence[i+1][0],true,true);return;}if(level==='model')openLevel('output');else if(level==='attention')openLevel('mlp');else if(level==='mlp')openLevel('block','residual');else if(level==='block'){if(block<config.blocks-1){block++;openLevel('block');}else openLevel('output');}};
 $('architectureBack').onclick=()=>{const i=sequence.findIndex(g=>g.includes(selected));if(i>0)selectNode(sequence[i-1][0],false,true);};
 $('architectureBreadcrumb').onclick=e=>{const button=e.target.closest('[data-level]');if(button)openLevel(button.dataset.level,button.dataset.level==='model'?`block-${block}`:'input');};
 const reset=()=>{camera={yaw:level==='model'?-.42:-.18,pitch:level==='model'?-.2:-.3,zoom:1};draw();};
 $('cameraReset').onclick=reset;$('cameraFront').onclick=()=>{camera.yaw=0;camera.pitch=0;draw();};
 const changeZoom=amount=>{camera.zoom=Math.max(.7,Math.min(1.35,camera.zoom+amount));draw();};
 $('cameraIn').onclick=()=>changeZoom(.1);$('cameraOut').onclick=()=>changeZoom(-.1);
 const svg=$('architectureCanvas');
 svg.addEventListener('click',e=>{if(suppressClick){suppressClick=false;return;}const node=e.target.closest('[data-node]');if(node)selectNode(node.dataset.node,true);});
 svg.addEventListener('dblclick',e=>{const node=e.target.closest('[data-node]');if(node){selectNode(node.dataset.node);openSelected();}});
 svg.addEventListener('pointerdown',e=>{if(e.button!==0||!e.isPrimary)return;suppressClick=false;drag={pointerId:e.pointerId,x:e.clientX,y:e.clientY,yaw:camera.yaw,pitch:camera.pitch,moved:false};});
 svg.addEventListener('pointermove',e=>{if(!drag||e.pointerId!==drag.pointerId)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.hypot(dx,dy)<5&&!drag.moved)return;drag.moved=true;if(!svg.hasPointerCapture(e.pointerId))svg.setPointerCapture(e.pointerId);stop();orbit(drag.yaw+dx*.004,drag.pitch-dy*.003);draw();});
 // Also end a press released outside the SVG before capture starts.
 window.addEventListener('pointerup',e=>{if(!drag||e.pointerId!==drag.pointerId)return;suppressClick=drag.moved;drag=null;if(svg.hasPointerCapture(e.pointerId))svg.releasePointerCapture(e.pointerId);});
 const cancelDrag=()=>{drag=null;suppressClick=false;};
 svg.addEventListener('pointercancel',cancelDrag);svg.addEventListener('lostpointercapture',()=>{drag=null;});window.addEventListener('blur',cancelDrag);
 svg.addEventListener('keydown',e=>{const node=e.target.closest('[data-node]');if(node&&['Enter',' '].includes(e.key)){e.preventDefault();selectNode(node.dataset.node,true);return;}const key=e.key.toLowerCase();if(!['a','d','w','s','+','-','=','0'].includes(key))return;e.preventDefault();stop();if(key==='0'){reset();return;}if(key==='a')orbit(camera.yaw-.08,camera.pitch);if(key==='d')orbit(camera.yaw+.08,camera.pitch);if(key==='w')orbit(camera.yaw,camera.pitch+.06);if(key==='s')orbit(camera.yaw,camera.pitch-.06);if(key==='+'||key==='=')changeZoom(.1);else if(key==='-')changeZoom(-.1);else draw();svg.focus({preventScroll:true});});
 document.addEventListener('keydown',e=>{if(e.defaultPrevented||e.ctrlKey||e.metaKey||e.altKey||document.querySelector('dialog[open]')||e.target.closest('input,select,textarea,summary'))return;if(e.key==='ArrowRight'){e.preventDefault();$('architectureNext').click();}if(e.key==='ArrowLeft'){e.preventDefault();$('architectureBack').click();}});
 document.addEventListener('visibilitychange',()=>{if(document.hidden){stop();draw();}});window.addEventListener('pagehide',stop);reduced.addEventListener('change',()=>{stop();draw();});
 let savedCamera;window.addEventListener('beforeprint',()=>{stop();savedCamera={...camera};camera={yaw:-.18,pitch:-.2,zoom:1};draw();});window.addEventListener('afterprint',()=>{if(savedCamera){camera=savedCamera;draw();}});
 narrow.addEventListener('change',()=>{stop();build();draw();});
 $('modelDimensions').innerHTML=`<dl><dt>Embedding</dt><dd>${math(T`50\,304\times2\,048=103\,022\,592`)}</dd><dt>Atenção por bloco</dt><dd>${math(T`4(2\,048^2+2\,048)=16\,785\,408`)}</dd><dt>MLP por bloco</dt><dd>${math(T`2(2\,048\times8\,192)+8\,192+2\,048=33\,564\,672`)}</dd><dt>Duas LayerNorms por bloco</dt><dd>${math(T`2(2\times2\,048)=8\,192`)}</dd><dt>Os 16 blocos</dt><dd>${math(T`16\times50\,358\,272=805\,732\,352`)}</dd><dt>LayerNorm final</dt><dd>${math(T`2\times2\,048=4\,096`)}</dd><dt>Projeção de saída</dt><dd>${math(T`2\,048\times50\,304=103\,022\,592`)}</dd><dt>Total de parâmetros</dt><dd>${math(T`1\,011\,781\,632`)}</dd></dl><p>Nas projeções internas, contamos os pesos das matrizes e os vetores somados ao resultado. Cada LayerNorm aprende dois valores por coordenada: escala e deslocamento.</p>`;
 typeset($('modelDimensions'));build();selectNode('tokens');
}
init();
})();
