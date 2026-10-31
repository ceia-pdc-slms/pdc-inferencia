/* Real Pythia dimensions; symbolic operations, with no invented checkpoint values. */
(() => {
'use strict';
const $=id=>document.getElementById(id),T=String.raw;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=n=>String(n).replace(/\B(?=(\d{3})+(?!\d))/g,'\\,');
const math=t=>`<span data-tex="${esc(t)}"></span>`;
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
let navigate,state={},step=0,projection='Q',animations=[],operation=null;
function typeset(root){root.querySelectorAll('[data-tex]').forEach(el=>katex.render(el.dataset.tex,el,{throwOnError:true,strict:false}));}
function stop(){animations.forEach(a=>a.cancel());animations=[];}
function pulse(el,delay=0){if(!el||reduced.matches||document.hidden)return;animations.push(el.animate([{opacity:.45,transform:'translateY(3px)'},{opacity:1,transform:'translateY(0)'}],{duration:640,delay,fill:'backwards',easing:'ease-out'}));}
function heads(){return `<div class="head-strip" aria-label="Oito cabeças de atenção">${Array.from({length:8},(_,i)=>`<span aria-label="Cabeça ${i+1}">${math(String(i+1))}</span>`).join('')}</div><p class="detail-caption">${math(T`8\times256=2\,048`)} coordenadas</p>`;}
function overview(){
 const block=state.selected.startsWith('block-')?Number(state.selected.slice(6)):state.block;
 return `<div class="block-overview"><p class="detail-caption">A mesma entrada segue pelos dois ramos.</p><div class="branch-grid"><div><button data-part="mean" data-target="block" data-block="${block}">LayerNorm da atenção</button><span class="branch-arrow" aria-hidden="true">↓</span><button data-part="qkv" data-target="attention" data-block="${block}">Atenção · 8 cabeças</button>${heads()}</div><div><button data-part="mlpnorm" data-target="mlp" data-block="${block}">LayerNorm da MLP</button><span class="branch-arrow" aria-hidden="true">↓</span><button data-part="expand" data-target="mlp" data-block="${block}">MLP</button><p class="branch-dimensions">${math(T`2\,048\to8\,192`)}<br>GELU<br>${math(T`8\,192\to2\,048`)}</p></div></div><button class="residual-detail" data-part="residual" data-target="block" data-block="${block}">Somar entrada + atenção + MLP</button></div>`;
}
function linear(){
 const id=state.selected;
 if(id==='qkv')return {input:2048,output:2048,matrix:`W_${projection}`,bias:true,result:projection==='Q'?'Query':projection==='K'?'Key':'Value',next:'Separar o resultado em 8 cabeças, com 256 coordenadas por cabeça.'};
 if(id==='project')return {input:2048,output:2048,matrix:'W_O',bias:true,result:'Atenção',next:'O resultado segue para a soma residual.'};
 if(id==='expand')return {input:2048,output:8192,matrix:T`W_{\uparrow}`,bias:true,result:'Expansão',next:'GELU transforma cada uma das 8.192 coordenadas.'};
 if(id==='contract')return {input:8192,output:2048,matrix:T`W_{\downarrow}`,bias:true,result:'MLP',next:'O resultado segue para a soma residual.'};
 if(id==='logits')return {input:2048,output:50304,matrix:T`W_{\mathrm{out}}`,bias:false,result:'Logits',next:'Os logits seguem para temperatura e softmax.'};
 return null;
}
function glyph(kind){return `<div class="tensor-glyph ${kind}" aria-hidden="true">${Array.from({length:kind==='matrix'?25:5},(_,i)=>`<i class="${kind==='matrix'&&i%5===2?'column':''}"></i>`).join('')}</div>`;}
function matrixMarkup(){
 const op=operation,d=fmt(op.input),out=fmt(op.output);
 return `${state.selected==='qkv'?`<div class="projection-picker" role="group" aria-label="Projeção">${['Q','K','V'].map(p=>`<button data-projection="${p}" aria-pressed="${p===projection}">${p==='Q'?'Query':p==='K'?'Key':'Value'}</button>`).join('')}</div>${heads()}`:''}
 <section class="operation" aria-label="Multiplicação matricial${op.bias?' e bias':''}">
 <h3>Como o vetor é transformado</h3><p class="detail-caption">O vetor de entrada ${math('x')} é multiplicado pela matriz.${op.bias?' O bias é um vetor aprendido, somado ao produto.':' A projeção de saída do Pythia-1B não soma bias.'}</p>
 <div class="tensor-product"><div class="tensor-card" id="inputTensor"><span>Entrada</span>${glyph('vector')}${math(T`1\times${d}`)}</div><span class="operator">${math(T`\times`)}</span><div class="tensor-card" id="matrixTensor"><span>Matriz ${math(op.matrix)}</span>${glyph('matrix')}${math(T`${d}\times${out}`)}</div></div>
 <ol class="operation-steps"><li data-calc="0"><strong>Multiplicar e somar</strong><div id="productEquation"></div><div id="productTensor" class="product-result">${glyph('vector')}<span>Produto</span></div></li>${op.bias?`<li data-calc="1"><strong>Somar o bias</strong><div class="bias-motion"><span>Produto</span><span>${math('+')}</span><span id="biasVector">${glyph('vector')}Bias ${math('b')}</span></div><div id="biasEquation"></div></li>`:''}<li data-calc="${op.bias?2:1}"><strong>${esc(op.result)}</strong><div class="result-vector">${math(T`\text{saída}=x${op.matrix}${op.bias?'+b':''}`)}<br>${math(T`1\times${out}`)}</div><p class="detail-caption">${op.next}</p></li></ol>
 <details class="coordinate-detail"><summary>Ver a conta de uma coordenada</summary><label>Coordenada da saída <input id="outputCoordinate" type="number" min="1" max="${op.output}" value="1" inputmode="numeric"></label><p class="detail-caption">Multiplicamos cada valor da entrada pelo peso da coluna escolhida e somamos.${op.bias?' Depois acrescentamos o bias dessa coordenada.':''}</p><div id="coordinateEquation" class="detail-equation"></div><p class="detail-caption">${math('x')}: vetor de entrada. ${math(op.matrix)}: matriz aprendida.${op.bias?` ${math('b')}: vetor de bias.`:''}</p></details>
 </section>`;
}
function generic(){
 const id=state.selected;
 const entries={
 mean:[['Vetor de entrada',T`2\,048\text{ coordenadas}`],['Normalizar','Ajustar a média e a escala das coordenadas.'],['Escala e deslocamento','Multiplicar cada coordenada pela escala aprendida e somar seu deslocamento.']],
 mlpnorm:[['Entrada original do bloco',T`2\,048\text{ coordenadas}`],['Normalizar','Ajustar a média e a escala das coordenadas.'],['Escala e deslocamento','Esta LayerNorm tem seus próprios parâmetros aprendidos.']],
 finalnorm:[['Última posição após o bloco 16',T`2\,048\text{ coordenadas}`],['Normalizar','Ajustar a média e a escala das coordenadas.'],['Escala e deslocamento','Aplicar a escala e o deslocamento aprendidos; seguir para a projeção de saída.']],
 activation:[['Entrada expandida',T`8\,192\text{ coordenadas}`],['Aplicar GELU','Transformar cada valor suavemente, atenuando entradas negativas.'],['Saída da ativação',T`8\,192\text{ coordenadas}`]],
 residual:[['Entrada original',T`2\,048\text{ coordenadas}`],['Resultados dos dois ramos','Atenção e MLP produzem vetores de 2.048 coordenadas.'],['Somar coordenadas correspondentes',T`\text{entrada}+\text{atenção}+\text{MLP}`]],
 scores:[['Comparar Query e Key',T`\underbrace{\text{Query}}_{1\times256}\underbrace{\text{Key em coluna}}_{256\times1}`],['Somar os produtos','Multiplicar coordenadas correspondentes e somar os 256 produtos.'],['Ajustar a escala',T`\text{score}=\frac{\text{soma dos produtos}}{16}`]],
 position:[['Query e Key',T`256\text{ coordenadas por cabeça}`],['Incluir a posição','Girar 32 pares de coordenadas conforme a posição do token.'],['Seguir para a comparação',T`64\text{ coordenadas giradas}+192\text{ preservadas}`]],
 mask:[['Scores antes da softmax','Um score para cada Key.'],['Ocultar posições futuras',T`\text{score futuro}=-\infty`],['Após a softmax',T`\text{peso futuro}=0`]],
 weights:[['Exponenciar os scores',T`e^{\text{score}}`],['Somar os exponenciais','Usar todos os scores desta consulta.'],['Dividir pela soma',T`\text{peso}=\frac{e^{\text{score}}}{\text{soma}}`]],
 weighted:[['Pesos e Values','Um peso para cada Value permitido.'],['Multiplicar','Cada peso multiplica todas as 256 coordenadas de seu Value.'],['Somar as contribuições',T`\text{peso}_1\text{Value}_1+\text{peso}_2\text{Value}_2+\cdots`]],
 concat:[['Oito resultados',T`8\text{ vetores de }256`],['Colocar lado a lado','Preservar todas as coordenadas, sem somá-las.'],['Um vetor reunido',T`8\times256=2\,048`]],
 temperature:[['Logits de saída',T`50\,304\text{ valores}`],['Dividir pela temperatura',T`\frac{\text{logit}}{\text{temperatura}}`],['Seguir para a softmax','A ordem dos valores permanece igual.']],
 probabilities:[['Exponenciar os logits ajustados',T`e^{\text{logit ajustado}}`],['Somar os exponenciais','Usar todas as 50.304 saídas.'],['Dividir pela soma',T`\text{probabilidade}=\frac{e^{\text{logit ajustado}}}{\text{soma}}`]],
 choose:[['Probabilidades','Uma distribuição sobre as saídas do modelo.'],['Escolher um token','Usar a distribuição para escolher a continuação.'],['Acrescentar à sequência','Repetir o percurso com a sequência ampliada.']]
 };
 const rows=entries[id];if(!rows)return '';
 return `<section class="operation"><h3>Acompanhe a transformação</h3><ol class="operation-steps">${rows.map(([title,content],i)=>`<li data-calc="${i}"><strong>${title}</strong><div>${content.includes('\\')||content.includes('=')||content.includes('^')?math(content):esc(content)}</div></li>`).join('')}</ol></section>`;
}
function equationForCoordinate(){
 const input=$('outputCoordinate');if(!input||!operation)return;
 const j=Math.max(1,Math.min(operation.output,Math.trunc(Number(input.value)||1))),d=fmt(operation.input),w=operation.matrix;
 input.value=j;
 $('coordinateEquation').innerHTML=math(T`\begin{aligned}\text{saída}_{${j}}={}&x_1(${w})_{1,${j}}+\cdots\\&+x_{${d}}(${w})_{${d},${j}}${operation.bias?T`+b_{${j}}`:''}\end{aligned}`);
 typeset($('coordinateEquation'));
}
function renderStep(animate=false){
 const rows=[...$('detailContent').querySelectorAll('[data-calc]')];if(!rows.length)return;
 rows.forEach((el,i)=>{el.classList.toggle('calc-current',i===step);el.classList.toggle('calc-complete',i<step);el.classList.toggle('calc-pending',i>step);el.setAttribute('aria-current',i===step?'step':'false');});
 $('calculationCount').textContent=`${String(step+1).padStart(2,'0')} / ${String(rows.length).padStart(2,'0')}`;
 $('calculationBack').disabled=step===0;$('calculationNext').disabled=step===rows.length-1;
 $('calculationNext').textContent=step===rows.length-1?'Concluído':operation?.bias&&step===0?'Somar bias':'Continuar';
 $('calculationStatus').textContent=rows[step].querySelector('strong').textContent;
 stop();if(animate){pulse(rows[step]);if(operation&&step===0){pulse($('inputTensor'));pulse($('matrixTensor'),160);pulse(rows[step],320);pulse($('productTensor'),480);if(!reduced.matches&&!document.hidden){$('detailContent').querySelectorAll('#inputTensor i,#matrixTensor .column').forEach((cell,i)=>animations.push(cell.animate([{backgroundColor:'#9ae4cb',opacity:1},{backgroundColor:'#58c4dd',opacity:.4}],{duration:640,delay:i*40,easing:'ease-out'})));}}if(operation?.bias&&step===1&&!reduced.matches&&!document.hidden)animations.push($('biasVector').animate([{transform:'translateX(18px)',opacity:.35},{transform:'translateX(0)',opacity:1}],{duration:800,easing:'ease-out'}));}
}
function buildDetail(){
 stop();step=reduced.matches?2:0;operation=linear();$('componentPanel').classList.toggle('has-matrix',!!operation);
 let html='';
 if(state.selected.startsWith('block-')||state.selected==='input')html=overview();
 else html=operation?matrixMarkup():generic();
 if(html.includes('data-calc'))html+=`<nav class="calculation-controls" aria-label="Etapas da transformação"><button id="calculationBack" aria-label="Etapa anterior">←</button><span id="calculationCount"></span><button id="calculationNext">Continuar</button><button id="calculationReplay" aria-label="Rever movimento desta etapa">↻</button></nav><span id="calculationStatus" class="sr-only" role="status" aria-live="polite"></span>`;
 $('detailContent').innerHTML=html;
 if(operation){$('productEquation').innerHTML=math(T`\underbrace{x}_{1\times${fmt(operation.input)}}\underbrace{${operation.matrix}}_{${fmt(operation.input)}\times${fmt(operation.output)}}=\underbrace{\text{produto}}_{1\times${fmt(operation.output)}}`);if(operation.bias)$('biasEquation').innerHTML=math(T`\underbrace{\text{produto}}_{1\times${fmt(operation.output)}}+\underbrace{b}_{1\times${fmt(operation.output)}}`);}
 typeset($('detailContent'));equationForCoordinate();
 const count=$('detailContent').querySelectorAll('[data-calc]').length;step=Math.min(step,count-1);if(count)renderStep(true);
 if($('calculationNext')){$('calculationNext').onclick=()=>{step++;renderStep(true);};$('calculationBack').onclick=()=>{step--;renderStep();};$('calculationReplay').onclick=()=>renderStep(true);}
 if($('outputCoordinate'))$('outputCoordinate').onchange=equationForCoordinate;
}
function update(next){
 state=next;const show=next.selected.startsWith('block-')||next.level!=='model'||['finalnorm','logits'].includes(next.selected);
 $('componentPanel').hidden=!show;$('architectureWorkspace').classList.toggle('has-details',show);document.querySelector('main').classList.toggle('with-details',show);
 (show?$('detailInspector'):$('inspectorHome')).append(document.querySelector('.architecture-inspector'));
 if(show)buildDetail();else{stop();$('detailContent').replaceChildren();}
}
function init(callback){
 navigate=callback;
 $('detailContent').addEventListener('click',e=>{const part=e.target.closest('[data-part]'),proj=e.target.closest('[data-projection]');if(part){navigate({target:part.dataset.target,id:part.dataset.part,targetBlock:Number(part.dataset.block)});$('componentTitle').focus({preventScroll:true});}if(proj){projection=proj.dataset.projection;buildDetail();$('detailContent').querySelector(`[data-projection="${projection}"]`)?.focus();}});
 $('componentPanel').addEventListener('keydown',e=>{if(e.target.closest('input,summary')||e.ctrlKey||e.altKey||e.metaKey)return;if(['ArrowLeft','ArrowRight'].includes(e.key)&&$('calculationNext')){e.preventDefault();$(e.key==='ArrowRight'?'calculationNext':'calculationBack').click();}});
 document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});window.addEventListener('pagehide',stop);
 let savedStep;window.addEventListener('beforeprint',()=>{savedStep=step;step=$('detailContent').querySelectorAll('[data-calc]').length-1;renderStep();});window.addEventListener('afterprint',()=>{if(savedStep!==undefined){step=savedStep;renderStep();}});
 reduced.addEventListener('change',()=>{stop();if(!reduced.matches)return;step=$('detailContent').querySelectorAll('[data-calc]').length-1;renderStep();});
}
window.PythiaDetails={init,update};
})();
