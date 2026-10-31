(() => {
  // Ritmo comum a todas as animações, incluindo vídeos locais.
  const ANIMATION_SPEED = 1.25;

  document.querySelectorAll('.latex[data-tex]').forEach(element => {
    if (typeof katex !== 'undefined') {
      katex.render(element.dataset.tex, element, { throwOnError: false, strict: false });
    } else {
      element.textContent = element.dataset.tex;
    }
  });

  document.getElementById('generatePdf').addEventListener('click', async () => {
    await document.fonts.ready;
    window.print();
  });

  function createCoverAnimation() {
    const svg = document.querySelector('.attention-diagram');
    const namespace = 'http://www.w3.org/2000/svg';
    const layer = document.createElementNS(namespace, 'g');
    layer.classList.add('cover-animation-layer');
    layer.setAttribute('aria-hidden', 'true');
    layer.setAttribute('pointer-events', 'none');
    svg.append(layer);
    const animations = [];
    const duration = 8000 / ANIMATION_SPEED;

    function animate(element, frames) {
      const animation = element.animate(frames, { duration, iterations: Infinity, easing: 'linear' });
      animation.pause();
      animation.currentTime = 0;
      animations.push(animation);
    }

    function highlight(rect, color, start, end, transform) {
      const outline = rect.cloneNode(false);
      outline.setAttribute('fill', 'none');
      outline.setAttribute('stroke', color);
      outline.setAttribute('stroke-width', '3');
      outline.setAttribute('opacity', '0');
      if (transform) outline.setAttribute('transform', transform);
      layer.append(outline);
      animate(outline, [
        { opacity: 0, offset: 0 }, { opacity: 0, offset: start },
        { opacity: 1, offset: start + 0.04 }, { opacity: 1, offset: end - 0.04 },
        { opacity: 0, offset: end }, { opacity: 0, offset: 1 },
      ]);
    }

    function signal(path, color, start, end) {
      const trace = path.cloneNode(false);
      const length = path.getTotalLength();
      trace.removeAttribute('class');
      trace.removeAttribute('marker-end');
      trace.setAttribute('stroke', color);
      trace.setAttribute('stroke-width', '4');
      trace.setAttribute('stroke-linecap', 'round');
      trace.setAttribute('fill', 'none');
      trace.setAttribute('stroke-dasharray', `14 ${length}`);
      trace.setAttribute('opacity', '0');
      layer.append(trace);
      animate(trace, [
        { strokeDashoffset: '0px', opacity: 0, offset: 0 },
        { strokeDashoffset: '0px', opacity: 0, offset: start },
        { strokeDashoffset: '0px', opacity: 1, offset: start + 0.02 },
        { strokeDashoffset: `-${length}px`, opacity: 1, offset: end - 0.02 },
        { strokeDashoffset: `-${length}px`, opacity: 0, offset: end },
        { strokeDashoffset: `-${length}px`, opacity: 0, offset: 1 },
      ]);
    }

    const colors = ['#58c4dd', '#9a72d8', '#f4ce72'];
    svg.querySelectorAll('.input-vectors > g > rect').forEach((rect, i) => {
      highlight(rect, colors[i], 0.02 + i * 0.04, 0.26 + i * 0.04);
    });
    svg.querySelectorAll('.attention-connections > path').forEach((path, i) => {
      signal(path, colors[i], 0.12 + i * 0.04, 0.4 + i * 0.04);
    });
    const matrix = svg.querySelector('.attention-weights');
    matrix.querySelectorAll('g:last-child > rect').forEach((rect, i) => {
      highlight(rect, '#f4ce72', 0.44 + i * 0.025, 0.65 + i * 0.025, matrix.getAttribute('transform'));
    });
    signal(svg.querySelector('.output-connection'), '#58c4dd', 0.66, 0.8);
    highlight(svg.querySelector('.context-vector > rect'), '#58c4dd', 0.78, 0.96);
    return animations;
  }

  const coverAnimations = createCoverAnimation();
  function syncCoverAnimation(active, restart = false) {
    coverAnimations.forEach(animation => {
      animation.pause();
      if (restart) animation.currentTime = 0;
      if (active && !document.hidden) animation.play();
    });
  }

  function createGenerationAnimation() {
    const svg = document.querySelector('.generation-diagram');
    if (!svg || matchMedia('(prefers-reduced-motion: reduce)').matches) return [];
    const animations = [];
    const duration = 14000 / ANIMATION_SPEED;
    function animate(selector, frames) {
      const animation = svg.querySelector(selector).animate(frames, {
        duration, iterations: Infinity, easing: 'linear',
      });
      animation.pause();
      animation.currentTime = 0;
      animations.push(animation);
    }
    function opacity(selector, points) {
      animate(selector, points.map(([offset, value]) => ({ offset, opacity: value })));
    }
    opacity('.distribution-first', [[0,0],[.10,0],[.16,1],[.43,1],[.48,0],[.96,0],[1,0]]);
    opacity('.distribution-second', [[0,0],[.50,0],[.56,1],[.91,1],[.97,0],[1,0]]);
    opacity('.appended-milk', [[0,0],[.39,0],[.43,1],[.91,1],[.98,0],[1,0]]);
    opacity('.appended-period', [[0,0],[.77,0],[.81,1],[.91,1],[.98,0],[1,0]]);
    opacity('.chosen-highlight', [[0,0],[.21,0],[.25,1],[.30,1],[.34,0],[.62,0],[.66,1],[.70,1],[.74,0],[1,0]]);
    opacity('.model-highlight', [[0,0],[.03,1],[.10,1],[.16,0],[.45,0],[.49,1],[.53,1],[.58,0],[1,0]]);
    function token(selector, start, destinationX) {
      animate(selector, [
        {offset:0, opacity:0, transform:'translate(970px, 140px)'},
        {offset:start, opacity:0, transform:'translate(970px, 140px)'},
        {offset:start+.02, opacity:1, transform:'translate(970px, 140px)'},
        {offset:start+.06, opacity:1, transform:'translate(1200px, 400px)', easing:'ease-in-out'},
        {offset:start+.10, opacity:1, transform:'translate(1150px, 440px)', easing:'ease-in-out'},
        {offset:start+.15, opacity:1, transform:`translate(${destinationX}px, 440px)`, easing:'ease-in-out'},
        {offset:start+.18, opacity:1, transform:`translate(${destinationX}px, 276px)`},
        {offset:start+.20, opacity:0, transform:`translate(${destinationX}px, 276px)`},
        {offset:1, opacity:0, transform:`translate(${destinationX}px, 276px)`},
      ]);
    }
    token('.flying-milk', .23, 30);
    token('.flying-period', .61, 114);
    return animations;
  }

  const generationAnimations = createGenerationAnimation();
  function syncGenerationAnimation(active, restart = false) {
    generationAnimations.forEach(animation => {
      animation.pause();
      if (restart) animation.currentTime = 0;
      if (active && !document.hidden) animation.play();
    });
  }

  function createEmbeddingAnimation() {
    const svg = document.querySelector('.embedding-diagram');
    if (!svg || matchMedia('(prefers-reduced-motion: reduce)').matches) return [];
    const layer = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    layer.classList.add('embedding-animation-layer');
    layer.setAttribute('aria-hidden', 'true');
    svg.append(layer);
    const animations = [];
    function animate(element, frames) {
      const animation = element.animate(frames, { duration: 14000 / ANIMATION_SPEED, iterations: Infinity });
      animation.pause(); animation.currentTime = 0; animations.push(animation);
    }
    for (let i = 0; i < 3; i++) {
      const start = .12 + i * .19;
      const token = svg.querySelector(`.token-${i}`);
      animate(token, [
        {offset:0,opacity:0,transform:'translateY(-20px)'},
        {offset:.02+i*.02,opacity:0,transform:'translateY(-20px)'},
        {offset:.10+i*.02,opacity:1,transform:'translateY(0px)',easing:'ease-out'},
        {offset:.92,opacity:1,transform:'translateY(0px)'},
        {offset:1,opacity:0,transform:'translateY(0px)'},
      ]);
      const row = svg.querySelector(`.row-${i}`);
      animate(row, [{offset:0,opacity:0},{offset:.08,opacity:0},{offset:.15,opacity:1},{offset:.92,opacity:1},{offset:1,opacity:0}]);
      const highlight = row.querySelector('rect').cloneNode(false);
      highlight.setAttribute('class','embedding-highlight'); layer.append(highlight);
      animate(highlight, [{offset:0,opacity:0},{offset:start,opacity:0},{offset:start+.03,opacity:1},{offset:start+.13,opacity:1},{offset:start+.16,opacity:0},{offset:1,opacity:0}]);
      animate(svg.querySelector(`.output-${i}`), [{offset:0,opacity:0},{offset:start+.10,opacity:0},{offset:start+.14,opacity:1},{offset:.92,opacity:1},{offset:1,opacity:0}]);
      row.querySelectorAll('.table-value').forEach((value,j) => {
        if (j === 3) return;
        const copy = value.cloneNode(true);copy.setAttribute('x','0');copy.setAttribute('y','0');copy.setAttribute('class','embedding-transfer');layer.append(copy);
        const from = `translate(${600+j*100}px, ${100+(140+i*90-100)*1.4}px)`;
        const to = `translate(${1040+i*160}px, ${100+(155+j*50-100)*1.4}px)`;
        animate(copy,[{offset:0,opacity:0,transform:from},{offset:start+.03,opacity:0,transform:from},{offset:start+.05,opacity:1,transform:from,easing:'ease-in-out'},{offset:start+.12,opacity:1,transform:to},{offset:start+.14,opacity:0,transform:to},{offset:1,opacity:0,transform:to}]);
      });
    }
    return animations;
  }
  const embeddingAnimations = createEmbeddingAnimation();
  function syncEmbeddingAnimation(active, restart = false) {
    embeddingAnimations.forEach(animation => {
      animation.pause();
      if (restart) animation.currentTime = 0;
      if (active && !document.hidden) animation.play();
    });
  }

  function createArchitectureAnimation() {
    const svg = document.querySelector('.architecture-diagram');
    if (!svg || matchMedia('(prefers-reduced-motion: reduce)').matches) return [];
    return [['.architecture-entry', .02, .18], ['.architecture-branches', .26, .53], ['.architecture-result', .62, .88]].map(([selector, start, end]) => {
      const animation = svg.querySelector(selector).animate([
        { offset: 0, opacity: 0 }, { offset: start, opacity: 0, easing: 'ease-in-out' },
        { offset: end, opacity: 1 }, { offset: 1, opacity: 1 }
      ], { duration: 6000 / ANIMATION_SPEED, iterations: 1, fill: 'both' });
      animation.pause(); animation.currentTime = 0;
      return animation;
    });
  }
  const architectureAnimations = createArchitectureAnimation();
  function syncArchitectureAnimation(active, restart = false) {
    syncSteppedAnimation('inside-transformer', active, restart);
  }

  function createContextAnimation() {
    const svg = document.querySelector('.context-diagram');
    if (!svg || matchMedia('(prefers-reduced-motion: reduce)').matches) return [];
    const animations = [];
    function animate(element, frames) {
      const animation = element.animate(frames, { duration: 12000 / ANIMATION_SPEED, iterations: Infinity });
      animation.pause(); animation.currentTime = 0; animations.push(animation);
    }
    const layer = document.createElementNS('http://www.w3.org/2000/svg','g');
    layer.classList.add('context-animation-layer');layer.setAttribute('aria-hidden','true');svg.append(layer);
    const main = svg.querySelector('.context-main-link');
    const trace = main.cloneNode(false);const length = main.getTotalLength();
    trace.removeAttribute('class');trace.removeAttribute('marker-end');trace.setAttribute('fill','none');trace.setAttribute('stroke','#58c4dd');trace.setAttribute('stroke-width','5');trace.setAttribute('stroke-linecap','round');trace.setAttribute('stroke-dasharray',`25 ${length}`);layer.append(trace);
    animate(trace,[{offset:0,opacity:0,strokeDashoffset:'0px'},{offset:.16,opacity:0,strokeDashoffset:'0px'},{offset:.20,opacity:1,strokeDashoffset:'0px'},{offset:.46,opacity:1,strokeDashoffset:`-${length}px`},{offset:.50,opacity:0,strokeDashoffset:`-${length}px`},{offset:1,opacity:0,strokeDashoffset:`-${length}px`}]);
    const updated=svg.querySelector('.context-updated');
    animate(updated,[{offset:0,opacity:.12},{offset:.44,opacity:.12},{offset:.58,opacity:1},{offset:.90,opacity:1},{offset:1,opacity:.12}]);
    updated.querySelectorAll('.context-coordinate').forEach((coordinate,i)=>{
      const ratio=[35/65,60/40,45/60][i];
      coordinate.style.transformOrigin='16px 0px';
      animate(coordinate,[{offset:0,transform:`scaleX(${ratio})`},{offset:.45,transform:`scaleX(${ratio})`,easing:'ease-in-out'},{offset:.60,transform:'scaleX(1)'},{offset:.90,transform:'scaleX(1)'},{offset:1,transform:`scaleX(${ratio})`}]);
    });
    return animations;
  }
  const contextAnimations = createContextAnimation();
  function syncContextAnimation(active, restart = false) {
    contextAnimations.forEach(animation => {
      animation.pause();if (restart) animation.currentTime = 0;
      if (active && !document.hidden) animation.play();
    });
  }

  function createQueryAnimation() {
    const svg = document.querySelector('.query-diagram');
    if (!svg || matchMedia('(prefers-reduced-motion: reduce)').matches) return [];
    const animations = [];
    function animate(element, frames) {
      const animation = element.animate(frames, {duration: 12000 / ANIMATION_SPEED, iterations:Infinity});
      animation.pause(); animation.currentTime=0; animations.push(animation);
    }
    function reveal(element, start, end) {
      animate(element,[{offset:0,opacity:0},{offset:start,opacity:0,easing:'ease-in-out'},{offset:end,opacity:1},{offset:.93,opacity:1},{offset:1,opacity:0}]);
    }
    reveal(svg.querySelector('.query-origin-link'),.04,.10);
    reveal(svg.querySelector('.query-source'),.08,.18);
    const moving = svg.querySelector('.query-input-vector').cloneNode(true);
    moving.classList.add('query-moving-vector'); moving.setAttribute('aria-hidden','true'); svg.append(moving);
    animate(moving,[{offset:0,opacity:0,transform:'translate(215px,198px)'},{offset:.20,opacity:0,transform:'translate(215px,198px)'},{offset:.23,opacity:1,transform:'translate(215px,198px)',easing:'ease-in-out'},{offset:.39,opacity:1,transform:'translate(629px,198px)'},{offset:.43,opacity:0,transform:'translate(629px,198px)'},{offset:1,opacity:0,transform:'translate(629px,198px)'}]);
    reveal(svg.querySelector('.query-result'),.42,.53);
    animate(svg.querySelector('.query-output-vector'),[{offset:0,transform:'translate(634px,215px)'},{offset:.42,transform:'translate(634px,215px)',easing:'ease-in-out'},{offset:.56,transform:'translate(1055px,215px)'},{offset:1,transform:'translate(1055px,215px)'}]);
    reveal(svg.querySelector('.query-question'),.59,.68);
    reveal(document.querySelector('.query-takeaway'),.69,.76);
    return animations;
  }
  const queryAnimations = createQueryAnimation();
  function syncQueryAnimation(active, restart = false) {
    if(syncSteppedAnimation('token-query',active,restart)) return;
    queryAnimations.forEach(animation=>{
      animation.pause(); if(restart) animation.currentTime=0;
      if(active && !document.hidden) animation.play();
    });
  }

  function createKeyAnimation() {
    const svg=document.querySelector('.key-diagram');
    if (!svg || matchMedia('(prefers-reduced-motion: reduce)').matches) return [];
    const animations=[];
    function reveal(element,start,end) {
      const animation=element.animate([{offset:0,opacity:0},{offset:start,opacity:0,easing:'ease-in-out'},{offset:end,opacity:1},{offset:1,opacity:1}],{duration: 12000 / ANIMATION_SPEED,iterations:1,fill:'both'});
      animation.pause();animation.currentTime=0;animations.push(animation);
    }
    [0,1,2].forEach(i=>{
      reveal(svg.querySelector(`.key-projection-${i}`),.03+i*.06,.11+i*.06);
      reveal(svg.querySelector(`.key-comparison-${i}`),.30+i*.06,.38+i*.06);
      reveal(svg.querySelector(`.key-score-${i}`),.58+i*.06,.66+i*.06);
    });
    return animations;
  }
  const keyAnimations=createKeyAnimation();
  function syncKeyAnimation(active,restart=false) {
    if(syncSteppedAnimation('token-key',active,restart)) return;
    keyAnimations.forEach(animation=>{
      animation.pause(); if(restart) animation.currentTime=0;
      if(active && !document.hidden) animation.play();
    });
  }

  function createPatternAnimation() {
    const svg=document.querySelector('.pattern-diagram');
    if (!svg || matchMedia('(prefers-reduced-motion: reduce)').matches) return [];
    const animations=[];
    function reveal(element,start,end) {
      const animation=element.animate([{offset:0,opacity:0},{offset:start,opacity:0,easing:'ease-in-out'},{offset:end,opacity:1},{offset:1,opacity:1}],{duration:7000/ANIMATION_SPEED,iterations:1,fill:'both'});
      animation.pause();animation.currentTime=0;animations.push(animation);
    }
    Array.from(svg.querySelectorAll('.pattern-column-values')).reverse().forEach((element,i)=>reveal(element,.04+i*.18,.17+i*.18));
    reveal(svg.querySelector('.pattern-focus'),.04,.17);
    return animations;
  }
  const patternAnimations=createPatternAnimation();
  function syncPatternAnimation(active,restart=false) {
    if(syncSteppedAnimation('attention-pattern',active,restart)) return;
    patternAnimations.forEach(animation=>{animation.pause();if(restart)animation.currentTime=0;if(active && !document.hidden)animation.play();});
  }

  function createValueAnimation() {
    const svg=document.querySelector('.value-diagram');
    if (!svg || matchMedia('(prefers-reduced-motion: reduce)').matches) return [];
    const animations=[];
    function animate(element,frames) {
      const animation=element.animate(frames,{duration: 16000 / ANIMATION_SPEED,iterations:Infinity});
      animation.pause();animation.currentTime=0;animations.push(animation);
    }
    function reveal(element,start,end) {
      animate(element,[{offset:0,opacity:0},{offset:start,opacity:0,easing:'ease-in-out'},{offset:end,opacity:1},{offset:.94,opacity:1},{offset:1,opacity:0}]);
    }
    svg.querySelectorAll('.value-created').forEach((element,i)=>reveal(element,.08+i*.10,.17+i*.10));
    svg.querySelectorAll('.value-weighted').forEach((element,i)=>{
      reveal(element,.30+i*.07,.38+i*.07);
      const weight=[.42,.58][i];
      element.querySelectorAll('.value-coordinate').forEach(coordinate=>{
        coordinate.style.transformBox='fill-box';coordinate.style.transformOrigin='left';
        animate(coordinate,[{offset:0,transform:`scaleX(${1/weight})`},{offset:.30+i*.07,transform:`scaleX(${1/weight})`,easing:'ease-in-out'},{offset:.42+i*.07,transform:'scaleX(1)'},{offset:1,transform:'scaleX(1)'}]);
      });
    });
    reveal(svg.querySelector('.value-sum'),.51,.60);
    reveal(svg.querySelector('.value-creature-fur'),.64,.73);
    reveal(svg.querySelector('.value-creature-blue'),.73,.80);
    reveal(svg.querySelector('.value-meaning'),.80,.86);
    return animations;
  }
  const valueAnimations=createValueAnimation();
  function syncValueAnimation(active,restart=false) {
    if(syncSteppedAnimation('token-value',active,restart)) return;
    valueAnimations.forEach(animation=>{animation.pause();if(restart)animation.currentTime=0;if(active && !document.hidden)animation.play();});
  }

  function createPortugueseAttentionAnimation() {
    const svg=document.querySelector('.example-diagram');
    if (!svg || matchMedia('(prefers-reduced-motion: reduce)').matches) return [];
    const animations=[];
    function animate(element,frames) {
      const animation=element.animate(frames,{duration: 18000 / ANIMATION_SPEED,iterations:Infinity});
      animation.pause();animation.currentTime=0;animations.push(animation);
    }
    function reveal(element,start,end) {
      animate(element,[{offset:0,opacity:0},{offset:start,opacity:0,easing:'ease-in-out'},{offset:end,opacity:1},{offset:.94,opacity:1},{offset:1,opacity:0}]);
    }
    reveal(svg.querySelector('.example-query'),.03,.10);
    svg.querySelectorAll('.example-keys').forEach(element=>reveal(element,.12,.20));
    svg.querySelectorAll('.example-comparison').forEach((element,i)=>reveal(element,.22+i*.018,.27+i*.018));
    svg.querySelectorAll('.example-weights').forEach(element=>reveal(element,.38,.44));
    svg.querySelectorAll('.example-values').forEach(element=>reveal(element,.49,.58));
    svg.querySelectorAll('.example-value-coordinate').forEach(coordinate=>{
      const group=coordinate.closest('.example-values');
      const i=Number(group.classList.value.match(/example-value-(\d)/)?.[1] || 0);
      const weight=[.07193523384355,.48095138015865,.09710240897875,.13107454199682,.08786189302541,.13107454199682][i];
      coordinate.style.transformBox='view-box';coordinate.style.transformOrigin=`${coordinate.getPointAtLength(0).x}px 0px`;
      animate(coordinate,[{offset:0,transform:'scaleX(1)'},{offset:.55,transform:'scaleX(1)',easing:'ease-in-out'},{offset:.65,transform:`scaleX(${weight})`},{offset:.94,transform:`scaleX(${weight})`},{offset:1,transform:'scaleX(1)'}]);
    });
    reveal(svg.querySelector('.example-update'),.68,.77);
    svg.querySelectorAll('.example-result-coordinate').forEach((coordinate,i)=>{
      coordinate.style.transformBox='view-box';coordinate.style.transformOrigin=`${coordinate.getPointAtLength(0).x}px 0px`;
      animate(coordinate,[{offset:0,transform:`scaleX(${[.5,.8,.4][i]})`},{offset:.69,transform:`scaleX(${[.5,.8,.4][i]})`,easing:'ease-in-out'},{offset:.80,transform:'scaleX(1)'},{offset:1,transform:'scaleX(1)'}]);
    });
    return animations;
  }
  const portugueseAttentionAnimations=createPortugueseAttentionAnimation();
  function syncPortugueseAttentionAnimation(active,restart=false) {
    if(syncSteppedAnimation('attention-portuguese',active,restart)) return;
    portugueseAttentionAnimations.forEach(animation=>{animation.pause();if(restart)animation.currentTime=0;if(active && !document.hidden)animation.play();});
  }

  function createMultiheadOutputAnimation() {
    const svg=document.querySelector('.multihead-diagram');
    if (!svg || matchMedia('(prefers-reduced-motion: reduce)').matches) return [];
    return [['.multihead-heads',.03,.25],['.multihead-concat',.34,.58],['.multihead-projection',.67,.91]].map(([selector,start,end])=>{
      const animation=svg.querySelector(selector).animate([
        {offset:0,opacity:0},{offset:start,opacity:0,easing:'ease-in-out'},
        {offset:end,opacity:1},{offset:1,opacity:1}
      ],{duration:7000/ANIMATION_SPEED,iterations:1,fill:'both',easing:'linear'});
      animation.pause();animation.currentTime=0;return animation;
    });
  }
  const multiheadOutputAnimations=createMultiheadOutputAnimation();
  function syncMultiheadOutputAnimation(active,restart=false) {
    syncSteppedAnimation('multihead-output',active,restart);
  }

  function createMlpAnimation() {
    const svg=document.querySelector('.mlp-diagram');
    if (!svg || matchMedia('(prefers-reduced-motion: reduce)').matches) return [];
    const animations=[];
    [['.mlp-expansion',.05,.23],['.mlp-activation',.28,.43],['.mlp-projection',.48,.63],['.mlp-residual',.69,.86]].forEach(([selector,start,end])=>{
      svg.querySelectorAll(selector).forEach(element=>{
        const animation=element.animate([{offset:0,opacity:0},{offset:start,opacity:0,easing:'ease-in-out'},{offset:end,opacity:1},{offset:1,opacity:1}],{duration:9000/ANIMATION_SPEED,iterations:1,fill:'both',easing:'linear'});
        animation.pause();animation.currentTime=0;animations.push(animation);
      });
    });
    return animations;
  }
  const mlpAnimations=createMlpAnimation();
  function syncMlpAnimation(active,restart=false) {
    if(syncSteppedAnimation('token-mlp',active,restart)) return;
    mlpAnimations.forEach(animation=>{animation.pause();if(restart)animation.currentTime=0;if(active && !document.hidden)animation.play();});
  }

  function createActivationComparisonAnimation() {
    const svg=document.querySelector('.activation-comparison-diagram');
    if (!svg || matchMedia('(prefers-reduced-motion: reduce)').matches) return [];
    const animations=[];
    [['.activation-gelu',.04,.40],['.activation-example',.49,.80]].forEach(([selector,start,end])=>{
      const animation=svg.querySelector(selector).animate([
        {offset:0,opacity:0},{offset:start,opacity:0,easing:'ease-in-out'},
        {offset:end,opacity:1},{offset:1,opacity:1}
      ],{duration:4000/ANIMATION_SPEED,iterations:1,fill:'both'});
      animation.pause();animation.currentTime=0;animations.push(animation);
    });
    return animations;
  }
  const activationComparisonAnimations=createActivationComparisonAnimation();
  function syncActivationComparisonAnimation(active,restart=false) {
    syncSteppedAnimation('relu-gelu-comparison',active,restart);
  }

  function createPythiaArchitectureAnimation() {
    const svg=document.querySelector('.pythia-diagram');
    if (!svg || matchMedia('(prefers-reduced-motion: reduce)').matches) return [];
    return [['.pythia-overview',.02,.18],['.pythia-branches',.26,.53],['.pythia-residual',.62,.88]].map(([selector,start,end])=>{
      const animation=svg.querySelector(selector).animate([
        {offset:0,opacity:0},{offset:start,opacity:0,easing:'ease-in-out'},
        {offset:end,opacity:1},{offset:1,opacity:1}
      ],{duration:6000/ANIMATION_SPEED,iterations:1,fill:'both'});
      animation.pause();animation.currentTime=0;return animation;
    });
  }
  const pythiaArchitectureAnimations=createPythiaArchitectureAnimation();
  function syncPythiaArchitectureAnimation(active,restart=false) {
    syncSteppedAnimation('pythia-architecture',active,restart);
  }

  function createSoftmaxAnimation() {
    const svg=document.querySelector('.softmax-diagram');
    if (!svg || matchMedia('(prefers-reduced-motion: reduce)').matches) return [];
    return [['.softmax-input',.02,.18],['.softmax-exponentials',.26,.53],['.softmax-normalized',.62,.88]].map(([selector,start,end])=>{
      const animation=svg.querySelector(selector).animate([
        {offset:0,opacity:0},{offset:start,opacity:0,easing:'ease-in-out'},
        {offset:end,opacity:1},{offset:1,opacity:1}
      ],{duration:6000/ANIMATION_SPEED,iterations:1,fill:'both'});
      animation.pause();animation.currentTime=0;return animation;
    });
  }
  const softmaxAnimations=createSoftmaxAnimation();
  function syncSoftmaxAnimation(active,restart=false) {
    syncSteppedAnimation('softmax-explained',active,restart);
  }

  function createTemperatureAnimation() {
    const svg=document.querySelector('.temperature-diagram');
    if (!svg || matchMedia('(prefers-reduced-motion: reduce)').matches) return [];
    const animations=[];
    [['.temperature-calculation',.02,.18],['.temperature-base',.25,.46],['.temperature-cold',.53,.72],['.temperature-warm',.79,.94]].forEach(([selector,start,end])=>{
      const group=svg.querySelector(selector);
      const timing={duration:7200/ANIMATION_SPEED,iterations:1,fill:'both'};
      const animation=group.animate([
        {offset:0,opacity:0},{offset:start,opacity:0,easing:'ease-in-out'},
        {offset:end,opacity:1},{offset:1,opacity:1}
      ],timing);
      animation.pause();animation.currentTime=0;animations.push(animation);
      group.querySelectorAll('.temperature-bar').forEach(bar=>{
        bar.style.transformBox='fill-box';bar.style.transformOrigin='left center';
        const growth=bar.animate([
          {offset:0,transform:'scaleX(0)'},{offset:start,transform:'scaleX(0)',easing:'ease-in-out'},
          {offset:end,transform:'scaleX(1)'},{offset:1,transform:'scaleX(1)'}
        ],timing);
        growth.pause();growth.currentTime=0;animations.push(growth);
      });
    });
    return animations;
  }
  const temperatureAnimations=createTemperatureAnimation();
  function syncTemperatureAnimation(active,restart=false) {
    syncSteppedAnimation('softmax-temperature',active,restart);
  }

  function createReluAnimation() {
    const svg=document.querySelector('.relu-diagram');
    if (!svg || matchMedia('(prefers-reduced-motion: reduce)').matches) return [];
    const animations=[];
    [['.relu-negatives',.04,.23],['.relu-positives',.30,.49],['.relu-location',.56,.76]].forEach(([selector,start,end])=>{
      svg.querySelectorAll(selector).forEach(element=>{
        const animation=element.animate([{offset:0,opacity:0},{offset:start,opacity:0,easing:'ease-in-out'},{offset:end,opacity:1},{offset:1,opacity:1}],{duration:6500/ANIMATION_SPEED,iterations:1,fill:'both',easing:'linear'});
        animation.pause();animation.currentTime=0;animations.push(animation);
      });
    });
    return animations;
  }
  const reluAnimations=createReluAnimation();
  function syncReluAnimation(active,restart=false) {
    if(syncSteppedAnimation('relu-activation',active,restart)) return;
    reluAnimations.forEach(animation=>{animation.pause();if(restart)animation.currentTime=0;if(active && !document.hidden)animation.play();});
  }

  function createOutputAnimation() {
    const svg=document.querySelector('.output-diagram');
    if (!svg || matchMedia('(prefers-reduced-motion: reduce)').matches) return [];
    const animations=[];
    const duration=8500/ANIMATION_SPEED;
    [['.output-selection',.02,.15],['.output-logits',.20,.38],['.output-probabilities',.43,.63],['.output-choice',.69,.87]].forEach(([selector,start,end])=>{
      svg.querySelectorAll(selector).forEach(element=>{
        const animation=element.animate([{offset:0,opacity:0},{offset:start,opacity:0,easing:'ease-in-out'},{offset:end,opacity:1},{offset:1,opacity:1}],{duration,iterations:1,fill:'both',easing:'linear'});
        animation.pause();animation.currentTime=0;animations.push(animation);
      });
    });
    svg.querySelectorAll('.output-bar').forEach(element=>{
      const animation=element.animate([{offset:0,transform:'scaleX(0)'},{offset:.44,transform:'scaleX(0)',easing:'ease-in-out'},{offset:.63,transform:'scaleX(1)'},{offset:1,transform:'scaleX(1)'}],{duration,fill:'both',easing:'linear'});
      animation.pause();animation.currentTime=0;animations.push(animation);
    });
    const path=svg.querySelector('.output-return'),length=path.getTotalLength();
    path.style.strokeDasharray=length;
    const animation=path.animate([{offset:0,strokeDashoffset:length},{offset:.69,strokeDashoffset:length,easing:'ease-in-out'},{offset:.85,strokeDashoffset:0},{offset:1,strokeDashoffset:0}],{duration,fill:'both',easing:'linear'});
    animation.pause();animation.currentTime=0;animations.push(animation);
    return animations;
  }
  const outputAnimations=createOutputAnimation();
  function syncOutputAnimation(active,restart=false) {
    if(syncSteppedAnimation('next-token-output',active,restart)) return;
    outputAnimations.forEach(animation=>{animation.pause();if(restart)animation.currentTime=0;if(active && !document.hidden)animation.play();});
  }

  function createParamsAnimation() {
    const svg=document.querySelector('.params-diagram');
    if (!svg || matchMedia('(prefers-reduced-motion: reduce)').matches) return [];
    const animations=[];
    [['.params-embedding',.02,.16],['.params-attention',.23,.39],['.params-mlp',.46,.62],['.params-output, .params-extra',.69,.87]].forEach(([selector,start,end])=>{
      svg.querySelectorAll(selector).forEach(element=>{
        const animation=element.animate([{offset:0,opacity:0},{offset:start,opacity:0,easing:'ease-in-out'},{offset:end,opacity:1},{offset:1,opacity:1}],{duration:8500/ANIMATION_SPEED,iterations:1,fill:'both',easing:'linear'});
        animation.pause();animation.currentTime=0;animations.push(animation);
      });
    });
    return animations;
  }
  const paramsAnimations=createParamsAnimation();
  function syncParamsAnimation(active,restart=false) {
    if(syncSteppedAnimation('parameter-map',active,restart)) return;
    paramsAnimations.forEach(animation=>{animation.pause();if(restart)animation.currentTime=0;if(active && !document.hidden)animation.play();});
  }

  function createCountAnimation() {
    const svg=document.querySelector('.count-diagram');
    if (!svg || matchMedia('(prefers-reduced-motion: reduce)').matches) return [];
    const animations=[],duration=11000/ANIMATION_SPEED;
    [['.count-attention',.02,.15],['.count-mlp',.20,.33],['.count-vectors, .count-subtotal',.38,.48],['.count-blocks, .count-components',.53,.76],['.count-closing',.82,.95]].forEach(([selector,start,end])=>{
      svg.querySelectorAll(selector).forEach(element=>{
        const animation=element.animate([{offset:0,opacity:0},{offset:start,opacity:0,easing:'ease-in-out'},{offset:end,opacity:1},{offset:1,opacity:1}],{duration,iterations:1,fill:'both',easing:'linear'});
        animation.pause();animation.currentTime=0;animations.push(animation);
      });
    });
    svg.querySelectorAll('.count-layer').forEach((element,i)=>{
      const start=.54+i*.013;
      const animation=element.animate([{offset:0,opacity:0},{offset:start,opacity:0,easing:'ease-in-out'},{offset:start+.018,opacity:.6},{offset:1,opacity:.6}],{duration,iterations:1,fill:'both',easing:'linear'});
      animation.pause();animation.currentTime=0;animations.push(animation);
    });
    return animations;
  }
  const countAnimations=createCountAnimation();
  function syncCountAnimation(active,restart=false) {
    if(syncSteppedAnimation('parameter-count',active,restart)) return;
    countAnimations.forEach(animation=>{animation.pause();if(restart)animation.currentTime=0;if(active && !document.hidden)animation.play();});
  }

  function createDiskAnimation() {
    const svg=document.querySelector('.disk-diagram');
    if (!svg || matchMedia('(prefers-reduced-motion: reduce)').matches) return [];
    const animations=[];
    [['.disk-weights',.03,.24],['.disk-file, .disk-total',.33,.54],['.disk-overhead',.63,.86]].forEach(([selector,start,end])=>{
      svg.querySelectorAll(selector).forEach(element=>{
        const animation=element.animate([{offset:0,opacity:0},{offset:start,opacity:0,easing:'ease-in-out'},{offset:end,opacity:1},{offset:1,opacity:1}],{duration:7000/ANIMATION_SPEED,iterations:1,fill:'both',easing:'linear'});
        animation.pause();animation.currentTime=0;animations.push(animation);
      });
    });
    return animations;
  }
  const diskAnimations=createDiskAnimation();
  function syncDiskAnimation(active,restart=false) {
    if(syncSteppedAnimation('parameters-on-disk',active,restart)) return;
    diskAnimations.forEach(animation=>{animation.pause();if(restart)animation.currentTime=0;if(active && !document.hidden)animation.play();});
  }

  // Paused WAAPI timelines are advanced together to explicit teaching checkpoints.
  const steppedAnimations = new Map([
    ['inside-transformer', {animations:architectureAnimations, stops:[.20,.55,.90], labels:['Entrada do bloco','Atenção e MLP em paralelo','Soma com o residual']}],
    ['token-query', {animations:queryAnimations, stops:[.19,.57,.78], labels:['Representação','Projeção em Query','Consulta']}],
    ['token-key', {animations:keyAnimations, stops:[.25,.52,.80], labels:['Vetor × matriz → Key','Key · Query','Scores']}],
    ['attention-pattern', {animations:patternAnimations, stops:[.18,.36,.54,.82], labels:['Query de creature','Query de blue','Query de fluffy','Query de A']}],
    ['token-value', {animations:valueAnimations, stops:[.29,.50,.62,.87], labels:['Values','Ponderação','Soma','Informação combinada']}],
    ['attention-portuguese', {animations:portugueseAttentionAnimations, stops:[.11,.36,.45,.66,.87], labels:['Query','Keys e comparação','Pesos de atenção','Values ponderados','Representação atualizada']}],
    ['multihead-output', {animations:multiheadOutputAnimations, stops:[.27,.60,.93], labels:['Oito cabeças de 256 dimensões','Concatenação em 2.048 dimensões','Mistura aprendida por W O']}],
    ['token-mlp', {animations:mlpAnimations, stops:[.24,.44,.64,.88], labels:['Expansão','Ativação','Projeção','Soma residual']}],
    ['relu-activation', {animations:reluAnimations, stops:[.24,.50,.77], labels:['Entradas negativas','Entrada positiva','Ativação na MLP']}],
    ['relu-gelu-comparison', {animations:activationComparisonAnimations, stops:[.42,.82], labels:['Curvas e transição suave','Comparação de uma entrada negativa']}],
    ['next-token-output', {animations:outputAnimations, stops:[.16,.39,.64,.88], labels:['Última posição','Projeção em logits','Probabilidades','Próximo token']}],
    ['softmax-explained', {animations:softmaxAnimations, stops:[.20,.55,.90], labels:['Logits','Exponenciais e soma','Probabilidades']}],
    ['softmax-temperature', {animations:temperatureAnimations, stops:[.20,.48,.74,.96], labels:['Ajustar os logits','Temperatura um','Temperatura menor','Temperatura maior']}],
    ['pythia-architecture', {animations:pythiaArchitectureAnimations, stops:[.20,.55,.90], labels:['Fluxo e 16 blocos','Atenção e MLP em paralelo','Soma residual']}],
    ['parameter-map', {animations:paramsAnimations, stops:[.17,.40,.63,.88], labels:['Embedding','Atenção','MLP','Saída e demais parâmetros']}],
    ['parameter-count', {animations:countAnimations, stops:[.16,.34,.49,.78,.96], labels:['Atenção e suas projeções','Expansão e projeção da MLP','LayerNorms e total por bloco','Embedding e 16 blocos','LayerNorm final, saída e total']}],
    ['parameters-on-disk', {animations:diskAnimations, stops:[.25,.55,.87], labels:['Bytes dos parâmetros FP16','Tamanho do arquivo','Buffers e cabeçalho']}],
  ].filter(([,state])=>state.animations.length));
  steppedAnimations.forEach(state=>Object.assign(state, {
    duration:state.animations[0].effect.getTiming().duration,
    current:0, stage:0, running:false, active:false, frame:null,
  }));

  function seekStep(state,time) {
    state.current=time;
    state.animations.forEach(animation=>{animation.pause();animation.currentTime=time;});
  }
  function cancelStepFrame(state) {
    if(state.frame!==null) cancelAnimationFrame(state.frame);
    state.frame=null;
  }
  function updateStepCounter() {
    const state=steppedAnimations.get(pages[index].id);
    const indicator=document.getElementById('stepCounter');
    indicator.hidden=!state;
    if(!state) return;
    indicator.textContent=`Etapa ${formatPage(state.stage+1)} / ${formatPage(state.stops.length)}`;
    indicator.setAttribute('aria-label',`Etapa ${state.stage+1} de ${state.stops.length}: ${state.labels[state.stage]}`);
    pages[index].dataset.animationStep=String(state.stage+1);
    pages[index].dataset.animationState=state.running?'running':'paused';
  }
  function runStep(state) {
    if(state.frame!==null || !state.active || document.hidden || !state.running) return;
    let previous=null;
    function tick(now) {
      state.frame=null;
      if(!state.active || document.hidden || !state.running) return;
      const target=state.duration*state.stops[state.stage];
      const elapsed=previous===null?0:Math.min(now-previous,100);
      previous=now;
      seekStep(state,Math.min(target,state.current+elapsed));
      if(state.current>=target) {
        state.running=false;
        updateStepCounter();
      } else state.frame=requestAnimationFrame(tick);
    }
    state.frame=requestAnimationFrame(tick);
  }
  function syncSteppedAnimation(id,active,restart) {
    const state=steppedAnimations.get(id);
    if(!state) return false;
    cancelStepFrame(state);
    state.active=active;
    if(restart) {
      state.stage=0;state.running=true;seekStep(state,0);
    }
    if(active) runStep(state);
    return true;
  }
  function advancePresentation() {
    if (index === pages.length - 1) { window.location.href = "laboratorio/index.html"; return; }
    const state=steppedAnimations.get(pages[index].id);
    if(!state) {showPage(index+1);return;}
    if(state.running) {
      // A press during a transition finishes only that transition.
      cancelStepFrame(state);seekStep(state,state.duration*state.stops[state.stage]);
      state.running=false;updateStepCounter();return;
    }
    if(state.stage===state.stops.length-1) {
      if(index<pages.length-1) showPage(index+1);
      return;
    }
    state.stage++;state.running=true;updateStepCounter();runStep(state);
  }
  function retreatPresentation() {
    const state=steppedAnimations.get(pages[index].id);
    if(state && state.stage>0) {
      cancelStepFrame(state);state.stage--;state.running=false;
      seekStep(state,state.duration*state.stops[state.stage]);updateStepCounter();return;
    }
    showPage(index-1);
    const previous=steppedAnimations.get(pages[index].id);
    if(previous) {
      cancelStepFrame(previous);previous.stage=previous.stops.length-1;previous.running=false;
      seekStep(previous,previous.duration*previous.stops[previous.stage]);updateStepCounter();
    }
  }

  const pages = [...document.querySelectorAll('[data-slide]')];
  const counter = document.getElementById('pageCounter');
  const app = document.getElementById('app');
  const zoomOut = document.getElementById('zoomOut');
  const zoomIn = document.getElementById('zoomIn');
  const levels = [1, 1.25, 1.5, 1.75, 2];
  const formatPage = value => String(value).padStart(2, '0');
  let index = 0;
  let zoomIndex = 0;

  const presentationMenu = document.getElementById('presentationMenu');
  const presentationMenuToggle = document.getElementById('presentationMenuToggle');
  const presentationMenuLinks = document.getElementById('presentationMenuLinks');
  pages.forEach((page, pageIndex) => {
    const heading = page.querySelector('h1, h2').cloneNode(true);
    heading.querySelectorAll('.guide-question').forEach(element => element.remove());
    heading.querySelectorAll('br').forEach(element => element.replaceWith(' '));
    heading.querySelectorAll('.latex').forEach(element => element.remove());
    const link = document.createElement('a');
    link.href = `#slide-${pageIndex}`;
    link.dataset.pageIndex = pageIndex;
    const number = document.createElement('span');
    number.className = 'presentation-menu-number';
    number.textContent = formatPage(pageIndex + 1);
    const label = document.createElement('span');
    label.textContent = heading.textContent.replace(/\(\s*\)/g, '').replace(/\s+/g, ' ').trim();
    link.append(number, label);
    link.addEventListener('click', event => {
      event.preventDefault();
      presentationMenu.close();
      showPage(pageIndex);
    });
    presentationMenuLinks.append(link);
  });
  presentationMenuToggle.addEventListener('click', () => {
    presentationMenu.showModal();
    presentationMenuToggle.setAttribute('aria-expanded', 'true');
    presentationMenuLinks.querySelector('[aria-current="page"]')?.scrollIntoView({ block: 'nearest' });
  });
  document.getElementById('presentationMenuClose').addEventListener('click', () => presentationMenu.close());
  presentationMenu.addEventListener('close', () => {
    presentationMenuToggle.setAttribute('aria-expanded', 'false');
    presentationMenuToggle.focus();
  });
  presentationMenu.addEventListener('click', event => {
    if (event.target !== presentationMenu) return;
    const rect = presentationMenu.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) presentationMenu.close();
  });

  let panX = 0;
  let panY = 0;
  let drag = null;

  function renderZoom() {
    const scale = Math.min(innerWidth / 1600, innerHeight / 900) * levels[zoomIndex];
    const maxX = Math.max(0, (1600 * scale - innerWidth) / 2);
    const maxY = Math.max(0, (900 * scale - innerHeight) / 2);
    panX = Math.max(-maxX, Math.min(maxX, panX));
    panY = Math.max(-maxY, Math.min(maxY, panY));
    const x = (innerWidth - 1600 * scale) / 2 + panX;
    const y = (innerHeight - 900 * scale) / 2 + panY;
    app.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
    app.classList.toggle('zoomed', zoomIndex > 0);
    zoomOut.disabled = zoomIndex === 0;
    zoomIn.disabled = zoomIndex === levels.length - 1;
  }

  function playVideo(video) {
    // A troca de página pode interromper uma reprodução ainda pendente.
    video.playbackRate = ANIMATION_SPEED;
    const attempt = video.play();
    if (attempt) attempt.catch(() => {});
  }

  function showPage(position, updateHash = true) {
    index = Math.max(0, Math.min(pages.length - 1, position));
    presentationMenuLinks.querySelectorAll('a').forEach(link => {
      if (Number(link.dataset.pageIndex) === index) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
    zoomIndex = 0;
    panX = panY = 0;
    drag = null;
    app.classList.remove('dragging');
    pages.forEach((page, i) => {
      const active = i === index;
      page.hidden = !active;
      page.setAttribute('aria-hidden', String(!active));
      page.dataset.pageLabel = `${formatPage(i + 1)} / ${formatPage(pages.length)}`;
      page.querySelectorAll('video').forEach(video => {
        video.pause();
        if (active) {
          video.currentTime = 0;
          if (!document.hidden) playVideo(video);
        }
      });
    });
    syncCoverAnimation(index === 0, index === 0);
    syncGenerationAnimation(pages[index].id === 'generation-loop', pages[index].id === 'generation-loop');
    syncEmbeddingAnimation(pages[index].id === 'token-embeddings', pages[index].id === 'token-embeddings');
    syncArchitectureAnimation(pages[index].id === 'inside-transformer', pages[index].id === 'inside-transformer');
    syncContextAnimation(pages[index].id === 'context-representation', pages[index].id === 'context-representation');
    syncQueryAnimation(pages[index].id === 'token-query', pages[index].id === 'token-query');
    syncKeyAnimation(pages[index].id === 'token-key', pages[index].id === 'token-key');
    syncPatternAnimation(pages[index].id === 'attention-pattern', pages[index].id === 'attention-pattern');
    syncValueAnimation(pages[index].id === 'token-value', pages[index].id === 'token-value');
    syncPortugueseAttentionAnimation(pages[index].id === 'attention-portuguese', pages[index].id === 'attention-portuguese');
    syncMultiheadOutputAnimation(pages[index].id === 'multihead-output', pages[index].id === 'multihead-output');
    syncMlpAnimation(pages[index].id === 'token-mlp', pages[index].id === 'token-mlp');
    syncReluAnimation(pages[index].id === 'relu-activation', pages[index].id === 'relu-activation');
    syncActivationComparisonAnimation(pages[index].id === 'relu-gelu-comparison', pages[index].id === 'relu-gelu-comparison');
    syncOutputAnimation(pages[index].id === 'next-token-output', pages[index].id === 'next-token-output');
    syncSoftmaxAnimation(pages[index].id === 'softmax-explained', pages[index].id === 'softmax-explained');
    syncTemperatureAnimation(pages[index].id === 'softmax-temperature', pages[index].id === 'softmax-temperature');
    syncPythiaArchitectureAnimation(pages[index].id === 'pythia-architecture', pages[index].id === 'pythia-architecture');
    syncParamsAnimation(pages[index].id === 'parameter-map', pages[index].id === 'parameter-map');
    syncCountAnimation(pages[index].id === 'parameter-count', pages[index].id === 'parameter-count');
    syncDiskAnimation(pages[index].id === 'parameters-on-disk', pages[index].id === 'parameters-on-disk');
    counter.textContent = `${formatPage(index + 1)} / ${formatPage(pages.length)}`;
    counter.setAttribute('aria-label', `Página ${index + 1} de ${pages.length}`);
    updateStepCounter();
    if (updateHash) history.replaceState(null, '', `#slide-${index}`);
    renderZoom();
    window.scrollTo(0, 0);
  }

  function changeZoom(delta) {
    zoomIndex = Math.max(0, Math.min(levels.length - 1, zoomIndex + delta));
    renderZoom();
  }

  document.getElementById('goHome').addEventListener('click', () => showPage(0));
  zoomOut.addEventListener('click', () => changeZoom(-1));
  zoomIn.addEventListener('click', () => changeZoom(1));
  app.addEventListener('pointerdown', event => {
    if (zoomIndex === 0 || event.button !== 0 || event.target.closest('button, a')) return;
    drag = { id: event.pointerId, x: event.clientX, y: event.clientY, panX, panY };
    app.setPointerCapture(event.pointerId);
    app.classList.add('dragging');
    event.preventDefault();
  });
  app.addEventListener('pointermove', event => {
    if (!drag || event.pointerId !== drag.id) return;
    panX = drag.panX + event.clientX - drag.x;
    panY = drag.panY + event.clientY - drag.y;
    renderZoom();
  });
  function stopDrag() { drag = null; app.classList.remove('dragging'); }
  app.addEventListener('pointerup', stopDrag);
  app.addEventListener('pointercancel', stopDrag);
  app.addEventListener('lostpointercapture', stopDrag);
  window.addEventListener('resize', renderZoom);
  window.addEventListener('load', renderZoom);
  document.addEventListener('visibilitychange', () => {
    syncCoverAnimation(index === 0);
    syncGenerationAnimation(pages[index].id === 'generation-loop');
    syncEmbeddingAnimation(pages[index].id === 'token-embeddings');
    syncArchitectureAnimation(pages[index].id === 'inside-transformer');
    syncContextAnimation(pages[index].id === 'context-representation');
    syncQueryAnimation(pages[index].id === 'token-query');
    syncKeyAnimation(pages[index].id === 'token-key');
    syncPatternAnimation(pages[index].id === 'attention-pattern');
    syncValueAnimation(pages[index].id === 'token-value');
    syncPortugueseAttentionAnimation(pages[index].id === 'attention-portuguese');
    syncMultiheadOutputAnimation(pages[index].id === 'multihead-output');
    syncMlpAnimation(pages[index].id === 'token-mlp');
    syncReluAnimation(pages[index].id === 'relu-activation');
    syncActivationComparisonAnimation(pages[index].id === 'relu-gelu-comparison');
    syncOutputAnimation(pages[index].id === 'next-token-output');
    syncSoftmaxAnimation(pages[index].id === 'softmax-explained');
    syncTemperatureAnimation(pages[index].id === 'softmax-temperature');
    syncPythiaArchitectureAnimation(pages[index].id === 'pythia-architecture');
    syncParamsAnimation(pages[index].id === 'parameter-map');
    syncCountAnimation(pages[index].id === 'parameter-count');
    syncDiskAnimation(pages[index].id === 'parameters-on-disk');
    pages[index].querySelectorAll('video').forEach(video => {
      if (document.hidden) video.pause();
      else playVideo(video);
    });
  });
  document.addEventListener('keydown', event => {
    if (presentationMenu.open) return;
    const target = event.target instanceof Element ? event.target : null;
    if (event.ctrlKey || event.metaKey || event.altKey || target?.closest('input, textarea, select, [contenteditable]')) return;
    if (event.key === ' ' && target?.closest('button, a')) return;
    if(event.repeat && ['ArrowRight','ArrowLeft','PageDown','PageUp',' '].includes(event.key)) return;
    if(event.shiftKey && ['ArrowRight','ArrowLeft'].includes(event.key)) {
      event.preventDefault();showPage(index+(event.key==='ArrowRight'?1:-1));return;
    }
    const actions = {
      ArrowRight: advancePresentation, PageDown: advancePresentation, ' ': advancePresentation,
      ArrowLeft: retreatPresentation, PageUp: retreatPresentation,
      Home: () => showPage(0), End: () => showPage(pages.length - 1),
      '+': () => changeZoom(1), '=': () => changeZoom(1), '-': () => changeZoom(-1),
    };
    if (actions[event.key]) { event.preventDefault(); actions[event.key](); }
  });
  function readHash() {
    const match = location.hash.match(/^#slide-(\d+)$/);
    showPage(match ? Number(match[1]) : 0, false);
  }
  window.addEventListener('hashchange', readHash);
  readHash();
})();
