// Testes das contas do laboratório de geração de texto:  node --test tests/
const test = require("node:test");
const assert = require("node:assert/strict");
const G = require("../laboratorios/geracao/geracao.js");

const perto = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} não está a ${tol} de ${b}`);
const soma = a => a.reduce((x, y) => x + y, 0);

test("KV cache do Qwen2.5-0.5B: 12 KB por token em FP16", () => {
  // 2 (K e V) × 24 camadas × 2 cabeças KV × 64 × 2 bytes
  assert.equal(G.kvPorToken(G.modelo("qwen05")), 12288);
});

test("KV cache do Llama-3.1-8B: 128 KB por token em FP16", () => {
  assert.equal(G.kvPorToken(G.modelo("llama8b")), 131072);
});

test("ler é mais rápido que escrever, para todo modelo, formato e máquina", () => {
  for (const m of G.MODELOS) for (const f of G.FORMATOS) for (const q of G.MAQUINAS) {
    const t = G.tempos({ modelo: m.id, formato: f.id, maquina: q.id, prompt: 500, resposta: 500 });
    assert.ok(t.decode.segundos > t.prefill.segundos, `${m.id} ${f.id} ${q.id}`);
    assert.equal(t.decode.limite, "memória", `decode de 1 token deveria ser limitado pela memória: ${m.id} ${q.id}`);
  }
});

test("Qwen2.5-0.5B Q4_K_M: arquivo perto dos 485 MB reais da oficina", () => {
  const b = G.bytesPesos(G.modelo("qwen05"), G.formato("Q4_K_M"));
  assert.ok(Math.abs(b - 485452288) / 485452288 < 0.06, `${b / 1e6} MB`);
  // por passada lê menos que o arquivo: a embedding de entrada não é lida inteira
  assert.ok(G.bytesLidos(G.modelo("qwen05"), G.formato("Q4_K_M")) < b);
});

test("notebook calibrado: Qwen2.5-0.5B Q4_K_M perto dos 60,6 tok/s medidos na oficina", () => {
  const t = G.tempos({ modelo: "qwen05", formato: "Q4_K_M", maquina: "notebook", prompt: 32, resposta: 64 });
  const real = 60.6;
  assert.ok(Math.abs(t.decode.tokPorSeg - real) / real < 0.25, `estimado ${t.decode.tokPorSeg} tok/s`);
});

test("prefill de 500 tokens é limitado por cálculo e faz centenas de contas por byte", () => {
  const t = G.tempos({ modelo: "qwen05", formato: "Q4_K_M", maquina: "notebook", prompt: 500, resposta: 1 });
  assert.equal(t.prefill.limite, "cálculo");
  assert.ok(t.prefill.intensidade > 500, `intensidade ${t.prefill.intensidade}`);
  assert.ok(t.decode.intensidade < 5, `intensidade do decode ${t.decode.intensidade}`);
});

test("quantizar acelera o decode quase na proporção dos bits", () => {
  const base = { modelo: "llama8b", maquina: "rtx4090", prompt: 100, resposta: 100 };
  const f16 = G.tempos({ ...base, formato: "F16" }).decode.segundos;
  const q4 = G.tempos({ ...base, formato: "Q4_K_M" }).decode.segundos;
  assert.ok(f16 / q4 > 2.5 && f16 / q4 < 16 / 4.89 + 0.01, `razão ${f16 / q4}`);
});

test("sem KV cache o trabalho cresce com o quadrado da resposta", () => {
  assert.equal(G.passagens(500, 500, true), 1000);
  const sem = G.passagens(500, 500, false);
  perto(sem, 500 + 499 * 500 + (499 * 500) / 2, 0, "soma aritmética");
  assert.ok(sem > 300 * G.passagens(500, 500, true));
});

test("softmax soma 1 e temperatura 0 vira greedy", () => {
  const l = G.EXEMPLOS[0].logits.map(x => x[1]);
  perto(soma(G.softmax(l, 1)), 1, 1e-12, "soma");
  assert.deepEqual(G.softmax(l, 0), [1, 0, 0, 0, 0, 0, 0, 0]);
  // mesmos números do slide e do gera_texto.py --sampling
  perto(G.softmax(l, 1)[0], 0.501, 0.001, "telhado T=1");
  perto(G.softmax(l, 0.5)[0], 0.788, 0.001, "telhado T=0,5");
  perto(G.softmax(l, 1.5)[0], 0.370, 0.001, "telhado T=1,5");
});

test("top-k, top-p e min-p cortam e renormalizam", () => {
  const l = G.EXEMPLOS[0].logits.map(x => x[1]);
  const k3 = G.distribuicao(l, { temperatura: 1, topK: 3 });
  assert.equal(k3.filter(v => v > 0).length, 3);
  perto(soma(k3), 1, 1e-12, "top-k soma");
  const p9 = G.distribuicao(l, { temperatura: 1, topP: 0.9 });
  const n = p9.filter(v => v > 0).length;
  assert.ok(n >= 3 && n < 8, `top-p 0,9 ficou com ${n}`);
  perto(soma(p9), 1, 1e-12, "top-p soma");
  const mp = G.distribuicao(l, { temperatura: 1, minP: 0.2 });
  assert.ok(mp.every((v, i) => v === 0 || G.softmax(l, 1)[i] >= 0.2 * G.softmax(l, 1)[0]));
});

test("sorteio com seed é reproduzível e segue as probabilidades", () => {
  const probs = G.softmax(G.EXEMPLOS[0].logits.map(x => x[1]), 1);
  const a = G.rng(42), b = G.rng(42);
  const sa = Array.from({ length: 50 }, () => G.sortear(probs, a));
  const sb = Array.from({ length: 50 }, () => G.sortear(probs, b));
  assert.deepEqual(sa, sb);
  const r = G.rng(7), cont = new Array(8).fill(0), N = 20000;
  for (let i = 0; i < N; i++) cont[G.sortear(probs, r)]++;
  perto(cont[0] / N, probs[0], 0.02, "frequência de telhado");
});
