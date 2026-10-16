// Testes das contas do laboratório de compressão:  node --test tests/
const test = require("node:test");
const assert = require("node:assert/strict");
const C = require("../laboratorios/compressao/calculo.js");

const MB = 1e6;
const perto = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} não está a ${tol} de ${b}`);

test("parâmetros: embedding + blocos somam o total publicado", () => {
  for (const m of C.MODELOS) {
    const mem = C.memoria({ modelo: m.id, formato: "F16", ctx: 0 });
    perto(mem.params, m.total, 1, m.id);
    assert.ok(mem.emb < m.total, `${m.id}: embedding maior que o modelo`);
  }
});

test("FP16 = 2 bytes por parâmetro", () => {
  const mem = C.memoria({ modelo: "qwen05", formato: "F16", ctx: 0 });
  perto(mem.pesos, 494.0e6 * 2, 1, "pesos FP16");
});

test("Qwen2.5-0.5B em Q4_K_M fica perto do arquivo real da oficina (485 MB)", () => {
  // o servidor reportou size = 485452288 bytes para Qwen2.5-0.5B-Instruct Q4_K_M
  const mem = C.memoria({ modelo: "qwen05", formato: "Q4_K_M", ctx: 0 });
  const real = 485452288;
  assert.ok(Math.abs(mem.pesos - real) / real < 0.06, `conta ${mem.pesos / MB} MB vs real ${real / MB} MB`);
  // e a conta ingênua (sem embedding separada) erra feio: é o ponto da aula
  const ingenua = C.memoria({ modelo: "qwen05", formato: "Q4_K_M", ctx: 0, embSeparada: false });
  assert.ok(ingenua.pesos < 0.65 * real);
});

test("KV cache por token bate com a fórmula 2 × camadas × cabeças KV × dim × bytes", () => {
  assert.equal(C.kvPorToken(C.modelo("qwen05"), 16), 2 * 24 * 2 * 64 * 2); // 12 288 B = 12 KB
  assert.equal(C.kvPorToken(C.modelo("llama1b"), 16), 2 * 16 * 8 * 64 * 2); // 32 KB
  const mem = C.memoria({ modelo: "qwen05", formato: "F16", ctx: 2048, pedidos: 1 });
  assert.equal(mem.kv, 12288 * 2048);
});

test("KV escala com contexto, pedidos e janela; quantizar o cache reduz proporcionalmente", () => {
  const base = { modelo: "llama1b", formato: "F16", ctx: 4096, pedidos: 1 };
  const a = C.memoria(base).kv;
  assert.equal(C.memoria({ ...base, pedidos: 4 }).kv, 4 * a);
  assert.equal(C.memoria({ ...base, ctx: 8192 }).kv, 2 * a);
  assert.equal(C.memoria({ ...base, janela: 1024 }).kv, a / 4);
  perto(C.memoria({ ...base, kv: "q8" }).kv / a, 8.5 / 16, 1e-9, "q8");
});

test("pruning estruturado reduz só os blocos, não a embedding", () => {
  const a = C.memoria({ modelo: "qwen15", formato: "F16", ctx: 0 });
  const b = C.memoria({ modelo: "qwen15", formato: "F16", ctx: 0, poda: 0.3 });
  perto(a.pesos - b.pesos, a.blocos * 0.3 * 2, 1, "economia");
  assert.equal(a.emb, b.emb);
});

test("receita: cascata começa em FP16, cada passo economiza e o fim bate com memoria()", () => {
  const base = { modelo: "qwen15", ctx: 2048, pedidos: 1, buffersMB: 60 };
  const r = C.receita(base, { formato: "Q4_K_M", kv: "q8", poda: 0.2, aluno: "qwen05" });
  assert.deepEqual(r.cascata.map(p => p.id), ["base", "quant", "kv", "poda", "dest"]);
  for (let i = 1; i < r.cascata.length; i++) {
    assert.ok(r.cascata[i].economia > 0, r.cascata[i].id + " não economizou");
    perto(r.cascata[i - 1].total - r.cascata[i].economia, r.cascata[i].total, 1e-6, "soma da cascata");
  }
  perto(r.fim.total, r.cascata.at(-1).total, 1e-6, "fim");
  const direto = C.memoria({ modelo: "qwen05", formato: "Q4_K_M", kv: "q8", poda: 0.2, ctx: 2048, pedidos: 1, buffersMB: 60 });
  perto(r.fim.total, direto.total, 1e-6, "receita = memória direta");
});

test("receita vazia não tem passos; aluno igual ao modelo base é ignorado", () => {
  const r = C.receita({ modelo: "qwen05", ctx: 2048, pedidos: 1 }, { formato: "F16", kv: "f16", poda: 0, aluno: "qwen05" });
  assert.equal(r.cascata.length, 1);
});

test("a resposta da semana: 0,5B em Q4 cabe em 500 MB só sem buffers grandes; 1,5B não cabe nem em Q2", () => {
  const q05 = C.memoria({ modelo: "qwen05", formato: "Q4_K_M", ctx: 2048, pedidos: 1, buffersMB: 60 });
  assert.ok(q05.total > 500 * MB, "o modelo da oficina, com cache e buffers, passa de 500 MB");
  const q15 = C.memoria({ modelo: "qwen15", formato: "Q2_K", ctx: 2048, pedidos: 1, buffersMB: 60 });
  assert.ok(q15.total > 500 * MB);
  const smol = C.memoria({ modelo: "smol360", formato: "Q4_K_M", ctx: 2048, pedidos: 1, buffersMB: 60 });
  assert.ok(smol.total < 500 * MB);
});

test("teto de decode: banda dividida pelos bytes lidos por passo", () => {
  const mem = C.memoria({ modelo: "qwen05", formato: "Q4_K_M", ctx: 2048, pedidos: 1 });
  const t = C.tetoTokS(mem, 51.2, 1);
  perto(t.porPedido, 51.2e9 / (mem.pesos + mem.kv), 1e-9, "teto");
  // o notebook-bia mediu 60,6 tok/s; o teto teórico tem de ficar acima do medido
  assert.ok(t.porPedido > 60.6);
});

test("custo: quantização e KV sem treino; pruning e destilação com treino", () => {
  const c = C.custo({ formato: "Q4_K_M", kv: "q8", poda: 0.3, aluno: "qwen05" });
  assert.deepEqual(c.map(x => [x.familia, x.treino]), [["quant", false], ["kv", false], ["poda", true], ["dest", true]]);
  assert.equal(C.custo({ formato: "Q2_K" })[0].risco, "alto");
});

test("quantizar: 2^bits níveis, erro cai com mais bits, extremos preservados", () => {
  const rnd = C.gerador(3), xs = Array.from({ length: 64 }, () => C.gauss(rnd));
  let anterior = Infinity;
  for (const b of [2, 3, 4, 6, 8]) {
    const r = C.quantizar(xs, b);
    assert.equal(r.niveis, 2 ** b);
    assert.ok(r.q.every(k => Number.isInteger(k) && k >= 0 && k < 2 ** b));
    assert.ok(r.erroRMS < anterior, `erro com ${b} bits não caiu`);
    assert.ok(r.erroRMS <= r.escala / 2 + 1e-12, "erro maior que meio degrau");
    anterior = r.erroRMS;
  }
  const r = C.quantizar(xs, 4);
  perto(Math.min(...r.deq), Math.min(...xs), 1e-12, "mínimo");
  perto(Math.max(...r.deq), Math.max(...xs), 1e-9, "máximo");
});

test("outlier estica a régua e aumenta o erro dos outros", () => {
  const rnd = C.gerador(5), xs = Array.from({ length: 32 }, () => C.gauss(rnd) * 0.5);
  const sem = C.quantizar(xs, 4), com = C.quantizar([...xs, 8], 4);
  assert.ok(com.escala > 2 * sem.escala);
  const erroOutros = Math.sqrt(xs.reduce((s, x, i) => s + (x - com.deq[i]) ** 2, 0) / xs.length);
  assert.ok(erroOutros > 2 * sem.erroRMS);
});

test("softmax soma 1, temperatura achata, KL é zero só para distribuições iguais", () => {
  const p = C.softmax([5, 3, 1], 1), q = C.softmax([5, 3, 1], 4);
  perto(p.reduce((a, b) => a + b), 1, 1e-12, "soma");
  assert.ok(q[0] < p[0] && q[2] > p[2]);
  perto(C.kl(p, p), 0, 1e-12, "kl igual");
  assert.ok(C.kl(p, q) > 0);
});
