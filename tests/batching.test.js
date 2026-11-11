// Testes das contas do laboratório de KV cache e batching:  node --test tests/
const test = require("node:test");
const assert = require("node:assert/strict");
const B = require("../laboratorios/batching/batching.js");

const GPU = { modelo: "llama8b", formato: "F16", maquina: "rtx4090", contexto: 1000 };

test("lote de 20 na GPU: muito mais barato que 20 × um, e cada pedido só um pouco mais lento", () => {
  const v = B.vinteVezesUm(GPU, 20, 100);
  assert.ok(v.economia > 10 && v.economia < 20, `economia ${v.economia}`);
  const p1 = B.passo({ ...GPU, lote: 1 }), p20 = B.passo({ ...GPU, lote: 20 });
  assert.ok(p20.tokPorSegPorPedido / p1.tokPorSegPorPedido > 0.8);
  assert.equal(p20.limite, "memória");
});

test("o custo por token nunca sobe com o lote, e o lote nunca passa de 20 × mais barato que 20 × um", () => {
  for (const maquina of ["rpi5", "notebook", "jetson", "rtx4090"]) {
    const cv = B.curva({ ...GPU, maquina }, [1, 2, 4, 8, 16, 20, 32, 64, 128, 256]);
    for (let i = 1; i < cv.length; i++) assert.ok(cv[i].custoPorToken <= cv[i - 1].custoPorToken + 1e-12, maquina);
    assert.ok(B.vinteVezesUm({ ...GPU, maquina }, 20, 10).economia <= 20 + 1e-9);
  }
});

test("sem KV, o ponto de virada é contas/bytes: ~170 pedidos para o Llama-8B FP16 na 4090", () => {
  const b = B.loteDeVirada({ ...GPU, contexto: 0 });
  assert.ok(b > 150 && b < 190, `virada em ${b}`);
  // na CPU do notebook, a virada chega muito antes
  assert.ok(B.loteDeVirada({ modelo: "qwen05", formato: "Q4_K_M", maquina: "notebook", contexto: 0 }) < 10);
});

test("memória: Llama-8B FP16 em 24 GB cabe 10 pedidos de 4k reservados e 41 em blocos", () => {
  const c = { modelo: "llama8b", formato: "F16", memoria: "rtx4090", contexto: 4096, medio: 1000 };
  const a = B.cabem({ ...c, modo: "contigua" }), p = B.cabem({ ...c, modo: "paginada" });
  assert.equal(a.porToken, 131072);
  assert.equal(a.porPedido, 512 * 2 ** 20);
  assert.equal(a.pedidos, 10);
  assert.equal(p.pedidos, 41);
  assert.ok(p.aproveitamento > 0.98 && a.aproveitamento < 0.25);
  assert.equal(B.cabem({ ...c, memoria: "jetson", modo: "paginada" }).cabeModelo, false);
});

test("paginador: blocos não vizinhos, só o último com sobra, liberar devolve tudo", () => {
  const P = B.novoPaginador(16, 4);
  assert.ok(B.acrescentar(P, "A", 9));                       // 3 blocos
  assert.ok(B.acrescentar(P, "B", 4));                       // 1 bloco
  assert.equal(P.pedidos.A.tabela.length, 3);
  assert.equal(new Set(P.pedidos.A.tabela.concat(P.pedidos.B.tabela)).size, 4);
  assert.ok(B.acrescentar(P, "A", 3));                       // 12 tokens: ainda 3 blocos
  assert.equal(P.pedidos.A.tabela.length, 3);
  assert.ok(B.acrescentar(P, "A", 1));                       // 13: 4º bloco
  assert.equal(P.pedidos.A.tabela.length, 4);
  const r = B.resumoPaginador(P);
  assert.equal(r.blocosUsados + r.blocosLivres, 16);
  assert.equal(r.tokens, 17);
  B.liberar(P, "A");
  assert.equal(B.resumoPaginador(P).blocosLivres, 15);
  assert.equal(B.acrescentar(P, "C", 61), false);            // 16 blocos: não cabe
});

test("agenda: contínuo nunca leva mais passos que estático e faz o mesmo trabalho útil", () => {
  for (let seed = 1; seed < 30; seed++) for (let vagas = 2; vagas <= 6; vagas++) {
    const ped = B.pedidosExemplo(10, seed, 2, 14);
    const e = B.agenda(ped, vagas, "estatico"), c = B.agenda(ped, vagas, "continuo");
    assert.ok(c.passos <= e.passos, `seed ${seed} vagas ${vagas}`);
    const uteis = g => g.grade.flat().filter(x => x && x !== "ocioso").length;
    assert.equal(uteis(e), uteis(c));
    assert.equal(uteis(c), ped.reduce((a, p) => a + p.tokens, 0));
    assert.ok(c.terminoMedio <= e.terminoMedio + 1e-9);
  }
});

test("exemplo dos slides: 8 pedidos, 4 vagas, 19 passos estático e 15 contínuo", () => {
  const ped = B.pedidosExemplo(8, 7, 2, 12);
  assert.deepEqual(ped.map(p => p.tokens), [2, 2, 12, 9, 7, 6, 7, 4]);
  assert.equal(B.agenda(ped, 4, "estatico").passos, 19);
  assert.equal(B.agenda(ped, 4, "continuo").passos, 15);
});

test("chunked prefill: o maior passo cai muito e o prefill do prompt longo quase não piora", () => {
  const c = { modelo: "llama8b", formato: "F16", maquina: "rtx4090", ativos: 20, contexto: 1000, prompt: 8000, orcamento: 512 };
  const sem = B.chunked({ ...c, chunked: false }), com = B.chunked({ ...c, chunked: true });
  assert.ok(sem.maiorPasso > 1 && sem.maiorPasso < 2.5, `sem: ${sem.maiorPasso}`);
  assert.ok(com.maiorPasso < 0.15, `com: ${com.maiorPasso}`);
  assert.ok(com.ttft >= sem.ttft && com.ttft / sem.ttft < 1.15);
  const somaPref = r => r.passos.reduce((a, p) => a + p.prefill, 0);
  assert.equal(somaPref(sem), 8000); assert.equal(somaPref(com), 8000);
});

test("prefix caching: 20 pedidos × (2.000 + 100) tokens viram 4.000 de prefill", () => {
  const r = B.prefixo({ modelo: "llama8b", prefixo: 2000, proprio: 100, pedidos: 20 });
  assert.equal(r.prefillSem, 42000);
  assert.equal(r.prefillCom, 4000);
  assert.equal(r.compartilhados, 2000);
  // só blocos cheios: 2.010 tokens de prefixo compartilham 2.000
  assert.equal(B.prefixo({ modelo: "llama8b", prefixo: 2010, proprio: 100, pedidos: 2 }).compartilhados, 2000);
  assert.equal(B.prefixo({ modelo: "llama8b", prefixo: 2000, proprio: 100, pedidos: 1 }).economiaPrefill, 1);
});
