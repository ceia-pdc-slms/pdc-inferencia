// Testes das contas do laboratório de quantização a fundo:  node --test tests/
const test = require("node:test");
const assert = require("node:assert/strict");
const Q = require("../laboratorios/quantizacao/quantizacao.js");

test("os máximos de cada formato batem com as especificações", () => {
  assert.equal(Q.maximo(Q.float("FP16")), 65504);
  assert.equal(Q.maximo(Q.float("E4M3")), 448);
  assert.equal(Q.maximo(Q.float("E5M2")), 57344);
  assert.equal(Q.maximo(Q.float("E2M1")), 6);
  assert.ok(Math.abs(Q.maximo(Q.float("BF16")) / 3.3895e38 - 1) < 1e-3);
  assert.equal(Q.bits(Q.float("E4M3")), 8);
  assert.equal(Q.bits(Q.float("E2M1")), 4);
});

test("FP4 E2M1 tem exatamente os valores 0, 0,5, 1, 1,5, 2, 3, 4 e 6", () => {
  assert.deepEqual(Q.grade(Q.float("E2M1")), [0, 0.5, 1, 1.5, 2, 3, 4, 6]);
  assert.deepEqual(Q.grade(Q.float("E2M1")), Q.FP4);
  assert.equal(Q.grade(Q.float("E4M3")).length, 127); // 126 positivos (sem o NaN) + o zero
});

test("arredondar: vizinho mais perto, saturação e infinito", () => {
  const f16 = Q.float("FP16"), e4 = Q.float("E4M3"), bf = Q.float("BF16");
  assert.equal(Q.arredonda(0.1, f16), 0.0999755859375);
  assert.equal(Q.arredonda(70000, f16), Infinity);   // FP16 estoura
  assert.equal(Q.arredonda(70000, bf), 70144);        // BF16 tem a faixa do FP32
  assert.equal(Q.arredonda(1000, e4), 448);           // E4M3 satura
  assert.equal(Q.arredonda(2.6, Q.float("E2M1")), 3);
  assert.equal(Q.arredonda(-0.3, Q.float("E2M1")), -0.5);
  for (const f of Q.FLOATS.filter(f => Q.bits(f) <= 8)) for (const v of Q.grade(f)) assert.equal(Q.arredonda(v, f), v, f.id);
});

test("campos de bits: 1,0 em FP16 é 0 01111 0000000000", () => {
  const c = Q.campos(1, Q.float("FP16"));
  assert.equal(c.sinal + c.expoente + c.mantissa, "0011110000000000");
  const s = Q.campos(0.5, Q.float("E2M1"));  // 0,5 é subnormal no E2M1
  assert.equal(s.expoente + s.mantissa, "001");
  assert.ok(s.subnormal);
  assert.equal(Q.campos(6, Q.float("E2M1")).expoente + Q.campos(6, Q.float("E2M1")).mantissa, "111");
});

test("o outlier apaga os pesos pequenos por tensor, e o grupo resolve", () => {
  const v = [0.12, -0.31, 0.05, 0.27, -0.08, 0.19, -0.22, 8.0];
  const t = Q.quantiza(v, { tipo: "int", bits: 4, grupo: 0 });
  assert.equal(t.zerados, 7);
  const g = Q.quantiza(v, { tipo: "int", bits: 4, grupo: 4 });
  assert.equal(g.zerados, 3);
  assert.ok(g.erroRel < t.erroRel);
  assert.deepEqual(g.q.slice(0, 4), [3, -7, 1, 6]);
  assert.equal(g.bitsPorPeso, 4 + 16 / 4);
  // FP4 por tensor também apaga: 8 / 6 = 1,33 de escala, e todo |peso| / 1,33 fica abaixo de 0,25
  assert.equal(Q.quantiza(v, { tipo: "fp4", grupo: 0 }).zerados, 7);
  assert.ok(Q.quantiza(v, { tipo: "fp4", grupo: 4 }).erroRel < t.erroRel);
});

test("a conta do token: menos bytes só viram tok/s na parte dos pesos", () => {
  const base = { camadas: "FP32", saida: "FP32", gbs: 25, fixoMs: 0, ctx: 0 };
  const b = Q.token(base);
  assert.ok(Math.abs(b.bytesPesos / 1e6 - 1976) < 2);
  assert.ok(Math.abs(b.msPesos - 79) < 1);
  const ideal = Q.token({ ...base, camadas: "Q4_0", saida: "Q4_0" });
  assert.ok(Math.abs(Q.compara(b, ideal).subida - 32 / 4.5) < 1e-6, "sem resto, tok/s sobe junto com os bytes");
  const real = { fixoMs: Q.MEDIDAS.restoFixoMs, ctx: 64, ctxUs: Q.MEDIDAS.restoCtxUs };
  const c = Q.compara(Q.token({ ...base, ...real }), Q.token({ ...base, ...real, camadas: "Q4_0", saida: "Q8_0" }));
  assert.ok(c.queda > 5 && c.subida < 3, `bytes ${c.queda}, tok/s ${c.subida}`);
  assert.ok(c.aproveitado < 0.5);
});

test("as medidas guardadas contam a mesma história", () => {
  const d = Q.MEDIDAS.decode, fp32 = d[0];
  const int4 = d.find(r => r.config === "INT4 tudo"), semInt = d.find(r => r.config === "INT4 tudo, sem conta int");
  assert.ok(fp32.ms1 / int4.ms1 > 5 && fp32.ms1 / semInt.ms1 < 3);
  for (const r of d) assert.ok(fp32.ms64 / r.ms64 < 1.2, "no prefill, quantizar os pesos não acelera");
  for (const r of Q.MEDIDAS.pontaAPonta) assert.ok(r.linear < r.ms);
  const g = Q.MEDIDAS.gguf;
  assert.ok((g[0].camadas + g[0].saida) / (g[3].camadas + g[3].saida) < 3, "F16 -> Q4_0 lê só ~2,9x menos por token");
});
