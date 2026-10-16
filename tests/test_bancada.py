"""
Testes da bancada: estatística do bench.py, medição contra um servidor falso com tempos
conhecidos, manifestos de pipeline, montagem de comando e a API HTTP de ponta a ponta.

  python -m unittest discover -s tests -v
"""
import importlib.util, json, shutil, subprocess, sys, tempfile, time, unittest, urllib.request, urllib.error
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "testbench"))
sys.path.insert(0, str(RAIZ / "tests"))

import servidor  # noqa: E402
import servidor_falso  # noqa: E402


def carregar_bench():
    spec = importlib.util.spec_from_file_location("bench", RAIZ / "docker" / "bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def http(metodo, url, corpo=None):
    dados = json.dumps(corpo).encode() if corpo is not None else None
    req = urllib.request.Request(url, data=dados, method=metodo, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


class Estatistica(unittest.TestCase):
    """As contas do bench.py, contra valores calculados à mão."""

    @classmethod
    def setUpClass(cls):
        cls.bench = carregar_bench()

    def test_percentil_interpola(self):
        pct = self.bench.pct
        self.assertEqual(pct([1, 2, 3, 4, 5], 50), 3)
        self.assertAlmostEqual(pct([1, 2, 3, 4], 50), 2.5)
        self.assertAlmostEqual(pct([10, 20], 95), 19.5)
        self.assertIsNone(pct([], 50))

    def test_resumo(self):
        s = self.bench.summarize([{"x": 2.0}, {"x": 4.0}, {"x": 6.0}, {"x": None}], "x")
        self.assertEqual(s["mean"], 4.0)
        self.assertAlmostEqual(s["stdev"], 2.0)
        self.assertEqual((s["min"], s["max"], s["p50"]), (2.0, 6.0, 4.0))


class MedicaoContraServidorFalso(unittest.TestCase):
    """O bench.py precisa medir o que o servidor falso entrega: tempos conhecidos de antemão."""

    def setUp(self):
        self.tokens, self.atraso, self.ttft = 20, 0.02, 0.05
        self.srv, self.url = servidor_falso.iniciar(servidor_falso.Config(self.tokens, self.atraso, self.ttft))
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        self.srv.shutdown()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_bench_mede_tempos_conhecidos(self):
        out = self.tmp / "r.json"
        p = subprocess.run([sys.executable, str(RAIZ / "docker" / "bench.py"), "--url", self.url, "--machine", "teste",
                            "--runs", "3", "--warmup", "1", "--max-tokens", str(self.tokens), "--out", str(out)],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr)
        r = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(r["procedencia"]["modelo"], "falso/modelo-teste:Q4_K_M")
        self.assertEqual(len(r["execucoes"]), 3)
        self.assertEqual(self.srv.cfg.pedidos, 4, "1 aquecimento + 3 execuções")
        for e in r["execucoes"]:
            self.assertEqual(e["n_chunks"], self.tokens, "o pedaço só com 'role' não pode contar como token")
        res = r["resumo"]
        # TTFT ~ 50 ms; TPOT ~ 20 ms; decode ~ 50 tok/s. Margem larga: o relógio do SO não é exato.
        self.assertAlmostEqual(res["ttft_s"]["mean"], self.ttft, delta=0.03)
        self.assertAlmostEqual(res["tpot_s"]["mean"], self.atraso, delta=0.008)
        self.assertAlmostEqual(res["tok_per_s_decode"]["mean"], 1 / self.atraso, delta=15)
        self.assertEqual(res["itl_s_agregado"]["n"], 3 * (self.tokens - 1))

    def test_sonda_ok_e_falha(self):
        out = self.tmp / "s.json"
        cmd = [sys.executable, str(RAIZ / "testbench" / "pipelines" / "sonda.py"), "--maquina", "teste", "--out", str(out)]
        p = subprocess.run(cmd + ["--url", self.url], capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        r = json.loads(out.read_text(encoding="utf-8"))
        self.assertTrue(r["ok"])
        self.assertEqual([c["nome"] for c in r["checagens"]], ["health", "modelos", "pedido curto", "streaming"])
        # servidor desligado: sai com 1 e ainda assim grava o JSON
        p = subprocess.run(cmd + ["--url", "http://127.0.0.1:9", "--timeout", "2"], capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 1)
        self.assertFalse(json.loads(out.read_text(encoding="utf-8"))["ok"])


class Manifestos(unittest.TestCase):
    def test_todos_os_manifestos_do_repositorio_sao_validos(self):
        pipes = servidor.carregar_pipelines()
        self.assertIn("latencia", pipes)
        self.assertIn("sonda", pipes)
        for m in pipes.values():
            for p in m["parametros"]:
                self.assertIn("rotulo", p, f"{m['id']}.{p['id']} sem rótulo")

    def test_latencia_cobre_todos_os_argumentos_do_bench(self):
        """Se alguém adicionar argumento ao bench.py, o manifesto precisa acompanhar."""
        ajuda = subprocess.run([sys.executable, str(RAIZ / "docker" / "bench.py"), "-h"],
                               capture_output=True, text=True).stdout
        import re
        do_bench = set(re.findall(r"--[a-z-]+", ajuda)) - {"--help", "--out"}
        m = servidor.carregar_pipelines()["latencia"]
        do_manifesto = {p["argumento"] for p in m["parametros"] if "argumento" in p}
        self.assertEqual(do_bench, do_manifesto)

    def test_validacao_recusa_manifestos_ruins(self):
        bom = {"id": "x", "nome": "x", "descricao": "x", "comando": ["python"], "parametros": []}
        servidor.validar_manifesto(dict(bom))
        ruins = [
            {k: v for k, v in bom.items() if k != "comando"},
            dict(bom, id="Com Espaço"),
            dict(bom, comando="python x.py"),
            dict(bom, parametros=[{"id": "a", "tipo": "cor"}]),
            dict(bom, parametros=[{"id": "a", "tipo": "texto"}, {"id": "a", "tipo": "texto"}]),
            dict(bom, parametros=[{"id": "a", "tipo": "escolha"}]),
            dict(bom, parametros=[{"id": "a", "tipo": "texto", "argumento": "a"}]),
            dict(bom, progresso="(\\d+)"),
        ]
        for m in ruins:
            with self.assertRaises(servidor.ErroValidacao, msg=str(m)):
                servidor.validar_manifesto(m)


class Comando(unittest.TestCase):
    M = {"id": "t", "nome": "t", "descricao": "t", "comando": ["{python}", "{raiz}/s.py"], "saida": {"argumento": "--out"},
         "parametros": [
             {"id": "n", "rotulo": "N", "tipo": "inteiro", "padrao": 5, "min": 1, "max": 10, "argumento": "--n"},
             {"id": "r", "rotulo": "R", "tipo": "real", "padrao": 0.5, "argumento": "--r"},
             {"id": "b", "rotulo": "B", "tipo": "booleano", "padrao": False, "argumento": "--b"},
             {"id": "e", "rotulo": "E", "tipo": "escolha", "opcoes": ["a", "b"], "padrao": "a", "argumento": "--e"},
             {"id": "t", "rotulo": "T", "tipo": "texto", "padrao": "", "argumento": "--t"},
             {"id": "u", "rotulo": "U", "tipo": "texto", "padrao": "{modelo_url}", "obrigatorio": True, "argumento": "--u"},
         ]}
    CTX = {"python": "PY", "raiz": "/r", "modelo_url": "http://m:1"}

    def test_padroes_e_placeholders(self):
        v = servidor.normalizar(self.M, {}, self.CTX)
        self.assertEqual(v, {"n": 5, "r": 0.5, "b": False, "e": "a", "t": None, "u": "http://m:1"})
        argv = servidor.montar_argv(self.M, v, self.CTX, "/o.json")
        self.assertEqual(argv, ["PY", "/r/s.py", "--n", "5", "--r", "0.5", "--e", "a", "--u", "http://m:1", "--out", "/o.json"])

    def test_conversao_e_flags(self):
        v = servidor.normalizar(self.M, {"n": "7", "r": "1,5", "b": True, "t": "com espaço; e aspas'"}, self.CTX)
        argv = servidor.montar_argv(self.M, v, self.CTX)
        self.assertIn("--b", argv)
        self.assertEqual(argv[argv.index("--t") + 1], "com espaço; e aspas'", "texto vai como um argumento só, sem shell")
        self.assertEqual(v["r"], 1.5)

    def test_erros_de_validacao(self):
        for valores, trecho in [({"n": 0}, "mínimo"), ({"n": 11}, "máximo"), ({"n": "x"}, "inválido"),
                                ({"n": 2.5}, "inválido"), ({"e": "z"}, "opção"), ({"u": ""}, "obrigatório"),
                                ({"zzz": 1}, "desconhecidos")]:
            with self.assertRaises(servidor.ErroValidacao) as c:
                servidor.normalizar(self.M, valores, self.CTX)
            self.assertIn(trecho, str(c.exception))

    def test_linha_shell_copiavel(self):
        argv = [sys.executable, "-u", str(servidor.RAIZ / "docker" / "bench.py"), "--notes", "na tomada"]
        self.assertEqual(servidor.linha_shell(argv, "sh"), "python docker/bench.py --notes 'na tomada'")
        self.assertEqual(servidor.linha_shell(argv, "ps"), "python docker/bench.py --notes 'na tomada'")


class API(unittest.TestCase):
    """Sobe a bancada de verdade (em thread) contra o servidor falso."""

    @classmethod
    def setUpClass(cls):
        cls.falso, cls.url_modelo = servidor_falso.iniciar(servidor_falso.Config(tokens=12, atraso=0.01, ttft=0.02))
        cls.tmp = Path(tempfile.mkdtemp())
        cls.srv = servidor.criar_servidor("127.0.0.1", 0, modelo_url=cls.url_modelo, maquina="maquina-teste")
        cls.srv.bancada.dir_resultados = cls.tmp
        import threading
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.falso.shutdown()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def esperar(self, eid, limite=60):
        t0 = time.time()
        while time.time() - t0 < limite:
            _, r = http("GET", f"{self.base}/api/execucoes/{eid}")
            if r["estado"] != "rodando":
                return r
            time.sleep(0.1)
        self.fail("execução não terminou")

    def test_estado_e_paginas(self):
        st, r = http("GET", self.base + "/api/estado")
        self.assertEqual(st, 200)
        self.assertEqual(r["maquina"], "maquina-teste")
        for pagina in ["/", "/index.html", "/testbench/index.html", "/laboratorios/compressao/laboratorio-compressao.html",
                       "/laboratorios/docker/docker-por-dentro.html", "/assets/tema.css"]:
            with urllib.request.urlopen(self.base + pagina) as resp:
                self.assertEqual(resp.status, 200, pagina)

    def test_nao_serve_fora_do_repositorio(self):
        with self.assertRaises(urllib.error.HTTPError):
            urllib.request.urlopen(self.base + "/../../../Windows/win.ini")

    def test_sonda_pelo_backend(self):
        _, r = http("GET", self.base + "/api/sonda")
        self.assertTrue(r["ok"])
        self.assertEqual(r["modelos"], ["falso/modelo-teste:Q4_K_M"])
        _, r = http("GET", self.base + "/api/sonda?url=http://127.0.0.1:9")
        self.assertFalse(r["ok"])

    def test_pipelines_trazem_padroes_resolvidos(self):
        _, ps = http("GET", self.base + "/api/pipelines")
        lat = next(p for p in ps if p["id"] == "latencia")
        self.assertEqual(lat["padroes"]["url"], self.url_modelo)
        self.assertEqual(lat["padroes"]["maquina"], "maquina-teste")

    def test_comando_previa(self):
        st, r = http("POST", self.base + "/api/comando", {"pipeline": "latencia", "parametros": {"runs": 3, "notas": "a b"}})
        self.assertEqual(st, 200)
        self.assertTrue(r["sh"].startswith("python docker/bench.py --url "))
        self.assertIn("--runs 3", r["sh"])
        self.assertIn("--notes 'a b'", r["sh"])
        st, r = http("POST", self.base + "/api/comando", {"pipeline": "latencia", "parametros": {"runs": 0}})
        self.assertEqual(st, 400)
        self.assertIn("Execuções", r["erro"])

    def test_execucao_completa_e_lista_de_resultados(self):
        st, r = http("POST", self.base + "/api/execucoes",
                     {"pipeline": "latencia", "parametros": {"runs": 2, "warmup": 0, "max_tokens": 12}})
        self.assertEqual(st, 201, r)
        fim = self.esperar(r["id"])
        self.assertEqual(fim["estado"], "ok", "\n".join(fim["log"]))
        self.assertEqual(fim["progresso"], [2, 2])
        self.assertEqual(fim["resultado"]["procedencia"]["maquina"], "maquina-teste")
        self.assertEqual(len(fim["resultado"]["execucoes"]), 2)
        _, lista = http("GET", self.base + "/api/resultados")
        self.assertTrue(any(i["maquina"] == "maquina-teste" for i in lista))
        self.assertTrue(any(i["legado"] for i in lista), "os resultados da oficina também aparecem")
        # log incremental
        _, parcial = http("GET", f"{self.base}/api/execucoes/{r['id']}?desde=2")
        self.assertEqual(parcial["log"], fim["log"][2:])

    def test_uma_execucao_por_vez_e_cancelamento(self):
        st, r = http("POST", self.base + "/api/execucoes",
                     {"pipeline": "latencia", "parametros": {"runs": 200, "warmup": 0, "max_tokens": 12}})
        self.assertEqual(st, 201)
        st, r2 = http("POST", self.base + "/api/execucoes", {"pipeline": "sonda", "parametros": {}})
        self.assertEqual(st, 409, "duas medições ao mesmo tempo estragariam as duas")
        st, c = http("POST", f"{self.base}/api/execucoes/{r['id']}/cancelar")
        self.assertEqual(st, 200)
        fim = self.esperar(r["id"])
        self.assertEqual(fim["estado"], "cancelado")

    def test_erro_do_processo_aparece(self):
        st, r = http("POST", self.base + "/api/execucoes",
                     {"pipeline": "latencia", "parametros": {"url": "http://127.0.0.1:9", "runs": 1, "warmup": 0}})
        fim = self.esperar(r["id"])
        self.assertEqual(fim["estado"], "erro")
        self.assertTrue(fim["erro"])

    def test_pedidos_invalidos(self):
        self.assertEqual(http("POST", self.base + "/api/execucoes", {"pipeline": "nao-existe"})[0], 400)
        self.assertEqual(http("GET", self.base + "/api/execucoes/xyz")[0], 404)
        self.assertEqual(http("GET", self.base + "/api/nada")[0], 404)
        req = urllib.request.Request(self.base + "/api/execucoes", data=b"{nao json", method="POST")
        with self.assertRaises(urllib.error.HTTPError) as c:
            urllib.request.urlopen(req)
        self.assertEqual(c.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
