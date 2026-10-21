"""
Teste de fumaça das páginas: abre cada uma num Chrome/Edge sem janela, servida pela bancada,
e confere que o script terminou (data-pronto) sem erro de JavaScript (data-erro).
É pulado se nenhum navegador baseado em Chromium for encontrado.

  python -m unittest tests.test_paginas -v
"""
import os, re, shutil, subprocess, sys, tempfile, threading, unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "testbench"))
sys.path.insert(0, str(RAIZ / "tests"))
import servidor  # noqa: E402
import servidor_falso  # noqa: E402

PAGINAS = [
    "index.html",
    "laboratorios/compressao/laboratorio-compressao.html",
    "testbench/index.html",
    "testbench/index.html#resultados",
    "laboratorios/docker/docker-por-dentro.html",
    "laboratorios/geracao/geracao-de-texto.html",
]


def achar_navegador():
    candidatos = [os.environ.get("NAVEGADOR")]
    for nome in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome", "msedge"):
        candidatos.append(shutil.which(nome))
    candidatos += [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    return next((c for c in candidatos if c and Path(c).exists()), None)


NAVEGADOR = achar_navegador()


@unittest.skipUnless(NAVEGADOR, "nenhum Chrome/Edge encontrado (defina NAVEGADOR=caminho)")
class Paginas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.falso, url_modelo = servidor_falso.iniciar()
        cls.tmp = Path(tempfile.mkdtemp())
        cls.srv = servidor.criar_servidor("127.0.0.1", 0, modelo_url=url_modelo, maquina="teste")
        cls.srv.bancada.dir_resultados = cls.tmp
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}/"
        cls.perfil = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.falso.shutdown()
        shutil.rmtree(cls.tmp, ignore_errors=True)
        shutil.rmtree(cls.perfil, ignore_errors=True)

    def dom(self, pagina, tema=None, budget=6000):
        perfil = tempfile.mkdtemp(dir=self.perfil)  # perfil novo por chamada: instâncias não se atrapalham
        args = [NAVEGADOR, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                f"--user-data-dir={perfil}", f"--virtual-time-budget={budget}", "--window-size=1400,1000",
                "--dump-dom", self.base + pagina]
        if tema:
            args.insert(1, f"--blink-settings=preferredColorScheme={1 if tema == 'claro' else 0}")
        p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
        return p.stdout

    def test_paginas_carregam_sem_erro(self):
        for pagina in PAGINAS:
            for tema in ("claro", "escuro"):
                with self.subTest(pagina=pagina, tema=tema):
                    html = self.dom(pagina, tema)
                    raiz = re.search(r"<html[^>]*>", html)
                    self.assertIsNotNone(raiz, f"{pagina}: sem <html> no DOM")
                    self.assertNotIn("data-erro", raiz.group(0), f"{pagina}: erro de JavaScript: {raiz.group(0)}")
                    self.assertIn('data-pronto="1"', raiz.group(0), f"{pagina}: o script não terminou")

    def test_testbench_conecta_e_lista_pipelines(self):
        html = self.dom("testbench/index.html")
        self.assertIn("Bancada conectada", html)
        self.assertIn('data-p="latencia"', html)
        self.assertIn('data-p="sonda"', html)
        self.assertIn("Modelo online", html)

    def test_interacoes_completas(self):
        """?autoteste faz a página clicar em tudo; precisa terminar sem erro."""
        for pagina in ["laboratorios/compressao/laboratorio-compressao.html?autoteste", "testbench/index.html?autoteste",
                       "laboratorios/geracao/geracao-de-texto.html?autoteste"]:
            with self.subTest(pagina=pagina):
                raiz = re.search(r"<html[^>]*>", self.dom(pagina, budget=20000)).group(0)
                self.assertNotIn("data-erro", raiz, raiz)
                self.assertIn('data-autoteste="ok"', raiz, raiz)

    def test_laboratorio_de_compressao_calcula(self):
        html = self.dom("laboratorios/compressao/laboratorio-compressao.html")
        # o veredito inicial (modelo da oficina em FP16) não cabe em 500 MB
        self.assertRegex(html, r'id="rChip"[^>]*>não cabe<')
        self.assertIn("Quantização primeiro", html)


if __name__ == "__main__":
    unittest.main()
