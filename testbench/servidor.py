#!/usr/bin/env python3
"""
servidor.py - backend da bancada de testbench do time de Compressão e Testbench.

Serve as páginas do repositório (hub, laboratórios, testbench) e uma API pequena que
executa pipelines de medição. Cada pipeline é um manifesto JSON em testbench/pipelines/
que descreve parâmetros e o comando a rodar; a interface monta o formulário a partir dele.

Só biblioteca padrão: roda em qualquer Python >= 3.9, local ou em container.

  python testbench/servidor.py                       # http://127.0.0.1:8088
  python testbench/servidor.py --modelo-url http://localhost:8080 --porta 8088

A medição roda aqui, no servidor, e não no navegador: assim não há CORS, e o número
é o mesmo que o bench.py daria no terminal.
"""
import argparse, json, os, platform, re, shlex, socket, subprocess, sys, threading, time, uuid
import urllib.request, urllib.error
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

VERSAO = "1.0.0"
RAIZ = Path(__file__).resolve().parent.parent
DIR_PIPELINES = RAIZ / "testbench" / "pipelines"
DIR_RESULTADOS = RAIZ / "testbench" / "resultados"
# resultados antigos, da oficina de Docker, também aparecem na comparação
DIRS_LEGADO = [RAIZ / "docker" / "resultados"]

TIPOS = {"texto", "texto_longo", "inteiro", "real", "escolha", "booleano"}
MAX_CORPO = 256 * 1024
MAX_LOG = 5000


class ErroValidacao(ValueError):
    pass


# ---------------------------------------------------------------- manifestos

def validar_manifesto(m, origem="?"):
    """Confere a forma de um manifesto. Levanta ErroValidacao com uma mensagem legível."""
    def falha(msg):
        raise ErroValidacao(f"{origem}: {msg}")
    for campo in ("id", "nome", "descricao", "comando", "parametros"):
        if campo not in m:
            falha(f"falta o campo '{campo}'")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", m["id"]):
        falha("id deve ter só letras minúsculas, números, _ e -")
    if not isinstance(m["comando"], list) or not m["comando"] or not all(isinstance(x, str) for x in m["comando"]):
        falha("comando deve ser uma lista de strings")
    ids = set()
    for p in m["parametros"]:
        pid = p.get("id")
        if not pid or not re.fullmatch(r"[a-z0-9_]+", pid):
            falha(f"parâmetro com id inválido: {pid!r}")
        if pid in ids:
            falha(f"parâmetro repetido: {pid}")
        ids.add(pid)
        if p.get("tipo") not in TIPOS:
            falha(f"parâmetro {pid}: tipo deve ser um de {sorted(TIPOS)}")
        if p["tipo"] == "escolha" and not p.get("opcoes"):
            falha(f"parâmetro {pid}: escolha precisa de 'opcoes'")
        if "argumento" in p and not str(p["argumento"]).startswith("-"):
            falha(f"parâmetro {pid}: argumento deve começar com '-'")
    saida = m.get("saida")
    if saida is not None and "argumento" not in saida:
        falha("saida precisa de 'argumento'")
    prog = m.get("progresso")
    if prog is not None:
        try:
            if re.compile(prog).groups != 2:
                falha("progresso deve ter dois grupos: atual e total")
        except re.error as e:
            falha(f"progresso não é regex válida ({e})")
    return m


def carregar_pipelines(diretorio=DIR_PIPELINES):
    pipes = {}
    for arq in sorted(Path(diretorio).glob("*.json")):
        m = validar_manifesto(json.loads(arq.read_text(encoding="utf-8")), arq.name)
        if m["id"] in pipes:
            raise ErroValidacao(f"{arq.name}: id repetido '{m['id']}'")
        pipes[m["id"]] = m
    return pipes


# ---------------------------------------------------------------- contexto e comando

def _digest_imagem(imagem="ghcr.io/ggml-org/llama.cpp:server"):
    try:
        out = subprocess.run(["docker", "image", "inspect", imagem, "--format", "{{index .RepoDigests 0}}"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def contexto_padrao(cfg):
    return {
        "python": sys.executable,
        "raiz": str(RAIZ),
        "modelo_url": cfg.get("modelo_url", "http://localhost:8080"),
        "maquina": cfg.get("maquina") or socket.gethostname(),
        "imagem_digest": cfg.get("imagem_digest", ""),
    }


def _substituir(texto, ctx):
    return re.sub(r"\{(\w+)\}", lambda mt: str(ctx.get(mt.group(1), mt.group(0))), texto)


def padroes(manifesto, ctx):
    """Valores padrão de cada parâmetro, com {placeholders} resolvidos."""
    out = {}
    for p in manifesto["parametros"]:
        v = p.get("padrao")
        out[p["id"]] = _substituir(v, ctx) if isinstance(v, str) else v
    return out


def normalizar(manifesto, valores, ctx):
    """Aplica padrões, converte tipos e valida limites. Devolve dict id -> valor."""
    valores = dict(valores or {})
    desconhecidos = set(valores) - {p["id"] for p in manifesto["parametros"]}
    if desconhecidos:
        raise ErroValidacao(f"parâmetros desconhecidos: {', '.join(sorted(desconhecidos))}")
    base = padroes(manifesto, ctx)
    out = {}
    for p in manifesto["parametros"]:
        pid, tipo, rot = p["id"], p["tipo"], p.get("rotulo", p["id"])
        v = valores.get(pid, base.get(pid))
        if v is None or (isinstance(v, str) and v.strip() == "" and tipo not in ("booleano",)):
            if p.get("obrigatorio"):
                raise ErroValidacao(f"{rot}: obrigatório")
            out[pid] = None
            continue
        try:
            if tipo == "inteiro":
                if isinstance(v, bool) or (isinstance(v, float) and not v.is_integer()):
                    raise ValueError
                v = int(str(v).strip())
            elif tipo == "real":
                v = float(str(v).strip().replace(",", "."))
            elif tipo == "booleano":
                if isinstance(v, str):
                    v = v.strip().lower() in ("1", "true", "sim", "on")
                v = bool(v)
            else:
                v = str(v)
        except ValueError:
            raise ErroValidacao(f"{rot}: valor inválido ({v!r})")
        if tipo in ("inteiro", "real"):
            if "min" in p and v < p["min"]:
                raise ErroValidacao(f"{rot}: mínimo {p['min']}")
            if "max" in p and v > p["max"]:
                raise ErroValidacao(f"{rot}: máximo {p['max']}")
        if tipo == "escolha" and v not in [str(o) if not isinstance(o, dict) else str(o["valor"]) for o in p["opcoes"]]:
            raise ErroValidacao(f"{rot}: opção inválida ({v})")
        if tipo in ("texto", "texto_longo") and len(v) > p.get("max_caracteres", 4000):
            raise ErroValidacao(f"{rot}: texto longo demais")
        out[pid] = v
    return out


def montar_argv(manifesto, valores, ctx, arquivo_saida=None):
    """Lista de argumentos (sem shell). Booleanos viram flag; vazios são omitidos."""
    argv = [_substituir(x, ctx) for x in manifesto["comando"]]
    for p in manifesto["parametros"]:
        arg, v = p.get("argumento"), valores.get(p["id"])
        if not arg or v is None:
            continue
        if p["tipo"] == "booleano":
            if v:
                argv.append(arg)
            continue
        argv += [arg, str(v)]
    if arquivo_saida and manifesto.get("saida"):
        argv += [manifesto["saida"]["argumento"], str(arquivo_saida)]
    return argv


def linha_shell(argv, estilo="sh"):
    """Comando copiável. Troca caminhos absolutos do repositório por relativos."""
    raiz = str(RAIZ)
    def rel(a):
        if a == sys.executable:
            return "python"
        if a.startswith(raiz):
            return os.path.relpath(a, raiz).replace("\\", "/")
        return a
    partes = [rel(a) for a in argv if a != "-u"]
    if estilo == "ps":
        def q(s):
            return s if re.fullmatch(r"[\w@%+=:,./\\-]+", s) else "'" + s.replace("'", "''") + "'"
        return " ".join(q(p) for p in partes)
    return " ".join(shlex.quote(p) for p in partes)


# ---------------------------------------------------------------- execuções

class Execucao:
    def __init__(self, manifesto, valores, argv, arquivo):
        self.id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.manifesto, self.valores, self.argv, self.arquivo = manifesto, valores, argv, arquivo
        self.estado, self.log, self.progresso = "rodando", [], None
        self.inicio, self.fim, self.codigo, self.erro = time.time(), None, None, None
        self.proc = None
        self._lock = threading.Lock()

    def anotar(self, linha):
        with self._lock:
            self.log.append(linha)
            if len(self.log) > MAX_LOG:
                del self.log[: len(self.log) - MAX_LOG]
        rx = self.manifesto.get("progresso")
        if rx:
            mt = re.search(rx, linha)
            if mt:
                self.progresso = [int(mt.group(1)), int(mt.group(2))]

    def resumo(self, desde=0):
        res = None
        if self.estado != "rodando" and self.arquivo and Path(self.arquivo).exists():
            try:
                res = json.loads(Path(self.arquivo).read_text(encoding="utf-8"))
            except Exception as e:
                res = {"erro_leitura": str(e)}
        with self._lock:
            log = self.log[desde:]
            total = len(self.log)
        return {
            "id": self.id, "pipeline": self.manifesto["id"], "estado": self.estado,
            "valores": self.valores, "comando": linha_shell(self.argv), "progresso": self.progresso,
            "inicio": self.inicio, "fim": self.fim, "codigo": self.codigo, "erro": self.erro,
            "log": log, "log_total": total,
            "arquivo": os.path.relpath(self.arquivo, RAIZ).replace("\\", "/") if self.arquivo else None,
            "resultado": res,
        }


class Bancada:
    """Estado do servidor: pipelines carregados e no máximo UMA execução por vez
    (duas medições ao mesmo tempo disputariam CPU e memória e estragariam as duas)."""

    def __init__(self, cfg, dir_pipelines=DIR_PIPELINES, dir_resultados=DIR_RESULTADOS):
        self.cfg = cfg
        self.dir_pipelines, self.dir_resultados = Path(dir_pipelines), Path(dir_resultados)
        self.pipelines = carregar_pipelines(self.dir_pipelines)
        self.execucoes = {}
        self.atual = None
        self._lock = threading.Lock()

    def ctx(self):
        return contexto_padrao(self.cfg)

    def preparar(self, pid, valores):
        if pid not in self.pipelines:
            raise ErroValidacao(f"pipeline desconhecido: {pid}")
        m = self.pipelines[pid]
        ctx = self.ctx()
        vals = normalizar(m, valores, ctx)
        arquivo = None
        if m.get("saida"):
            maq = re.sub(r"[^\w.-]+", "-", str(vals.get("maquina") or ctx["maquina"]))
            arquivo = self.dir_resultados / m["id"] / f"{maq}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        return m, vals, montar_argv(m, vals, ctx, arquivo), arquivo

    def iniciar(self, pid, valores):
        m, vals, argv, arquivo = self.preparar(pid, valores)
        with self._lock:
            if self.atual and self.atual.estado == "rodando":
                raise RuntimeError(f"já existe uma execução rodando ({self.atual.id})")
            ex = Execucao(m, vals, argv, arquivo)
            self.execucoes[ex.id] = ex
            self.atual = ex
        if arquivo:
            arquivo.parent.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self._rodar, args=(ex,), daemon=True).start()
        return ex

    def _rodar(self, ex):
        ex.anotar("$ " + linha_shell(ex.argv))
        try:
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            ex.proc = subprocess.Popen(ex.argv, cwd=str(RAIZ), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)
            for linha in ex.proc.stdout:
                ex.anotar(linha.rstrip("\n"))
            ex.codigo = ex.proc.wait()
            if ex.estado == "cancelado":
                pass
            elif ex.codigo == 0 and (not ex.arquivo or Path(ex.arquivo).exists()):
                ex.estado = "ok"
            else:
                ex.estado = "erro"
                ex.erro = f"o processo terminou com código {ex.codigo}" if ex.codigo else "o processo não gerou o arquivo de resultado"
        except Exception as e:
            ex.estado, ex.erro = "erro", f"{type(e).__name__}: {e}"
            ex.anotar("erro: " + ex.erro)
        ex.fim = time.time()

    def cancelar(self, eid):
        ex = self.execucoes.get(eid)
        if not ex:
            raise KeyError(eid)
        if ex.estado == "rodando":
            ex.estado = "cancelado"
            ex.anotar("cancelado pelo usuário")
            if ex.proc and ex.proc.poll() is None:
                ex.proc.terminate()
        return ex

    def resultados(self):
        itens = []
        for base in [self.dir_resultados, *DIRS_LEGADO]:
            if not base.exists():
                continue
            for arq in sorted(base.rglob("*.json")):
                try:
                    j = json.loads(arq.read_text(encoding="utf-8"))
                except Exception:
                    continue
                proc = j.get("procedencia", {}) if isinstance(j, dict) else {}
                itens.append({
                    "caminho": os.path.relpath(arq, RAIZ).replace("\\", "/"),
                    "esquema": j.get("esquema", "bench-v1" if "resumo" in j else "desconhecido"),
                    "maquina": proc.get("maquina"), "modelo": proc.get("modelo"),
                    "data_utc": proc.get("data_utc"),
                    "legado": base != self.dir_resultados,
                })
        itens.sort(key=lambda x: x["data_utc"] or "", reverse=True)
        return itens


def sondar(url, timeout=3):
    """Pergunta /v1/models ao servidor do modelo, daqui do backend (sem CORS)."""
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=timeout) as r:
            j = json.load(r)
        ids = [d.get("id") for d in j.get("data", [])] if isinstance(j, dict) else []
        return {"ok": True, "url": url, "modelos": ids, "ms": round((time.perf_counter() - t0) * 1000, 1)}
    except urllib.error.HTTPError as e:
        return {"ok": False, "url": url, "erro": f"HTTP {e.code}"}
    except Exception as e:
        motivo = getattr(e, "reason", e)
        return {"ok": False, "url": url, "erro": str(motivo)}


# ---------------------------------------------------------------- HTTP

def criar_handler(bancada):
    class Handler(SimpleHTTPRequestHandler):
        server_version = "BancadaPDC/" + VERSAO

        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(RAIZ), **kw)

        def log_message(self, fmt, *args):
            if bancada.cfg.get("verboso"):
                super().log_message(fmt, *args)

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def _json(self, obj, status=200):
            corpo = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(corpo)))
            self.end_headers()
            self.wfile.write(corpo)

        def _corpo(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_CORPO:
                raise ErroValidacao("corpo grande demais")
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n).decode("utf-8"))
            except Exception:
                raise ErroValidacao("JSON inválido")

        def do_GET(self):
            u = urlparse(self.path)
            if not u.path.startswith("/api/"):
                if u.path == "/":
                    self.path = "/index.html"
                return super().do_GET()
            q = parse_qs(u.query)
            partes = u.path.strip("/").split("/")[1:]
            try:
                if partes == ["estado"]:
                    ctx = bancada.ctx()
                    return self._json({"versao": VERSAO, "modelo_url": ctx["modelo_url"], "maquina": ctx["maquina"],
                                       "python": platform.python_version(), "plataforma": platform.platform(),
                                       "pipelines": len(bancada.pipelines),
                                       "rodando": bancada.atual.id if bancada.atual and bancada.atual.estado == "rodando" else None})
                if partes == ["pipelines"]:
                    ctx = bancada.ctx()
                    return self._json([dict(m, padroes=padroes(m, ctx)) for m in bancada.pipelines.values()])
                if partes == ["sonda"]:
                    return self._json(sondar(q.get("url", [bancada.ctx()["modelo_url"]])[0]))
                if partes == ["execucoes"]:
                    return self._json([e.resumo(desde=10**9) for e in sorted(bancada.execucoes.values(), key=lambda e: -e.inicio)][:50])
                if len(partes) == 2 and partes[0] == "execucoes":
                    ex = bancada.execucoes.get(partes[1])
                    if not ex:
                        return self._json({"erro": "execução não encontrada"}, 404)
                    return self._json(ex.resumo(desde=int(q.get("desde", ["0"])[0] or 0)))
                if partes == ["resultados"]:
                    return self._json(bancada.resultados())
                return self._json({"erro": "rota não encontrada"}, 404)
            except ErroValidacao as e:
                return self._json({"erro": str(e)}, 400)

        def do_POST(self):
            partes = urlparse(self.path).path.strip("/").split("/")
            if partes[:1] != ["api"]:
                return self._json({"erro": "rota não encontrada"}, 404)
            partes = partes[1:]
            try:
                corpo = self._corpo()
                if partes == ["comando"]:
                    m, vals, argv, _ = bancada.preparar(corpo.get("pipeline"), corpo.get("parametros"))
                    return self._json({"argv": argv, "sh": linha_shell(argv, "sh"), "ps": linha_shell(argv, "ps"), "valores": vals})
                if partes == ["execucoes"]:
                    ex = bancada.iniciar(corpo.get("pipeline"), corpo.get("parametros"))
                    return self._json(ex.resumo(), 201)
                if len(partes) == 3 and partes[0] == "execucoes" and partes[2] == "cancelar":
                    return self._json(bancada.cancelar(partes[1]).resumo())
                return self._json({"erro": "rota não encontrada"}, 404)
            except ErroValidacao as e:
                return self._json({"erro": str(e)}, 400)
            except RuntimeError as e:
                return self._json({"erro": str(e)}, 409)
            except KeyError:
                return self._json({"erro": "execução não encontrada"}, 404)

    return Handler


def criar_servidor(host="127.0.0.1", porta=8000, **cfg):
    bancada = Bancada(cfg)
    srv = ThreadingHTTPServer((host, porta), criar_handler(bancada))
    srv.daemon_threads = True
    srv.bancada = bancada
    return srv


def main():
    ap = argparse.ArgumentParser(description="Bancada de testbench do time de Compressão e Testbench")
    ap.add_argument("--host", default=os.environ.get("BANCADA_HOST", "127.0.0.1"),
                    help="use 0.0.0.0 só dentro de container ou de rede confiável: a API executa comandos")
    ap.add_argument("--porta", type=int, default=int(os.environ.get("BANCADA_PORTA", "8088")))
    ap.add_argument("--modelo-url", default=os.environ.get("BANCADA_MODELO_URL", "http://localhost:8080"))
    ap.add_argument("--maquina", default=os.environ.get("BANCADA_MAQUINA"),
                    help="nome padrão da máquina nos resultados (ex.: notebook-bia)")
    ap.add_argument("--verboso", action="store_true")
    a = ap.parse_args()

    srv = criar_servidor(a.host, a.porta, modelo_url=a.modelo_url, maquina=a.maquina,
                         imagem_digest=os.environ.get("IMAGE_DIGEST") or _digest_imagem(), verboso=a.verboso)
    b = srv.bancada
    print(f"Bancada {VERSAO} · {len(b.pipelines)} pipeline(s): {', '.join(b.pipelines)}")
    print(f"Modelo esperado em {a.modelo_url} · máquina padrão: {b.ctx()['maquina']}")
    print(f"Abra http://{'localhost' if a.host in ('0.0.0.0', '127.0.0.1') else a.host}:{a.porta}  (Ctrl+C para sair)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nencerrado")


if __name__ == "__main__":
    main()
