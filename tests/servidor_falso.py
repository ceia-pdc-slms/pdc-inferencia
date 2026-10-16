"""
Servidor falso compatível com OpenAI, para testar a bancada sem modelo nenhum.

Entrega N pedaços de texto em streaming com atraso controlado, então dá para saber
de antemão qual TTFT, TPOT e tok/s o bench.py deveria medir.

  python tests/servidor_falso.py --porta 8089 --tokens 32 --atraso 0.01
"""
import argparse, json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Config:
    def __init__(self, tokens=16, atraso=0.005, ttft=0.02, modelo="falso/modelo-teste:Q4_K_M"):
        self.tokens, self.atraso, self.ttft, self.modelo = tokens, atraso, ttft, modelo
        self.pedidos = 0


def criar(cfg):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, obj, status=200):
            b = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/health":
                return self._json({"status": "ok"})
            if self.path == "/v1/models":
                return self._json({"object": "list", "data": [{"id": cfg.modelo, "object": "model"}]})
            self._json({"erro": "nao encontrado"}, 404)

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self._json({"erro": "nao encontrado"}, 404)
            corpo = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            cfg.pedidos += 1
            n = min(cfg.tokens, corpo.get("max_tokens", cfg.tokens))
            if not corpo.get("stream"):
                time.sleep(cfg.ttft)
                return self._json({"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                                   "usage": {"completion_tokens": 1}})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def enviar(obj):
                linha = ("data: " + (obj if isinstance(obj, str) else json.dumps(obj)) + "\n\n").encode()
                self.wfile.write(f"{len(linha):x}\r\n".encode() + linha + b"\r\n")
                self.wfile.flush()

            time.sleep(cfg.ttft)
            # primeiro pedaço só com o papel, sem texto: o bench.py não deve contar
            enviar({"choices": [{"delta": {"role": "assistant"}}]})
            for i in range(n):
                if i:
                    time.sleep(cfg.atraso)
                enviar({"choices": [{"delta": {"content": f"t{i} "}}]})
            enviar({"choices": [], "usage": {"completion_tokens": n}})
            enviar("[DONE]")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    return H


def iniciar(cfg=None, porta=0):
    """Sobe em thread e devolve (servidor, url)."""
    cfg = cfg or Config()
    srv = ThreadingHTTPServer(("127.0.0.1", porta), criar(cfg))
    srv.daemon_threads = True
    srv.cfg = cfg
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--porta", type=int, default=8089)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--atraso", type=float, default=0.01)
    a = ap.parse_args()
    srv, url = iniciar(Config(a.tokens, a.atraso), a.porta)
    print(f"servidor falso em {url} (Ctrl+C para sair)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
