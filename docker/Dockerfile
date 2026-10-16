# Imagem da bancada de medicao: so Python + bench.py (biblioteca padrao, sem pip install).
FROM python:3.12-slim

COPY bench.py /app/bench.py

# Os JSON de resultado sao gravados aqui; monte uma pasta do host neste caminho.
WORKDIR /resultados

ENTRYPOINT ["python", "/app/bench.py"]
