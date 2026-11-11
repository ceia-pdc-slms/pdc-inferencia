# Quantização a fundo: FP16 → FP8 → INT4 → FP4, o que quebra e o que acelera

Apresentação principal da semana 8 (06/11/2026) · Beatriz

| Material | O que é |
|---|---|
| [Slides (PDF)](quantizacao-a-fundo.pdf) | 26 slides. Antes de cada slide mais difícil há um slide de ilustração (a régua, o peso gigante, o empacotamento, o arredondamento que compensa, o cano da memória e o byte a desembalar). Quase todos têm embaixo uma faixa "Em palavras simples". O último é a resposta da pergunta da semana |
| [`quantizacao_a_fundo.py`](quantizacao_a_fundo.py) | Mede o erro e a perplexidade de cada formato, os kernels, um passo de decode, o modelo de ponta a ponta e o que há dentro dos GGUF |
| [Laboratório interativo](../../../laboratorios/quantizacao/quantizacao-a-fundo.html) | Os bits de cada formato, escala e outliers, o que quebra, o que acelera e a conta do tok/s com os números medidos |

## Pergunta da semana

> O tok/s aumentou na mesma proporção em que os bytes por token caíram? Se não, o que consumiu o ganho?

**Resposta curta.** Não. Do F16 para o Q4_0 do Qwen2.5-0.5B, o nome promete 4×, cada token lê 2,9× menos bytes e o tok/s sobe cerca de 1,6×. Três coisas consomem o ganho:

1. **Os bytes caem menos que o nome.** Cada grupo de 32 pesos guarda uma escala (+0,5 bit por peso), e a matriz de saída fica em Q8_0: 145 MB, 42% do que o token lê.
2. **O kernel tem que aproveitar os bytes menores.** Com conta inteira, as multiplicações acompanharam os bytes (5,6× mais rápidas). Desempacotando para FP32, 2,5×. Com o kernel INT4 do PyTorch para CPU, 20× mais lentas.
3. **O resto do token não encolhe.** Atenção sobre o KV cache, normalizações, escolha do token e o runtime são de 29% a 63% do token, e crescem com o contexto.

No prefill, quantizar só os pesos não acelera nada: lá quem limita são as contas.

## Medido no notebook-bia

Intel Core i5-1135G7, 4 núcleos, só CPU, 10/11/2026. Python 3.12, PyTorch 2.7.1, transformers 4.49, ONNX Runtime 1.24.1. Modelo: Qwen2.5-0.5B-Instruct.

### O que quebra: os pesos e o modelo com cada formato

Só os pesos das camadas foram arredondados (358 M); o embedding, as ativações e o KV cache ficaram em FP32. Perplexidade num texto fixo de 385 tokens; menor é melhor.

| Formato | Bits por peso | Erro nos pesos | Perplexidade | Mudou |
|---|---|---|---|---|
| FP32 (original) | 32 | — | 17,98 | — |
| FP16 | 16 | 0,00% | 17,98 | +0,0% |
| FP8 E4M3 | 8 | 2,65% | 17,94 | −0,2% |
| FP8 E5M2 | 8 | 5,27% | 19,42 | +8,0% |
| INT8 por canal | 8,01 | 1,00% | 17,99 | +0,1% |
| INT4 por tensor | 4 | 75,76% | 37 milhões | quebrou |
| INT4 por canal | 4,01 | 17,90% | 44,94 | +150% |
| INT4 grupo 32 | 4,5 | 10,25% | 22,74 | +26% |
| FP4 por tensor | 4 | 52,42% | 758.563 | quebrou |
| MXFP4 (bloco 32) | 4,25 | 11,71% | 22,66 | +26% |
| NVFP4 (bloco 16) | 4,5 | 9,45% | 19,76 | +9,9% |

### O que acelera: as multiplicações de um token

Todas as multiplicações por pesos de um passo (24 camadas e o lm_head), no ONNX Runtime com 4 threads.

| Configuração | MB por passo | Bytes caíram | Lote 1: tok/s | Subiu | Lote 64: tok/s |
|---|---|---|---|---|---|
| FP32 | 1.976 | 1,00× | 14,1 | 1,00× | 189 |
| FP16 | 988 | 2,00× | 13,5 | 0,96× | 187 |
| INT8 | 556 | 3,55× | 55,3 | 3,93× | 201 |
| INT4 + lm_head INT8 | 377 | 5,24× | 73,6 | 5,23× | 172 |
| INT4 tudo | 309 | 6,40× | 78,1 | 5,55× | 200 |
| INT4 tudo, sem conta inteira | 309 | 6,40× | 34,6 | 2,46× | 138 |

INT8 e INT4 usam o `MatMulNBits` em blocos de 32. Com `accuracy_level=4`, a ativação vira INT8 e a conta é inteira (VNNI). Com `accuracy_level=0` ("sem conta inteira"), cada peso é desempacotado para FP32. Esta CPU não faz contas em FP16, por isso o FP16 não ganhou nada.

### O resto do token: o modelo de verdade

transformers em FP32, 32 tokens gerados com KV cache.

| Prompt | ms por token | Multiplicações por pesos | Resto | Resto % |
|---|---|---|---|---|
| 64 tokens | 107,4 | 76,6 ms | 30,8 ms | 29% |
| 1.024 tokens | 137,5 | 78,9 ms | 58,6 ms | 43% |
| 4.096 tokens | 206,4 | 76,4 ms | 130,0 ms | 63% |

### Dentro dos GGUF oficiais

| Arquivo | MB | Bits por peso | Camadas | Matriz de saída | MB lidos por token |
|---|---|---|---|---|---|
| F16 | 1.260 | 16,00 | F16 | F16 | 988 |
| Q8_0 | 670 | 8,50 | Q8_0 | Q8_0 | 525 |
| Q4_K_M | 485 | 6,16 | Q5_0 + Q6_K + Q4_K | Q8_0 | 392 |
| Q4_0 | 423 | 5,37 | Q4_0 | Q8_0 | 346 |
| Q2_K | 409 | 5,20 | IQ4_NL + Q3_K | Q8_0 | 332 |

Os k-quants usam blocos de 256 pesos, e a largura do Qwen2.5-0.5B é 896. Onde não cabe, o llama.cpp usa outro tipo: por isso o Q2_K quase empata com o Q4_0.

### O que não foi medido

O `--llama-bench` não rodou nesta máquina, porque o ambiente bloqueou a execução do binário baixado. O ~1,6× da resposta é uma conta com duas medidas: o tempo das multiplicações (ONNX Runtime) e o resto do token (transformers). Para medir direto no llama.cpp, rode o modo 4 abaixo e compare. Ativações e KV cache quantizados também ficaram de fora.

## Como rodar

```bash
pip install numpy torch safetensors        # o básico
pip install transformers                   # --qualidade e --ponta-a-ponta
pip install onnx onnxruntime               # --decode

python quantizacao_a_fundo.py --exemplo       # 0. a conta na mão: 8 números e um outlier
python quantizacao_a_fundo.py                 # 1. erro de cada formato nos pesos do Qwen2.5-0.5B
python quantizacao_a_fundo.py --qualidade     # 2. perplexidade e respostas com cada formato
python quantizacao_a_fundo.py --decode --lotes 1,64    # 3. um passo de decode por formato
python quantizacao_a_fundo.py --ponta-a-ponta # 3b. quanto do token é multiplicação por pesos
python quantizacao_a_fundo.py --velocidade    # 3c. os kernels do PyTorch, bytes x tempo
python quantizacao_a_fundo.py --gguf pasta/dos/gguf    # 4. o que há dentro de cada GGUF
python quantizacao_a_fundo.py --llama-bench caminho/llama-bench --gguf pasta/dos/gguf
                                              # 4b. tok/s de verdade no llama.cpp x bytes
```

O modelo é lido do cache do Hugging Face (`Qwen/Qwen2.5-0.5B-Instruct`, 1 GB); `--modelo pasta` aponta outro. Os GGUF estão em `Qwen/Qwen2.5-0.5B-Instruct-GGUF`. Opções: `--threads 4`, `--reps 9`, `--json arquivo.json` salva as medidas.

**Memória.** `--qualidade` e `--ponta-a-ponta` carregam o modelo em FP32 (2 GB) e `--decode` monta grafos de até 1 GB. Os três conferem a RAM livre antes de começar e param se houver menos de 2,5 a 3,5 GB, para não travar a máquina.

## Para estudar

- Dettmers et al., 2022, *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale* (arXiv:2208.07339): outliers nas ativações.
- Xiao et al., 2022, *SmoothQuant* (arXiv:2211.10438): passar a dificuldade das ativações para os pesos.
- Frantar et al., 2022, *GPTQ* (arXiv:2210.17323): arredondar compensando o erro.
- Lin et al., 2023, *AWQ* (arXiv:2306.00978): proteger os pesos que mais importam.
- Micikevicius et al., 2022, *FP8 Formats for Deep Learning* (arXiv:2209.05433): E4M3 e E5M2.
- Open Compute Project, 2023, *OCP Microscaling Formats (MX) Specification v1.0*: MXFP4.
