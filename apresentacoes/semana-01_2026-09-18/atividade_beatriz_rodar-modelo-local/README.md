# Guia de Execução de Small Language Models (SLMs)

Esse documento detalha a arquitetura e a função de cada componente do script `run_slm.py`, projetado para inicializar, configurar e executar inferências em um Small Language Model (SLM) localmente.

## 1. Importações e Dependências (`torch` e `transformers`)
O código inicia importando as bibliotecas essenciais para o funcionamento do modelo:
- `transformers`: Fornecida pela Hugging Face, essa biblioteca abstrai a complexidade do download de arquiteturas de redes neurais, oferecendo classes prontas como `AutoModelForCausalLM` e `AutoTokenizer`.
- `torch` (PyTorch): É o framework de tensor subjacente que executa os cálculos matemáticos pesados e gerencia o uso de memória na CPU ou GPU (através do ecossistema CUDA).

## 2. Seleção do Modelo
Utilizei o `"Qwen/Qwen2.5-0.5B-Instruct"`. O sufixo `0.5B` indica que a rede possui aproximadamente 500 milhões de parâmetros. Sendo um SLM, ele é leve o suficiente para rodar em hardwares modestos, mantendo um desempenho razoável para tarefas de linguagem natural e seguimento de instruções (`Instruct`).

## 3. Tokenização
A classe `Tokenizer` atua como o codificador/decodificador do modelo. Redes neurais não processam strings nativamente, mas sim tensores numéricos.
O Tokenizer converte o texto de entrada (o *prompt*) em uma sequência de IDs (números inteiros) que o modelo pode processar, e depois realiza o trabalho inverso para transformar a saída numérica gerada em texto legível humano.

## 4. Alocação de Dispositivo (CPU vs GPU)
O script verifica automaticamente o ambiente através da função `torch.cuda.is_available()`. Se uma GPU compatível com CUDA for detectada, o modelo é alocado para a VRAM (memória de vídeo) e configurado para precisão de ponto flutuante `float16`. Isso reduz drasticamente o espaço necessário na memória e acelera consideravelmente o cálculo matricial. Caso contrário, ele fará *fallback* nativo para a CPU.

## 5. Preparação e Formatação de Prompt
Modelos instrucionais (Chat/Instruct) esperam que a entrada siga um *template* de marcação específico (frequentemente usando tags para separar contexto do sistema da mensagem do usuário). Utilizei o `apply_chat_template` para garantir que as instruções do usuário e o *system prompt* sejam encapsulados exatamente na sintaxe em que a IA foi otimizada durante sua fase de *fine-tuning*.

## 6. Inferência (`model.generate`)
A geração de texto em modelos autorregressivos ocorre em um processo iterativo, calculando a distribuição de probabilidade do próximo token (palavra) passo a passo. O parâmetro `max_new_tokens` é crucial para limitar o loop de inferência, evitando ciclos infinitos e controlando o uso da memória de contexto.

## 7. Coleta e Exportação de Métricas
Após a conclusão da inferência, o script computa a performance da execução medindo:
- O consumo de memória alocada do modelo (pegada da rede neural na memória).
- O tempo total de execução.
- O *Throughput* (Taxa de Transferência): Mensurada em **Tokens por segundo**. Essa é a métrica para aferir a capacidade e a latência de processamento da infraestrutura escolhida.
- Finalmente, o código inicializa uma função de File I/O para exportar persistir esses benchmarks em um arquivo local em disco (`metricas.txt`), garantindo reprodutibilidade das análises de performance.
