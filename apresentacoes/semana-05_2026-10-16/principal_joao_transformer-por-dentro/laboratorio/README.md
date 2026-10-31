# Laboratório Pythia-1B

Abra `index.html` diretamente ou acesse `/laboratorio/` pelo servidor da apresentação. A página apresenta somente a arquitetura completa do Pythia-1B: entrada, embedding, todos os 16 blocos, LayerNorm final e saída. A implementação anterior com dimensões reduzidas, parâmetros próprios e geração calibrada foi removida.

## Percurso e linguagem

Uma única área mostra o mapa em três dimensões; abaixo, uma operação, sua explicação e seu fluxo matemático. Selecione um componente ou use **Seguir o fluxo** e **Voltar** (também setas do teclado). **Abrir este bloco** revela as normalizações independentes, atenção, MLP e soma residual. É possível explorar qualquer um dos 16 blocos.

Dentro da atenção: Query/Key/Value, posição, produto interno, máscara causal, softmax, soma ponderada dos Values, concatenação das oito cabeças e projeção. Dentro da MLP: LayerNorm, expansão, GELU e projeção. A saída mostra normalização final, logits, temperatura, probabilidades e retorno à entrada.

A linguagem segue os slides atuais: embedding (03), residual paralelo (04), Query/Key/Value e Score/peso (06–10), concatenação e projeção (11), MLP (12), GELU (14), saída (15), softmax (16), temperatura (17) e arquitetura/contagem (19–20). Cada etapa oferece um link ao slide correspondente. Score é o valor anterior à softmax; peso é seu resultado normalizado. As equações usam KaTeX local e rótulos explícitos, sem introduzir índices ou operadores não explicados.

A informação de posição, necessária para representar a arquitetura completa, é descrita diretamente como rotação de pares de coordenadas, sem exigir a sigla RoPE ou sua matriz de rotação. O slide 19 contextualiza essa etapa na arquitetura, mas não detalha a rotação. LayerNorm é explicada em palavras; GELU usa a explicação já presente no slide 14, sem introduzir a função de distribuição normal. As projeções internas incluem os vetores aprendidos somados aos produtos, mencionados antes dos fluxos de dimensão. As contagens completas ficam em detalhes expansíveis.

## Painel do bloco e transformações

Ao selecionar qualquer bloco, um painel ao lado do mapa mostra as duas LayerNorms, atenção com oito cabeças, expansão/GELU/projeção da MLP e soma residual. Seus botões abrem o componente dentro do bloco selecionado. A partir de 1.200 pixels, mapa e painel ficam lado a lado; em telas menores, o painel vem abaixo. A explicação existente é movida para o painel, sem duplicá-la abaixo do mapa.

As projeções de Query, Key, Value, atenção, expansão, contração e saída mostram o vetor-linha de entrada, a matriz com suas dimensões reais e o resultado. A construção segue multiplicação → bias → saída, preservando os passos concluídos. O bias é explicado como um vetor aprendido, somado coordenada por coordenada. Na projeção final não há bias, e o percurso tem apenas multiplicação e saída. As projeções Q/K/V são exibidas separadamente para acompanhar as operações; a implementação do GPT-NeoX pode armazená-las em uma projeção conjunta de 6.144 saídas.

Os destaques percorrem a linha de entrada e uma coluna da matriz; o vetor resultante aparece depois. Na soma, o vetor de bias se aproxima do produto. Os desenhos compactos de matrizes e vetores representam suas estruturas, sem células numéricas: não são matrizes menores nem ativações calculadas. As dimensões indicadas são as do Pythia-1B. Cada transformação acompanha um token; as mesmas matrizes e biases são aplicados aos outros tokens.

**Ver a conta de uma coordenada** permite escolher qualquer coordenada válida e exibe a soma dos produtos correspondentes, com o bias quando aplicável. Os símbolos são definidos no painel; nenhum peso ou resultado de checkpoint é inventado. LayerNorm, GELU, posição, scores, máscara, softmax, soma dos Values, concatenação e residual têm percursos próprios, sem tratar operações não lineares como projeções matriciais.

Os controles do painel avançam, voltam ou repetem a animação da etapa atual. Com o foco no painel, as setas controlam a transformação; com o foco no mapa, percorrem a arquitetura. Animações encerram ao trocar de componente, ocultar a aba ou imprimir. Movimento reduzido e impressão mostram a construção completa. O código fica em `detalhes.js`, usando KaTeX e Web Animations API locais ao navegador.

## Fidelidade ao modelo

- 16 blocos; 2.048 coordenadas; oito cabeças de 256 coordenadas.
- MLP com dimensão intermediária 8.192 e ativação GELU.
- Atenção e MLP normalizam a mesma entrada com parâmetros independentes. Os dois resultados são somados à entrada original.
- A posição gira 64 coordenadas de Query e Key por cabeça (25%); não altera Value.
- Tabelas de entrada e saída distintas, ambas com 50.304 posições. Esse é o tamanho acolchoado das matrizes do modelo; não se deve confundir com a quantidade de itens efetivamente definidos no tokenizer.
- Contexto máximo de 2.048 tokens; projeção final sem vetor adicional (bias).
- Total: 1.011.781.632 parâmetros, conforme os arquivos de verificação locais.

O navegador explora a arquitetura e suas operações; não carrega o checkpoint nem calcula ativações ou previsões reais. Os fluxos matemáticos mostram operações e dimensões reais, sem fabricar resultados numéricos de inferência. As dimensões das matrizes são apresentadas na convenção de vetores-linha adotada pelos slides; o armazenamento do framework pode usar a orientação transposta.

## Interação e animação

A geometria usa paralelepípedos em coordenadas tridimensionais, rotação em dois eixos e projeção ortográfica em SVG. As faces são opacas, com iluminação suave a partir do alto à esquerda e sombras discretas. Faces voltadas para trás não são desenhadas; os objetos são ordenados do mais distante ao mais próximo. Isso elimina as arestas traseiras que tornavam ambígua a orientação. A vista inicial mostra a face superior dos blocos. Arraste para girar, use **Vista frontal**, **Restaurar** e os botões de zoom. Com o mapa em foco: A/D e W/S giram; +/− aproximam/afastam; 0 restaura. Tab e Enter/Espaço selecionam componentes. A face frontal acompanha o arraste horizontal e vertical; W/S seguem a mesma convenção. No celular, arrastar dentro do mapa gira o objeto; a rolagem da página continua disponível fora dele. Soltar fora do mapa, cancelar o toque ou tirar o foco da janela encerra o arraste. Na impressão, as faces usam tons claros e as sombras são removidas.

Os sinais animados indicam o caminho dos dados. Cada avanço dura 0,8 s (velocidade comum de 1,25×), mantém o resultado e espera outra ação. As duas normalizações e os dois ramos são destacados simultaneamente no percurso do bloco. A ordem de estudo atenção → MLP não representa dependência sequencial entre esses ramos nem latência do hardware. Voltar recupera a etapa anterior. Movimento reduzido, ocultação da aba e impressão encerram os movimentos.

A página é responsiva, separada da sequência de slides 16:9. Em telas estreitas, os grafos são reorganizados em duas colunas. Equações longas têm rolagem local. Não há recursos de rede; KaTeX e fontes são locais. A impressão preserva o mapa e a etapa atual.

## Verificação

```bash
python -m http.server 8765 --directory apresentacao_transformer/apresentacao
# Em outro terminal, com Playwright disponível:
node apresentacao_transformer/verificacao/testar_arquitetura_ui.cjs
node apresentacao_transformer/verificacao/testar_camera_ui.cjs
node apresentacao_transformer/verificacao/testar_detalhes_ui.cjs
```

O teste aceita `PLAYWRIGHT_PATH`, `CHROME_PATH`, `LAB_URL` e `LAB_ARTIFACTS`. Cobre todos os blocos e operações, residual paralelo, navegação entre níveis, câmera, fórmulas, quatro larguras, impressão e acesso offline com movimento reduzido. Capturas e PDFs são gravados em `/tmp/transformer-architecture-qa` por padrão. O teste da câmera verifica os quatro sentidos de arraste, teclas WASD, ida e volta, ocultação das faces traseiras, ordenação por profundidade, encerramento do gesto fora do mapa e rotação por toque. Seus artefatos ficam em `/tmp/transformer-camera-qa`.

O teste de detalhes cobre o painel lateral, os oito indicadores de cabeças, navegação no bloco correto, Q/K/V, dimensões de expansão, bias, saída sem bias, etapas cumulativas, animações, seleção de coordenadas, teclado, quatro larguras, PDF e arquivo local com movimento reduzido. Os artefatos ficam em `/tmp/transformer-details-qa`.

## Fontes verificadas

- [Configuração oficial de treinamento](https://github.com/EleutherAI/pythia/blob/main/models/1B/pythia-1b.yml).
- [Configuração distribuída do Pythia-1B](https://huggingface.co/EleutherAI/pythia-1b/raw/main/config.json); cópia local em `../../verificacao/pythia-config.json`.
- [Implementação GPT-NeoX no Transformers 4.24](https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py): residual paralelo, posição, projeções e normalizações.
- [Manim — construção por objetos e transformações](https://docs.manim.community/en/stable/tutorials/building_blocks.html): referência de linguagem visual; a implementação usa SVG e JavaScript.

- [MDN — feDropShadow](https://developer.mozilla.org/en-US/docs/Web/SVG/Reference/Element/feDropShadow): sombra suave em SVG, com deslocamento, desfoque e opacidade controlados.
