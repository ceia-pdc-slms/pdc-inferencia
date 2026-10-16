# Quantização — Guia Completo de Apresentação e Estudo

## Parte 1 — Os Conceitos, do Zero

### 1.1 Por que isso importa

Um computador só entende zeros e uns. Todo número — seja um peso de rede neural ou a ativação de um neurônio — precisa ser guardado numa quantidade fixa de bits. A forma como organizamos esses bits decide três coisas ao mesmo tempo: **quão preciso** o número é, **quão grande ou pequeno** ele pode ser sem "estourar", e **quanto espaço** ele ocupa na memória. Quantização é a técnica de reduzir os bits usados por número — trocando parte dessa precisão por memória e velocidade.

### 1.2 Ponto Flutuante — a "notação científica" do computador

Pense em ponto flutuante como a notação científica (ex: 3,14 × 10²): ele consegue representar tanto números microscópicos quanto gigantescos porque separa a **magnitude** da **precisão**. Todo número em ponto flutuante tem três partes:

| Parte | Função |
|---|---|
| **Sinal** | É positivo ou negativo? (sempre 1 bit) |
| **Expoente** | Controla o **alcance** — o quão grande/pequeno o número pode ser |
| **Fração (mantissa)** | Controla a **precisão** — quantas casas decimais significativas |

Essa tabela é a base de tudo o que vem depois:

| Formato | Total de bits | Sinal | Expoente | Fração | Memória/número | Característica |
|---|---|---|---|---|---|---|
| **FP32** | 32 | 1 | 8 | 23 | 4 bytes | Padrão-ouro: extrema precisão |
| **FP16** | 16 | 1 | 5 | 10 | 2 bytes | Expoente curto → risco de overflow |
| **BF16** | 16 | 1 | 8 | 7 | 2 bytes | Expoente igual ao FP32 → padrão de IA |

**FP16** corta o expoente para só 5 bits. O maior número que ele consegue representar é **65.504** — qualquer coisa acima disso "estoura" (overflow) e vira infinito, o que faz o modelo gerar lixo. **BF16** (criado pelo Google Brain) resolve isso de um jeito inteligente: mantém os mesmos 8 bits de expoente do FP32 — logo, tem o mesmo alcance gigantesco e não estoura fácil — mas sacrifica a fração (só 7 bits). Em IA, errar a oitava casa decimal quase não importa; estourar o número, sim. Por isso BF16 virou o padrão de treinamento de LLMs.

### 1.3 Números Inteiros — INT8 e INT4

Inteiros não têm expoente nem fração — são só uma régua linear de valores:

| Formato | Bits | Valores possíveis | Faixa (signed) |
|---|---|---|---|
| **INT8** | 8 | 256 | −128 a 127 |
| **INT4** | 4 | 16 | −8 a 7 |

O problema: como representar um número quebrado como 3,7 usando só 16 ou 256 "baldes" inteiros? É aqui que entram escala e zero-point.

### 1.4 Escala e Zero-Point — a ponte entre os dois mundos

Duas variáveis resolvem o problema:

- **Escala**: o tamanho do "degrau" entre um inteiro e o próximo, traduzido para o mundo real.
- **Zero-point**: qual número inteiro representa exatamente o zero real.

**Fórmula de quantização** (mundo real → inteiro, para guardar na memória):
```
INT = round(valor / escala) + zero_point
```

**Fórmula de dequantização** (inteiro → mundo real, para a GPU calcular):
```
valor = (INT − zero_point) × escala
```

> ⚠️ Atenção aos parênteses na segunda fórmula: sem eles, a ordem das operações muda e o resultado sai errado.

Se você tem um intervalo real `[min, max]` e quer mapear para um intervalo inteiro `[qmin, qmax]`, calcula assim:
```
escala = (max − min) / (qmax − qmin)
zero_point = qmin − round(min / escala)
```

**Simétrica vs. Assimétrica** — a distinção mais importante deste tópico:

- **Simétrica** (`zero_point = 0`): usada quando o intervalo real é centrado em zero — o caso típico dos **pesos** de uma rede neural (distribuição em sino, mais ou menos igual para positivo e negativo).
- **Assimétrica** (`zero_point ≠ 0`): necessária quando o intervalo não é centrado em zero — o caso típico de **ativações** depois de uma ReLU (sempre ≥ 0) ou GELU (majoritariamente positiva, com uma pequena cauda negativa).

Dois exemplos rápidos para fixar:
- Ativação ReLU, intervalo `[0, 6]` → UINT8 `[0, 255]`: escala = 6/255 ≈ 0,0235; zero_point = 0 (o intervalo já começa em zero, não precisa deslocar).
- Ativação tipo GELU, intervalo `[−2, 6]` → UINT8 `[0, 255]`: escala = 8/255 ≈ 0,0314; zero_point = 0 − round(−2/0,0314) = **64** (precisa deslocar, porque o zero real não cai no início da régua inteira).

### 1.5 O Preço: o Erro de Quantização

`round()` é uma via de mão única: uma vez arredondado, você não recupera a precisão original. A boa notícia é que esse erro é **limitado**: como você sempre arredonda para o degrau mais próximo, o erro máximo nunca passa de **metade do tamanho do degrau**:
```
erro máximo ≤ escala / 2
```
Isso conecta diretamente com "menos bits": menos bits → menos degraus disponíveis no mesmo intervalo → degraus maiores (escala maior) → erro máximo maior. É a régua de ouro para explicar "o que se perde".

### 1.6 Outliers e Saturação (Clipping)

E se um valor real for maior do que a régua inteira consegue guardar? Ele é "cortado" (saturado) no teto ou piso da régua — e isso pode gerar um erro bem maior do que o erro normal de arredondamento (veja o Exemplo 3 na Parte 2). Esse é o motivo de existirem técnicas de quantização mais sofisticadas como **GPTQ** e **AWQ**, que identificam e protegem esses valores "fora da curva" (outliers) para não destruir a qualidade do modelo.

### 1.7 A Fronteira FP4

Por que usar FP4 em vez de INT4, se os dois têm 4 bits? Porque FP4 ainda reserva alguns bits para expoente (no formato mais comum, 1 sinal + 2 expoente + 1 fração), então seus 16 valores possíveis **não são espaçados uniformemente** — ficam mais densos perto do zero. Isso importa porque os pesos de uma rede treinada seguem uma distribuição em sino: a maioria fica pertinho de zero, e poucos são grandes. INT4 (linear) desperdiça "resolução" em regiões onde quase não há peso nenhum; FP4 concentra a precisão exatamente onde ela é mais usada.

### 1.8 Por que fazer tudo isso? O gargalo é a memória, não a conta

Um ponto que eleva o nível da discussão: ao gerar texto token a token (inferência com lote pequeno), a GPU não fica limitada pela quantidade de multiplicações que consegue fazer — ela fica limitada pela **velocidade de trazer os pesos da memória RAM até o processador**. Isso é conhecido como estar "limitado por banda de memória" (memory-bandwidth bound). Por isso, quantizar de FP32 (4 bytes) para INT8 (1 byte) não deixa a conta mais fácil — deixa a **leitura de memória até 4x mais rápida**, porque trafegam 4x menos bytes pelo barramento a cada token gerado. É essa a razão prática, e não só teórica, de todo o esforço de quantização.

Em números concretos, para um modelo de 7 bilhões de parâmetros:

| Formato | Memória necessária |
|---|---|
| FP32 | 28 GB |
| FP16 / BF16 | 14 GB |
| INT8 | 7 GB |
| INT4 / FP4 | 3,5 GB |

---

## Parte 2 — Exemplos Resolvidos, Passo a Passo

### Exemplo 1 — O caso perfeito
**Valor real:** 3,7 · **Escala:** 0,05

1. Escalonar: 3,7 / 0,05 = 74
2. Arredondar: 74 já é inteiro → continua 74
3. Guardar na memória (INT8): **74** (1 byte, em vez de 4 bytes em FP32)
4. Dequantizar: 74 × 0,05 = 3,7
5. **Erro: 0,0** — aconteceu porque 3,7 é um múltiplo exato de 0,05. Na prática, é raro ter tanta sorte.

### Exemplo 2 — O caso real (onde o erro aparece)
**Valor real:** 3,72 · **Escala:** 0,05

1. Escalonar: 3,72 / 0,05 = 74,4
2. Arredondar: 74,4 está mais perto de 74 do que de 75 → arredonda para **74** (o ",4" é descartado para sempre)
3. Guardar: **74**
4. Dequantizar: 74 × 0,05 = 3,70
5. **Erro absoluto:** 3,72 − 3,70 = 0,02
6. **Erro relativo:** 0,02 / 3,72 ≈ **0,54%**
7. **É aceitável?** Sim. Erros relativos abaixo de ~1% costumam se somar ao ruído normal de inferência sem afetar a qualidade percebida do modelo — e o ganho é 4x menos memória lida por parâmetro.

### Exemplo 3 — Outlier e saturação (clipping)
**Valor real:** 8,5 · **Escala:** 0,05 (mesma escala do tensor do Exemplo 1/2)

1. Escalonar: 8,5 / 0,05 = 170
2. Arredondar: continua 170
3. **Problema:** INT8 (signed) só vai até 127 — 170 não cabe na régua!
4. **Saturação:** o valor é "cortado" no teto máximo → guarda-se **127**
5. Dequantizar: 127 × 0,05 = 6,35
6. **Erro: 8,5 − 6,35 = 2,15** — bem maior que o erro normal de arredondamento.
7. **Lição:** um único valor "fora da curva" pode gerar um erro enorme se a escala foi calibrada para a maioria dos valores "normais". É exatamente o problema que métodos como AWQ e GPTQ tentam resolver, tratando outliers com cuidado especial.

### Exemplo 4 — Zero-point assimétrico na prática
**Cenário:** ativação ReLU, sempre no intervalo [0, 6] → UINT8 [0, 255]. **Valor a quantizar:** 4,5

1. Escala: (6 − 0) / 255 ≈ 0,0235
2. Zero-point: 0 − round(0/0,0235) = **0** (o intervalo já começa em zero, não precisa deslocar)
3. Quantizar: round(4,5 / 0,0235) = round(191,49) = **191**
4. Dequantizar: 191 × 0,0235 ≈ **4,49**
5. **Erro: ≈ 0,01 (≈ 0,2%)** — bem aceitável.
6. **Contraste:** se o intervalo fosse [−2, 6] (uma ativação tipo GELU, que pode ser levemente negativa), o zero-point seria **64**, não 0 — porque aí o zero real não cai mais no início da régua inteira. É o mesmo raciocínio do item 1.4.

---

## Parte 3 — Roteiro Slide a Slide (versão final, 11 slides)

### Slide 1 — Título
**No slide:** Quantização: Representação Numérica · FP32, FP16, BF16, INT8, INT4, FP4 · A escala, o zero-point e o preço real da compressão.

**Fale:** "Hoje vou falar sobre a base da quantização: como os números são representados na memória, e o que exatamente perdemos quando espremos um modelo de 32 bits para 8 ou até 4 bits."

### Slide 2 — Ponto Flutuante (FP32, FP16, BF16)
**No slide:** os três cards comparando bits de sinal/expoente/fração.

**Fale:** "O computador guarda números quebrando-os em sinal, expoente e fração. Ao ir de FP32 para FP16, ganhamos metade do espaço, mas perdemos alcance — o expoente encolhe de 8 para 5 bits, e o maior número que cabe é 65.504; passar disso quebra o modelo. O BF16, do Google, resolveu isso mantendo os 8 bits de expoente do FP32 e sacrificando a fração. Por isso virou o padrão de treinamento de IA: errar a última casa decimal importa muito menos do que estourar o número."

### Slide 3 — Descendo para os Inteiros (INT8, INT4)
**No slide:** INT8 = 256 valores (−128 a 127) · INT4 = 16 valores (−8 a 7).

**Fale:** "A grande virada em inferência foi passar a usar inteiros puros. INT8 tem só 256 valores possíveis, INT4 tem 16. Pra colocar todos os pesos decimais do modelo dentro dessas réguas tão curtas, usamos duas ferramentas: escala e zero-point — é o que vem no próximo slide."

### Slide 4 — Escala e Zero-Point
**No slide:** régua numérica mostrando o mapeamento de [−2, 6] real para [0, 255] inteiro, com zero-point = 64 destacado; cards de simétrico vs. assimétrico.

**Fale:** "A escala é o tamanho do degrau — quanto vale andar 1 no mundo inteiro, traduzido pro mundo real. O zero-point diz qual número inteiro representa exatamente o zero real. Quando o intervalo é centrado em zero — o caso típico dos pesos de uma rede — usamos zero-point igual a zero, chamamos de quantização simétrica. Mas ativações depois de uma ReLU são sempre positivas, então o intervalo não é centrado em zero — aí precisamos de um zero-point diferente de zero, a quantização assimétrica. Nesse exemplo, o intervalo vai de −2 a 6, e o zero-point calculado dá 64." *(Use o Exemplo 4 da Parte 2 se alguém pedir um caso mais simples de cabeça.)*

### Slide 5 — Matemática da Conversão
**No slide:** `INT = round(valor/escala) + zero_point` e `valor = (INT − zero_point) × escala`.

**Fale:** "Formalizando: para quantizar, dividimos pela escala, arredondamos e somamos o zero-point. Para recuperar o valor, fazemos o caminho inverso. O ponto de atenção é o `round()` — é ali que a precisão original é destruída de forma irreversível. Uma vez arredondado, você nunca mais recupera exatamente o número original."

### Slide 6 — Onde Mora o Erro
**No slide:** gráfico de escada — valor real (linha diagonal) vs. valor quantizado (degraus), com a distância entre os dois destacada.

**Fale:** "Esse gráfico mostra visualmente o que a fórmula do slide anterior faz na prática: todo valor real é arredondado pro degrau mais próximo, e a distância até esse degrau é o erro. A regra prática que fecha o raciocínio: o erro máximo nunca passa de metade do tamanho da escala. E é aí que conectamos com 'menos bits': menos bits significa menos degraus no mesmo intervalo, degraus maiores, e erro máximo maior."

### Slide 7 — Pratique: Zero-Point em Ação
**No slide:** exercício resolvido passo a passo (ReLU [0,6], valor 4,5) + comparação com o caso assimétrico do Slide 4.

**Fale:** "Vamos praticar com zero-point. Uma ativação ReLU está sempre entre 0 e 6. Escala: 6 dividido por 255, aproximadamente 0,0235. Como o intervalo já começa em zero, o zero-point é zero. Quantizando 4,5: 4,5 dividido pela escala dá 191. Dequantizando: 191 vezes a escala dá aproximadamente 4,49 — um erro de só 0,01, bem abaixo de 1%. A diferença para o exemplo do slide 4 é só isso: ali o intervalo cruzava o zero fora do centro, então precisava de zero-point 64; aqui não precisa, porque já começa em zero."

### Slide 8 — O Preço da Compressão
**No slide:** gráfico de alocação de bits (todos os 6 formatos lado a lado) + gráfico de memória para um modelo de 7B parâmetros.

**Fale:** "Juntando tudo: o primeiro gráfico mostra como cada formato gasta seus bits — os de ponto flutuante dividem entre sinal, expoente e fração; os inteiros gastam tudo num valor linear só. O segundo gráfico traduz isso em algo concreto: um modelo de 7 bilhões de parâmetros pesa 28 GB em FP32, 14 GB em FP16 ou BF16, 7 GB em INT8, e só 3,5 GB em INT4 ou FP4. A frase que eu quero que vocês levem desse slide: a pergunta certa nunca é 'perde-se algo?' — sempre se perde algo. A pergunta certa é 'esse erro é aceitável para a minha aplicação?'"

### Slide 9 — Fronteira FP4
**No slide:** curva de sino com os 16 níveis do FP4 concentrados perto do zero.

**Fale:** "Por que FP4 em vez de INT4, se os dois usam 4 bits? Porque o FP4 ainda reserva bits pra expoente, então seus 16 valores não são espaçados igualmente — ficam mais densos perto do zero. Como os pesos de uma rede treinada se concentram perto de zero (formato de sino), o FP4 aproveita melhor cada um dos seus 4 bits do que o INT4, que desperdiça resolução em regiões quase vazias."

### Slide 10 — Regra da Vulnerabilidade (Discussão Aberta)
**No slide:** "Como as GPUs modernas executam matemática de matrizes em INT4? O chip faz o cálculo nativamente ou precisa desempacotar tudo para FP16 internamente?"

**Fale (para abrir a discussão, não para fechar com uma resposta única):** "Essa eu quero deixar em aberto pro grupo, mas posso adiantar o que eu descobri estudando: GPUs modernas (a partir da geração Ampere/Hopper da NVIDIA, e ainda mais na Blackwell) têm instruções nativas para multiplicar matrizes direto em INT8, INT4 e até FP4 — o cálculo em si acontece em baixa precisão. Mas o *acúmulo* da soma (que pode crescer muito ao somar milhares de produtos) é feito em precisão mais alta, tipo INT32 ou FP32, pra não estourar. Então não é simplesmente 'desempacota tudo pra FP16' — só que hardware mais antigo ou sem essas instruções específicas realmente precisa converter antes de calcular. Deixo a pergunta pra turma: vale a pena ter hardware dedicado pra cada formato, ou isso trava a evolução dos formatos numéricos?" *(Veja mais respostas de apoio na Parte 4.)*

### Slide 11 — Resposta da Semana
**No slide:** o exercício pedido, resolvido em 4 passos, com os três cenários (perfeito, real, veredito).

**Fale:** "E pra fechar, o exercício da semana: representar 3,7 em INT8 com escala 0,05. Dividindo, 3,7 por 0,05 dá exatamente 74 — não precisa arredondar, então o erro nesse caso específico é zero. Mas eu fui um passo além: se o valor original fosse 3,72 em vez de 3,7 exato, o processo dá o mesmo 74 armazenado, só que ao recuperar teríamos 3,70 — um erro de 0,02, ou seja, um erro relativo de aproximadamente 0,54%. Isso é perfeitamente aceitável: a regra prática é que erros relativos abaixo de 1% se somam ao ruído normal de inferência, e em troca a gente ganha até 4 vezes mais velocidade de leitura de memória, trocando 4 bytes por número (FP32) por só 1 byte (INT8)."

---

## Parte 4 — Perguntas Frequentes (para se preparar)

**"Por que não usar sempre zero-point = 0? Seria mais simples."**
Porque forçaria todo intervalo a ser representado como se fosse centrado em zero — desperdiçando metade da régua inteira em valores que nunca ocorrem. Ativações depois de ReLU, por exemplo, nunca são negativas; usar zero-point = 0 aí jogaria fora metade da resolução disponível.

**"Por que BF16 e não FP16 hoje em dia, se os dois têm 16 bits?"**
Porque na prática o alcance (não estourar durante o treinamento) importa mais do que a precisão fina das últimas casas decimais. BF16 garante o mesmo alcance do FP32 com metade do espaço; FP16 economiza o mesmo espaço mas corre risco real de overflow.

**"O erro de quantização se acumula conforme passa pelas camadas da rede?"**
Sim, pode acumular — é uma preocupação real em modelos muito profundos. É exatamente por isso que existem técnicas mais sofisticadas do que a quantização "ingênua" mostrada aqui: calibração cuidadosa da escala por camada (ou até por canal), proteção de outliers (AWQ, GPTQ) e, em casos mais extremos, treinamento consciente da quantização (QAT — Quantization-Aware Training), que ensina o modelo a já compensar esse ruído durante o próprio treino.

**"Isso vale tanto para treinar quanto para usar (inferir) o modelo?"**
Na prática, hoje a quantização agressiva (INT8, INT4, FP4) é usada principalmente para **inferência** — depois que o modelo já foi treinado. O treinamento normalmente usa BF16 ou uma combinação de precisões (mixed precision), porque o processo de treino é mais sensível a ruído numérico do que simplesmente rodar o modelo já pronto.

**"Todo mundo usa a mesma escala pro modelo inteiro?"**
Não necessariamente — cada tensor (cada camada de pesos, cada conjunto de ativações) normalmente tem sua própria escala e zero-point, calculados a partir do intervalo real de valores daquele tensor específico. É por isso que os exemplos deste guia usam escalas diferentes em contextos diferentes (0,05 para um tensor de pesos, 0,0235 para uma ativação).

---

## Parte 5 — Revisão 

**Fórmulas:**
```
Quantizar:    INT   = round(valor / escala) + zero_point
Dequantizar:  valor = (INT − zero_point) × escala
Calcular:     escala = (max − min) / (qmax − qmin)
              zero_point = qmin − round(min / escala)
Erro máximo:  ≤ escala / 2
```

**Formatos:**

| Formato | Bits | Sinal+Exp+Frac | Memória | Onde é usado |
|---|---|---|---|---|
| FP32 | 32 | 1+8+23 | 4 bytes | Padrão-ouro |
| FP16 | 16 | 1+5+10 | 2 bytes | Risco de overflow (máx. 65.504) |
| BF16 | 16 | 1+8+7 | 2 bytes | Padrão de treino de IA |
| INT8 | 8 | — | 1 byte | Inferência comum |
| INT4 | 4 | — | 0,5 byte | Inferência agressiva |
| FP4 | 4 | 1+2+1 | 0,5 byte | Inferência agressiva, não-uniforme |

**Memória de um modelo de 7B parâmetros:** FP32 = 28 GB · FP16/BF16 = 14 GB · INT8 = 7 GB · INT4/FP4 = 3,5 GB

**Exercício da semana (3,7 em INT8, escala 0,05):** guarda-se 74 · erro no caso exato = 0 · erro num caso próximo (3,72) ≈ 0,02 (≈0,54% relativo) · veredito: aceitável, com ganho de 4x em memória.