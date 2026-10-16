# Máquinas

Um dossiê por máquina física do time. O dossiê é a entrega individual do semestre e é o que cada pessoa apresenta na reunião de 18/12.

## Meta de cada dossiê (até dezembro)

1. A máquina liga e é acessada por SSH.
2. Roda um modelo.
3. O dossiê é um passo a passo que outra pessoa segue sozinha.
4. Tem pelo menos uma medição de tok/s em `runs/`, com procedência completa.

## Como criar

Copie [`_modelo.md`](_modelo.md) para `<nome-da-maquina>.md` e preencha. O passo a passo do PR está no [guia de contribuição](../CONTRIBUTING.md#seu-primeiro-pr-o-dossiê-da-sua-máquina).

O nome da máquina é o mesmo usado nos registros de `runs/`: a verificação automática confere se a máquina de cada medição tem dossiê aqui.

## Máquinas do time

| Máquina | Hardware | Responsável | Liga e SSH | Roda modelo | README testado | Medida |
|---|---|---|---|---|---|---|
| *cada PR de dossiê adiciona uma linha aqui* | | | | | | |

### Alocação proposta no plano

Enquanto a lista de hardware não fecha, esta é a divisão do plano. Quem receber a máquina troca a linha acima pelo dossiê de verdade.

| Pessoa | Máquina proposta | Papel |
|---|---|---|
| Beatriz | Máquina com GPU / acesso ao Spark | Receitas de compressão |
| Wevton | Spark / ambiente | Ambiente e reprodutibilidade |
| João Vitor | Jetson | Modelo analítico e kernels |
| Pedro | Raspberry Pi | Bancada de medição |
| Marcos | Raspberry Pi / segundo dispositivo de borda | Corpus de avaliação e perfis de carga |

Os notebooks usados na oficina de Docker (`notebook-bia`, `noteboook-pedro`) já têm resultado em [`docker/resultados/`](../docker/resultados/). Se forem continuar sendo usados para medir, merecem dossiê também.
