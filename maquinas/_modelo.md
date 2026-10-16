# <nome-da-maquina>

> Copie este arquivo para `maquinas/<nome-da-maquina>.md` e substitua tudo que está entre `< >`.
> O que você ainda não sabe, escreva `PENDENTE`. Pendência marcada é informação; campo vazio não é.

## Identificação

| Campo | Valor |
|---|---|
| Nome (o mesmo usado em `runs/`) | `<nome-da-maquina>` |
| Hardware | <ex.: Raspberry Pi 5, 8 GB> |
| Responsável | <seu nome> (@<usuario-github>) |
| Patrimônio | <número de patrimônio, se tiver> |
| Onde está | <casa de quem, laboratório> |

## Status do dossiê

- [ ] Liga e aceita acesso por SSH
- [ ] Roda um modelo
- [ ] Este README foi seguido por outra pessoa do time sem ajuda (quem: <nome>)
- [ ] Pelo menos uma medição registrada em `runs/<nome-da-maquina>/`

## Especificações

| Item | Valor | De onde saiu |
|---|---|---|
| CPU | | |
| GPU, NPU ou acelerador | | |
| Memória (tipo e capacidade) | | |
| **Banda de memória teórica (GB/s)** | | <datasheet, link> |
| Armazenamento | | |
| Sistema operacional e versão | | `cat /etc/os-release` |
| Kernel | | `uname -r` |
| Alimentação e refrigeração | <fonte, cooler, dissipador> | |

A banda de memória é o número mais importante desta tabela: é o numerador da conta do time. Registre de onde você tirou.

## Acesso por SSH

<Como ligar, como descobrir o endereço na rede local, como conectar.>

```bash
ssh <usuario>@<nome-ou-endereco-local>
```

Não coloque aqui senha, IP público nem chave privada. Use autenticação por chave (`ssh-keygen` e `ssh-copy-id`).

## Do zero até o primeiro modelo

O passo a passo exato, com os comandos, na ordem. O teste é: alguém que nunca viu esta máquina segue isto sem te chamar?

1. <...>
2. <...>
3. <...>

## Medições

| Data | Modelo e formato | Ferramenta | Decode (tok/s) | Registro |
|---|---|---|---|---|
| | | | | [`runs/<nome-da-maquina>/...`](../runs/) |

## Previsão versus medição

Antes de medir, preveja com a conta do time:

```
tok/s previsto ≈ banda de memória (GB/s) ÷ tamanho do modelo em memória (GB)
```

| Modelo e formato | Tamanho (GB) | Previsto (tok/s) | Medido (tok/s) | Medido ÷ previsto |
|---|---|---|---|---|
| | | | | |

Se o medido ficou bem abaixo do previsto, o que consumiu a diferença? Essa é a pergunta da semana 7.

## Problemas encontrados

| O que aconteceu | Mensagem de erro | Como resolveu |
|---|---|---|
| | | |

Erro relatado vale tanto quanto sucesso: é o que vai poupar a próxima pessoa.
