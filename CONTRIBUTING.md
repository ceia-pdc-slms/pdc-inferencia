# Como contribuir

Este guia assume que você nunca usou Git. Se você já usa, leia só **Fluxo de toda contribuição** e **Regras**.

## Configuração (uma vez só)

### 1. Instale o Git

- **Linux, Raspberry, Jetson:** `sudo apt install git`
- **macOS:** `xcode-select --install`
- **Windows:** instale o [Git for Windows](https://git-scm.com/download/win) e use o *Git Bash* para os comandos abaixo.

### 2. Diga ao Git quem você é

```bash
git config --global user.name "Seu Nome"
git config --global user.email "o-email-da-sua-conta-github@exemplo.com"
```

### 3. Autentique no GitHub

O jeito mais simples é o [GitHub CLI](https://cli.github.com):

```bash
gh auth login
```

Escolha *GitHub.com → HTTPS → Login with a web browser* e siga o que aparecer.

### 4. Clone o repositório

```bash
gh repo clone ceia-pdc-slms/pdc-inferencia
cd pdc-inferencia
```

## Fluxo de toda contribuição

Toda mudança segue o mesmo caminho, sem exceção:

```bash
# 1. Comece da versão mais recente
git switch main
git pull

# 2. Crie um branch para o seu trabalho
git switch -c seunome/assunto-curto        # ex.: pedro/dossie-rpi5

# 3. Edite os arquivos. Depois veja o que mudou
git status
git diff

# 4. Registre a mudança (commit)
git add caminho/do/arquivo
git commit -m "Adiciona dossiê da Raspberry Pi 5"

# 5. Envie o branch para o GitHub
git push -u origin seunome/assunto-curto

# 6. Abra o pull request
gh pr create --fill                        # ou pelo link que aparece depois do push
```

Depois do PR aberto:

- Peça revisão para pelo menos uma pessoa (no PR, em *Reviewers*).
- Se pedirem mudança, edite no mesmo branch, faça um novo commit e `git push`. O PR se atualiza sozinho.
- Quando aprovado, quem abriu faz o merge com **Squash and merge**.

## Seu primeiro PR: o dossiê da sua máquina

```bash
git switch main && git pull
git switch -c seunome/dossie-maquina
cp maquinas/_modelo.md maquinas/NOME-DA-MAQUINA.md
# preencha o arquivo; o que você ainda não sabe, deixe marcado como PENDENTE
git add maquinas/NOME-DA-MAQUINA.md
git commit -m "Adiciona dossiê da NOME-DA-MAQUINA"
git push -u origin seunome/dossie-maquina
gh pr create --fill
```

Use um nome curto para a máquina, sem espaço nem acento, que diga o que ela é: `rpi5-01`, `jetson-orin-nano`, `spark`. **O nome é da máquina, não da pessoa:** a máquina pode trocar de mãos e o histórico continua.

O dossiê não precisa estar completo para abrir o PR. Um PR com metade preenchida e as pendências marcadas é melhor do que nenhum PR.

## Subindo a sua apresentação

Toda apresentação de sexta entra em [`apresentacoes/`](apresentacoes/), numa pasta da semana e, dentro dela, numa pasta sua:

```
apresentacoes/semana-05_2026-10-16/keyword_marcos_compressao-1-as-quatro-familias/
```

O nome exato da sua pasta já está no README da semana. Coloque nela:

- os slides **em PDF** (abre direto no GitHub, sem baixar);
- se quiser, o original (`.pptx`) ou o link do Canva/Google Slides num `README.md`;
- notebooks e código que você mostrou.

Nomes de arquivo sem espaço e sem acento: `compressao-quatro-familias.pdf`, não `Compressão (final) v2.pdf`. Arquivo acima de 25 MB não sobe pelo navegador; exporte o PDF com imagens comprimidas.

### Pelo terminal

```bash
git switch main && git pull
git switch -c marcos/semana-05-compressao
mkdir -p apresentacoes/semana-05_2026-10-16/keyword_marcos_compressao-1-as-quatro-familias
# copie os arquivos para a pasta
git add apresentacoes/semana-05_2026-10-16/
git commit -m "Adiciona slides da keyword de compressão (semana 5)"
git push -u origin marcos/semana-05-compressao
gh pr create --fill
```

### Pelo navegador (sem instalar nada)

1. Abra a pasta da semana no GitHub, por exemplo `apresentacoes/semana-05_2026-10-16/`.
2. **Add file → Upload files**. Arraste a sua **pasta** inteira (já com o nome certo) para a área de upload: o GitHub mantém o nome da pasta.
3. Embaixo, escolha **Create a new branch for this commit and start a pull request**, escreva a mensagem ("Adiciona slides da keyword de compressão") e confirme.
4. Na tela seguinte, **Create pull request**.

Depois de aprovado, atualize a linha da sua apresentação no README da semana e no [`apresentacoes/README.md`](apresentacoes/README.md) trocando *sem material* pelos links. Se esquecer, quem revisar faz.

## Registrando uma medição

Medições feitas pela bancada (`testbench/`) já saem com procedência em `testbench/resultados/`. Para as feitas à mão (Ollama, llama-bench, LM Studio), veja [`runs/README.md`](runs/README.md). Em resumo: copie `runs/_modelo.yaml` para `runs/<nome-da-maquina>/<data>_<modelo>_<formato>.yaml`, preencha e abra um PR. Uma verificação automática confere a procedência; se faltar algo, o PR mostra exatamente o quê.

## Regras

1. **Nunca faça commit direto na `main`.** Toda mudança entra por PR.
2. **Nunca suba pesos de modelo, datasets grandes, senhas, tokens, chaves ou IPs.** Se subiu por engano, avise no Discord na hora: apagar num commit novo não basta, porque continua no histórico.
3. **Commits pequenos, mensagem no imperativo e em português:** "Adiciona dossiê da Jetson", "Corrige comando de medição". Não "update", nem "ajustes".
4. **PRs pequenos.** Um assunto por PR. Fica mais fácil revisar e mais fácil aprender.
5. **Revisar é parte do trabalho.** A pergunta principal da revisão é: *eu conseguiria repetir isso sem perguntar nada a quem fez?* Se não, comente o que faltou.

## Travou?

Mande no canal de dúvidas do Discord no mesmo dia, com o comando que você rodou e a mensagem de erro completa (copie o texto; print só quando não der). Travar é normal. Travar em silêncio é o único problema.

### Erros comuns

| Erro | O que fazer |
|---|---|
| `Permission denied` ou `403` no push | Rode `gh auth status`. Se estiver logado e mesmo assim falhar, peça ao Hugo para conferir se você está no time no GitHub |
| `Updates were rejected because the remote contains work...` | `git pull --rebase` no seu branch e `git push` de novo |
| `CONFLICT` depois de um pull | Abra o arquivo, procure os marcadores `<<<<<<<`, escolha o que fica, apague os marcadores, depois `git add` no arquivo e `git rebase --continue` |
| Fiz commit na `main` sem querer (ainda sem push) | `git switch -c seunome/assunto` leva o commit para um branch novo. Depois, se não houver nada sem salvar: `git switch main && git reset --hard origin/main` |
