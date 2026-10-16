# recipes

Receitas de compressão reproduzíveis: quantização, pruning, destilação. Uma pasta por receita, por exemplo `recipes/qwen2.5-0.5b_gguf-q4_k_m/`.

Cada receita tem um README com:

- **Entrada:** modelo de origem e link.
- **Ferramenta e versão:** ex.: llama.cpp no commit X, llm-compressor na versão Y.
- **Comandos exatos**, na ordem.
- **Saída:** formato, tamanho em disco e onde o artefato foi publicado (os pesos não entram no repositório).
- **Perda esperada e perda medida**, com link para o registro em `runs/` ou `quality/`.

Puxa o tema: Beatriz (receitas de compressão).
