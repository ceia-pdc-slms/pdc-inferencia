import tiktoken

# Tokenizador 
encoding = tiktoken.get_encoding("o200k_base")

# Lista de tuplas contendo as frases em portugês e inglês
frases = [
    (
        "O gato está dormindo no sofá.",
        "The cat is sleeping on the couch."
    ),
    (
        "Eu estou aprendendo programação em Python.",
        "I am learning programming in Python."
    ),
    (
        "A inteligência artificial está transformando a tecnologia.",
        "Artificial intelligence is transforming technology."
    ),
    (
        "Precisamos analisar os dados antes de tomar uma decisão.",
        "We need to analyze the data before making a decision."
    ),
    (
        "O computador processou todas as informações rapidamente.",
        "The computer processed all the information quickly."
    ),
    (
        "Os modelos de linguagem conseguem gerar textos automaticamente.",
        "Language models can generate text automatically."
    ),
    (
        "Estamos desenvolvendo uma aplicação para análise de dados.",
        "We are developing an application for data analysis."
    ),
    (
        "A equipe precisa melhorar o fluxo de trabalho.",
        "The team needs to improve the workflow."
    ),
    (
        "Esse sistema utiliza inteligência artificial para identificar padrões.",
        "This system uses artificial intelligence to identify patterns."
    ),
    (
        "Quanto mais dados tivermos, melhor poderemos treinar o modelo.",
        "The more data we have, the better we can train the model."
    )
]

total_pt = 0
total_en = 0

for i, (pt, en) in enumerate(frases, start=1):

    tokens_pt = encoding.encode(pt)
    tokens_en = encoding.encode(en)

    total_pt += len(tokens_pt)
    total_en += len(tokens_en)

    diferenca = len(tokens_pt) - len(tokens_en)
    percentual = (diferenca / len(tokens_en)) * 100

    print(f"\n--- Frase {i} ---")
    print(f"PT: {pt}")
    print(f"EN: {en}")

    print(f"\nPortuguês: {len(tokens_pt)} tokens")
    print(f"Inglês:    {len(tokens_en)} tokens")
    print(f"Diferença: {diferenca:+} tokens ({percentual:+.1f}%)")


print("\n==============================")
print("RESULTADO FINAL")
print("==============================")

print(f"Português: {total_pt} tokens")
print(f"Inglês:    {total_en} tokens")

diferenca_total = total_pt - total_en
percentual_total = (diferenca_total / total_en) * 100

print(f"Diferença: {diferenca_total:+} tokens")
print(f"Português usa {percentual_total:.1f}% mais tokens")