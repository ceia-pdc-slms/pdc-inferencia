import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# nome do modelo 
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

def main():
    print(f"Carregando o modelo: {model_name}")
    
    # 1. Carregar o Tokenizer (responsável por transformar o texto humano em tokens)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Configurar o dispositivo (GPU ou CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo selecionado: {device}")
    
    # 3. Carregar o Modelo
    # Baixa e carrega os pesos pré-treinados. Usar bfloat16 ou float16 (se suportado) para economizar memória na GPU.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    
    # Tamanho do modelo em GB na memória
    model_size_bytes = model.get_memory_footprint()
    model_size_gb = model_size_bytes / (1024 ** 3)
    print(f"Tamanho aproximado do modelo na memória: {model_size_gb:.2f} GB")

    # 4. Preparar o texto de entrada (Prompt)
    prompt = "Escreva um poema curto sobre a beleza da programação e dos códigos."
    messages = [
        {"role": "system", "content": "Você é um assistente prestativo."},
        {"role": "user", "content": prompt}
    ]
    # Aplicar o template de chat específico do modelo (formato que o modelo espera)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Transforma o texto final em tokens e enviamos para o dispositivo correto (GPU/CPU)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 5. Geração e Medição de Tempo
    print("\nGerando resposta...")
    start_time = time.time()
    
    # O modelo começa a prever as próximas palavras
    # max_new_tokens limita a quantidade de tokens gerados
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256
    )
    
    end_time = time.time()
    
    # 6. Processar os Resultados
    # Apenas pega a parte da resposta gerada, ignorando o que enviei de entrada
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    # Transforma os tokens de volta em texto legível
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Calcular os Tokens por Segundo (velocidade de geração)
    num_tokens = len(generated_ids[0])
    time_taken = end_time - start_time
    tokens_per_second = num_tokens / time_taken
    
    print("\n--- RESPOSTA ---")
    print(response)
    print("----------------")
    
    print("\n--- MÉTRICAS ---")
    print(f"Tempo total de geração: {time_taken:.2f} segundos")
    print(f"Tokens gerados: {num_tokens}")
    print(f"Velocidade: {tokens_per_second:.2f} tokens/segundo")
    
    # Salvar as métricas em um arquivo de texto
    with open("metricas.txt", "w", encoding="utf-8") as f:
        f.write(f"Modelo: {model_name}\n")
        f.write(f"Tamanho na memoria: {model_size_gb:.2f} GB\n")
        f.write(f"Tempo total de geracao: {time_taken:.2f} segundos\n")
        f.write(f"Tokens gerados: {num_tokens}\n")
        f.write(f"Velocidade: {tokens_per_second:.2f} tokens/segundo\n")
    print("\n[Métricas salvas com sucesso no arquivo metricas.txt]")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERRO ENCONTRADO]: {e}")
