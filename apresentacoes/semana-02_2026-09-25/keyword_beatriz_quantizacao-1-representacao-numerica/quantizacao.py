import math

def quantize_to_int8(value, scale, zero_point=0):
    """
    Quantiza um valor de ponto flutuante para INT8.
    INT8 vai de -128 a 127.
    """
    # 1. Escalar o valor
    scaled_value = value / scale
    
    # 2. Adicionar o zero-point
    shifted_value = scaled_value + zero_point
    
    # 3. Arredondar para o inteiro mais próximo
    rounded_value = round(shifted_value)
    
    # 4. Limitar (clip) aos valores possíveis do INT8 (-128 a 127)
    q_value = max(-128, min(127, rounded_value))
    
    return q_value

def dequantize(q_value, scale, zero_point=0):
    """
    Dequantiza um valor INT8 de volta para ponto flutuante.
    """
    return (q_value - zero_point) * scale

if __name__ == "__main__":
    print("="*40)
    print(" EXEMPLO PRÁTICO DE QUANTIZAÇÃO (INT8)")
    print("="*40)
    
    # Valores da Pergunta da Semana
    original_value_perfect = 3.7
    scale = 0.05
    zero_point = 0
    
    print("\n--- CASO 1: A PERGUNTA EXATA (3.7) ---")
    print(f"Valor original: {original_value_perfect}")
    print(f"Escala: {scale}")
    
    # Quantizando
    q_value = quantize_to_int8(original_value_perfect, scale, zero_point)
    print(f"Valor quantizado (armazenado em INT8): {q_value}")
    
    # Dequantizando para ver a perda
    recovered_value = dequantize(q_value, scale, zero_point)
    print(f"Valor recuperado (Dequantizado): {recovered_value}")
    
    # Calculando o erro
    erro = abs(original_value_perfect - recovered_value)
    print(f"Erro absoluto (Perda de Precisão): {erro:.4f}")
    
    print("\n--- CASO 2: DEMONSTRANDO O ERRO DE ARREDONDAMENTO (3.72) ---")
    original_value_imperfect = 3.72
    print(f"Valor original: {original_value_imperfect}")
    
    q_value_imp = quantize_to_int8(original_value_imperfect, scale, zero_point)
    print(f"Valor quantizado (armazenado em INT8): {q_value_imp}")
    
    recovered_value_imp = dequantize(q_value_imp, scale, zero_point)
    print(f"Valor recuperado (Dequantizado): {recovered_value_imp:.2f}")
    
    erro_imp = abs(original_value_imperfect - recovered_value_imp)
    print(f"Erro absoluto (Perda de Precisão): {erro_imp:.4f}")
    
    print("\n[Conclusão do Código]")
    print(f"No Caso 1 (3.7), o valor era perfeitamente divisível pela escala, resultando em erro zero.")
    print(f"No Caso 2 (3.72), o arredondamento 'cortou' as casas decimais extras, gerando um erro aceitável de {erro_imp:.4f}.")
