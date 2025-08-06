import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import random

def generate_lotomania_numbers():
    """Gera 20 números únicos entre 0 e 99 para um concurso da Lotomania"""
    return sorted(random.sample(range(0, 100), 20))

def create_sample_lotomania_data(num_contests=2805):
    """
    Cria dados de exemplo da Lotomania para demonstração do sistema
    
    Args:
        num_contests (int): Número de concursos para gerar
        
    Returns:
        pd.DataFrame: DataFrame com dados simulados
    """
    print(f"Gerando {num_contests} concursos simulados da Lotomania...")
    
    data = []
    start_date = datetime(2000, 1, 1)
    
    for concurso in range(1, num_contests + 1):
        # Data do concurso (aproximadamente 3 vezes por semana)
        days_offset = (concurso - 1) * 2.3  # Aproximadamente 3 sorteios por semana
        contest_date = start_date + timedelta(days=int(days_offset))
        
        # Gera números sorteados
        numbers = generate_lotomania_numbers()
        
        # Simula dados adicionais
        arrecadacao = random.randint(5000000, 20000000)  # Arrecadação em centavos
        ganhadores_20 = random.randint(0, 3)  # Ganhadores com 20 acertos
        ganhadores_19 = random.randint(5, 50)  # Ganhadores com 19 acertos
        ganhadores_18 = random.randint(50, 300)  # Ganhadores com 18 acertos
        ganhadores_17 = random.randint(500, 2000)  # Ganhadores com 17 acertos
        ganhadores_16 = random.randint(5000, 15000)  # Ganhadores com 16 acertos
        ganhadores_0 = random.randint(2, 10)  # Ganhadores com 0 acertos
        
        contest_data = {
            'concurso': concurso,
            'data': contest_date,
            'numeros': numbers,
            'numeros_str': '-'.join([f"{n:02d}" for n in numbers]),
            'arrecadacao': arrecadacao,
            'ganhadores_20': ganhadores_20,
            'ganhadores_19': ganhadores_19,
            'ganhadores_18': ganhadores_18,
            'ganhadores_17': ganhadores_17,
            'ganhadores_16': ganhadores_16,
            'ganhadores_0': ganhadores_0,
            # Adiciona cada número em uma coluna separada para compatibilidade
            'n1': numbers[0], 'n2': numbers[1], 'n3': numbers[2], 'n4': numbers[3],
            'n5': numbers[4], 'n6': numbers[5], 'n7': numbers[6], 'n8': numbers[7],
            'n9': numbers[8], 'n10': numbers[9], 'n11': numbers[10], 'n12': numbers[11],
            'n13': numbers[12], 'n14': numbers[13], 'n15': numbers[14], 'n16': numbers[15],
            'n17': numbers[16], 'n18': numbers[17], 'n19': numbers[18], 'n20': numbers[19]
        }
        
        data.append(contest_data)
        
        # Progresso
        if concurso % 500 == 0:
            print(f"Gerados {concurso} concursos...")
    
    df = pd.DataFrame(data)
    
    # Salva em diferentes formatos
    df.to_csv('lotomania_sample.csv', index=False)
    df.to_excel('lotomania_sample.xlsx', index=False)
    
    # Salva também como JSON processado
    data_dict = df.to_dict('records')
    for record in data_dict:
        record['data'] = record['data'].strftime('%Y-%m-%d')
    
    with open('lotomania_processed.json', 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)
    
    print(f"Dados salvos em:")
    print("- lotomania_sample.csv")
    print("- lotomania_sample.xlsx") 
    print("- lotomania_processed.json")
    
    return df

def analyze_sample_data(df):
    """Analisa os dados de exemplo gerados"""
    print("\n=== ANÁLISE DOS DADOS DE EXEMPLO ===")
    print(f"Total de concursos: {len(df)}")
    print(f"Período: {df['data'].min()} a {df['data'].max()}")
    
    # Análise de frequência dos números
    all_numbers = []
    for _, row in df.iterrows():
        all_numbers.extend(row['numeros'])
    
    frequency = pd.Series(all_numbers).value_counts().sort_index()
    
    print(f"\nNúmeros mais sorteados:")
    print(frequency.head(10))
    
    print(f"\nNúmeros menos sorteados:")
    print(frequency.tail(10))
    
    print(f"\nEstatísticas de frequência:")
    print(f"Média: {frequency.mean():.2f}")
    print(f"Desvio padrão: {frequency.std():.2f}")
    print(f"Mínimo: {frequency.min()}")
    print(f"Máximo: {frequency.max()}")
    
    return frequency

if __name__ == "__main__":
    # Gera dados de exemplo
    df = create_sample_lotomania_data(2805)
    
    # Analisa os dados
    frequency = analyze_sample_data(df)
    
    print("\nDados de exemplo criados com sucesso!")
    print("Agora você pode testar o sistema completo da Lotomania.")