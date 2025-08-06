import pandas as pd
import numpy as np

# Carrega com tratamento de valores especiais
try:
    print("Carregando Excel com tratamento de valores especiais...")
    
    # Usa na_values para tratar "-" como NaN
    df = pd.read_excel('Lotomania.xlsx', engine='openpyxl', header=None, na_values=['-', '', ' '])
    
    print(f"Shape: {df.shape}")
    print("Primeiras 10 linhas:")
    print(df.head(10))
    
    print("\nTipos de dados:")
    print(df.dtypes)
    
    # Identifica colunas que parecem conter números da Lotomania
    print("\nAnalisando colunas...")
    number_cols = []
    for col in df.columns:
        # Verifica se a coluna tem valores numéricos no range da Lotomania
        non_null_values = df[col].dropna()
        if len(non_null_values) > 0:
            try:
                # Tenta converter para numérico
                numeric_values = pd.to_numeric(non_null_values, errors='coerce').dropna()
                if len(numeric_values) > 0:
                    min_val = numeric_values.min()
                    max_val = numeric_values.max()
                    print(f"Coluna {col}: min={min_val}, max={max_val}, count={len(numeric_values)}")
                    
                    # Se está no range da Lotomania (0-99)
                    if min_val >= 0 and max_val <= 99:
                        number_cols.append(col)
            except:
                pass
    
    print(f"\nColunas identificadas como números da Lotomania: {number_cols}")
    
    # Salva uma amostra para análise manual
    df.head(20).to_csv("sample_with_headers.csv", index=False)
    print("\nAmostra salva em sample_with_headers.csv")
    
    # Tenta identificar o header
    print("\nPrimeira linha (possível header):")
    print(df.iloc[0].tolist())
    
except Exception as e:
    print(f"Erro: {e}")
    import traceback
    traceback.print_exc()