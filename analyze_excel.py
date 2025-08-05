import pandas as pd
import numpy as np

# Primeiro, vamos tentar carregar com diferentes parâmetros
try:
    print("Tentando carregar com openpyxl sem especificar header...")
    df = pd.read_excel('Lotomania.xlsx', engine='openpyxl', header=None)
    print(f"Shape: {df.shape}")
    print("Primeiras 10 linhas:")
    print(df.head(10))
    print("\nÚltimas 5 linhas:")
    print(df.tail(5))
    print("\nInfo do DataFrame:")
    print(df.info())
    
    # Salva uma amostra para análise
    df.head(20).to_csv("sample_lotomania.csv", index=False)
    print("\nAmostra salva em sample_lotomania.csv")
    
except Exception as e:
    print(f"Erro: {e}")
    
    # Tenta com xlrd2
    try:
        import xlrd2 as xlrd
        print("\nTentando com xlrd2...")
        book = xlrd.open_workbook('Lotomania.xlsx')
        print(f"Número de planilhas: {book.nsheets}")
        
        for i, sheet_name in enumerate(book.sheet_names()):
            print(f"Planilha {i}: {sheet_name}")
            
        sheet = book.sheet_by_index(0)
        print(f"Número de linhas: {sheet.nrows}")
        print(f"Número de colunas: {sheet.ncols}")
        
        # Mostra primeiras linhas
        print("\nPrimeiras 5 linhas:")
        for row_idx in range(min(5, sheet.nrows)):
            row_data = []
            for col_idx in range(min(10, sheet.ncols)):  # Primeiras 10 colunas
                cell_value = sheet.cell_value(row_idx, col_idx)
                row_data.append(cell_value)
            print(f"Linha {row_idx}: {row_data}")
            
    except Exception as e2:
        print(f"Erro com xlrd2: {e2}")