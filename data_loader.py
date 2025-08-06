"""
Módulo de Carregamento e Processamento de Dados da Lotomania
Responsável por carregar, limpar e estruturar os dados históricos da Lotomania
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import logging

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LotomaniaDataLoader:
    """Classe responsável pelo carregamento e processamento dos dados da Lotomania"""
    
    def __init__(self, excel_file_path='Lotomania.xlsx'):
        """
        Inicializa o carregador de dados
        
        Args:
            excel_file_path (str): Caminho para o arquivo Excel com dados históricos
        """
        self.excel_file_path = excel_file_path
        self.df = None
        self.clean_data = None
        
    def load_excel_data(self):
        """
        Carrega dados do arquivo Excel existente
        
        Returns:
            pd.DataFrame: DataFrame com dados carregados
        """
        try:
            logger.info(f"Carregando dados do arquivo: {self.excel_file_path}")
            
            # Tenta diferentes engines e métodos para ler o Excel
            engines_to_try = ['openpyxl', 'xlrd', 'xlrd2']
            
            for engine in engines_to_try:
                try:
                    logger.info(f"Tentando carregar com engine: {engine}")
                    if engine == 'xlrd2':
                        import xlrd2 as xlrd
                        # Para xlrd2, precisamos abrir manualmente
                        book = xlrd.open_workbook(self.excel_file_path)
                        sheet = book.sheet_by_index(0)
                        
                        # Converte para lista de listas
                        data = []
                        for row_idx in range(sheet.nrows):
                            row = []
                            for col_idx in range(sheet.ncols):
                                cell_value = sheet.cell_value(row_idx, col_idx)
                                row.append(cell_value)
                            data.append(row)
                        
                        # Primeira linha como header
                        if data:
                            headers = [f"Col_{i}" if not str(data[0][i]).strip() else str(data[0][i]) 
                                     for i in range(len(data[0]))]
                            self.df = pd.DataFrame(data[1:], columns=headers)
                        else:
                            continue
                    else:
                        self.df = pd.read_excel(self.excel_file_path, engine=engine)
                    
                    logger.info(f"Dados carregados com sucesso usando {engine}. Shape: {self.df.shape}")
                    logger.info(f"Colunas disponíveis: {list(self.df.columns)}")
                    break
                    
                except Exception as e:
                    logger.warning(f"Falha com engine {engine}: {str(e)}")
                    continue
            else:
                # Se nenhum engine funcionou, tenta como CSV
                logger.info("Tentando carregar como arquivo CSV...")
                try:
                    self.df = pd.read_csv(self.excel_file_path, sep=';', encoding='utf-8')
                except:
                    try:
                        self.df = pd.read_csv(self.excel_file_path, sep=',', encoding='latin-1')
                    except:
                        self.df = pd.read_csv(self.excel_file_path, sep='\t', encoding='utf-8')
                
                logger.info(f"Dados carregados como CSV. Shape: {self.df.shape}")
            
            if self.df is not None:
                return self.df
            else:
                raise Exception("Não foi possível carregar os dados com nenhum método")
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados do Excel: {str(e)}")
            return None
    
    def clean_and_structure_data(self):
        """
        Limpa e estrutura os dados carregados
        
        Returns:
            pd.DataFrame: DataFrame limpo e estruturado
        """
        if self.df is None:
            logger.error("Dados não carregados. Execute load_excel_data() primeiro.")
            return None
        
        try:
            logger.info("Iniciando limpeza e estruturação dos dados...")
            
            # Cria uma cópia para trabalhar
            clean_df = self.df.copy()
            
            # Identifica as colunas de números sorteados (assumindo que são as primeiras 20 colunas numéricas)
            number_columns = []
            
            # Procura por colunas que contenham números sorteados
            for col in clean_df.columns:
                # Se a coluna contém valores numéricos entre 0 e 100 (números da Lotomania)
                if clean_df[col].dtype in ['int64', 'float64']:
                    # Verifica se os valores estão no range da Lotomania (0-99)
                    if clean_df[col].min() >= 0 and clean_df[col].max() <= 99:
                        number_columns.append(col)
            
            # Se não encontrou colunas automáticamente, assume as primeiras 20
            if len(number_columns) < 20:
                number_columns = clean_df.columns[:20].tolist()
            
            # Garante que temos exatamente 20 colunas de números
            number_columns = number_columns[:20]
            
            logger.info(f"Colunas de números identificadas: {len(number_columns)}")
            
            # Cria estrutura padronizada
            structured_data = []
            
            for idx, row in clean_df.iterrows():
                # Extrai os números sorteados
                numbers = []
                for col in number_columns:
                    if pd.notna(row[col]):
                        numbers.append(int(row[col]))
                
                # Remove duplicatas e ordena
                numbers = sorted(list(set(numbers)))
                
                # Garante que temos exatamente 20 números
                if len(numbers) == 20:
                    contest_data = {
                        'concurso': idx + 1,  # Assume que o índice representa o concurso
                        'data': row.get('Data', None) if 'Data' in clean_df.columns else None,
                        'numeros': numbers,
                        'numeros_str': '-'.join([f"{n:02d}" for n in numbers])
                    }
                    
                    # Adiciona outras informações se disponíveis
                    if 'Arrecadacao' in clean_df.columns:
                        contest_data['arrecadacao'] = row.get('Arrecadacao', 0)
                    
                    if 'Ganhadores_20' in clean_df.columns:
                        contest_data['ganhadores_20'] = row.get('Ganhadores_20', 0)
                    
                    structured_data.append(contest_data)
            
            # Converte para DataFrame
            self.clean_data = pd.DataFrame(structured_data)
            
            # Converte coluna de data se existir
            if 'data' in self.clean_data.columns:
                self.clean_data['data'] = pd.to_datetime(self.clean_data['data'], errors='coerce')
            
            logger.info(f"Dados estruturados com sucesso. {len(self.clean_data)} concursos processados.")
            
            return self.clean_data
            
        except Exception as e:
            logger.error(f"Erro na limpeza e estruturação dos dados: {str(e)}")
            return None
    
    def scrape_latest_results(self, num_contests=10):
        """
        Realiza web scraping dos resultados mais recentes
        
        Args:
            num_contests (int): Número de concursos mais recentes para buscar
            
        Returns:
            list: Lista com dados dos concursos mais recentes
        """
        try:
            logger.info("Iniciando web scraping dos resultados mais recentes...")
            
            url = "https://loterias.caixa.gov.br/Paginas/Lotomania.aspx"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Esta é uma implementação básica - seria necessário ajustar baseado na estrutura real do site
            logger.warning("Web scraping não implementado completamente. Usando dados do Excel.")
            
            return []
            
        except Exception as e:
            logger.error(f"Erro no web scraping: {str(e)}")
            return []
    
    def save_processed_data(self, filename='lotomania_processed.json'):
        """
        Salva os dados processados em arquivo JSON
        
        Args:
            filename (str): Nome do arquivo para salvar os dados
        """
        if self.clean_data is None:
            logger.error("Dados não processados. Execute clean_and_structure_data() primeiro.")
            return False
        
        try:
            # Converte DataFrame para dicionário para salvar em JSON
            data_dict = self.clean_data.to_dict('records')
            
            # Converte datetime para string para serialização JSON
            for record in data_dict:
                if 'data' in record and pd.notna(record['data']):
                    record['data'] = record['data'].strftime('%Y-%m-%d')
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Dados salvos em: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar dados: {str(e)}")
            return False
    
    def load_processed_data(self, filename='lotomania_processed.json'):
        """
        Carrega dados processados de arquivo JSON
        
        Args:
            filename (str): Nome do arquivo para carregar os dados
            
        Returns:
            pd.DataFrame: DataFrame com dados carregados
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)
            
            self.clean_data = pd.DataFrame(data_dict)
            
            # Converte string de volta para datetime se existir
            if 'data' in self.clean_data.columns:
                self.clean_data['data'] = pd.to_datetime(self.clean_data['data'], errors='coerce')
            
            logger.info(f"Dados carregados de: {filename}")
            return self.clean_data
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados processados: {str(e)}")
            return None
    
    def get_data_summary(self):
        """
        Retorna resumo dos dados carregados
        
        Returns:
            dict: Dicionário com estatísticas dos dados
        """
        if self.clean_data is None:
            return None
        
        summary = {
            'total_contests': len(self.clean_data),
            'date_range': {
                'start': self.clean_data['data'].min() if 'data' in self.clean_data.columns else None,
                'end': self.clean_data['data'].max() if 'data' in self.clean_data.columns else None
            },
            'last_contest': self.clean_data['concurso'].max() if 'concurso' in self.clean_data.columns else None,
            'columns': list(self.clean_data.columns)
        }
        
        return summary

def main():
    """Função principal para testar o carregador de dados"""
    
    # Inicializa o carregador
    loader = LotomaniaDataLoader()
    
    # Carrega dados do Excel
    df = loader.load_excel_data()
    
    if df is not None:
        print(f"Dados carregados: {df.shape}")
        print(f"Primeiras 5 linhas:\n{df.head()}")
        
        # Limpa e estrutura os dados
        clean_df = loader.clean_and_structure_data()
        
        if clean_df is not None:
            print(f"\nDados limpos: {clean_df.shape}")
            print(f"Resumo dos dados:")
            summary = loader.get_data_summary()
            print(json.dumps(summary, indent=2, default=str))
            
            # Salva dados processados
            loader.save_processed_data()
            
            print("\nPrimeiros 3 concursos processados:")
            for i, row in clean_df.head(3).iterrows():
                print(f"Concurso {row['concurso']}: {row['numeros_str']}")

if __name__ == "__main__":
    main()