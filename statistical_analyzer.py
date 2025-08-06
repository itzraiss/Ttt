"""
Módulo de Análise Estatística da Lotomania
Responsável por análises avançadas, detecção de padrões e estatísticas dos dados históricos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import logging

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LotomaniaStatisticalAnalyzer:
    """Classe para análise estatística avançada dos dados da Lotomania"""
    
    def __init__(self, data_source='lotomania_processed.json'):
        """
        Inicializa o analisador estatístico
        
        Args:
            data_source (str): Caminho para o arquivo de dados processados
        """
        self.data_source = data_source
        self.df = None
        self.frequency_analysis = {}
        self.pattern_analysis = {}
        self.trend_analysis = {}
        
    def load_data(self):
        """Carrega os dados processados"""
        try:
            if self.data_source.endswith('.json'):
                with open(self.data_source, 'r', encoding='utf-8') as f:
                    data_dict = json.load(f)
                self.df = pd.DataFrame(data_dict)
                # Converte coluna de data
                if 'data' in self.df.columns:
                    self.df['data'] = pd.to_datetime(self.df['data'])
            else:
                self.df = pd.read_csv(self.data_source)
                if 'data' in self.df.columns:
                    self.df['data'] = pd.to_datetime(self.df['data'])
            
            logger.info(f"Dados carregados: {len(self.df)} concursos")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {str(e)}")
            return False
    
    def analyze_number_frequency(self):
        """
        Analisa a frequência de cada número (0-99)
        
        Returns:
            dict: Análise completa de frequência
        """
        logger.info("Analisando frequência dos números...")
        
        # Coleta todos os números sorteados
        all_numbers = []
        for _, row in self.df.iterrows():
            if isinstance(row['numeros'], list):
                all_numbers.extend(row['numeros'])
            else:
                # Se estiver como string, converte
                numbers = eval(row['numeros']) if isinstance(row['numeros'], str) else []
                all_numbers.extend(numbers)
        
        # Calcula frequências
        frequency_count = Counter(all_numbers)
        total_draws = len(self.df)
        
        # Cria análise detalhada
        analysis = {
            'frequency_count': dict(frequency_count),
            'frequency_percentage': {num: (count/total_draws)*100 for num, count in frequency_count.items()},
            'total_draws': total_draws,
            'expected_frequency': total_draws * (20/100),  # 20 números em 100 possíveis
            'statistics': {
                'mean': np.mean(list(frequency_count.values())),
                'std': np.std(list(frequency_count.values())),
                'min': min(frequency_count.values()),
                'max': max(frequency_count.values())
            }
        }
        
        # Classifica números como quentes, frios e neutros
        mean_freq = analysis['statistics']['mean']
        std_freq = analysis['statistics']['std']
        
        hot_numbers = []
        cold_numbers = []
        neutral_numbers = []
        
        for num in range(100):
            freq = frequency_count.get(num, 0)
            if freq > mean_freq + std_freq:
                hot_numbers.append(num)
            elif freq < mean_freq - std_freq:
                cold_numbers.append(num)
            else:
                neutral_numbers.append(num)
        
        analysis['classification'] = {
            'hot_numbers': sorted(hot_numbers),
            'cold_numbers': sorted(cold_numbers),
            'neutral_numbers': sorted(neutral_numbers)
        }
        
        # Top e bottom números
        sorted_by_freq = sorted(frequency_count.items(), key=lambda x: x[1], reverse=True)
        analysis['top_10_numbers'] = sorted_by_freq[:10]
        analysis['bottom_10_numbers'] = sorted_by_freq[-10:]
        
        self.frequency_analysis = analysis
        logger.info(f"Análise de frequência concluída: {len(hot_numbers)} quentes, {len(cold_numbers)} frios, {len(neutral_numbers)} neutros")
        
        return analysis
    
    def analyze_decade_patterns(self):
        """
        Analisa padrões por décadas (0-9, 10-19, ..., 90-99)
        
        Returns:
            dict: Análise de padrões por década
        """
        logger.info("Analisando padrões por década...")
        
        decade_analysis = {}
        decade_counts = defaultdict(int)
        decade_numbers = defaultdict(list)
        
        # Mapeia números para décadas
        for _, row in self.df.iterrows():
            numbers = row['numeros'] if isinstance(row['numeros'], list) else eval(row['numeros'])
            
            contest_decades = defaultdict(int)
            for num in numbers:
                decade = num // 10
                contest_decades[decade] += 1
                decade_counts[decade] += 1
                decade_numbers[decade].append(num)
        
        # Calcula estatísticas por década
        for decade in range(10):
            decade_analysis[decade] = {
                'range': f"{decade*10}-{decade*10+9}",
                'total_occurrences': decade_counts[decade],
                'average_per_contest': decade_counts[decade] / len(self.df),
                'percentage': (decade_counts[decade] / (len(self.df) * 20)) * 100,
                'numbers_in_decade': sorted(set(decade_numbers[decade])),
                'most_common_in_decade': Counter(decade_numbers[decade]).most_common(3)
            }
        
        # Identifica padrões de distribuição
        total_by_decade = [decade_counts[i] for i in range(10)]
        decade_analysis['distribution_stats'] = {
            'mean': np.mean(total_by_decade),
            'std': np.std(total_by_decade),
            'most_active_decade': max(decade_counts.items(), key=lambda x: x[1]),
            'least_active_decade': min(decade_counts.items(), key=lambda x: x[1])
        }
        
        return decade_analysis
    
    def analyze_consecutive_patterns(self):
        """
        Analisa padrões de números consecutivos
        
        Returns:
            dict: Análise de consecutivos
        """
        logger.info("Analisando padrões de números consecutivos...")
        
        consecutive_patterns = {
            'pairs': defaultdict(int),
            'triplets': defaultdict(int),
            'sequences': defaultdict(int)
        }
        
        for _, row in self.df.iterrows():
            numbers = sorted(row['numeros'] if isinstance(row['numeros'], list) else eval(row['numeros']))
            
            # Analisa pares consecutivos
            for i in range(len(numbers)-1):
                if numbers[i+1] - numbers[i] == 1:
                    consecutive_patterns['pairs'][(numbers[i], numbers[i+1])] += 1
            
            # Analisa sequências de 3 ou mais números consecutivos
            consecutive_count = 1
            for i in range(len(numbers)-1):
                if numbers[i+1] - numbers[i] == 1:
                    consecutive_count += 1
                else:
                    if consecutive_count >= 3:
                        consecutive_patterns['sequences'][consecutive_count] += 1
                    consecutive_count = 1
            
            # Verifica última sequência
            if consecutive_count >= 3:
                consecutive_patterns['sequences'][consecutive_count] += 1
        
        # Converte para formato mais amigável
        analysis = {
            'most_common_pairs': sorted(consecutive_patterns['pairs'].items(), 
                                      key=lambda x: x[1], reverse=True)[:10],
            'sequence_frequency': dict(consecutive_patterns['sequences']),
            'total_consecutive_pairs': sum(consecutive_patterns['pairs'].values()),
            'total_sequences_3plus': sum(consecutive_patterns['sequences'].values()),
            'percentage_with_pairs': (len([row for _, row in self.df.iterrows() 
                                         if self._has_consecutive_pair(row['numeros'])]) / len(self.df)) * 100
        }
        
        return analysis
    
    def _has_consecutive_pair(self, numbers):
        """Verifica se uma lista de números tem pelo menos um par consecutivo"""
        if isinstance(numbers, str):
            numbers = eval(numbers)
        numbers = sorted(numbers)
        for i in range(len(numbers)-1):
            if numbers[i+1] - numbers[i] == 1:
                return True
        return False
    
    def analyze_temporal_trends(self):
        """
        Analisa tendências temporais nos sorteios
        
        Returns:
            dict: Análise de tendências temporais
        """
        logger.info("Analisando tendências temporais...")
        
        if 'data' not in self.df.columns:
            logger.warning("Coluna de data não encontrada. Pulando análise temporal.")
            return {}
        
        # Análise por ano
        self.df['ano'] = self.df['data'].dt.year
        self.df['mes'] = self.df['data'].dt.month
        self.df['dia_semana'] = self.df['data'].dt.dayofweek
        
        trends = {}
        
        # Frequência por ano
        trends['by_year'] = {}
        for year in sorted(self.df['ano'].unique()):
            year_data = self.df[self.df['ano'] == year]
            year_numbers = []
            for _, row in year_data.iterrows():
                numbers = row['numeros'] if isinstance(row['numeros'], list) else eval(row['numeros'])
                year_numbers.extend(numbers)
            
            year_freq = Counter(year_numbers)
            trends['by_year'][year] = {
                'contests': len(year_data),
                'most_common': year_freq.most_common(5),
                'unique_numbers': len(set(year_numbers))
            }
        
        # Análise por dia da semana
        trends['by_weekday'] = {}
        for dow in range(7):
            dow_data = self.df[self.df['dia_semana'] == dow]
            if len(dow_data) > 0:
                dow_numbers = []
                for _, row in dow_data.iterrows():
                    numbers = row['numeros'] if isinstance(row['numeros'], list) else eval(row['numeros'])
                    dow_numbers.extend(numbers)
                
                trends['by_weekday'][dow] = {
                    'contests': len(dow_data),
                    'most_common': Counter(dow_numbers).most_common(5)
                }
        
        return trends
    
    def analyze_gap_patterns(self):
        """
        Analisa padrões de intervalos entre sorteios dos números
        
        Returns:
            dict: Análise de gaps/intervalos
        """
        logger.info("Analisando padrões de intervalos...")
        
        # Para cada número, calcula o intervalo entre aparições
        number_gaps = defaultdict(list)
        last_appearance = defaultdict(int)
        
        for idx, row in self.df.iterrows():
            contest_num = row['concurso']
            numbers = row['numeros'] if isinstance(row['numeros'], list) else eval(row['numeros'])
            
            for num in numbers:
                if num in last_appearance:
                    gap = contest_num - last_appearance[num]
                    number_gaps[num].append(gap)
                last_appearance[num] = contest_num
        
        # Calcula estatísticas de gaps
        gap_analysis = {}
        for num in range(100):
            if num in number_gaps and len(number_gaps[num]) > 0:
                gaps = number_gaps[num]
                gap_analysis[num] = {
                    'mean_gap': np.mean(gaps),
                    'std_gap': np.std(gaps),
                    'min_gap': min(gaps),
                    'max_gap': max(gaps),
                    'total_gaps': len(gaps),
                    'current_gap': len(self.df) - last_appearance.get(num, 0)
                }
        
        return gap_analysis
    
    def detect_anomalies(self):
        """
        Detecta anomalias e padrões raros nos sorteios
        
        Returns:
            dict: Análise de anomalias
        """
        logger.info("Detectando anomalias e padrões raros...")
        
        anomalies = {
            'unusual_patterns': [],
            'extreme_sequences': [],
            'rare_combinations': []
        }
        
        for idx, row in self.df.iterrows():
            numbers = sorted(row['numeros'] if isinstance(row['numeros'], list) else eval(row['numeros']))
            contest = row['concurso']
            
            # Detecta sequências muito longas
            max_consecutive = self._find_max_consecutive_sequence(numbers)
            if max_consecutive >= 5:
                anomalies['extreme_sequences'].append({
                    'contest': contest,
                    'consecutive_length': max_consecutive,
                    'numbers': numbers
                })
            
            # Detecta concentração em poucas décadas
            decades = [num // 10 for num in numbers]
            unique_decades = len(set(decades))
            if unique_decades <= 3:
                anomalies['unusual_patterns'].append({
                    'contest': contest,
                    'pattern': f'concentrated_in_{unique_decades}_decades',
                    'decades': sorted(set(decades)),
                    'numbers': numbers
                })
            
            # Detecta números muito próximos (dentro de range pequeno)
            number_range = max(numbers) - min(numbers)
            if number_range <= 30:
                anomalies['rare_combinations'].append({
                    'contest': contest,
                    'pattern': 'narrow_range',
                    'range': number_range,
                    'numbers': numbers
                })
        
        return anomalies
    
    def _find_max_consecutive_sequence(self, numbers):
        """Encontra a maior sequência consecutiva em uma lista de números"""
        if not numbers:
            return 0
        
        max_length = 1
        current_length = 1
        
        for i in range(1, len(numbers)):
            if numbers[i] - numbers[i-1] == 1:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 1
        
        return max_length
    
    def generate_comprehensive_report(self):
        """
        Gera relatório estatístico completo
        
        Returns:
            dict: Relatório completo de análises
        """
        logger.info("Gerando relatório estatístico completo...")
        
        if self.df is None:
            self.load_data()
        
        report = {
            'metadata': {
                'total_contests': len(self.df),
                'analysis_date': datetime.now().isoformat(),
                'data_range': {
                    'start': self.df['data'].min().isoformat() if 'data' in self.df.columns else None,
                    'end': self.df['data'].max().isoformat() if 'data' in self.df.columns else None
                }
            },
            'frequency_analysis': self.analyze_number_frequency(),
            'decade_patterns': self.analyze_decade_patterns(),
            'consecutive_patterns': self.analyze_consecutive_patterns(),
            'temporal_trends': self.analyze_temporal_trends(),
            'gap_analysis': self.analyze_gap_patterns(),
            'anomalies': self.detect_anomalies()
        }
        
        # Converte numpy types para tipos Python nativos antes de salvar
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        # Salva relatório
        with open('statistical_report.json', 'w', encoding='utf-8') as f:
            json.dump(convert_types(report), f, ensure_ascii=False, indent=2, default=str)
        
        logger.info("Relatório estatístico salvo em 'statistical_report.json'")
        
        return report
    
    def get_hot_cold_neutral_numbers(self):
        """
        Retorna classificação atual dos números (quentes, frios, neutros)
        
        Returns:
            dict: Classificação dos números
        """
        if not self.frequency_analysis:
            self.analyze_number_frequency()
        
        return self.frequency_analysis.get('classification', {})
    
    def predict_next_hot_numbers(self, top_n=10):
        """
        Prevê os números mais prováveis baseado na análise estatística
        
        Args:
            top_n (int): Número de números quentes para retornar
            
        Returns:
            list: Lista dos números mais prováveis
        """
        if not self.frequency_analysis:
            self.analyze_number_frequency()
        
        classification = self.frequency_analysis.get('classification', {})
        hot_numbers = classification.get('hot_numbers', [])
        
        # Se não há números quentes suficientes, inclui neutros
        if len(hot_numbers) < top_n:
            neutral_numbers = classification.get('neutral_numbers', [])
            hot_numbers.extend(neutral_numbers[:top_n - len(hot_numbers)])
        
        return hot_numbers[:top_n]

def main():
    """Função principal para demonstrar o analisador estatístico"""
    
    # Inicializa o analisador
    analyzer = LotomaniaStatisticalAnalyzer()
    
    # Carrega dados
    if analyzer.load_data():
        print("=== ANÁLISE ESTATÍSTICA COMPLETA DA LOTOMANIA ===\n")
        
        # Gera relatório completo
        report = analyzer.generate_comprehensive_report()
        
        # Mostra resumo dos resultados
        print("FREQUÊNCIA DOS NÚMEROS:")
        freq_analysis = report['frequency_analysis']
        print(f"Números quentes: {freq_analysis['classification']['hot_numbers']}")
        print(f"Números frios: {freq_analysis['classification']['cold_numbers']}")
        print(f"Top 10 mais sorteados: {[f'{num}({count})' for num, count in freq_analysis['top_10_numbers']]}")
        
        print("\nPADRÕES POR DÉCADA:")
        decade_analysis = report['decade_patterns']
        most_active = decade_analysis['distribution_stats']['most_active_decade']
        print(f"Década mais ativa: {most_active[0]*10}-{most_active[0]*10+9} ({most_active[1]} ocorrências)")
        
        print("\nPADRÕES CONSECUTIVOS:")
        consec_analysis = report['consecutive_patterns']
        print(f"Pares consecutivos mais comuns: {consec_analysis['most_common_pairs'][:5]}")
        print(f"Porcentagem de sorteios com pares consecutivos: {consec_analysis['percentage_with_pairs']:.2f}%")
        
        print("\nANOMALIAS DETECTADAS:")
        anomalies = report['anomalies']
        print(f"Sequências extremas encontradas: {len(anomalies['extreme_sequences'])}")
        print(f"Padrões incomuns: {len(anomalies['unusual_patterns'])}")
        
        print("\nPREDIÇÃO - NÚMEROS QUENTES PARA PRÓXIMO SORTEIO:")
        hot_predictions = analyzer.predict_next_hot_numbers(15)
        print(f"Top 15 números recomendados: {hot_predictions}")
        
        print(f"\nRelatório completo salvo em 'statistical_report.json'")

if __name__ == "__main__":
    main()