"""
Sistema de Simulação Retroativa (Backtesting) para Lotomania
Testa estratégias de predição em concursos passados para validar eficácia
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from collections import Counter, defaultdict
import logging
from statistical_analyzer import LotomaniaStatisticalAnalyzer

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LotomaniaBacktester:
    """Classe para simulação retroativa de estratégias de predição"""
    
    def __init__(self, data_source='lotomania_processed.json'):
        """
        Inicializa o sistema de backtesting
        
        Args:
            data_source (str): Fonte dos dados históricos
        """
        self.data_source = data_source
        self.df = None
        self.results = {}
        
    def load_data(self):
        """Carrega dados históricos"""
        try:
            if self.data_source.endswith('.json'):
                with open(self.data_source, 'r', encoding='utf-8') as f:
                    data_dict = json.load(f)
                self.df = pd.DataFrame(data_dict)
            else:
                self.df = pd.read_csv(self.data_source)
            
            if 'data' in self.df.columns:
                self.df['data'] = pd.to_datetime(self.df['data'])
            
            logger.info(f"Dados carregados para backtesting: {len(self.df)} concursos")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {str(e)}")
            return False
    
    def frequency_based_strategy(self, train_data, num_predictions=20):
        """
        Estratégia baseada em frequência histórica
        
        Args:
            train_data (pd.DataFrame): Dados de treino
            num_predictions (int): Número de números para predizer
            
        Returns:
            list: Números preditos
        """
        all_numbers = []
        for _, row in train_data.iterrows():
            numbers = row['numeros'] if isinstance(row['numeros'], list) else eval(row['numeros'])
            all_numbers.extend(numbers)
        
        # Conta frequências
        frequency_count = Counter(all_numbers)
        
        # Retorna os mais frequentes
        most_common = frequency_count.most_common(num_predictions)
        return [num for num, _ in most_common]
    
    def hot_cold_strategy(self, train_data, num_predictions=20):
        """
        Estratégia baseada em números quentes e frios
        
        Args:
            train_data (pd.DataFrame): Dados de treino
            num_predictions (int): Número de números para predizer
            
        Returns:
            list: Números preditos
        """
        # Usa o analisador estatístico para identificar números quentes
        analyzer = LotomaniaStatisticalAnalyzer()
        analyzer.df = train_data
        
        freq_analysis = analyzer.analyze_number_frequency()
        hot_numbers = freq_analysis['classification']['hot_numbers']
        neutral_numbers = freq_analysis['classification']['neutral_numbers']
        
        # Combina números quentes e neutros
        predicted_numbers = hot_numbers[:num_predictions]
        
        # Se não há números quentes suficientes, adiciona neutros
        if len(predicted_numbers) < num_predictions:
            predicted_numbers.extend(neutral_numbers[:num_predictions - len(predicted_numbers)])
        
        return predicted_numbers[:num_predictions]
    
    def gap_based_strategy(self, train_data, num_predictions=20):
        """
        Estratégia baseada em gaps (intervalos desde última aparição)
        
        Args:
            train_data (pd.DataFrame): Dados de treino
            num_predictions (int): Número de números para predizer
            
        Returns:
            list: Números preditos
        """
        # Calcula gap atual para cada número
        last_appearance = {}
        current_gaps = {}
        
        for idx, row in train_data.iterrows():
            contest = row['concurso']
            numbers = row['numeros'] if isinstance(row['numeros'], list) else eval(row['numeros'])
            
            for num in numbers:
                last_appearance[num] = contest
        
        # Calcula gaps atuais
        last_contest = train_data['concurso'].max()
        for num in range(100):
            if num in last_appearance:
                current_gaps[num] = last_contest - last_appearance[num]
            else:
                current_gaps[num] = len(train_data)  # Nunca apareceu
        
        # Ordena por gap (números que estão há mais tempo sem aparecer)
        sorted_by_gap = sorted(current_gaps.items(), key=lambda x: x[1], reverse=True)
        
        return [num for num, _ in sorted_by_gap[:num_predictions]]
    
    def decade_balance_strategy(self, train_data, num_predictions=20):
        """
        Estratégia que balanceia números por décadas
        
        Args:
            train_data (pd.DataFrame): Dados de treino
            num_predictions (int): Número de números para predizer
            
        Returns:
            list: Números preditos
        """
        # Calcula frequência por década nos dados de treino
        decade_counts = defaultdict(int)
        number_counts = Counter()
        
        for _, row in train_data.iterrows():
            numbers = row['numeros'] if isinstance(row['numeros'], list) else eval(row['numeros'])
            for num in numbers:
                decade_counts[num // 10] += 1
                number_counts[num] += 1
        
        # Seleciona números balanceando décadas
        predicted_numbers = []
        numbers_per_decade = num_predictions // 10
        remainder = num_predictions % 10
        
        for decade in range(10):
            # Números desta década ordenados por frequência
            decade_numbers = [(num, count) for num, count in number_counts.items() 
                            if num // 10 == decade]
            decade_numbers.sort(key=lambda x: x[1], reverse=True)
            
            # Adiciona números desta década
            quota = numbers_per_decade + (1 if decade < remainder else 0)
            for i in range(min(quota, len(decade_numbers))):
                predicted_numbers.append(decade_numbers[i][0])
        
        return predicted_numbers[:num_predictions]
    
    def combined_strategy(self, train_data, num_predictions=20):
        """
        Estratégia combinada que usa múltiplas abordagens
        
        Args:
            train_data (pd.DataFrame): Dados de treino
            num_predictions (int): Número de números para predizer
            
        Returns:
            list: Números preditos
        """
        # Executa todas as estratégias
        freq_pred = self.frequency_based_strategy(train_data, num_predictions)
        hot_cold_pred = self.hot_cold_strategy(train_data, num_predictions)
        gap_pred = self.gap_based_strategy(train_data, num_predictions)
        decade_pred = self.decade_balance_strategy(train_data, num_predictions)
        
        # Conta votos de cada número
        vote_counts = Counter()
        for predictions in [freq_pred, hot_cold_pred, gap_pred, decade_pred]:
            for num in predictions:
                vote_counts[num] += 1
        
        # Retorna números com mais votos
        most_voted = vote_counts.most_common(num_predictions)
        
        # Se não há votos suficientes, completa com frequência
        result = [num for num, _ in most_voted]
        if len(result) < num_predictions:
            freq_pred_set = set(freq_pred)
            for num in freq_pred:
                if num not in result:
                    result.append(num)
                    if len(result) >= num_predictions:
                        break
        
        return result[:num_predictions]
    
    def evaluate_prediction(self, predicted_numbers, actual_numbers):
        """
        Avalia uma predição comparando com números reais
        
        Args:
            predicted_numbers (list): Números preditos
            actual_numbers (list): Números reais sorteados
            
        Returns:
            dict: Métricas de avaliação
        """
        if isinstance(actual_numbers, str):
            actual_numbers = eval(actual_numbers)
        
        predicted_set = set(predicted_numbers)
        actual_set = set(actual_numbers)
        
        # Contagem de acertos
        hits = len(predicted_set.intersection(actual_set))
        
        # Precisão e recall
        precision = hits / len(predicted_set) if predicted_set else 0
        recall = hits / len(actual_set) if actual_set else 0
        
        # F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'hits': hits,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'predicted_count': len(predicted_set),
            'actual_count': len(actual_set)
        }
    
    def run_backtesting(self, start_contest=100, end_contest=None, window_size=50):
        """
        Executa backtesting completo
        
        Args:
            start_contest (int): Concurso inicial para teste
            end_contest (int): Concurso final para teste
            window_size (int): Tamanho da janela de dados históricos para treino
            
        Returns:
            dict: Resultados do backtesting
        """
        if not self.load_data():
            return None
        
        if end_contest is None:
            end_contest = self.df['concurso'].max()
        
        logger.info(f"Iniciando backtesting: concursos {start_contest} a {end_contest}")
        
        strategies = {
            'frequency': self.frequency_based_strategy,
            'hot_cold': self.hot_cold_strategy,
            'gap_based': self.gap_based_strategy,
            'decade_balance': self.decade_balance_strategy,
            'combined': self.combined_strategy
        }
        
        results = {strategy: [] for strategy in strategies}
        
        # Para cada concurso no período de teste
        for test_contest in range(start_contest, min(end_contest + 1, len(self.df))):
            # Dados de treino: janela antes do concurso de teste
            train_start = max(0, test_contest - window_size)
            train_data = self.df.iloc[train_start:test_contest]
            
            if len(train_data) < 10:  # Precisa de dados mínimos
                continue
            
            # Dados reais do concurso de teste
            test_row = self.df.iloc[test_contest]
            actual_numbers = test_row['numeros'] if isinstance(test_row['numeros'], list) else eval(test_row['numeros'])
            
            # Testa cada estratégia
            for strategy_name, strategy_func in strategies.items():
                try:
                    predicted_numbers = strategy_func(train_data)
                    evaluation = self.evaluate_prediction(predicted_numbers, actual_numbers)
                    
                    evaluation['contest'] = test_row['concurso']
                    evaluation['predicted_numbers'] = predicted_numbers
                    evaluation['actual_numbers'] = actual_numbers
                    
                    results[strategy_name].append(evaluation)
                    
                except Exception as e:
                    logger.warning(f"Erro na estratégia {strategy_name} para concurso {test_contest}: {str(e)}")
            
            # Progresso
            if test_contest % 100 == 0:
                logger.info(f"Progresso: concurso {test_contest}")
        
        # Calcula estatísticas finais
        final_results = {}
        for strategy_name, strategy_results in results.items():
            if strategy_results:
                hits_list = [r['hits'] for r in strategy_results]
                precision_list = [r['precision'] for r in strategy_results]
                recall_list = [r['recall'] for r in strategy_results]
                f1_list = [r['f1_score'] for r in strategy_results]
                
                final_results[strategy_name] = {
                    'total_tests': len(strategy_results),
                    'avg_hits': np.mean(hits_list),
                    'std_hits': np.std(hits_list),
                    'max_hits': max(hits_list),
                    'min_hits': min(hits_list),
                    'avg_precision': np.mean(precision_list),
                    'avg_recall': np.mean(recall_list),
                    'avg_f1_score': np.mean(f1_list),
                    'hit_distribution': Counter(hits_list),
                    'detailed_results': strategy_results[-10:]  # Últimos 10 resultados
                }
        
        # Salva resultados
        with open('backtesting_results.json', 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info("Backtesting concluído. Resultados salvos em 'backtesting_results.json'")
        
        return final_results
    
    def generate_performance_report(self, results):
        """
        Gera relatório de performance das estratégias
        
        Args:
            results (dict): Resultados do backtesting
            
        Returns:
            str: Relatório formatado
        """
        report = "=== RELATÓRIO DE BACKTESTING - LOTOMANIA ===\n\n"
        
        # Ordena estratégias por performance
        strategy_performance = [(name, data['avg_hits']) for name, data in results.items()]
        strategy_performance.sort(key=lambda x: x[1], reverse=True)
        
        for i, (strategy_name, avg_hits) in enumerate(strategy_performance):
            data = results[strategy_name]
            
            report += f"{i+1}. ESTRATÉGIA: {strategy_name.upper()}\n"
            report += f"   Testes realizados: {data['total_tests']}\n"
            report += f"   Acertos médios: {data['avg_hits']:.2f} ± {data['std_hits']:.2f}\n"
            report += f"   Máximo de acertos: {data['max_hits']}\n"
            report += f"   Mínimo de acertos: {data['min_hits']}\n"
            report += f"   Precisão média: {data['avg_precision']:.3f}\n"
            report += f"   F1 Score médio: {data['avg_f1_score']:.3f}\n"
            
            # Distribuição de acertos
            hit_dist = data['hit_distribution']
            report += f"   Distribuição de acertos:\n"
            for hits in sorted(hit_dist.keys()):
                percentage = (hit_dist[hits] / data['total_tests']) * 100
                report += f"     {hits} acertos: {hit_dist[hits]} vezes ({percentage:.1f}%)\n"
            
            report += "\n"
        
        return report

def main():
    """Função principal para demonstrar o backtesting"""
    
    # Inicializa o backtester
    backtester = LotomaniaBacktester()
    
    print("=== SISTEMA DE SIMULAÇÃO RETROATIVA (BACKTESTING) ===\n")
    print("Testando estratégias de predição em concursos históricos...\n")
    
    # Executa backtesting em uma amostra menor para demonstração
    print("⚠️  Executando backtesting em amostra (últimos 200 concursos)...")
    
    # Usa os últimos 200 concursos para teste
    end_contest = 2800  # Últimos dados disponíveis
    start_contest = 2600  # 200 concursos para teste
    
    results = backtester.run_backtesting(
        start_contest=start_contest, 
        end_contest=end_contest, 
        window_size=100
    )
    
    if results:
        print("✅ Backtesting concluído!\n")
        
        # Gera e exibe relatório
        report = backtester.generate_performance_report(results)
        print(report)
        
        # Salva relatório
        with open('performance_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("📊 Relatório detalhado salvo em 'performance_report.txt'")
        print("📁 Dados completos em 'backtesting_results.json'")
        
        # Melhor estratégia
        best_strategy = max(results.items(), key=lambda x: x[1]['avg_hits'])
        print(f"\n🏆 MELHOR ESTRATÉGIA: {best_strategy[0].upper()}")
        print(f"   Acertos médios: {best_strategy[1]['avg_hits']:.2f}")
        print(f"   Máximo conseguido: {best_strategy[1]['max_hits']} acertos")
        
    else:
        print("❌ Erro no backtesting")

if __name__ == "__main__":
    main()