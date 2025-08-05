"""
Sistema de Otimização de Apostas para Lotomania
Calcula estratégias ótimas de apostas baseadas em orçamento, risco e análise estatística
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from collections import Counter
import logging
from itertools import combinations

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LotomaniaBetOptimizer:
    """Classe para otimização de estratégias de apostas"""
    
    def __init__(self, data_source='lotomania_processed.json'):
        """
        Inicializa o otimizador de apostas
        
        Args:
            data_source (str): Fonte dos dados históricos
        """
        self.data_source = data_source
        self.df = None
        self.statistical_data = None
        self.cost_per_game = 3.0  # Custo padrão por jogo
        
        # Probabilidades calculadas estatisticamente
        self.hit_probabilities = {
            20: 1 / 11372635,  # Probabilidade de acertar 20 números
            19: 1 / 352038,
            18: 1 / 17158,
            17: 1 / 1292,
            16: 1 / 129,
            15: 1 / 17,
            0: 1 / 11372  # Probabilidade de acertar 0 números
        }
        
        # Prêmios médios históricos (em reais)
        self.average_prizes = {
            20: 500000,
            19: 25000,
            18: 500,
            17: 25,
            16: 5,
            15: 2,
            0: 2
        }
    
    def load_data(self):
        """Carrega dados históricos e estatísticos"""
        try:
            # Carrega dados principais
            if self.data_source.endswith('.json'):
                with open(self.data_source, 'r', encoding='utf-8') as f:
                    data_dict = json.load(f)
                self.df = pd.DataFrame(data_dict)
            else:
                self.df = pd.read_csv(self.data_source)
            
            # Carrega análise estatística se disponível
            try:
                with open('statistical_report.json', 'r', encoding='utf-8') as f:
                    self.statistical_data = json.load(f)
            except:
                logger.warning("Análise estatística não encontrada")
            
            logger.info(f"Dados carregados: {len(self.df)} concursos")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {str(e)}")
            return False
    
    def calculate_expected_value(self, strategy_numbers):
        """
        Calcula o valor esperado de uma aposta
        
        Args:
            strategy_numbers (list): Números da estratégia
            
        Returns:
            float: Valor esperado da aposta
        """
        expected_value = 0
        
        for hits, probability in self.hit_probabilities.items():
            prize = self.average_prizes.get(hits, 0)
            expected_value += probability * prize
        
        # Subtrai o custo da aposta
        expected_value -= self.cost_per_game
        
        return expected_value
    
    def generate_hot_strategy(self, num_numbers=20):
        """
        Gera estratégia baseada em números quentes
        
        Args:
            num_numbers (int): Número de números para selecionar
            
        Returns:
            dict: Estratégia com números e métricas
        """
        if not self.statistical_data:
            return self._fallback_strategy(num_numbers)
        
        freq_analysis = self.statistical_data.get('frequency_analysis', {})
        hot_numbers = freq_analysis.get('classification', {}).get('hot_numbers', [])
        neutral_numbers = freq_analysis.get('classification', {}).get('neutral_numbers', [])
        
        # Combina números quentes e neutros
        selected_numbers = hot_numbers[:num_numbers]
        
        if len(selected_numbers) < num_numbers:
            selected_numbers.extend(neutral_numbers[:num_numbers - len(selected_numbers)])
        
        # Garante exatamente num_numbers números
        if len(selected_numbers) > num_numbers:
            selected_numbers = selected_numbers[:num_numbers]
        elif len(selected_numbers) < num_numbers:
            all_numbers = list(range(100))
            remaining = [n for n in all_numbers if n not in selected_numbers]
            selected_numbers.extend(remaining[:num_numbers - len(selected_numbers)])
        
        expected_value = self.calculate_expected_value(selected_numbers)
        
        return {
            'strategy_name': 'Hot Numbers',
            'numbers': sorted(selected_numbers),
            'expected_value': expected_value,
            'cost': self.cost_per_game,
            'roi_percent': (expected_value / self.cost_per_game) * 100,
            'confidence': 85
        }
    
    def generate_balanced_strategy(self, num_numbers=20):
        """
        Gera estratégia balanceada por décadas
        
        Args:
            num_numbers (int): Número de números para selecionar
            
        Returns:
            dict: Estratégia balanceada
        """
        if not self.statistical_data:
            return self._fallback_strategy(num_numbers)
        
        # Distribui números uniformemente pelas décadas
        numbers_per_decade = num_numbers // 10
        remainder = num_numbers % 10
        
        selected_numbers = []
        
        # Para cada década, seleciona os melhores números
        for decade in range(10):
            decade_quota = numbers_per_decade + (1 if decade < remainder else 0)
            
            # Números desta década ordenados por frequência
            if self.statistical_data:
                freq_count = self.statistical_data.get('frequency_analysis', {}).get('frequency_count', {})
                decade_numbers = []
                
                for num in range(decade * 10, (decade + 1) * 10):
                    frequency = freq_count.get(str(num), 0)
                    decade_numbers.append((num, frequency))
                
                # Ordena por frequência e seleciona os melhores
                decade_numbers.sort(key=lambda x: x[1], reverse=True)
                selected_from_decade = [num for num, _ in decade_numbers[:decade_quota]]
            else:
                # Fallback: seleciona números aleatórios da década
                decade_range = list(range(decade * 10, (decade + 1) * 10))
                selected_from_decade = np.random.choice(decade_range, 
                                                       min(decade_quota, len(decade_range)), 
                                                       replace=False).tolist()
            
            selected_numbers.extend(selected_from_decade)
        
        expected_value = self.calculate_expected_value(selected_numbers)
        
        return {
            'strategy_name': 'Balanced Decades',
            'numbers': sorted(selected_numbers),
            'expected_value': expected_value,
            'cost': self.cost_per_game,
            'roi_percent': (expected_value / self.cost_per_game) * 100,
            'confidence': 75
        }
    
    def generate_gap_strategy(self, num_numbers=20):
        """
        Gera estratégia baseada em gaps (números que não saem há muito tempo)
        
        Args:
            num_numbers (int): Número de números para selecionar
            
        Returns:
            dict: Estratégia baseada em gaps
        """
        if not self.statistical_data:
            return self._fallback_strategy(num_numbers)
        
        gap_analysis = self.statistical_data.get('gap_analysis', {})
        
        if gap_analysis:
            # Ordena números por gap atual (maior gap = mais tempo sem sair)
            gap_data = []
            for num_str, gap_info in gap_analysis.items():
                current_gap = gap_info.get('current_gap', 0)
                gap_data.append((int(num_str), current_gap))
            
            # Seleciona números com maiores gaps
            gap_data.sort(key=lambda x: x[1], reverse=True)
            selected_numbers = [num for num, _ in gap_data[:num_numbers]]
        else:
            selected_numbers = self._fallback_strategy(num_numbers)['numbers']
        
        expected_value = self.calculate_expected_value(selected_numbers)
        
        return {
            'strategy_name': 'Gap Analysis',
            'numbers': sorted(selected_numbers),
            'expected_value': expected_value,
            'cost': self.cost_per_game,
            'roi_percent': (expected_value / self.cost_per_game) * 100,
            'confidence': 70
        }
    
    def generate_composite_strategy(self, num_numbers=20):
        """
        Gera estratégia composta combinando múltiplas abordagens
        
        Args:
            num_numbers (int): Número de números para selecionar
            
        Returns:
            dict: Estratégia composta
        """
        # Gera diferentes estratégias
        hot_strategy = self.generate_hot_strategy(num_numbers)
        balanced_strategy = self.generate_balanced_strategy(num_numbers)
        gap_strategy = self.generate_gap_strategy(num_numbers)
        
        # Combina números com sistema de votação
        number_votes = Counter()
        
        for strategy in [hot_strategy, balanced_strategy, gap_strategy]:
            for num in strategy['numbers']:
                number_votes[num] += 1
        
        # Seleciona números com mais votos
        most_voted = number_votes.most_common(num_numbers)
        selected_numbers = [num for num, votes in most_voted]
        
        # Se não temos números suficientes, completa com números quentes
        if len(selected_numbers) < num_numbers:
            hot_numbers = hot_strategy['numbers']
            for num in hot_numbers:
                if num not in selected_numbers:
                    selected_numbers.append(num)
                    if len(selected_numbers) >= num_numbers:
                        break
        
        expected_value = self.calculate_expected_value(selected_numbers[:num_numbers])
        
        return {
            'strategy_name': 'Composite Strategy',
            'numbers': sorted(selected_numbers[:num_numbers]),
            'expected_value': expected_value,
            'cost': self.cost_per_game,
            'roi_percent': (expected_value / self.cost_per_game) * 100,
            'confidence': 90
        }
    
    def optimize_budget_allocation(self, total_budget, risk_level='moderate'):
        """
        Otimiza a alocação de orçamento entre diferentes estratégias
        
        Args:
            total_budget (float): Orçamento total disponível
            risk_level (str): Nível de risco ('conservative', 'moderate', 'aggressive')
            
        Returns:
            dict: Plano otimizado de apostas
        """
        max_games = int(total_budget // self.cost_per_game)
        
        if max_games == 0:
            return {
                'error': 'Orçamento insuficiente para uma aposta',
                'minimum_budget': self.cost_per_game
            }
        
        # Define estratégias baseadas no nível de risco
        if risk_level == 'conservative':
            strategies = [
                ('hot', 0.6),      # 60% números quentes
                ('balanced', 0.4)  # 40% balanceado
            ]
            max_games = min(max_games, 3)  # Máximo 3 jogos
            
        elif risk_level == 'moderate':
            strategies = [
                ('hot', 0.4),      # 40% números quentes
                ('balanced', 0.3), # 30% balanceado
                ('composite', 0.3) # 30% composto
            ]
            max_games = min(max_games, 5)  # Máximo 5 jogos
            
        else:  # aggressive
            strategies = [
                ('hot', 0.3),      # 30% números quentes
                ('balanced', 0.2), # 20% balanceado
                ('gap', 0.2),      # 20% gap
                ('composite', 0.3) # 30% composto
            ]
            # Usa todo o orçamento disponível
        
        # Gera apostas otimizadas
        optimized_bets = []
        total_expected_value = 0
        
        for i in range(max_games):
            # Seleciona estratégia baseada na distribuição
            strategy_choice = np.random.choice(
                [s[0] for s in strategies],
                p=[s[1] for s in strategies]
            )
            
            # Gera a aposta
            if strategy_choice == 'hot':
                bet = self.generate_hot_strategy()
            elif strategy_choice == 'balanced':
                bet = self.generate_balanced_strategy()
            elif strategy_choice == 'gap':
                bet = self.generate_gap_strategy()
            else:  # composite
                bet = self.generate_composite_strategy()
            
            bet['game_number'] = i + 1
            optimized_bets.append(bet)
            total_expected_value += bet['expected_value']
        
        total_cost = max_games * self.cost_per_game
        overall_roi = (total_expected_value / total_cost) * 100 if total_cost > 0 else 0
        
        return {
            'total_budget': total_budget,
            'total_cost': total_cost,
            'remaining_budget': total_budget - total_cost,
            'number_of_games': max_games,
            'risk_level': risk_level,
            'total_expected_value': total_expected_value,
            'overall_roi_percent': overall_roi,
            'optimized_bets': optimized_bets,
            'strategy_distribution': strategies
        }
    
    def simulate_betting_outcomes(self, betting_plan, num_simulations=1000):
        """
        Simula resultados de apostas baseado no plano otimizado
        
        Args:
            betting_plan (dict): Plano de apostas otimizado
            num_simulations (int): Número de simulações
            
        Returns:
            dict: Resultados da simulação
        """
        if 'optimized_bets' not in betting_plan:
            return {'error': 'Plano de apostas inválido'}
        
        simulation_results = []
        
        for simulation in range(num_simulations):
            total_winnings = 0
            total_cost = betting_plan['total_cost']
            
            for bet in betting_plan['optimized_bets']:
                # Simula sorteio (20 números aleatórios)
                drawn_numbers = set(np.random.choice(100, 20, replace=False))
                bet_numbers = set(bet['numbers'])
                
                # Conta acertos
                hits = len(drawn_numbers.intersection(bet_numbers))
                
                # Calcula prêmio baseado nos acertos
                prize = self.average_prizes.get(hits, 0)
                total_winnings += prize
            
            net_result = total_winnings - total_cost
            simulation_results.append({
                'total_winnings': total_winnings,
                'net_result': net_result,
                'roi': (net_result / total_cost) * 100 if total_cost > 0 else 0
            })
        
        # Calcula estatísticas das simulações
        winnings = [r['total_winnings'] for r in simulation_results]
        net_results = [r['net_result'] for r in simulation_results]
        rois = [r['roi'] for r in simulation_results]
        
        # Probabilidade de lucro
        profit_probability = len([r for r in net_results if r > 0]) / num_simulations
        
        return {
            'num_simulations': num_simulations,
            'average_winnings': np.mean(winnings),
            'average_net_result': np.mean(net_results),
            'average_roi': np.mean(rois),
            'profit_probability': profit_probability,
            'best_case': max(net_results),
            'worst_case': min(net_results),
            'std_deviation': np.std(net_results),
            'percentiles': {
                '10th': np.percentile(net_results, 10),
                '25th': np.percentile(net_results, 25),
                '50th': np.percentile(net_results, 50),
                '75th': np.percentile(net_results, 75),
                '90th': np.percentile(net_results, 90)
            }
        }
    
    def _fallback_strategy(self, num_numbers=20):
        """Estratégia de fallback quando dados estatísticos não estão disponíveis"""
        selected_numbers = np.random.choice(100, num_numbers, replace=False).tolist()
        expected_value = self.calculate_expected_value(selected_numbers)
        
        return {
            'strategy_name': 'Random Selection',
            'numbers': sorted(selected_numbers),
            'expected_value': expected_value,
            'cost': self.cost_per_game,
            'roi_percent': (expected_value / self.cost_per_game) * 100,
            'confidence': 50
        }
    
    def generate_report(self, betting_plan, simulation_results=None):
        """
        Gera relatório detalhado do plano de apostas
        
        Args:
            betting_plan (dict): Plano de apostas
            simulation_results (dict): Resultados da simulação (opcional)
            
        Returns:
            str: Relatório formatado
        """
        report = f"""
=== RELATÓRIO DE OTIMIZAÇÃO DE APOSTAS - LOTOMANIA ===

📊 RESUMO GERAL
Orçamento Total: R$ {betting_plan['total_budget']:.2f}
Custo Total: R$ {betting_plan['total_cost']:.2f}
Sobra: R$ {betting_plan['remaining_budget']:.2f}
Número de Jogos: {betting_plan['number_of_games']}
Nível de Risco: {betting_plan['risk_level'].title()}

💰 ANÁLISE FINANCEIRA
Valor Esperado Total: R$ {betting_plan['total_expected_value']:.2f}
ROI Esperado: {betting_plan['overall_roi_percent']:.2f}%

🎯 JOGOS RECOMENDADOS
"""
        
        for bet in betting_plan['optimized_bets']:
            numbers_str = " - ".join([f"{num:02d}" for num in bet['numbers']])
            report += f"""
Jogo {bet['game_number']}: {bet['strategy_name']}
Números: {numbers_str}
Valor Esperado: R$ {bet['expected_value']:.2f}
Confiança: {bet['confidence']}%
"""
        
        if simulation_results:
            report += f"""
🧪 RESULTADOS DA SIMULAÇÃO ({simulation_results['num_simulations']} simulações)
Ganho Médio: R$ {simulation_results['average_winnings']:.2f}
Resultado Líquido Médio: R$ {simulation_results['average_net_result']:.2f}
ROI Médio: {simulation_results['average_roi']:.2f}%
Probabilidade de Lucro: {simulation_results['profit_probability']*100:.1f}%

Melhor Caso: R$ {simulation_results['best_case']:.2f}
Pior Caso: R$ {simulation_results['worst_case']:.2f}
Desvio Padrão: R$ {simulation_results['std_deviation']:.2f}

Percentis:
10%: R$ {simulation_results['percentiles']['10th']:.2f}
25%: R$ {simulation_results['percentiles']['25th']:.2f}
50%: R$ {simulation_results['percentiles']['50th']:.2f}
75%: R$ {simulation_results['percentiles']['75th']:.2f}
90%: R$ {simulation_results['percentiles']['90th']:.2f}
"""
        
        report += """
⚠️ AVISO IMPORTANTE
Este sistema é baseado em análise estatística e não garante resultados.
Jogos de azar envolvem risco. Jogue com responsabilidade.
"""
        
        return report

def main():
    """Função principal para demonstrar o otimizador"""
    
    print("=== SISTEMA DE OTIMIZAÇÃO DE APOSTAS DA LOTOMANIA ===\n")
    
    # Inicializa o otimizador
    optimizer = LotomaniaBetOptimizer()
    
    if not optimizer.load_data():
        print("❌ Erro ao carregar dados")
        return
    
    # Exemplo de otimização para diferentes orçamentos e níveis de risco
    budgets = [50, 100, 200]
    risk_levels = ['conservative', 'moderate', 'aggressive']
    
    for budget in budgets:
        print(f"\n💰 ORÇAMENTO: R$ {budget}")
        print("=" * 50)
        
        for risk_level in risk_levels:
            print(f"\n🎯 Nível de Risco: {risk_level.upper()}")
            
            # Gera plano otimizado
            betting_plan = optimizer.optimize_budget_allocation(budget, risk_level)
            
            if 'error' in betting_plan:
                print(f"❌ {betting_plan['error']}")
                continue
            
            # Executa simulação
            simulation = optimizer.simulate_betting_outcomes(betting_plan, 1000)
            
            # Mostra resumo
            print(f"Jogos: {betting_plan['number_of_games']}")
            print(f"ROI Esperado: {betting_plan['overall_roi_percent']:.2f}%")
            print(f"Prob. Lucro: {simulation['profit_probability']*100:.1f}%")
            print(f"Resultado Médio: R$ {simulation['average_net_result']:.2f}")
            
            # Salva relatório detalhado
            report = optimizer.generate_report(betting_plan, simulation)
            filename = f"betting_report_{budget}_{risk_level}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"📄 Relatório salvo: {filename}")
    
    print(f"\n✅ Otimização concluída!")
    print("📁 Verifique os relatórios gerados para detalhes completos.")

if __name__ == "__main__":
    main()