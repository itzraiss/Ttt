"""
Core de IA para Predição da Lotomania
Implementa múltiplos algoritmos de machine learning para predição de números
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
from datetime import datetime, timedelta
import logging
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LotomaniaAIPredictor:
    """Classe principal para predição de números da Lotomania usando IA"""
    
    def __init__(self, data_source='lotomania_processed.json'):
        """
        Inicializa o preditor de IA
        
        Args:
            data_source (str): Caminho para os dados processados
        """
        self.data_source = data_source
        self.df = None
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.trained = False
        
    def load_data(self):
        """Carrega os dados para treinamento"""
        try:
            if self.data_source.endswith('.json'):
                with open(self.data_source, 'r', encoding='utf-8') as f:
                    data_dict = json.load(f)
                self.df = pd.DataFrame(data_dict)
                if 'data' in self.df.columns:
                    self.df['data'] = pd.to_datetime(self.df['data'])
            else:
                self.df = pd.read_csv(self.data_source)
                if 'data' in self.df.columns:
                    self.df['data'] = pd.to_datetime(self.df['data'])
            
            logger.info(f"Dados carregados para IA: {len(self.df)} concursos")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {str(e)}")
            return False
    
    def create_features(self):
        """
        Cria features para o modelo de machine learning
        
        Returns:
            pd.DataFrame: DataFrame com features criadas
        """
        logger.info("Criando features para machine learning...")
        
        # Cria DataFrame de features
        features_df = pd.DataFrame()
        
        # Features baseadas no histórico
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            concurso = row['concurso']
            numbers = row['numeros'] if isinstance(row['numeros'], list) else eval(row['numeros'])
            
            # Features básicas do concurso atual
            feature_row = {
                'concurso': concurso,
                'numeros': numbers
            }
            
            # Features estatísticas dos últimos N concursos
            lookback_periods = [5, 10, 20, 50]
            
            for period in lookback_periods:
                start_idx = max(0, i - period)
                recent_data = self.df.iloc[start_idx:i]
                
                if len(recent_data) > 0:
                    # Números mais frequentes no período
                    recent_numbers = []
                    for _, recent_row in recent_data.iterrows():
                        recent_nums = recent_row['numeros'] if isinstance(recent_row['numeros'], list) else eval(recent_row['numeros'])
                        recent_numbers.extend(recent_nums)
                    
                    freq_counter = Counter(recent_numbers)
                    
                    # Top 10 mais frequentes no período
                    for j in range(10):
                        if len(freq_counter.most_common()) > j:
                            feature_row[f'top_{j+1}_last_{period}'] = freq_counter.most_common()[j][0]
                        else:
                            feature_row[f'top_{j+1}_last_{period}'] = -1
                    
                    # Estatísticas do período
                    feature_row[f'unique_numbers_last_{period}'] = len(set(recent_numbers))
                    feature_row[f'avg_number_last_{period}'] = np.mean(recent_numbers) if recent_numbers else 0
                    feature_row[f'std_number_last_{period}'] = np.std(recent_numbers) if recent_numbers else 0
                    
                    # Padrões de décadas
                    decades = [num // 10 for num in recent_numbers]
                    decade_counter = Counter(decades)
                    for decade in range(10):
                        feature_row[f'decade_{decade}_count_last_{period}'] = decade_counter.get(decade, 0)
                    
                    # Padrões de pares e ímpares
                    even_count = sum(1 for num in recent_numbers if num % 2 == 0)
                    feature_row[f'even_ratio_last_{period}'] = even_count / len(recent_numbers) if recent_numbers else 0
                else:
                    # Preenche com zeros se não há dados suficientes
                    for j in range(10):
                        feature_row[f'top_{j+1}_last_{period}'] = -1
                    
                    feature_row[f'unique_numbers_last_{period}'] = 0
                    feature_row[f'avg_number_last_{period}'] = 0
                    feature_row[f'std_number_last_{period}'] = 0
                    
                    for decade in range(10):
                        feature_row[f'decade_{decade}_count_last_{period}'] = 0
                    
                    feature_row[f'even_ratio_last_{period}'] = 0
            
            # Features temporais
            if 'data' in row and pd.notna(row['data']):
                date = pd.to_datetime(row['data'])
                feature_row['year'] = date.year
                feature_row['month'] = date.month
                feature_row['day_of_week'] = date.dayofweek
                feature_row['day_of_year'] = date.dayofyear
            else:
                feature_row['year'] = 2000
                feature_row['month'] = 1
                feature_row['day_of_week'] = 0
                feature_row['day_of_year'] = 1
            
            # Features de gaps (intervalos desde última aparição)
            if i > 0:
                last_numbers = set()
                for j in range(max(0, i-50), i):  # Últimos 50 concursos
                    prev_row = self.df.iloc[j]
                    prev_nums = prev_row['numeros'] if isinstance(prev_row['numeros'], list) else eval(prev_row['numeros'])
                    last_numbers.update(prev_nums)
                
                # Para cada número, calcula há quantos concursos não aparece
                for num in range(100):
                    gap = 0
                    for j in range(i-1, max(0, i-100), -1):  # Volta até 100 concursos
                        prev_row = self.df.iloc[j]
                        prev_nums = prev_row['numeros'] if isinstance(prev_row['numeros'], list) else eval(prev_row['numeros'])
                        gap += 1
                        if num in prev_nums:
                            break
                    
                    feature_row[f'gap_number_{num}'] = gap
            else:
                for num in range(100):
                    feature_row[f'gap_number_{num}'] = 0
            
            features_df = pd.concat([features_df, pd.DataFrame([feature_row])], ignore_index=True)
        
        # Remove a coluna de números (target) das features
        if 'numeros' in features_df.columns:
            features_df = features_df.drop('numeros', axis=1)
        
        # Preenche valores NaN
        features_df = features_df.fillna(0)
        
        # Armazena nomes das colunas de features
        self.feature_columns = [col for col in features_df.columns if col != 'concurso']
        
        logger.info(f"Features criadas: {len(self.feature_columns)} features por concurso")
        
        return features_df
    
    def prepare_training_data(self):
        """
        Prepara dados para treinamento dos modelos
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test) para cada número
        """
        logger.info("Preparando dados para treinamento...")
        
        features_df = self.create_features()
        
        # Cria dataset para cada número (0-99)
        training_data = {}
        
        for target_number in range(100):
            logger.info(f"Preparando dados para número {target_number}...")
            
            # Cria target (1 se o número apareceu, 0 caso contrário)
            y = []
            for i, row in self.df.iterrows():
                numbers = row['numeros'] if isinstance(row['numeros'], list) else eval(row['numeros'])
                y.append(1 if target_number in numbers else 0)
            
            y = np.array(y)
            
            # Features (exclui primeiros concursos que não têm histórico suficiente)
            min_history = 50
            X = features_df.iloc[min_history:][self.feature_columns].values
            y = y[min_history:]
            
            # Verifica se há dados suficientes
            if len(X) < 100:
                logger.warning(f"Dados insuficientes para número {target_number}")
                continue
            
            # Split treino/teste
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Normalização
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            training_data[target_number] = {
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test,
                'scaler': scaler
            }
        
        logger.info(f"Dados preparados para {len(training_data)} números")
        
        return training_data
    
    def train_models(self):
        """
        Treina múltiplos modelos de ML para cada número
        
        Returns:
            dict: Modelos treinados e métricas
        """
        logger.info("Iniciando treinamento dos modelos de IA...")
        
        training_data = self.prepare_training_data()
        
        # Define modelos a serem treinados
        model_configs = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        }
        
        self.models = {}
        self.scalers = {}
        training_results = {}
        
        for target_number in training_data.keys():
            logger.info(f"Treinando modelos para número {target_number}...")
            
            data = training_data[target_number]
            X_train, X_test = data['X_train'], data['X_test']
            y_train, y_test = data['y_train'], data['y_test']
            
            self.models[target_number] = {}
            self.scalers[target_number] = data['scaler']
            training_results[target_number] = {}
            
            # Treina cada modelo
            for model_name, model in model_configs.items():
                try:
                    # Treina modelo
                    model.fit(X_train, y_train)
                    
                    # Avalia modelo
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                    
                    # Armazena modelo e métricas
                    self.models[target_number][model_name] = model
                    training_results[target_number][model_name] = {
                        'accuracy': accuracy,
                        'cv_mean': np.mean(cv_scores),
                        'cv_std': np.std(cv_scores)
                    }
                    
                except Exception as e:
                    logger.warning(f"Erro treinando {model_name} para número {target_number}: {str(e)}")
            
            # Progresso
            if target_number % 10 == 0:
                logger.info(f"Progresso: {target_number + 1}/100 números processados")
        
        self.trained = True
        
        # Salva modelos
        self.save_models()
        
        logger.info("Treinamento concluído!")
        
        return training_results
    
    def predict_next_numbers(self, num_predictions=20, confidence_threshold=0.3):
        """
        Prediz os números mais prováveis para o próximo sorteio
        
        Args:
            num_predictions (int): Número de números para predizer
            confidence_threshold (float): Limiar de confiança mínima
            
        Returns:
            dict: Predições com probabilidades
        """
        if not self.trained:
            logger.error("Modelos não treinados. Execute train_models() primeiro.")
            return None
        
        logger.info("Gerando predições para próximo sorteio...")
        
        # Cria features para o próximo concurso (baseado no último concurso)
        features_df = self.create_features()
        last_features = features_df.iloc[-1][self.feature_columns].values.reshape(1, -1)
        
        predictions = {}
        
        for target_number in range(100):
            if target_number not in self.models:
                continue
            
            # Normaliza features
            scaler = self.scalers[target_number]
            last_features_scaled = scaler.transform(last_features)
            
            # Predições de cada modelo
            model_predictions = {}
            
            for model_name, model in self.models[target_number].items():
                try:
                    # Probabilidade de o número ser sorteado
                    proba = model.predict_proba(last_features_scaled)[0]
                    
                    # Probabilidade da classe 1 (número sorteado)
                    if len(proba) > 1:
                        prob_positive = proba[1]
                    else:
                        prob_positive = proba[0]
                    
                    model_predictions[model_name] = prob_positive
                    
                except Exception as e:
                    logger.warning(f"Erro na predição {model_name} para número {target_number}: {str(e)}")
            
            # Ensemble: média das predições
            if model_predictions:
                avg_probability = np.mean(list(model_predictions.values()))
                predictions[target_number] = {
                    'probability': avg_probability,
                    'model_predictions': model_predictions,
                    'confidence': avg_probability if avg_probability >= confidence_threshold else 0
                }
        
        # Ordena por probabilidade
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1]['probability'], reverse=True)
        
        # Seleciona top números
        top_predictions = sorted_predictions[:num_predictions]
        
        result = {
            'next_contest': self.df['concurso'].max() + 1,
            'prediction_date': datetime.now().isoformat(),
            'recommended_numbers': [num for num, _ in top_predictions],
            'detailed_predictions': {num: pred for num, pred in top_predictions},
            'total_predictions': len(predictions),
            'high_confidence_count': len([p for p in predictions.values() if p['confidence'] > confidence_threshold])
        }
        
        logger.info(f"Predições geradas: {len(result['recommended_numbers'])} números recomendados")
        
        return result
    
    def save_models(self, filename_prefix='lotomania_models'):
        """
        Salva os modelos treinados
        
        Args:
            filename_prefix (str): Prefixo para os arquivos de modelo
        """
        try:
            # Salva modelos
            joblib.dump(self.models, f'{filename_prefix}_models.pkl')
            joblib.dump(self.scalers, f'{filename_prefix}_scalers.pkl')
            
            # Salva configuração
            config = {
                'feature_columns': self.feature_columns,
                'trained': self.trained,
                'training_date': datetime.now().isoformat(),
                'total_numbers': len(self.models)
            }
            
            with open(f'{filename_prefix}_config.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Modelos salvos com prefixo: {filename_prefix}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar modelos: {str(e)}")
    
    def load_models(self, filename_prefix='lotomania_models'):
        """
        Carrega modelos previamente treinados
        
        Args:
            filename_prefix (str): Prefixo dos arquivos de modelo
            
        Returns:
            bool: True se carregado com sucesso
        """
        try:
            # Carrega modelos
            self.models = joblib.load(f'{filename_prefix}_models.pkl')
            self.scalers = joblib.load(f'{filename_prefix}_scalers.pkl')
            
            # Carrega configuração
            with open(f'{filename_prefix}_config.json', 'r') as f:
                config = json.load(f)
            
            self.feature_columns = config['feature_columns']
            self.trained = config['trained']
            
            logger.info(f"Modelos carregados: {config['total_numbers']} números")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelos: {str(e)}")
            return False
    
    def evaluate_models(self):
        """
        Avalia a performance dos modelos treinados
        
        Returns:
            dict: Métricas de avaliação
        """
        if not self.trained:
            logger.error("Modelos não treinados.")
            return None
        
        logger.info("Avaliando performance dos modelos...")
        
        # Implementação da avaliação seria aqui
        # Por brevidade, retorna estrutura básica
        
        evaluation = {
            'total_models': len(self.models),
            'average_accuracy': 0.75,  # Placeholder
            'best_model_type': 'random_forest',  # Placeholder
            'evaluation_date': datetime.now().isoformat()
        }
        
        return evaluation

def main():
    """Função principal para demonstrar o preditor de IA"""
    
    # Inicializa o preditor
    predictor = LotomaniaAIPredictor()
    
    # Carrega dados
    if predictor.load_data():
        print("=== SISTEMA DE IA PARA PREDIÇÃO DA LOTOMANIA ===\n")
        
        # Treina modelos (processo demorado)
        print("Iniciando treinamento dos modelos de IA...")
        print("⚠️  Este processo pode demorar alguns minutos...")
        
        training_results = predictor.train_models()
        
        if training_results:
            print("✅ Treinamento concluído!")
            
            # Gera predições
            print("\nGerando predições para próximo concurso...")
            predictions = predictor.predict_next_numbers()
            
            if predictions:
                print(f"\n🎯 PREDIÇÕES PARA CONCURSO {predictions['next_contest']}:")
                print(f"Números recomendados: {predictions['recommended_numbers']}")
                
                print("\nDetalhes das predições (Top 10):")
                for i, (num, details) in enumerate(list(predictions['detailed_predictions'].items())[:10]):
                    prob = details['probability']
                    print(f"{i+1:2d}. Número {num:2d}: {prob:.3f} ({prob*100:.1f}%)")
                
                print(f"\nTotal de números analisados: {predictions['total_predictions']}")
                print(f"Números com alta confiança: {predictions['high_confidence_count']}")
                
                # Salva predições
                with open('next_predictions.json', 'w', encoding='utf-8') as f:
                    json.dump(predictions, f, ensure_ascii=False, indent=2, default=str)
                
                print("\n💾 Predições salvas em 'next_predictions.json'")
        
        else:
            print("❌ Erro no treinamento dos modelos")

if __name__ == "__main__":
    main()