#!/usr/bin/env python3
"""
Script Principal do Sistema de IA para Lotomania
Orquestra a execução completa do pipeline de análise e predição

Execute com: python main.py
Para interface web: python main.py --web
"""

import argparse
import sys
import os
from datetime import datetime
import subprocess

def run_command(command, description):
    """Executa um comando e trata erros"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Concluído com sucesso!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro durante execução: {e}")
        if e.stderr:
            print(f"Detalhes do erro: {e.stderr}")
        return False

def check_dependencies():
    """Verifica se as dependências estão instaladas"""
    print("🔍 Verificando dependências...")
    
    required_packages = ['pandas', 'numpy', 'sklearn', 'streamlit', 'plotly']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - FALTANDO")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Instale as dependências faltantes:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ Todas as dependências estão instaladas!")
    return True

def setup_data():
    """Configura dados necessários"""
    print("\n📊 Configurando dados...")
    
    # Verifica se já existem dados processados
    if os.path.exists('lotomania_processed.json'):
        print("✅ Dados já existem: lotomania_processed.json")
        return True
    
    print("📊 Gerando dados de demonstração...")
    if run_command("python3 create_sample_data.py", "Gerando dados de demonstração"):
        return True
    else:
        print("❌ Erro ao gerar dados. Tentando com python...")
        return run_command("python create_sample_data.py", "Gerando dados (fallback)")

def run_statistical_analysis():
    """Executa análise estatística"""
    if os.path.exists('statistical_report.json'):
        print("✅ Relatório estatístico já existe")
        return True
    
    print("📈 Executando análise estatística...")
    if run_command("python3 statistical_analyzer.py", "Análise Estatística"):
        return True
    else:
        return run_command("python statistical_analyzer.py", "Análise Estatística (fallback)")

def run_backtesting():
    """Executa validação do sistema"""
    if os.path.exists('backtesting_results.json'):
        print("✅ Resultados de backtesting já existem")
        return True
    
    print("🧪 Executando validação (backtesting)...")
    if run_command("python3 backtesting_simulator.py", "Sistema de Backtesting"):
        return True
    else:
        return run_command("python backtesting_simulator.py", "Sistema de Backtesting (fallback)")

def run_optimization():
    """Executa otimização de apostas"""
    print("🎯 Executando otimização de apostas...")
    if run_command("python3 bet_optimizer.py", "Otimização de Apostas"):
        return True
    else:
        return run_command("python bet_optimizer.py", "Otimização de Apostas (fallback)")

def launch_web_interface():
    """Lança a interface web"""
    print("\n🌐 Iniciando interface web...")
    print("📱 Acesse: http://localhost:8501")
    print("⏹️  Pressione Ctrl+C para parar")
    
    try:
        if os.system("which streamlit") == 0 or os.system("where streamlit") == 0:
            os.system("streamlit run streamlit_app.py")
        else:
            # Tenta usar python -m streamlit
            os.system("python -m streamlit run streamlit_app.py")
    except KeyboardInterrupt:
        print("\n👋 Interface web finalizada.")

def generate_summary_report():
    """Gera relatório resumo do sistema"""
    print("\n📋 Gerando relatório resumo...")
    
    report = f"""
=== RELATÓRIO RESUMO - SISTEMA LOTOMANIA AI ===
Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 ARQUIVOS GERADOS:
{"✅" if os.path.exists('lotomania_processed.json') else "❌"} lotomania_processed.json - Dados processados
{"✅" if os.path.exists('statistical_report.json') else "❌"} statistical_report.json - Análise estatística
{"✅" if os.path.exists('backtesting_results.json') else "❌"} backtesting_results.json - Validação do sistema
{"✅" if os.path.exists('performance_report.txt') else "❌"} performance_report.txt - Relatório de performance

🎯 PRÓXIMOS PASSOS:
1. Execute 'python main.py --web' para interface gráfica
2. Ou use os módulos individualmente conforme documentação
3. Consulte o README.md para instruções detalhadas

⚠️  LEMBRE-SE:
- Este sistema é para fins educacionais e de pesquisa
- Não garante resultados em jogos reais
- Use com responsabilidade e consciência dos riscos

📞 SUPORTE:
- README.md: Documentação completa
- Comentários no código: Explicações detalhadas
- Relatórios JSON: Análises técnicas completas
"""
    
    with open('system_summary.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print("📄 Relatório salvo em: system_summary.txt")

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Sistema de IA para Lotomania')
    parser.add_argument('--web', action='store_true', help='Inicia apenas a interface web')
    parser.add_argument('--skip-analysis', action='store_true', help='Pula análises se já existirem')
    parser.add_argument('--force-refresh', action='store_true', help='Força regeneração de todos os dados')
    
    args = parser.parse_args()
    
    print("🎲 SISTEMA DE IA PARA PREDIÇÃO DA LOTOMANIA")
    print("=" * 60)
    print("Sistema completo de análise estatística e machine learning")
    print("para predição inteligente de números da Lotomania")
    print("=" * 60)
    
    # Se apenas interface web
    if args.web:
        if not check_dependencies():
            sys.exit(1)
        launch_web_interface()
        return
    
    # Verifica dependências
    if not check_dependencies():
        print("\n❌ Instale as dependências antes de continuar:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Remove arquivos existentes se force-refresh
    if args.force_refresh:
        files_to_remove = [
            'lotomania_processed.json',
            'statistical_report.json', 
            'backtesting_results.json',
            'performance_report.txt'
        ]
        for file in files_to_remove:
            if os.path.exists(file):
                os.remove(file)
                print(f"🗑️  Removido: {file}")
    
    # Pipeline completo
    success_count = 0
    total_steps = 4
    
    # 1. Configuração de dados
    if setup_data():
        success_count += 1
    else:
        print("❌ Falha crítica na configuração de dados")
        sys.exit(1)
    
    # 2. Análise estatística
    if not args.skip_analysis or not os.path.exists('statistical_report.json'):
        if run_statistical_analysis():
            success_count += 1
    else:
        print("⏭️  Pulando análise estatística (já existe)")
        success_count += 1
    
    # 3. Backtesting
    if not args.skip_analysis or not os.path.exists('backtesting_results.json'):
        if run_backtesting():
            success_count += 1
    else:
        print("⏭️  Pulando backtesting (já existe)")
        success_count += 1
    
    # 4. Otimização (opcional)
    if run_optimization():
        success_count += 1
    
    # Relatório final
    generate_summary_report()
    
    print(f"\n{'='*60}")
    print(f"🏁 PIPELINE CONCLUÍDO: {success_count}/{total_steps} etapas")
    print(f"{'='*60}")
    
    if success_count == total_steps:
        print("✅ Sistema configurado com sucesso!")
        print("\n🚀 PRÓXIMAS AÇÕES:")
        print("1. python main.py --web     # Interface gráfica")
        print("2. python streamlit_app.py  # Direto pelo Streamlit")
        print("3. Consulte README.md para uso avançado")
    else:
        print("⚠️  Algumas etapas falharam. Verifique os logs acima.")
        print("💡 Tente executar novamente ou consulte a documentação.")

if __name__ == "__main__":
    main()