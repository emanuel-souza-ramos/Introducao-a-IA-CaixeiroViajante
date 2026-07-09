import sys
import os
import time
import tracemalloc
import numpy as np
import tsplib95
import matplotlib.pyplot as plt

# Adiciona o diretório do script ao sys.path para garantir que os imports locais funcionem
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from ga_v1 import Ga as GaV1
from ga_v2 import Ga as GaV2
from ga_v3 import Ga as GaV3

def run_experiment(GaClass, name, city_dist_mat, individual_num, tournament_size, num_runs=10):
    print("\n" + "=" * 60)
    print(f"INICIANDO EXPERIMENTO DE 10 EXECUÇÕES: {name}")
    print(f"Parâmetros: População = {individual_num} | Torneio = {tournament_size}")
    print("=" * 60)
    
    runs = []
    for i in range(num_runs):
        print(f"-> Executando rodada {i+1}/{num_runs}...", end="", flush=True)
        
        tracemalloc.start()
        start_time = time.perf_counter()
        
        # Cria uma instância silenciosa do GA (sem imprimir a cada 50 gerações para não poluir o terminal)
        # Para silenciar, redirecionamos temporariamente o stdout ou simplesmente ignoramos, 
        # mas como o train() possui prints, faremos um pequeno desvio do stdout para manter o log limpo.
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        try:
            ga = GaClass(city_dist_mat, individual_num=individual_num, gen_num=2000, mutate_prob=0.2, patience=300, tournament_size=tournament_size)
            result_list, fitness_list = ga.train()
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout
            
        end_time = time.perf_counter()
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        elapsed_time = end_time - start_time
        peak_memory_mb = peak_memory / (1024 * 1024)
        
        runs.append({
            "final_fitness": fitness_list[-1],
            "fitness_list": fitness_list,
            "elapsed_time": elapsed_time,
            "peak_memory_mb": peak_memory_mb
        })
        
        print(f" Concluído! Fitness: {fitness_list[-1]:.2f} | Tempo: {elapsed_time:.2f}s")
        
    # Calcular as médias e desvios padrão
    fitness_values = [r["final_fitness"] for r in runs]
    times = [r["elapsed_time"] for r in runs]
    memories = [r["peak_memory_mb"] for r in runs]
    
    # Calcular a curva de aprendizado média
    max_len = max(len(r["fitness_list"]) for r in runs)
    padded_curves = []
    for r in runs:
        curve = r["fitness_list"]
        if len(curve) < max_len:
            curve = curve + [curve[-1]] * (max_len - len(curve))
        padded_curves.append(curve)
    average_curve = np.mean(padded_curves, axis=0)
    
    stats = {
        "runs": runs,
        "fitness_values": fitness_values,
        "times": times,
        "memories": memories,
        "avg_fitness": np.mean(fitness_values),
        "std_fitness": np.std(fitness_values),
        "avg_time": np.mean(times),
        "std_time": np.std(times),
        "avg_memory": np.mean(memories),
        "std_memory": np.std(memories),
        "average_curve": average_curve
    }
    
    return stats

def main():
    # Determina o diretório de mapas (pasta 'mapas' na raiz do projeto)
    mapas_dir = os.path.abspath(os.path.join(script_dir, '..', 'mapas'))

    # Lista arquivos .tsp na pasta de mapas
    if os.path.exists(mapas_dir):
        tsp_files = [f for f in os.listdir(mapas_dir) if f.endswith('.tsp')]
    else:
        tsp_files = []

    # Permite passar o mapa por linha de comando ou escolher interativamente
    if len(sys.argv) > 1:
        arg_file = sys.argv[1]
        if os.path.exists(arg_file):
            selected_file = arg_file
        else:
            potential_file = os.path.join(mapas_dir, arg_file)
            if os.path.exists(potential_file):
                selected_file = potential_file
            else:
                print(f"[ERRO] O arquivo '{arg_file}' nao foi encontrado diretamente nem na pasta '{mapas_dir}'.")
                sys.exit(1)
    else:
        if not tsp_files:
            print(f"[ERRO] Nenhum arquivo .tsp encontrado na pasta de mapas ('{mapas_dir}').")
            sys.exit(1)

        print("Arquivos .tsp encontrados na pasta 'mapas':")
        for idx, f in enumerate(tsp_files):
            print(f"{idx + 1}: {f}")
        
        try:
            choice = input("Escolha o numero do arquivo (padrao 1): ").strip()
            if choice == "":
                selected_file = tsp_files[0]
            else:
                selected_file = tsp_files[int(choice) - 1]
        except (ValueError, IndexError, KeyboardInterrupt, EOFError):
            selected_file = tsp_files[0]
        
        selected_file = os.path.join(mapas_dir, selected_file)

    print(f"\n[INFO] Carregando o arquivo de mapa: {selected_file}...")
    problem = tsplib95.load(selected_file)
    nodes = list(problem.get_nodes())
    n = len(nodes)
    
    print(f"[INFO] Mapa carregado com sucesso. Total de cidades (nos): {n}")
    
    # Constrói a matriz de distância
    city_dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            city_dist_mat[i, j] = problem.get_weight(nodes[i], nodes[j])

    # Executa os experimentos (10 rodadas de cada versão)
    num_runs = 10
    stats = {}
    stats["GA V1"] = run_experiment(GaV1, "GA V1", city_dist_mat, individual_num=100, tournament_size=5, num_runs=num_runs)
    stats["GA V2"] = run_experiment(GaV2, "GA V2", city_dist_mat, individual_num=200, tournament_size=5, num_runs=num_runs)
    stats["GA V3"] = run_experiment(GaV3, "GA V3", city_dist_mat, individual_num=500, tournament_size=10, num_runs=num_runs)

    # Imprime tabela de logs estatísticos
    print("\n" + "=" * 90)
    print(f"RELATÓRIO ESTATÍSTICO COMPILADO (MÉDIA DE {num_runs} EXECUÇÕES) - MAPA: {os.path.basename(selected_file)}")
    print("=" * 90)
    print(f"{'Algoritmo':<12} | {'Distância Média (Fitness)':<27} | {'Tempo Médio (segundos)':<24} | {'Memória Média':<18}")
    print("-" * 90)
    for name, data in stats.items():
        fit_str = f"{data['avg_fitness']:.2f} ± {data['std_fitness']:.2f}"
        time_str = f"{data['avg_time']:.3f}s ± {data['std_time']:.3f}s"
        mem_str = f"{data['avg_memory']:.3f} MB ± {data['std_memory']:.3f} MB"
        print(f"{name:<12} | {fit_str:<27} | {time_str:<24} | {mem_str:<18}")
    print("=" * 90 + "\n")

    map_name = os.path.basename(selected_file)
    run_indices = np.arange(1, num_runs + 1)

    # Plot 1: Curva de Convergência Média (Fitness médio vs. Geração)
    plt.figure(figsize=(10, 6))
    plt.plot(stats["GA V1"]["average_curve"], color='#1f77b4', linewidth=2.5, label='GA V1 (Pop: 100, Tor: 5)')
    plt.plot(stats["GA V2"]["average_curve"], color='#ff7f0e', linewidth=2.5, label='GA V2 (Pop: 200, Tor: 5)')
    plt.plot(stats["GA V3"]["average_curve"], color='#2ca02c', linewidth=2.5, label='GA V3 (Pop: 500, Tor: 10)')
    plt.title(f"Curva de Convergência Média ({num_runs} Execuções) - {map_name}", fontsize=12, fontweight='bold')
    plt.xlabel("Geração (Iteração)")
    plt.ylabel("Distância Média (Fitness)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')

    # Plot 2: Painel Geral de Métricas Individuais por Rodada
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # 2.1 Distância Final por Rodada
    axes[0].plot(run_indices, stats["GA V1"]["fitness_values"], 'o-', color='#1f77b4', label='GA V1')
    axes[0].plot(run_indices, stats["GA V2"]["fitness_values"], 's-', color='#ff7f0e', label='GA V2')
    axes[0].plot(run_indices, stats["GA V3"]["fitness_values"], '^-', color='#2ca02c', label='GA V3')
    # Adiciona linhas pontilhadas com a média
    axes[0].axhline(stats["GA V1"]["avg_fitness"], color='#1f77b4', linestyle='--', alpha=0.7)
    axes[0].axhline(stats["GA V2"]["avg_fitness"], color='#ff7f0e', linestyle='--', alpha=0.7)
    axes[0].axhline(stats["GA V3"]["avg_fitness"], color='#2ca02c', linestyle='--', alpha=0.7)
    axes[0].set_title("Distância Mínima Encontrada por Rodada", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Rodada / Execução")
    axes[0].set_ylabel("Distância (Fitness)")
    axes[0].set_xticks(run_indices)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='best')

    # 2.2 Tempo de Execução por Rodada
    axes[1].plot(run_indices, stats["GA V1"]["times"], 'o-', color='#1f77b4', label='GA V1')
    axes[1].plot(run_indices, stats["GA V2"]["times"], 's-', color='#ff7f0e', label='GA V2')
    axes[1].plot(run_indices, stats["GA V3"]["times"], '^-', color='#2ca02c', label='GA V3')
    # Adiciona linhas pontilhadas com a média
    axes[1].axhline(stats["GA V1"]["avg_time"], color='#1f77b4', linestyle='--', alpha=0.7)
    axes[1].axhline(stats["GA V2"]["avg_time"], color='#ff7f0e', linestyle='--', alpha=0.7)
    axes[1].axhline(stats["GA V3"]["avg_time"], color='#2ca02c', linestyle='--', alpha=0.7)
    axes[1].set_title("Tempo de Execução por Rodada", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Rodada / Execução")
    axes[1].set_ylabel("Tempo (segundos)")
    axes[1].set_xticks(run_indices)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='best')

    # 2.3 Pico de Consumo de Memória por Rodada
    axes[2].plot(run_indices, stats["GA V1"]["memories"], 'o-', color='#1f77b4', label='GA V1')
    axes[2].plot(run_indices, stats["GA V2"]["memories"], 's-', color='#ff7f0e', label='GA V2')
    axes[2].plot(run_indices, stats["GA V3"]["memories"], '^-', color='#2ca02c', label='GA V3')
    # Adiciona linhas pontilhadas com a média
    axes[2].axhline(stats["GA V1"]["avg_memory"], color='#1f77b4', linestyle='--', alpha=0.7)
    axes[2].axhline(stats["GA V2"]["avg_memory"], color='#ff7f0e', linestyle='--', alpha=0.7)
    axes[2].axhline(stats["GA V3"]["avg_memory"], color='#2ca02c', linestyle='--', alpha=0.7)
    axes[2].set_title("Pico de Consumo de Memória por Rodada", fontsize=11, fontweight='bold')
    axes[2].set_xlabel("Rodada / Execução")
    axes[2].set_ylabel("Pico de Memória (MB)")
    axes[2].set_xticks(run_indices)
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].legend(loc='best')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
