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

from sa_v1 import Sa as SaV1
from sa_v2 import Sa as SaV2
from sa_v3 import Sa as SaV3

def run_sa_version(SaClass, name, city_dist_mat, T_0, alpha, patience):
    tracemalloc.start()
    start_time = time.perf_counter()
    
    print("\n" + "=" * 60)
    print(f"INICIANDO EXECUÇÃO: {name}")
    print(f"Parâmetros: T_0 = {T_0} | Alpha = {alpha} | Paciência = {patience}")
    print("=" * 60)
    
    sa = SaClass(city_dist_mat, T_0=T_0, T_f=1.5, alpha=alpha, patience=patience, max_iter=2000)
    result_list, fitness_list = sa.train()
    
    end_time = time.perf_counter()
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    elapsed_time = end_time - start_time
    peak_memory_mb = peak_memory / (1024 * 1024)
    
    print(f"\n[FIM {name}] Concluído em {elapsed_time:.3f}s | Melhor Fitness: {fitness_list[-1]:.2f}")
    
    return {
        "result": result_list[-1],
        "fitness_list": fitness_list,
        "final_fitness": fitness_list[-1],
        "elapsed_time": elapsed_time,
        "peak_memory_mb": peak_memory_mb
    }

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
    
    # Extrai as coordenadas X e Y de cada cidade do arquivo de mapa
    city_pos_list = np.array([problem.node_coords[node] for node in nodes])
    
    # Constrói a matriz de distância
    city_dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            city_dist_mat[i, j] = problem.get_weight(nodes[i], nodes[j])

    # Executa as 3 versões de SA
    results = {}
    results["SA V1"] = run_sa_version(SaV1, "SA V1", city_dist_mat, T_0=1000, alpha=0.95, patience=300)
    results["SA V2"] = run_sa_version(SaV2, "SA V2", city_dist_mat, T_0=1000, alpha=0.98, patience=300)
    results["SA V3"] = run_sa_version(SaV3, "SA V3", city_dist_mat, T_0=500, alpha=0.90, patience=150)

    # Imprime tabela comparativa no console
    print("\n" + "=" * 80)
    print(f"TABELA COMPARATIVA DE RESULTADOS SA - MAPA: {os.path.basename(selected_file)}")
    print("=" * 80)
    print(f"{'Algoritmo':<15} | {'Melhor Fitness (Distância)':<27} | {'Tempo (segundos)':<16} | {'Pico de Memória':<15}")
    print("-" * 80)
    for name, data in results.items():
        print(f"{name:<15} | {data['final_fitness']:<27.2f} | {data['elapsed_time']:<16.3f} | {data['peak_memory_mb']:<11.3f} MB")
    print("=" * 80 + "\n")

    map_name = os.path.basename(selected_file)

    # Gráfico 1: Comparativo da Curva de Evolução de Fitness (Linha)
    plt.figure(figsize=(10, 6))
    plt.plot(results["SA V1"]["fitness_list"], color='#1f77b4', linewidth=2, label='SA V1 (T0: 1000, a: 0.95, Pat: 300)')
    plt.plot(results["SA V2"]["fitness_list"], color='#ff7f0e', linewidth=2, label='SA V2 (T0: 1000, a: 0.98, Pat: 300)')
    plt.plot(results["SA V3"]["fitness_list"], color='#2ca02c', linewidth=2, label='SA V3 (T0: 500, a: 0.90, Pat: 150)')
    
    plt.title(f"Comparativo de Evolução do Fitness SA (Convergência) - {map_name}")
    plt.xlabel("Iteração")
    plt.ylabel("Distância Total (Menor é melhor)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')

    # Gráfico 2: Subplots com as Rotas Físicas de cada versão lado a lado
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Comparação de Rotas Físicas SA - {map_name}", fontsize=16, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    versions = ["SA V1", "SA V2", "SA V3"]

    for idx, name in enumerate(versions):
        ax = axes[idx]
        route = list(results[name]["result"])
        # Conecta de volta ao ponto de início
        route.append(route[0])
        route_coords = city_pos_list[route, :]
        
        ax.plot(route_coords[:, 0], route_coords[:, 1], 'o-', color=colors[idx], alpha=0.7, label='Rota')
        # Destaca o ponto inicial/final
        ax.plot(route_coords[0, 0], route_coords[0, 1], 'g^', markersize=10, label='Início/Fim')
        
        ax.set_title(f"{name}\nDistância: {results[name]['final_fitness']:.2f}")
        ax.set_xlabel("Coordenada X")
        ax.set_ylabel("Coordenada Y")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='best')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
