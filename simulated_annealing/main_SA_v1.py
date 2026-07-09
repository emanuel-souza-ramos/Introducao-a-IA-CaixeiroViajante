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

from sa_v1 import Sa

# Heurística para resolver o problema do Caixeiro Viajante (TSP) usando Simulated Annealing V1.
# Parâmetros padrão: T_0 = 1000, alpha = 0.95, patience = 300

def main():
    tracemalloc.start()
    start_time = time.perf_counter()

    mapas_dir = os.path.abspath(os.path.join(script_dir, '..', 'mapas'))

    if os.path.exists(mapas_dir):
        tsp_files = [f for f in os.listdir(mapas_dir) if f.endswith('.tsp')]
    else:
        tsp_files = []

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
    
    city_pos_list = np.array([problem.node_coords[node] for node in nodes])
    
    city_dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            city_dist_mat[i, j] = problem.get_weight(nodes[i], nodes[j])
            
    print("[INFO] Matriz de distancias gerada. Iniciando o Simulated Annealing V1...")
    
    sa = Sa(city_dist_mat, T_0=1000, T_f=1.5, alpha=0.95, patience=300)
    result_list, fitness_list = sa.train()
    result = list(result_list[-1])
    result.append(result[0])
    
    result_pos_list = city_pos_list[result, :]
    
    print(f"\n[SUCESSO] Execucao concluida!")
    print(f"Melhor distancia encontrada (Fitness final): {fitness_list[-1]:.2f}")
    
    map_name = os.path.basename(selected_file)
    
    # Gráfico 1: Plot da Rota Física Encontrada
    plt.figure(figsize=(8, 6))
    plt.plot(result_pos_list[:, 0], result_pos_list[:, 1], 'o-r', label='Trecho da Rota')
    plt.plot(result_pos_list[0, 0], result_pos_list[0, 1], 'g^', markersize=12, label='Ponto Inicial/Final')
    plt.title(f"Simulated Annealing V1 - Melhor Rota - {map_name}")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    
    # Gráfico 2: Evolução da Aptidão (Fitness)
    plt.figure(figsize=(8, 5))
    plt.plot(fitness_list, color='blue', linewidth=2, label='Distancia da Melhor Rota')
    plt.title(f"Simulated Annealing V1 - Evolução do Fitness - {map_name}")
    plt.xlabel("Geração (Iteração)")
    plt.ylabel("Distancia Total")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    
    end_time = time.perf_counter()
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    elapsed_time = end_time - start_time
    peak_memory_mb = peak_memory / (1024 * 1024)
    
    print("\n" + "=" * 50)
    print("ESTATÍSTICAS DE PERFORMANCE DA EXECUÇÃO (SA V1)")
    print("=" * 50)
    print(f"Tempo total de processamento: {elapsed_time:.3f} segundos")
    print(f"Pico de consumo de memória: {peak_memory_mb:.3f} MB")
    print("=" * 50 + "\n")
    
    plt.show()

if __name__ == "__main__":
    main()
