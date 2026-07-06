import sys
import os
import time
import tracemalloc
import random
import math
import numpy as np
import tsplib95
import matplotlib.pyplot as plt

# Adiciona o diretório do script ao sys.path para garantir que os imports locais funcionem
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from random_initial_solution_tsp import random_initial_solution_tsp
from swap_permutation import swap_permutation
from objective_function_tsp import objective_function_tsp

def main():
    # Inicia o rastreamento de memória e tempo para medir tudo usado na execução
    tracemalloc.start()
    start_time = time.perf_counter()

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
    
    # Constrói a matriz de distância com os pesos das arestas oficiais do formato TSPLIB (EUC_2D)
    city_dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            city_dist_mat[i, j] = problem.get_weight(nodes[i], nodes[j])
            
    print("[INFO] Matriz de distancias gerada. Iniciando o Simulated Annealing...")

    # Parameters
    T_0 = 1000
    T_f = 1.5
    alpha = 0.95
    patience = 300 # Num. de iterações sem melhora antes de parar o algoritmo

    # Initial setting
    num_iter = 0
    max_iter = 2000
    best_solution_found = []
    T = T_0
    s_0 = random_initial_solution_tsp(n)
    f_0 = objective_function_tsp(n, s_0, city_dist_mat)

    s_best = s_0.copy()
    f_best = f_0
    last_improvement_iter = 0

    print("Solução inicial gerada.")
    print(f"Melhor distância da solução inicial: {f_best:.2f}")

    while num_iter < max_iter:
        s_1 = swap_permutation(s_0, n)
        f_1 = objective_function_tsp(n, s_1, city_dist_mat)
        if (f_1 - f_0 < 0) or (random.random() < math.exp((f_0 - f_1) / T)):
            s_0 = s_1.copy()
            f_0 = f_1
        if f_0 < f_best:
            f_best = f_0
            s_best = s_0.copy()
            last_improvement_iter = num_iter

        if T <= T_f:
            T = T_0
        else:
            T = alpha * T

        best_solution_found.append(f_best)
        
        # Critério de parada antecipada se não houver melhora
        reached_patience = num_iter - last_improvement_iter >= patience
        
        # Gera logs detalhados a cada 50 iterações, na última iteração ou caso pare antecipadamente
        if (num_iter + 1) % 50 == 0 or (num_iter + 1) == max_iter or reached_patience:
            print(f"Iteração {num_iter + 1:4d}/{max_iter} | Melhor Distância = {f_best:.2f}")
            
        if reached_patience:
            print(f"\n[CRITÉRIO DE PARADA] O algoritmo parou na iteração {num_iter + 1} porque a distância mínima não diminuiu por {patience} iterações consecutivas.")
            break
            
        num_iter += 1

    # Mapeia os índices das cidades na melhor rota final para as suas coordenadas geométricas correspondentes
    result_route = list(s_best)
    result_route.append(result_route[0])
    result_pos_list = city_pos_list[result_route, :]
    
    print(f"\n[SUCESSO] Execucao concluida!")
    print(f"Melhor distancia encontrada (Fitness final): {f_best:.2f}")
    
    # Gráfico 1: Plot da Rota Física Encontrada
    map_name = os.path.basename(selected_file)
    plt.figure(figsize=(8, 6))
    plt.plot(result_pos_list[:, 0], result_pos_list[:, 1], 'o-r', label='Trecho da Rota')
    # Destaca o ponto inicial/final do caixeiro
    plt.plot(result_pos_list[0, 0], result_pos_list[0, 1], 'g^', markersize=12, label='Ponto Inicial/Final')
    plt.title(f"Simulated Annealing - Melhor Rota - {map_name}")
    plt.xlabel("Coordenada X (Leste-Oeste)")
    plt.ylabel("Coordenada Y (Norte-Sul)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    
    # Gráfico 2: Evolução da Aptidão (Fitness)
    plt.figure(figsize=(8, 5))
    plt.plot(best_solution_found, color='blue', linewidth=2, label='Distancia da Melhor Rota')
    plt.title(f"Simulated Annealing - Evolução do Fitness - {map_name}")
    plt.xlabel("Geracao (Iteracao)")
    plt.ylabel("Distancia Total do Caminho (Menor e melhor)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    
    # Captura o tempo total e pico de memória antes do plt.show() (que bloqueia a execução)
    end_time = time.perf_counter()
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    elapsed_time = end_time - start_time
    peak_memory_mb = peak_memory / (1024 * 1024)
    
    print("\n" + "=" * 50)
    print("ESTATÍSTICAS DE PERFORMANCE DA EXECUÇÃO")
    print("=" * 50)
    print(f"Tempo total de processamento: {elapsed_time:.3f} segundos")
    print(f"Pico de consumo de memória: {peak_memory_mb:.3f} MB")
    print("=" * 50 + "\n")
    
    plt.show()

if __name__ == "__main__":
    main()
