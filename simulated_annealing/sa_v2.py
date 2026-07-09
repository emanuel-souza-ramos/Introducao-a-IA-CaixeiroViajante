import sys
import os
import random
import math
import numpy as np

# Adiciona o diretório do script ao sys.path para garantir que os imports locais funcionem
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from random_initial_solution_tsp import random_initial_solution_tsp
from swap_permutation import swap_permutation
from objective_function_tsp import objective_function_tsp

# Classe que gerencia o ciclo do Simulated Annealing V2 (Resfriamento Mais Lento / Maior Exploração)
class Sa:
    def __init__(self, city_dist_mat, T_0=1000, T_f=1.5, alpha=0.98, patience=300, max_iter=2000):
        self.city_dist_mat = city_dist_mat
        self.n = len(city_dist_mat)
        self.T_0 = T_0
        self.T_f = T_f
        self.alpha = alpha
        self.patience = patience
        self.max_iter = max_iter
        self.result_list = []
        self.fitness_list = []

    def train(self):
        num_iter = 0
        T = self.T_0
        s_0 = random_initial_solution_tsp(self.n)
        f_0 = objective_function_tsp(self.n, s_0, self.city_dist_mat)

        s_best = s_0.copy()
        f_best = f_0
        last_improvement_iter = 0

        print("Solução inicial gerada.")
        print(f"Melhor distância da solução inicial: {f_best:.2f}")

        while num_iter < self.max_iter:
            s_1 = swap_permutation(s_0, self.n)
            f_1 = objective_function_tsp(self.n, s_1, self.city_dist_mat)
            if (f_1 - f_0 < 0) or (random.random() < math.exp((f_0 - f_1) / T)):
                s_0 = s_1.copy()
                f_0 = f_1
            if f_0 < f_best:
                f_best = f_0
                s_best = s_0.copy()
                last_improvement_iter = num_iter

            if T <= self.T_f:
                T = self.T_0
            else:
                T = self.alpha * T

            self.result_list.append(s_best.copy())
            self.fitness_list.append(f_best)
            
            # Critério de parada antecipada se não houver melhora
            reached_patience = num_iter - last_improvement_iter >= self.patience
            
            # Gera logs detalhados a cada 50 iterações, na última iteração ou caso pare antecipadamente
            if (num_iter + 1) % 50 == 0 or (num_iter + 1) == self.max_iter or reached_patience:
                print(f"Iteração {num_iter + 1:4d}/{self.max_iter} | Melhor Distância = {f_best:.2f}")
                
            if reached_patience:
                print(f"\n[CRITÉRIO DE PARADA] O algoritmo parou na iteração {num_iter + 1} porque a distância mínima não diminuiu por {self.patience} iterações consecutivas.")
                break
                
            num_iter += 1

        return self.result_list, self.fitness_list
