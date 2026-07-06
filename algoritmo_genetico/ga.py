import random

# A heurística de Algoritmo Genético (AG) para o Caixeiro Viajante trabalha com uma POPULAÇÃO
# de caminhos possíveis simultaneamente. Cada "Individual" representa uma rota candidata.
# Através das gerações, as melhores rotas são selecionadas e combinadas para gerar rotas ainda melhores.

city_dist_mat = None
gene_len = 0

def copy_list(old_arr: [int]):
    new_arr = []
    for element in old_arr:
        new_arr.append(element)
    return new_arr

# Classe que representa uma rota (indivíduo) na população de caminhos simultâneos
class Individual:
    def __init__(self, genes=None):
        if genes is None:
            # Se não houver rota prévia, cria uma rota aleatória (permutação de cidades)
            genes = [i for i in range(gene_len)]
            random.shuffle(genes)
        self.genes = genes
        self.fitness = self.evaluate_fitness()

    def evaluate_fitness(self):
        # O "fitness" (aptidão) aqui é a distância total da rota.
        # Como queremos minimizar a distância, um valor menor de fitness é melhor.
        fitness = 0.0
        for i in range(gene_len - 1):
            from_idx = self.genes[i]
            to_idx = self.genes[i + 1]
            fitness += city_dist_mat[from_idx, to_idx]
        # Conecta a última cidade de volta à primeira para fechar o ciclo
        fitness += city_dist_mat[self.genes[-1], self.genes[0]]
        return fitness

# Classe que gerencia o ciclo evolutivo da população de caminhos
class Ga:
    def __init__(self, input_, individual_num=100, gen_num=2000, mutate_prob=0.2, patience=300, tournament_size=5):
        global city_dist_mat, gene_len
        city_dist_mat = input_
        gene_len = len(input_)
        self.individual_num = individual_num
        self.gen_num = gen_num
        self.mutate_prob = mutate_prob
        self.patience = patience
        self.tournament_size = tournament_size
        self.best = None
        self.individual_list = []
        self.result_list = []
        self.fitness_list = []

    def cross(self):
        # Cruzamento PMX (Partially Mapped Crossover):
        # Tática de recombinação para permutações que evita cidades duplicadas ou ausentes.
        # Seleciona uma seção da rota de um pai e mapeia na rota do outro pai.
        new_gen = []
        random.shuffle(self.individual_list)
        for i in range(0, self.individual_num - 1, 2):
            genes1 = copy_list(self.individual_list[i].genes)
            genes2 = copy_list(self.individual_list[i + 1].genes)
            index1 = random.randint(0, gene_len - 2)
            index2 = random.randint(index1, gene_len - 1)
            pos1_recorder = {value: idx for idx, value in enumerate(genes1)}
            pos2_recorder = {value: idx for idx, value in enumerate(genes2)}
            for j in range(index1, index2):
                value1, value2 = genes1[j], genes2[j]
                pos1, pos2 = pos1_recorder[value2], pos2_recorder[value1]
                genes1[j], genes1[pos1] = genes1[pos1], genes1[j]
                genes2[j], genes2[pos2] = genes2[pos2], genes2[j]
                pos1_recorder[value1], pos1_recorder[value2] = pos1, j
                pos2_recorder[value1], pos2_recorder[value2] = j, pos2
            new_gen.append(Individual(genes1))
            new_gen.append(Individual(genes2))
        return new_gen

    def mutate(self, new_gen):
        # Mutação por Inversão (Inversion Mutation):
        # Escolhe um trecho aleatório do caminho e inverte a ordem das cidades.
        # Essa mutação é uma heurística excelente para o TSP pois preserva a maioria das conexões adjacentes.
        for individual in new_gen:
            if random.random() < self.mutate_prob:
                old_genes = copy_list(individual.genes)
                index1 = random.randint(0, gene_len - 2)
                index2 = random.randint(index1, gene_len - 1)
                genes_mutate = old_genes[index1:index2]
                genes_mutate.reverse()
                individual.genes = old_genes[:index1] + genes_mutate + old_genes[index2:]
        self.individual_list += new_gen

    def select(self):
        # Seleção por Torneio (Tournament Selection):
        # Para preencher cada uma das vagas da nova geração (self.individual_num),
        # sorteamos 'tournament_size' indivíduos aleatórios da população atual (pais + filhos)
        # e o de melhor aptidão (menor distância no TSP) vence o torneio e é selecionado.
        winners = []
        for _ in range(self.individual_num):
            competitors = random.sample(self.individual_list, k=self.tournament_size)
            winner = min(competitors, key=lambda ind: ind.fitness)
            # Instancia um novo Individual para copiar os genes e evitar compartilhamento de referência em memória
            winners.append(Individual(winner.genes))
        self.individual_list = winners

    @staticmethod
    def rank(group):
        # Ordenação bolha simples para ordenar as rotas da menor distância para a maior
        for i in range(1, len(group)):
            for j in range(0, len(group) - i):
                if group[j].fitness > group[j + 1].fitness:
                    group[j], group[j + 1] = group[j + 1], group[j]
        return group

    def next_gen(self):
        # Executa uma geração completa do ciclo evolutivo
        new_gen = self.cross()
        self.mutate(new_gen)
        self.select()
        for individual in self.individual_list:
            if individual.fitness < self.best.fitness:
                self.best = individual

    def train(self):
        # Inicializa a população com caminhos aleatórios gerados simultaneamente
        self.individual_list = [Individual() for _ in range(self.individual_num)]
        
        # Encontra a melhor rota na população inicial para servir de base comparativa inicial
        self.best = self.individual_list[0]
        for individual in self.individual_list:
            if individual.fitness < self.best.fitness:
                self.best = individual
        
        print(f"População inicial gerada com {self.individual_num} caminhos simultâneos.")
        print(f"Melhor distância da população inicial: {self.best.fitness:.2f}")
        
        # no_improvement_count conta quantas gerações consecutivas se passaram sem que a menor distância diminuísse
        no_improvement_count = 0
        
        # Executa a busca heurística evolutiva pelas gerações definidas
        for i in range(self.gen_num):
            prev_best_fitness = self.best.fitness
            self.next_gen()
            
            # Se a melhor distância encontrada diminuiu nesta geração, reseta o contador de estagnação.
            # Caso contrário, incrementa o contador.
            if self.best.fitness < prev_best_fitness:
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Copia os genes do melhor indivíduo e adiciona o ponto inicial ao final para fechar o ciclo do TSP
            result = copy_list(self.best.genes)
            result.append(result[0])
            self.result_list.append(result)
            self.fitness_list.append(self.best.fitness)
            
            # Verifica se atingimos o limite de gerações consecutivas sem melhorias (critério de parada / patience)
            reached_patience = self.patience is not None and no_improvement_count >= self.patience
            
            # Gera logs detalhados a cada 50 gerações, na última geração ou caso pare antecipadamente
            if (i + 1) % 50 == 0 or (i + 1) == self.gen_num or reached_patience:
                print(f"Geração {i + 1:3d}/{self.gen_num} | Melhor Distância = {self.best.fitness:.2f}")
            
            # Executa a parada prematura por estagnação
            if reached_patience:
                print(f"\n[CRITÉRIO DE PARADA] O algoritmo parou na geração {i + 1} porque a distância mínima não diminuiu por {self.patience} gerações consecutivas.")
                break
                
        return self.result_list, self.fitness_list
