# Introdução à Inteligência Artificial - Problema do Caixeiro Viajante (TSP)

Repositório dedicado à implementação de soluções para o **Problema do Caixeiro Viajante (TSP)** utilizando duas abordagens clássicas de inteligência artificial:
1. **Algoritmo Genético (GA)**
2. **Simulated Annealing (SA)**

---

## 🛠️ Requisitos e Instalação

Para rodar os algoritmos deste repositório, você precisará do Python (versão 3.8 ou superior) instalado, além das seguintes bibliotecas externas:

* **numpy**: Para processamento numérico e manipulação de matrizes.
* **matplotlib**: Para geração dos gráficos de rotas e curvas de aprendizado.
* **tsplib95**: Para carregar e interpretar os arquivos de mapa do formato oficial da TSPLIB.

Você pode instalar todas as dependências necessárias executando o comando abaixo no seu terminal:

```bash
pip install numpy matplotlib tsplib95
```

---

## Como Executar os Algoritmos

Ambas as implementações estão configuradas para ler arquivos de instâncias de mapas `.tsp` presentes na pasta `/mapas/`. Você pode rodar os códigos de duas formas:

### 1. Modo Interativo (Escolha de Mapa)
Se você rodar o script sem nenhum argumento, o terminal exibirá a lista de mapas `.tsp` disponíveis na pasta `/mapas/` para você escolher qual deseja rodar.

```bash
# Executar Algoritmo Genético
python algoritmo_genetico/main.py

# Executar Simulated Annealing
python simulated_annealing/main_SA.py
```

### 2. Modo Direto (Passando o Mapa por Argumento)
Você também pode passar o nome do arquivo do mapa diretamente por argumento da linha de comando para iniciar a execução imediatamente.

```bash
# Executar passando apenas o nome do arquivo na pasta 'mapas' ou o caminho completo
python <caminho_do_algoritmo>/<arquivo_principal>.py <nome_do_mapa>.tsp
```

#### Exemplos práticos:
* **Para rodar o Algoritmo Genético:**
  ```bash
  python algoritmo_genetico/main.py berlin52.tsp
  ```
* **Para rodar o Simulated Annealing:**
  ```bash
  python simulated_annealing/main_SA.py berlin52.tsp
  ```

---

## 📊 Saídas e Logs

Ao término da execução de qualquer um dos algoritmos, o programa exibirá:
1. Os logs do progresso de convergência a cada 50 iterações/gerações.
2. O critério de parada atingido (convergência máxima ou estagnação por patience de 300 iterações).
3. **Estatísticas de Performance:** O tempo total de processamento em segundos e o pico de consumo de memória em MB.
4. **Gráficos:**
   * Um plot visual do melhor caminho traçado entre as cidades (com o ponto inicial destacado).
   * A curva de aprendizado (evolução da distância do melhor caminho ao longo do tempo).
