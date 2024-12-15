const aiResponses = {
  1: `

  BFS and DFS in Python: Breadth-First Search (BFS) explores nodes level by level, using a queue to store nodes, while Depth-First Search (DFS) explores as far as possible down each branch before backtracking, using a stack. Both algorithms traverse graphs, but BFS is ideal for finding shortest paths.
  from collections import deque

 # Breadth-First Search (BFS)
 def bfs(graph, start):
     visited = set()
     queue = deque([start])
     result = []

     while queue:
         node = queue.popleft()
         if node not in visited:
             visited.add(node)
             result.append(node)
             queue.extend(graph[node] - visited)

     return result

 # Depth-First Search (DFS)
 def dfs(graph, start, visited=None, result=None):
     if visited is None:
         visited = set()
     if result is None:
         result = []

     visited.add(start)
     result.append(start)

     for neighbor in graph[start] - visited:
         dfs(graph, neighbor, visited, result)

     return result

 # Example graph represented as an adjacency list
 graph = {
     'A': {'B', 'C'},
     'B': {'A', 'D', 'E'},
     'C': {'A', 'F'},
     'D': {'B'},
     'E': {'B', 'F'},
     'F': {'C', 'E'}
 }

 # Example usage of BFS and DFS
 start_node = 'A'

 print("BFS result:", bfs(graph, start_node))
 print("DFS result:", dfs(graph, start_node))`,

  2: `
  Water Jug Problem: The water jug problem involves two jugs with different capacities, and the task is to measure a specific amount of water. The program uses states and transitions to explore the possible combinations and solve the problem using search techniques like BFS or DFS.
  from collections import deque

  def water_jug_problem(capacity_x, capacity_y, target):
      # Initialize the queue for BFS and the visited set
      queue = deque([(0, 0)])  # Initial state (0, 0) meaning both jugs are empty
      visited = set()  # To track visited states
      visited.add((0, 0))

      while queue:
          x, y = queue.popleft()

          # Check if we've reached the target
          if x == target or y == target:
              print(f"Solution found: ({x}, {y})")
              return True

          # Generate possible next states and add them to the queue
          next_states = [
              (capacity_x, y),  # Fill jug X
              (x, capacity_y),  # Fill jug Y
              (0, y),            # Empty jug X
              (x, 0),            # Empty jug Y
              (x - min(x, capacity_y - y), y + min(x, capacity_y - y)),  # Pour from X to Y
              (x + min(y, capacity_x - x), y - min(y, capacity_x - x)),  # Pour from Y to X
          ]

          for state in next_states:
              if state not in visited:
                  visited.add(state)
                  queue.append(state)

      print("No solution exists")
      return False

  # Test the function with capacities and target
  capacity_x = 4  # Capacity of jug X
  capacity_y = 3  # Capacity of jug Y
  target = 2      # Target amount to measure

  water_jug_problem(capacity_x, capacity_y, target)`,

  3: `
  8-Puzzle in Python: The 8-Puzzle problem involves arranging tiles numbered 1-8 in a 3x3 grid with one empty space. The goal is to reach a target configuration using sliding moves. The program uses search algorithms like BFS, DFS, or A* to find the optimal solution.
  from collections import deque

  # Goal state of the puzzle
  goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

  # Directions for possible moves: up, down, left, right
  directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

  def is_solved(state):
      return state == goal_state

  def get_possible_moves(state):
      empty_pos = [(i, row.index(0)) for i, row in enumerate(state) if 0 in row][0]
      x, y = empty_pos
      moves = []

      for dx, dy in directions:
          new_x, new_y = x + dx, y + dy
          if 0 <= new_x < 3 and 0 <= new_y < 3:
              new_state = [row[:] for row in state]
              new_state[x][y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[x][y]
              moves.append(new_state)

      return moves

  def bfs(initial_state):
      queue = deque([(initial_state, [])])
      visited = set()

      while queue:
          current_state, path = queue.popleft()

          if is_solved(current_state):
              return path

          for next_state in get_possible_moves(current_state):
              state_tuple = tuple(tuple(row) for row in next_state)
              if state_tuple not in visited:
                  visited.add(state_tuple)
                  queue.append((next_state, path + [next_state]))

      return None

  # Example initial state
  initial_state = [[1, 2, 3], [4, 5, 6], [0, 7, 8]]

  # Solving the puzzle using BFS
  solution = bfs(initial_state)

  if solution:
      for step in solution:
          for row in step:
              print(row)
          print()
  else:
      print("No solution found")
`,

  4: `
  A Algorithm in Python*: A* is a pathfinding algorithm that finds the shortest path by combining the actual cost to reach a node and the estimated cost to reach the goal. The program uses a priority queue to evaluate nodes and ensures an optimal path using heuristics.


  import heapq

  # A* Algorithm
  def a_star(start, goal, grid):
      def heuristic(a, b):
          return abs(a[0] - b[0]) + abs(a[1] - b[1])

      def get_neighbors(node):
          x, y = node
          neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
          return [n for n in neighbors if 0 <= n[0] < len(grid) and 0 <= n[1] < len(grid[0]) and grid[n[0]][n[1]] == 0]

      open_list = []
      heapq.heappush(open_list, (0 + heuristic(start, goal), 0, start, []))
      closed_list = set()

      while open_list:
          _, cost, current, path = heapq.heappop(open_list)
          if current == goal:
              return path + [current]
          if current in closed_list:
              continue
          closed_list.add(current)
          for neighbor in get_neighbors(current):
              heapq.heappush(open_list, (cost + 1 + heuristic(neighbor, goal), cost + 1, neighbor, path + [current]))
      return None

  # Example grid (0: free space, 1: obstacle)
  grid = [
      [0, 0, 0, 0, 0],
      [0, 1, 1, 1, 0],
      [0, 0, 0, 1, 0],
      [0, 1, 0, 0, 0],
      [0, 0, 0, 0, 0]
  ]

  start = (0, 0)
  goal = (4, 4)

  # Running the A* algorithm
  path = a_star(start, goal, grid)
  print("Path:", path)
`,

  5: `
  Tic-Tac-Toe Game in Python: This game simulates a two-player Tic-Tac-Toe game on a 3x3 grid. The program allows players to take turns marking 'X' or 'O', checks for win conditions, and prints the current game state after each move. It also detects if the game is a draw.
  # Tic-Tac-Toe game

  def print_board(board):
      for row in board:
          print(" | ".join(row))
          print("-" * 5)

  def check_win(board, player):
      # Check rows, columns, and diagonals for a win
      for i in range(3):
          if all([board[i][j] == player for j in range(3)]) or \
             all([board[j][i] == player for j in range(3)]):
              return True
      if board[0][0] == player and board[1][1] == player and board[2][2] == player:
          return True
      if board[0][2] == player and board[1][1] == player and board[2][0] == player:
          return True
      return False

  def check_draw(board):
      return all(board[i][j] != ' ' for i in range(3) for j in range(3))

  def tic_tac_toe():
      board = [[' ' for _ in range(3)] for _ in range(3)]
      current_player = 'X'

      while True:
          print_board(board)
          print(f"Player {current_player}'s turn")

          # Get player's move
          row, col = map(int, input("Enter row and column (0-2) separated by space: ").split())

          # Check if the move is valid
          if board[row][col] != ' ':
              print("Cell already taken. Try again.")
              continue

          # Make the move
          board[row][col] = current_player

          # Check if the current player wins
          if check_win(board, current_player):
              print_board(board)
              print(f"Player {current_player} wins!")
              break

          # Check for a draw
          if check_draw(board):
              print_board(board)
              print("It's a draw!")
              break

          # Switch player
          current_player = 'O' if current_player == 'X' else 'X'

  # Run the game
  tic_tac_toe()
`,

  6: `Traveling Salesman Problem (TSP): The Traveling Salesman Problem (TSP) involves finding the shortest possible route that visits a set of cities and returns to the origin. The program uses brute force, dynamic programming, or approximation algorithms to solve this optimization problem efficiently.
  # Tic-Tac-Toe Game
  import itertools

  # Function to calculate the total distance of a given path
  def calculate_distance(path, distance_matrix):
      distance = 0
      for i in range(len(path) - 1):
          distance += distance_matrix[path[i]][path[i+1]]
      distance += distance_matrix[path[-1]][path[0]]  # Return to the starting point
      return distance

  # Traveling Salesman Problem (TSP) function
  def traveling_salesman(distance_matrix):
      num_cities = len(distance_matrix)
      cities = list(range(num_cities))
      min_distance = float('inf')
      best_path = None

      # Generate all possible permutations of cities (excluding the first city)
      for path in itertools.permutations(cities[1:]):
          # Add the starting city (0) at the beginning and end of the path
          full_path = [0] + list(path)
          distance = calculate_distance(full_path, distance_matrix)

          # Update minimum distance and best path if a shorter path is found
          if distance < min_distance:
              min_distance = distance
              best_path = full_path

      return best_path, min_distance

  # Example distance matrix for 4 cities (symmetric)
  distance_matrix = [
      [0, 10, 15, 20],
      [10, 0, 35, 25],
      [15, 35, 0, 30],
      [20, 25, 30, 0]
  ]

  # Find the shortest path and its distance
  best_path, min_distance = traveling_salesman(distance_matrix)
  print(f"Best path: {best_path}")
  print(f"Minimum distance: {min_distance}")

`,

  7: `
  Bayesian Network in Python: A Bayesian network represents probabilistic relationships among variables. The program constructs a directed acyclic graph (DAG) where nodes represent variables, and edges represent conditional dependencies. It uses probability theory to calculate the likelihood of events based on prior knowledge and observed evidence.
  # Import necessary libraries
  from pgmpy.models import BayesianNetwork
  from pgmpy.factors.discrete import TabularCPD
  from pgmpy.inference import VariableElimination

  # Create a Bayesian Network
  model = BayesianNetwork([('Rain', 'Traffic'), ('Traffic', 'Accident')])

  # Define the Conditional Probability Distributions (CPDs)
  cpd_rain = TabularCPD(variable='Rain', variable_card=2, values=[[0.7], [0.3]])  # P(Rain)
  cpd_traffic = TabularCPD(variable='Traffic', variable_card=2, values=[[0.8, 0.4], [0.2, 0.6]],
                           evidence=['Rain'], evidence_card=[2])  # P(Traffic | Rain)
  cpd_accident = TabularCPD(variable='Accident', variable_card=2, values=[[0.9, 0.7], [0.1, 0.3]],
                            evidence=['Traffic'], evidence_card=[2])  # P(Accident | Traffic)

  # Add CPDs to the model
  model.add_cpds(cpd_rain, cpd_traffic, cpd_accident)

  # Check if the model is valid
  model.check_model()

  # Perform inference to query the probability of an accident given that it is raining
  inference = VariableElimination(model)

  # Query P(Accident | Rain)
  query_result = inference.query(variables=['Accident'], evidence={'Rain': 1})
  print(query_result)

  # Query P(Traffic | Rain)
  query_traffic = inference.query(variables=['Traffic'], evidence={'Rain': 1})
  print(query_traffic)
`,

  8: `
  Fuzzy Sets in Python: Fuzzy sets represent uncertainty by allowing partial membership in a set. The program performs operations like union, intersection, and complement on fuzzy sets, which are used in fuzzy logic systems for decision-making and approximating human reasoning in uncertain conditions.
  # Import necessary libraries
  from pgmpy.models import BayesianNetwork
  from pgmpy.factors.discrete import TabularCPD
  from pgmpy.inference import VariableElimination

  # Step 1: Create the structure of the Bayesian Network (edges between nodes)
  model = BayesianNetwork([('Rain', 'Traffic'), ('Traffic', 'Accident')])

  # Step 2: Define the Conditional Probability Distributions (CPDs)

  # P(Rain)
  cpd_rain = TabularCPD(variable='Rain', variable_card=2, values=[[0.7], [0.3]])  # 70% chance of no rain, 30% chance of rain

  # P(Traffic | Rain)
  # If Rain = 0 (no rain), there's a 20% chance of traffic.
  # If Rain = 1 (rain), there's a 60% chance of traffic.
  cpd_traffic = TabularCPD(variable='Traffic', variable_card=2,
                           values=[[0.8, 0.4], [0.2, 0.6]],
                           evidence=['Rain'], evidence_card=[2])

  # P(Accident | Traffic)
  # If there's no traffic, the chance of an accident is 10%.
  # If there's traffic, the chance of an accident is 50%.
  cpd_accident = TabularCPD(variable='Accident', variable_card=2,
                            values=[[0.9, 0.5], [0.1, 0.5]],
                            evidence=['Traffic'], evidence_card=[2])

  # Step 3: Add the CPDs to the model
  model.add_cpds(cpd_rain, cpd_traffic, cpd_accident)

  # Step 4: Check if the model is valid
  model.check_model()

  # Step 5: Perform inference to query the probability of an accident, given that it is raining
  inference = VariableElimination(model)

  # Query P(Accident | Rain)
  query_result = inference.query(variables=['Accident'], evidence={'Rain': 1})
  print("Probability of Accident given it is raining:")
  print(query_result)

  # Query P(Traffic | Rain)
  query_traffic = inference.query(variables=['Traffic'], evidence={'Rain': 1})
  print("\nProbability of Traffic given it is raining:")
  print(query_traffic)
`,

  9: `
  Constraint Satisfaction Problem (CSP) in Python: CSPs involve finding solutions to problems with constraints. The program defines variables, domains, and constraints, and uses techniques like backtracking or constraint propagation to find valid assignments that satisfy all constraints, commonly applied in scheduling, puzzles, and optimization problems.
  # Constraint Satisfaction Problem (CSP) - N-Queens Problem
  def is_safe(board, row, col, N):
      # Check column
      for i in range(row):
          if board[i] == col or abs(board[i] - col) == abs(i - row):
              return False
      return True

  def solve_n_queens_util(board, row, N):
      # If all queens are placed
      if row >= N:
          return True

      # Try all columns in this row
      for col in range(N):
          if is_safe(board, row, col, N):
              board[row] = col  # Place queen
              if solve_n_queens_util(board, row + 1, N):
                  return True
              board[row] = -1  # Backtrack

      return False

  def solve_n_queens(N):
      # Initialize board with -1 indicating no queen placed
      board = [-1] * N

      if solve_n_queens_util(board, 0, N):
          # Solution found, print the board
          solution = []
          for row in range(N):
              solution.append(['Q' if col == board[row] else '.' for col in range(N)])
          return solution
      else:
          return None

  def print_board(solution):
      if solution:
          for row in solution:
              print(" ".join(row))
      else:
          print("No solution found")

  # Example: Solve 4-Queens Problem
  N = 4
  solution = solve_n_queens(N)
  print_board(solution)
`,

  10: `
  Perceptron Model in Python: The perceptron is a type of neural network used for binary classification. It learns from input data by adjusting weights using the perceptron learning rule. The program trains the model on labeled data and makes predictions using the step activation function, suitable for linearly separable problems.
  import numpy as np

  class Perceptron:
      def __init__(self, input_size, learning_rate=0.1, epochs=1000):
          # Initialize weights and bias
          self.weights = np.zeros(input_size)
          self.bias = 0
          self.learning_rate = learning_rate
          self.epochs = epochs

      def activation_function(self, x):
          # Step activation function (returns 1 if x >= 0, else -1)
          return 1 if x >= 0 else -1

      def predict(self, X):
          # Weighted sum and apply activation function
          linear_output = np.dot(X, self.weights) + self.bias
          return self.activation_function(linear_output)

      def fit(self, X, y):
          # Training the perceptron model
          for epoch in range(self.epochs):
              for i in range(len(X)):
                  # Calculate the prediction
                  prediction = self.predict(X[i])

                  # Update weights and bias if there is an error
                  if prediction != y[i]:
                      # Weight update rule: w = w + learning_rate * (y - y_pred) * x
                      self.weights += self.learning_rate * (y[i] - prediction) * X[i]
                      self.bias += self.learning_rate * (y[i] - prediction)

      def evaluate(self, X, y):
          # Evaluate the model's performance
          correct_predictions = 0
          for i in range(len(X)):
              if self.predict(X[i]) == y[i]:
                  correct_predictions += 1
          accuracy = correct_predictions / len(X) * 100
          return accuracy


  # Example usage with a simple AND dataset
  if __name__ == "__main__":
      # Input data for AND operation (X1, X2)
      X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
      # Output data for AND operation (Y)
      y = np.array([-1, -1, -1, 1])  # Use -1 for False and 1 for True in Perceptron

      # Initialize the Perceptron
      perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=100)

      # Train the model
      perceptron.fit(X, y)

      # Evaluate the model
      accuracy = perceptron.evaluate(X, y)
      print(f"Model Accuracy: {accuracy}%")

      # Predicting results
      for i in range(len(X)):
          print(f"Input: {X[i]}, Predicted Output: {perceptron.predict(X[i])}, Actual Output: {y[i]}")
`,
};

module.exports = aiResponses;
