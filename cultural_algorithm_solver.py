# CulturalAlgorithmSolver.py - Enhanced Version with GA & CA (FIXED with Timeout)
# Includes: plateau detection, restart mechanism, local search, timeout, and adaptive parameters

import random
import time
from typing import List, Optional, Tuple, Dict
from Solver import Solver


class Individual:
    """Represents a candidate solution with enhanced fitness calculation"""
    
    def __init__(self, board, fixed_mask):
        self.board = board.copy()
        self.fixed_mask = fixed_mask
        self.fitness = 0.0
        self.conflicts = 0
        self.calculate_fitness()

    def calculate_fitness(self):
        """Calculate fitness with exponential penalty for conflicts"""
        self.conflicts = self.board.get_conflicts()
        if self.conflicts == 0:
            self.fitness = float('inf')
        else:
            self.fitness = 1.0 / (1.0 + self.conflicts ** 2.0)

    def copy(self):
        new_ind = Individual.__new__(Individual)
        new_ind.board = self.board.copy()
        new_ind.fixed_mask = self.fixed_mask
        new_ind.fitness = self.fitness
        new_ind.conflicts = self.conflicts
        return new_ind
    
    def update_fitness(self):
        self.calculate_fitness()
    
    def __lt__(self, other):
        return self.fitness < other.fitness


class BeliefSpace:
    """Enhanced Belief Space with plateau detection and adaptive influence"""
    
    def __init__(self, size: int):
        self.size = size
        self.box_size = int(size ** 0.5)
        self.best_individual: Optional[Individual] = None
        self.best_fitness_history: List[float] = []
        self.cell_preferences = [[{} for _ in range(size)] for _ in range(size)]
        self.stagnation_counter = 0
        self.last_best_conflicts = float('inf')

    def update(self, accepted_individuals: List[Individual]):
        for ind in accepted_individuals:
            if (self.best_individual is None or ind.fitness > self.best_individual.fitness):
                self.best_individual = ind.copy()
            weight = ind.fitness ** 2
            for i in range(self.size):
                for j in range(self.size):
                    val = ind.board.grid[i][j]
                    if val != 0:
                        if val not in self.cell_preferences[i][j]:
                            self.cell_preferences[i][j][val] = 0
                        self.cell_preferences[i][j][val] += weight
        
        if self.best_individual:
            if self.best_individual.conflicts == self.last_best_conflicts:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
                self.last_best_conflicts = self.best_individual.conflicts

    def influence(self, individual: Individual, fixed_mask: List[List[bool]], influence_rate: float = 0.6):
        board = individual.board
        is_stagnant = self.stagnation_counter > 20
        actual_rate = influence_rate * (0.3 if is_stagnant else 1.0)
        
        for i in range(self.size):
            for j in range(self.size):
                if not fixed_mask[i][j]:
                    prefs = self.cell_preferences[i][j]
                    if prefs and random.random() < actual_rate:
                        candidates = sorted(prefs.items(), key=lambda x: -x[1])
                        if is_stagnant:
                            if random.random() < 0.7:
                                nums = list(range(1, self.size + 1))
                                random.shuffle(nums)
                                for num in nums:
                                    if board.is_valid(i, j, num):
                                        board.grid[i][j] = num
                                        break
                            else:
                                top_n = min(len(candidates), 5)
                                val, _ = random.choice(candidates[-top_n:])
                                if board.is_valid(i, j, val):
                                    board.grid[i][j] = val
                        else:
                            top_n = min(5, len(candidates))
                            val, _ = random.choice(candidates[:top_n])
                            if board.is_valid(i, j, val):
                                board.grid[i][j] = val
                            else:
                                nums = list(range(1, self.size + 1))
                                random.shuffle(nums)
                                for num in nums:
                                    if board.is_valid(i, j, num):
                                        board.grid[i][j] = num
                                        break


class ParameterOptimizer:
    """Optimized parameters based on board size and difficulty"""
    
    @staticmethod
    def get_optimal_params(size: int, difficulty: float = 0.5) -> Dict:
        base_pop = {4: 150, 6: 300, 9: 400}
        base_gens = {4: 400, 6: 800, 9: 1000}
        pop_size = base_pop.get(size, 400)
        generations = base_gens.get(size, 1000)
        pop_size = int(pop_size * (1 + difficulty * 0.4))
        generations = int(generations * (1 + difficulty * 0.5))
        
        return {
            'pop_size': pop_size,
            'generations': generations,
            'accept_rate': 0.4,
            'elitism_rate': 0.15,
            'mutation_rate': 0.25,
            'tournament_size': max(7, size // 2 + 2),
            'influence_rate': 0.6,
            'restart_threshold': 60,
            'max_time': 120
        }


class GeneticAlgorithmSolver(Solver):
    """Enhanced GA with plateau detection, restart, and local search"""
    
    def __init__(self, board, pop_size=None, generations=None, params=None):
        super().__init__(board)
        self.original_board = board.copy()
        self.size = board.size
        if params is None:
            params = ParameterOptimizer.get_optimal_params(self.size)
        self.pop_size = pop_size or params['pop_size']
        self.max_generations = generations or params['generations']
        self.elitism_rate = params.get('elitism_rate', 0.15)
        self.mutation_rate = params.get('mutation_rate', 0.25)
        self.tournament_size = params.get('tournament_size', 7)
        self.restart_threshold = params.get('restart_threshold', 60)
        self.max_time = params.get('max_time', 120)
        self.fixed_mask = [[board.grid[i][j] != 0 for j in range(self.size)] for i in range(self.size)]
        self.population: List[Individual] = []
        self.current_generation = 0
        self.is_running = True
        self.best_conflicts_history = []
        self.stagnation_counter = 0

    def initialize_population(self):
        self.population = []
        strategies = ['random', 'smart_random', 'row_based', 'constraint']
        for i in range(self.pop_size):
            board = self.original_board.copy()
            strategy = strategies[i % len(strategies)]
            if strategy == 'random':
                for r in range(self.size):
                    for c in range(self.size):
                        if not self.fixed_mask[r][c]:
                            board.grid[r][c] = random.randint(1, self.size)
            elif strategy == 'smart_random':
                for r in range(self.size):
                    for c in range(self.size):
                        if not self.fixed_mask[r][c]:
                            candidates = list(range(1, self.size + 1))
                            random.shuffle(candidates)
                            placed = False
                            for num in candidates:
                                if board.is_valid(r, c, num):
                                    board.grid[r][c] = num
                                    placed = True
                                    break
                            if not placed:
                                board.grid[r][c] = random.randint(1, self.size)
            elif strategy == 'row_based':
                for r in range(self.size):
                    available = list(range(1, self.size + 1))
                    random.shuffle(available)
                    for c in range(self.size):
                        if not self.fixed_mask[r][c]:
                            board.grid[r][c] = available.pop() if available else random.randint(1, self.size)
            else:
                for r in range(self.size):
                    for c in range(self.size):
                        if not self.fixed_mask[r][c]:
                            valid = [n for n in range(1, self.size + 1) if board.is_valid(r, c, n)]
                            board.grid[r][c] = random.choice(valid) if valid else random.randint(1, self.size)
            self.population.append(Individual(board, self.fixed_mask))

    def restart_population(self, keep_best: int = 5):
        best_individuals = self.population[:keep_best]
        for ind in best_individuals:
            if ind.conflicts <= 5:
                self.local_search(ind, max_iterations=30)
        self.population = best_individuals
        strategies = ['pure_random', 'constraint_heavy', 'row_permutation', 'hybrid']
        while len(self.population) < self.pop_size:
            board = self.original_board.copy()
            strategy = random.choice(strategies)
            if strategy == 'pure_random':
                for i in range(self.size):
                    for j in range(self.size):
                        if not self.fixed_mask[i][j]:
                            board.grid[i][j] = random.randint(1, self.size)
            elif strategy == 'constraint_heavy':
                for i in range(self.size):
                    for j in range(self.size):
                        if not self.fixed_mask[i][j]:
                            valid = [n for n in range(1, self.size + 1) if board.is_valid(i, j, n)]
                            board.grid[i][j] = random.choice(valid) if valid else random.randint(1, self.size)
            elif strategy == 'row_permutation':
                for i in range(self.size):
                    available = [n for n in range(1, self.size + 1)]
                    random.shuffle(available)
                    idx = 0
                    for j in range(self.size):
                        if not self.fixed_mask[i][j]:
                            board.grid[i][j] = available[idx]
                            idx += 1
            else:
                for i in range(self.size):
                    for j in range(self.size):
                        if not self.fixed_mask[i][j]:
                            if random.random() < 0.5:
                                board.grid[i][j] = random.randint(1, self.size)
                            else:
                                valid = [n for n in range(1, self.size + 1) if board.is_valid(i, j, n)]
                                board.grid[i][j] = random.choice(valid) if valid else random.randint(1, self.size)
            self.population.append(Individual(board, self.fixed_mask))
        self.stagnation_counter = 0

    def selection(self) -> Individual:
        tournament = random.sample(self.population, self.tournament_size)
        progress = self.current_generation / self.max_generations if self.max_generations > 0 else 0
        random_chance = 0.25 * (1 - progress * 0.5)
        if random.random() < random_chance:
            return random.choice(tournament)
        return max(tournament, key=lambda x: x.fitness)

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        child_board = self.original_board.copy()
        strategy = random.choice(['uniform', 'row_wise', 'box_wise'])
        if strategy == 'uniform':
            for i in range(self.size):
                for j in range(self.size):
                    if not self.fixed_mask[i][j]:
                        child_board.grid[i][j] = (parent1.board.grid[i][j] if random.random() < 0.5 else parent2.board.grid[i][j])
        elif strategy == 'row_wise':
            for i in range(self.size):
                parent = parent1 if random.random() < 0.5 else parent2
                for j in range(self.size):
                    if not self.fixed_mask[i][j]:
                        child_board.grid[i][j] = parent.board.grid[i][j]
        else:
            box_size = int(self.size ** 0.5)
            for box_r in range(0, self.size, box_size):
                for box_c in range(0, self.size, box_size):
                    parent = parent1 if random.random() < 0.5 else parent2
                    for i in range(box_r, box_r + box_size):
                        for j in range(box_c, box_c + box_size):
                            if not self.fixed_mask[i][j]:
                                child_board.grid[i][j] = parent.board.grid[i][j]
        return Individual(child_board, self.fixed_mask)

    def local_search(self, individual: Individual, max_iterations: int = 50):
        board = individual.board
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and individual.conflicts > 0:
            improved = False
            iterations += 1
            conflict_cells = []
            for i in range(self.size):
                for j in range(self.size):
                    if not self.fixed_mask[i][j]:
                        conflicts = self._count_cell_conflicts(board, i, j)
                        if conflicts > 0:
                            conflict_cells.append((conflicts, i, j))
            if not conflict_cells:
                break
            conflict_cells.sort(reverse=True)
            for _, i, j in conflict_cells[:min(3, len(conflict_cells))]:
                current_val = board.grid[i][j]
                best_val = current_val
                best_total_conflicts = individual.conflicts
                for num in range(1, self.size + 1):
                    if num == current_val:
                        continue
                    board.grid[i][j] = num
                    new_conflicts = board.get_conflicts()
                    if new_conflicts < best_total_conflicts:
                        best_val = num
                        best_total_conflicts = new_conflicts
                        improved = True
                board.grid[i][j] = best_val
                individual.calculate_fitness()
                if individual.conflicts == 0:
                    return
    
    def mutate(self, individual: Individual):
        board = individual.board
        if individual.conflicts <= 5 and individual.conflicts > 0:
            self.local_search(individual, max_iterations=50)
            if individual.conflicts == 0:
                return
        adaptive_rate = self.mutation_rate
        if individual.fitness < 0.3:
            adaptive_rate *= 2.0
        elif individual.fitness < 0.5:
            adaptive_rate *= 1.5
        if self.stagnation_counter > 30:
            adaptive_rate *= 1.8
        mutation_type = random.choice(['smart_cell', 'swap', 'row_shuffle'])
        if mutation_type == 'smart_cell':
            conflict_cells = []
            for i in range(self.size):
                for j in range(self.size):
                    if not self.fixed_mask[i][j]:
                        conflicts = self._count_cell_conflicts(board, i, j)
                        if conflicts > 0:
                            conflict_cells.append((conflicts, i, j))
            conflict_cells.sort(reverse=True)
            for _, i, j in conflict_cells[:max(3, int(self.size * adaptive_rate))]:
                best_val = board.grid[i][j]
                best_conflicts = float('inf')
                candidates = list(range(1, self.size + 1))
                random.shuffle(candidates)
                for num in candidates:
                    if board.is_valid(i, j, num):
                        board.grid[i][j] = num
                        conflicts = self._count_cell_conflicts(board, i, j)
                        if conflicts < best_conflicts:
                            best_val = num
                            best_conflicts = conflicts
                board.grid[i][j] = best_val
        elif mutation_type == 'swap':
            for i in range(self.size):
                if random.random() < adaptive_rate:
                    non_fixed = [j for j in range(self.size) if not self.fixed_mask[i][j]]
                    if len(non_fixed) >= 2:
                        j1, j2 = random.sample(non_fixed, 2)
                        board.grid[i][j1], board.grid[i][j2] = board.grid[i][j2], board.grid[i][j1]
        else:
            if random.random() < adaptive_rate * 0.7:
                row = random.randint(0, self.size - 1)
                non_fixed = [j for j in range(self.size) if not self.fixed_mask[row][j]]
                if len(non_fixed) > 2:
                    vals = [board.grid[row][j] for j in non_fixed]
                    random.shuffle(vals)
                    for idx, j in enumerate(non_fixed):
                        board.grid[row][j] = vals[idx]
        individual.calculate_fitness()
    
    def _count_cell_conflicts(self, board, row: int, col: int) -> int:
        conflicts = 0
        val = board.grid[row][col]
        if val == 0:
            return 100
        conflicts += sum(1 for j in range(board.size) if j != col and board.grid[row][j] == val)
        conflicts += sum(1 for i in range(board.size) if i != row and board.grid[i][col] == val)
        box_r = (row // board.box_size) * board.box_size
        box_c = (col // board.box_size) * board.box_size
        for i in range(box_r, box_r + board.box_size):
            for j in range(box_c, box_c + board.box_size):
                if (i, j) != (row, col) and board.grid[i][j] == val:
                    conflicts += 1
        return conflicts

    def solve(self):
        self.metrics.start()
        self.is_running = True
        self.initialize_population()
        start_time = time.time()
        for gen in range(self.max_generations):
            if time.time() - start_time > self.max_time:
                print(f"⚠️ Timeout reached after {self.max_time} seconds")
                self.metrics.stop(False)
                best = max(self.population, key=lambda x: x.fitness)
                return False, best.board
            if not self.is_running:
                self.metrics.stop(False)
                return False, None
            self.current_generation = gen + 1
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            best = self.population[0]
            if best.conflicts <= 10 and best.conflicts > 0:
                self.local_search(best, max_iterations=50)
            self.metrics.add_generation(best.fitness if best.fitness != float('inf') else 1.0)
            self.best_conflicts_history.append(best.conflicts)
            if best.conflicts == 0:
                self.metrics.stop(True)
                return True, best.board
            if len(self.best_conflicts_history) > 15:
                recent = self.best_conflicts_history[-15:]
                if len(set(recent)) <= 2:
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = max(0, self.stagnation_counter - 3)
            if self.stagnation_counter >= self.restart_threshold:
                self.restart_population(keep_best=max(3, self.pop_size // 20))
            elite_size = max(1, int(self.pop_size * self.elitism_rate))
            new_population = [ind.copy() for ind in self.population[:elite_size]]
            while len(new_population) < self.pop_size:
                parent1 = self.selection()
                parent2 = self.selection()
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)
            self.population = new_population
        best = max(self.population, key=lambda x: x.fitness)
        if best.conflicts <= 15:
            self.local_search(best, max_iterations=100)
            if best.conflicts == 0:
                self.metrics.stop(True)
                return True, best.board
        self.metrics.stop(False)
        return False, best.board
    
    def stop(self):
        self.is_running = False


class CulturalAlgorithmSolver(Solver):
    """Enhanced CA with belief space, plateau handling, restart, and timeout"""
    
    def __init__(self, board, pop_size=None, generations=None, params=None):
        super().__init__(board)
        self.original_board = board.copy()
        self.size = board.size
        if params is None:
            params = ParameterOptimizer.get_optimal_params(self.size)
        self.pop_size = pop_size or params['pop_size']
        self.max_generations = generations or params['generations']
        self.accept_rate = params.get('accept_rate', 0.4)
        self.elitism_rate = params.get('elitism_rate', 0.15)
        self.mutation_rate = params.get('mutation_rate', 0.25)
        self.tournament_size = params.get('tournament_size', 7)
        self.influence_rate = params.get('influence_rate', 0.6)
        self.restart_threshold = params.get('restart_threshold', 60)
        self.max_time = params.get('max_time', 120)
        self.fixed_mask = [[board.grid[i][j] != 0 for j in range(self.size)] for i in range(self.size)]
        self.population: List[Individual] = []
        self.belief_space = BeliefSpace(self.size)
        self.current_generation = 0
        self.is_running = True
        self.best_conflicts_history = []
        self.stagnation_counter = 0

    def initialize_population(self):
        ga = GeneticAlgorithmSolver(self.original_board, pop_size=self.pop_size, generations=1)
        ga.fixed_mask = self.fixed_mask
        ga.initialize_population()
        self.population = ga.population

    def restart_population(self, keep_best: int = 5):
        best_individuals = self.population[:keep_best]
        for i in range(self.size):
            for j in range(self.size):
                if not self.fixed_mask[i][j]:
                    for val in self.belief_space.cell_preferences[i][j]:
                        self.belief_space.cell_preferences[i][j][val] *= 0.5
        self.belief_space.stagnation_counter = 0
        self.population = best_individuals
        while len(self.population) < self.pop_size:
            board = self.original_board.copy()
            for i in range(self.size):
                for j in range(self.size):
                    if not self.fixed_mask[i][j]:
                        board.grid[i][j] = random.randint(1, self.size)
            self.population.append(Individual(board, self.fixed_mask))
        self.stagnation_counter = 0

    def selection(self) -> Individual:
        tournament = random.sample(self.population, self.tournament_size)
        progress = self.current_generation / self.max_generations if self.max_generations > 0 else 0
        random_chance = 0.25 * (1 - progress * 0.5)
        if random.random() < random_chance:
            return random.choice(tournament)
        return max(tournament, key=lambda x: x.fitness)

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        ga = GeneticAlgorithmSolver(self.original_board)
        ga.fixed_mask = self.fixed_mask
        ga.size = self.size
        return ga.crossover(parent1, parent2)

    def mutate(self, individual: Individual):
        if individual.conflicts <= 5 and individual.conflicts > 0:
            ga = GeneticAlgorithmSolver(self.original_board)
            ga.fixed_mask = self.fixed_mask
            ga.size = self.size
            ga.local_search(individual, max_iterations=50)
            if individual.conflicts == 0:
                return
        adaptive_rate = self.mutation_rate
        if individual.fitness < 0.3:
            adaptive_rate *= 2.0
        elif individual.fitness < 0.5:
            adaptive_rate *= 1.5
        if self.stagnation_counter > 30:
            adaptive_rate *= 1.8
        ga = GeneticAlgorithmSolver(self.original_board)
        ga.fixed_mask = self.fixed_mask
        ga.mutation_rate = adaptive_rate
        ga.size = self.size
        ga.current_generation = self.current_generation
        ga.max_generations = self.max_generations
        ga.stagnation_counter = self.stagnation_counter
        ga.mutate(individual)
        self.belief_space.influence(individual, self.fixed_mask, self.influence_rate)
        individual.calculate_fitness()

    def solve(self):
        self.metrics.start()
        self.is_running = True
        self.initialize_population()
        start_time = time.time()
        for gen in range(self.max_generations):
            if time.time() - start_time > self.max_time:
                print(f"⚠️ Timeout reached after {self.max_time} seconds")
                self.metrics.stop(False)
                best = max(self.population, key=lambda x: x.fitness)
                return False, best.board
            if not self.is_running:
                self.metrics.stop(False)
                return False, None
            self.current_generation = gen + 1
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            best = self.population[0]
            if best.conflicts <= 10 and best.conflicts > 0:
                ga = GeneticAlgorithmSolver(self.original_board)
                ga.fixed_mask = self.fixed_mask
                ga.size = self.size
                ga.local_search(best, max_iterations=50)
            self.metrics.add_generation(best.fitness if best.fitness != float('inf') else 1.0)
            self.best_conflicts_history.append(best.conflicts)
            if best.conflicts == 0:
                self.metrics.stop(True)
                return True, best.board
            n_accept = max(1, int(self.pop_size * self.accept_rate))
            accepted = self.population[:n_accept]
            self.belief_space.update(accepted)
            if len(self.best_conflicts_history) > 15:
                recent = self.best_conflicts_history[-15:]
                if len(set(recent)) <= 2:
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = max(0, self.stagnation_counter - 3)
            if self.stagnation_counter >= self.restart_threshold:
                self.restart_population(keep_best=max(3, self.pop_size // 20))
            elite_size = max(1, int(self.pop_size * self.elitism_rate))
            new_population = [ind.copy() for ind in self.population[:elite_size]]
            while len(new_population) < self.pop_size:
                parent1 = self.selection()
                parent2 = self.selection()
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)
            self.population = new_population
        best = max(self.population, key=lambda x: x.fitness)
        if best.conflicts <= 15:
            ga = GeneticAlgorithmSolver(self.original_board)
            ga.fixed_mask = self.fixed_mask
            ga.size = self.size
            ga.local_search(best, max_iterations=100)
            if best.conflicts == 0:
                self.metrics.stop(True)
                return True, best.board
        self.metrics.stop(False)
        return False, best.board

    def stop(self):
        self.is_running = False