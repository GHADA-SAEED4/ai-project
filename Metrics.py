# Metrics.py - Simple Version
# Tracks performance statistics

import time

class Metrics:
    """Keeps track of solver performance"""
    
    def __init__(self):
        """Initialize empty metrics"""
        self.start_time = None
        self.end_time = None
        
        # For Backtracking
        self.iterations = 0
        self.backtracks = 0
        
        # For Cultural Algorithm
        self.generations = 0
        self.best_fitness_history = []
        
        # Common
        self.solution_found = False
    
    def start(self):
        """Start the timer"""
        self.start_time = time.time()
        self.iterations = 0
        self.backtracks = 0
        self.generations = 0
        self.best_fitness_history = []
        self.solution_found = False
    
    def stop(self, found=True):
        """Stop the timer"""
        self.end_time = time.time()
        self.solution_found = found
    
    def increment_iteration(self):
        """Add one iteration (for Backtracking)"""
        self.iterations += 1
    
    def increment_backtrack(self):
        """Add one backtrack (for Backtracking)"""
        self.backtracks += 1
    
    def add_generation(self, fitness):
        """Add a generation (for Cultural Algorithm)"""
        self.generations += 1
        self.best_fitness_history.append(fitness)
    
    def get_elapsed_time(self):
        """Get time taken"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return 0.0
    
    def to_dict(self):
        """Convert metrics to dictionary"""
        result = {
            'time': round(self.get_elapsed_time(), 4),
            'solution_found': self.solution_found
        }
        
        # Add Backtracking metrics if used
        if self.iterations > 0 or self.backtracks > 0:
            result['iterations'] = self.iterations
            result['backtracks'] = self.backtracks
        
        # Add Cultural Algorithm metrics if used
        if self.generations > 0:
            result['generations'] = self.generations
            if self.best_fitness_history:
                result['best_fitness'] = self.best_fitness_history[-1]
                result['initial_fitness'] = self.best_fitness_history[0]
        
        return result