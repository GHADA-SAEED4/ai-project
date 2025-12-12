# Solver.py - Simple Version
# Base class for all solvers

from SudokuBoard import SudokuBoard
from Metrics import Metrics

class Solver:
    """Base class that all solvers inherit from"""
    
    def __init__(self, board):
        """Initialize with a board"""
        self.board = board
        self.metrics = Metrics()
        self.steps = []  # Store steps for visualization
    
    def solve(self):
        """Solve the puzzle - must be implemented by child classes"""
        raise NotImplementedError("Each solver must implement solve()")
    
    def get_metrics(self):
        """Get performance statistics"""
        return self.metrics.to_dict()
    
    def reset_metrics(self):
        """Reset all statistics"""
        self.metrics = Metrics()
        self.steps = []
    
    def get_steps(self):
        """Get list of steps taken"""
        return self.steps
    
    def get_board(self):
        """Get the current board"""
        return self.board