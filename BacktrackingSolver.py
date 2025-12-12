# BacktrackingSolver.py - Size-Agnostic Version
# Solves Sudoku of any size (4x4, 6x6, 9x9) using backtracking

from Solver import Solver

class BacktrackingSolver(Solver):
    """Backtracking algorithm solver that works with any grid size"""
    
    def __init__(self, board, use_heuristic=True):
        """Initialize solver"""
        super().__init__(board)
        self.use_heuristic = use_heuristic
        self.size = board.size  # ✅ Get the board size dynamically
    
    def solve(self):
        """Solve the puzzle"""
        self.metrics.start()
        result = self._backtrack()
        self.metrics.stop(result)
        return result
    
    def _backtrack(self):
        """Main recursive backtracking function"""
        self.metrics.increment_iteration()
        
        # Find next empty cell
        if self.use_heuristic:
            # Smart way: find cell with fewest options
            empty_cell = self._find_best_cell()
        else:
            # Simple way: find first empty cell
            empty_cell = self.board.get_empty_cell()
        
        # If no empty cells, puzzle is solved!
        if empty_cell is None:
            return True
        
        row, col = empty_cell
        
        # ✅ Try numbers from 1 to board size (not hardcoded to 9)
        for num in range(1, self.size + 1):
            if self.board.is_valid(row, col, num):
                # Place the number
                self.board.place_number(row, col, num)
                
                # Record this step
                self.steps.append({
                    'row': row,
                    'col': col,
                    'value': num,
                    'action': 'place'
                })
                
                # Try to solve rest of puzzle
                if self._backtrack():
                    return True
                
                # If we reach here, this number didn't work
                # Remove it and try next number
                self.board.remove_number(row, col)
                self.metrics.increment_backtrack()
                
                # Record backtrack step
                self.steps.append({
                    'row': row,
                    'col': col,
                    'value': 0,
                    'action': 'backtrack'
                })
        
        # No number worked
        return False
    
    def _find_best_cell(self):
        """Find empty cell with fewest valid options (MRV heuristic)"""
        min_options = self.size + 1  # ✅ Use board size instead of hardcoded 10
        best_cell = None
        
        # ✅ Loop through actual board size, not hardcoded 9
        for i in range(self.size):
            for j in range(self.size):
                if self.board.grid[i][j] == 0:
                    # Count how many numbers can go here
                    options = self._count_valid_options(i, j)
                    
                    if options < min_options:
                        min_options = options
                        best_cell = (i, j)
                    
                    # If 0 options, return immediately
                    if min_options == 0:
                        return best_cell
        
        return best_cell
    
    def _count_valid_options(self, row, col):
        """Count how many valid numbers for this cell"""
        count = 0
        # ✅ Try numbers from 1 to board size (not hardcoded to 9)
        for num in range(1, self.size + 1):
            if self.board.is_valid(row, col, num):
                count += 1
        return count
    
    def get_steps(self):
        """Get list of steps"""
        return self.steps