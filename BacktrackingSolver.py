# BacktrackingSolver.py - COMPLETE FIXED VERSION
# Works with ANY board size (4×4, 9×9, 16×16, 25×25, etc.)
# IMPORTANT: COPY THIS ENTIRE FILE - DELETE YOUR OLD ONE FIRST!

from Solver import Solver


class BacktrackingSolver(Solver):
    """Backtracking algorithm solver - works with any size"""
    
    def __init__(self, board, use_heuristic=True):
        """Initialize solver"""
        super().__init__(board)
        self.use_heuristic = use_heuristic
        # IMPORTANT: Store the board size
        self.size = board.size
    
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
        
        # IMPORTANT: Try numbers from 1 to board size (not just 1-9!)
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
        # IMPORTANT: Use board size, not hardcoded 10
        min_options = self.size + 1
        best_cell = None
        
        # IMPORTANT: Loop through ALL cells based on board size
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
        # IMPORTANT: Try all numbers from 1 to board size
        for num in range(1, self.size + 1):
            if self.board.is_valid(row, col, num):
                count += 1
        return count
    
    def get_steps(self):
        """Get list of steps"""
        return self.steps


# Test the solver
if __name__ == "__main__":
    from SudokuBoard import SudokuBoard
    
    print("Testing 4x4 solver:")
    board4 = SudokuBoard(4, [
        [1, 0, 0, 4],
        [0, 4, 1, 0],
        [4, 0, 3, 0],
        [0, 3, 0, 1]
    ])
    print("Original:")
    board4.display()
    
    solver4 = BacktrackingSolver(board4)
    print(f"\nSolver size: {solver4.size}")
    success = solver4.solve()
    print(f"\nSolved: {success}")
    if success:
        board4.display()
    
    print("\n" + "="*50)
    print("\nTesting 9x9 solver:")
    board9 = SudokuBoard(9, [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])
    print("Original:")
    board9.display()
    
    solver9 = BacktrackingSolver(board9)
    print(f"\nSolver size: {solver9.size}")
    success = solver9.solve()
    print(f"\nSolved: {success}")
    if success:
        board9.display()