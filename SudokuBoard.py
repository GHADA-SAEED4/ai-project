# SudokuBoard.py - FIXED VERSION
# Now properly supports ANY board size
# COPY THIS ENTIRE FILE and replace your current SudokuBoard.py

class SudokuBoard:
    def __init__(self, size=9, grid=None):
        """Create a Sudoku board of any size"""
        self.size = size
        self.box_size = int(size ** 0.5)  # Calculate box size from board size
        
        # Validate that size is a perfect square
        if self.box_size * self.box_size != size:
            raise ValueError(f"Board size {size} must be a perfect square (4, 9, 16, 25, etc.)")
        
        # Create or copy grid
        if grid:
            self.grid = [row[:] for row in grid]
        else:
            self.grid = [[0] * size for _ in range(size)]
        
        # Remember which cells were given (not empty)
        self.fixed_cells = set()
        for i in range(size):
            for j in range(size):
                if self.grid[i][j] != 0:
                    self.fixed_cells.add((i, j))
    
    def is_valid(self, row, col, num):
        """Check if we can put 'num' at position (row, col)"""
        # Check row - is number already there?
        if num in self.grid[row]:
            return False
        
        # Check column - is number already there?
        for i in range(self.size):  # FIXED: Use self.size instead of hardcoded 9
            if self.grid[i][col] == num:
                return False
        
        # Check box - is number already there?
        box_row_start = (row // self.box_size) * self.box_size
        box_col_start = (col // self.box_size) * self.box_size
        
        for i in range(box_row_start, box_row_start + self.box_size):  # FIXED: Use box_size
            for j in range(box_col_start, box_col_start + self.box_size):  # FIXED: Use box_size
                if self.grid[i][j] == num:
                    return False
        
        return True
    
    def place_number(self, row, col, num):
        """Put a number in the cell (if not fixed)"""
        if (row, col) not in self.fixed_cells:
            self.grid[row][col] = num
    
    def remove_number(self, row, col):
        """Remove number from cell (if not fixed)"""
        if (row, col) not in self.fixed_cells:
            self.grid[row][col] = 0
    
    def get_empty_cell(self):
        """Find first empty cell (with 0)"""
        for i in range(self.size):  # FIXED: Use self.size
            for j in range(self.size):  # FIXED: Use self.size
                if self.grid[i][j] == 0:
                    return (i, j)
        return None
    
    def get_conflicts(self):
        """Count how many conflicts (errors) in the board"""
        conflicts = 0
        
        # Count empty cells as conflicts
        for i in range(self.size):  # FIXED: Use self.size
            for j in range(self.size):  # FIXED: Use self.size
                if self.grid[i][j] == 0:
                    conflicts += 1
        
        # Check rows for duplicates
        for i in range(self.size):  # FIXED: Use self.size
            seen = set()
            for j in range(self.size):  # FIXED: Use self.size
                val = self.grid[i][j]
                if val != 0:
                    if val in seen:
                        conflicts += 1
                    seen.add(val)
        
        # Check columns for duplicates
        for j in range(self.size):  # FIXED: Use self.size
            seen = set()
            for i in range(self.size):  # FIXED: Use self.size
                val = self.grid[i][j]
                if val != 0:
                    if val in seen:
                        conflicts += 1
                    seen.add(val)
        
        # Check boxes for duplicates
        for box_r in range(0, self.size, self.box_size):  # FIXED: Use self.size and box_size
            for box_c in range(0, self.size, self.box_size):  # FIXED: Use self.size and box_size
                seen = set()
                for i in range(box_r, box_r + self.box_size):  # FIXED: Use box_size
                    for j in range(box_c, box_c + self.box_size):  # FIXED: Use box_size
                        val = self.grid[i][j]
                        if val != 0:
                            if val in seen:
                                conflicts += 1
                            seen.add(val)
        
        return conflicts
    
    def copy(self):
        """Make a copy of this board"""
        return SudokuBoard(self.size, self.grid)
    
    def display(self):
        """Print the board nicely"""
        for i in range(self.size):  # FIXED: Use self.size
            if i % self.box_size == 0 and i != 0:  # FIXED: Use box_size
                print("-" * (self.size * 2 + self.box_size - 1))  # FIXED: Dynamic length
            
            for j in range(self.size):  # FIXED: Use self.size
                if j % self.box_size == 0 and j != 0:  # FIXED: Use box_size
                    print("|", end=" ")
                
                num = self.grid[i][j]
                if num == 0:
                    print(".", end=" ")
                else:
                    # For numbers > 9, use letters (A=10, B=11, etc.)
                    if num < 10:
                        print(num, end=" ")
                    else:
                        print(chr(55 + num), end=" ")  # A=10, B=11, C=12, etc.
            
            print()


# Test if run directly
if __name__ == "__main__":
    print("Testing 4x4 board:")
    board4 = SudokuBoard(4, [
        [1, 0, 0, 4],
        [0, 4, 1, 0],
        [4, 0, 3, 0],
        [0, 3, 0, 1]
    ])
    board4.display()
    print(f"Conflicts: {board4.get_conflicts()}")
    
    print("\n\nTesting 9x9 board:")
    board9 = SudokuBoard(9)
    board9.grid[0] = [5, 3, 0, 0, 7, 0, 0, 0, 0]
    board9.display()
    print(f"Conflicts: {board9.get_conflicts()}")
    
    print("\n\nTesting 16x16 board:")
    board16 = SudokuBoard(16)
    print(f"Board size: {board16.size}")
    print(f"Box size: {board16.box_size}")
    print(f"Valid check (0,0,5): {board16.is_valid(0, 0, 5)}")