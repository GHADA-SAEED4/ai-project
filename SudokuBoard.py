# SudokuBoard.py - Size-Agnostic Version
# Represents a Sudoku puzzle board of any size (4x4, 6x6, 9x9)

class SudokuBoard:
    def __init__(self, size=9, grid=None):
        """Create a Sudoku board"""
        self.size = size
        
        # ✅ Calculate box size based on grid size
        if size == 6:
            self.box_rows = 2
            self.box_cols = 3
            self.box_size = 3  # ✅ ضيفي السطر ده
        else:  # 4x4 or 9x9 (perfect squares)
            self.box_rows = int(size ** 0.5)
            self.box_cols = int(size ** 0.5)
            self.box_size = int(size ** 0.5)  # ✅ وده كمان
        
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
        for i in range(self.size):
            if self.grid[i][col] == num:
                return False
        
        # ✅ Check box with dynamic size
        box_row_start = (row // self.box_rows) * self.box_rows
        box_col_start = (col // self.box_cols) * self.box_cols
        
        for i in range(box_row_start, box_row_start + self.box_rows):
            for j in range(box_col_start, box_col_start + self.box_cols):
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
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] == 0:
                    return (i, j)
        return None
    
    def get_conflicts(self):
        """Count how many conflicts (errors) in the board"""
        conflicts = 0
        
        # Count empty cells as conflicts
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] == 0:
                    conflicts += 1
        
        # Check rows for duplicates
        for i in range(self.size):
            seen = set()
            for j in range(self.size):
                val = self.grid[i][j]
                if val != 0:
                    if val in seen:
                        conflicts += 1
                    seen.add(val)
        
        # Check columns for duplicates
        for j in range(self.size):
            seen = set()
            for i in range(self.size):
                val = self.grid[i][j]
                if val != 0:
                    if val in seen:
                        conflicts += 1
                    seen.add(val)
        
        # ✅ Check boxes with dynamic size
        for box_r in range(0, self.size, self.box_rows):
            for box_c in range(0, self.size, self.box_cols):
                seen = set()
                for i in range(box_r, box_r + self.box_rows):
                    for j in range(box_c, box_c + self.box_cols):
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
        """Print the board nicely with proper formatting for any size"""
        # ✅ Dynamic separator calculation
        separator_length = self.size * 2 + (self.size // self.box_cols) - 1
        
        for i in range(self.size):
            # Print horizontal separator between boxes
            if i % self.box_rows == 0 and i != 0:
                print("-" * separator_length)
            
            for j in range(self.size):
                # Print vertical separator between boxes
                if j % self.box_cols == 0 and j != 0:
                    print("|", end=" ")
                
                num = self.grid[i][j]
                print(num if num != 0 else ".", end=" ")
            
            print()