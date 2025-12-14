# PuzzleLoader.py - UPDATED VERSION
# Loads and generates Sudoku puzzles of ANY size
# COPY THIS ENTIRE FILE and replace your current PuzzleLoader.py

from SudokuBoard import SudokuBoard
import random


class PuzzleLoader:
    """Loads and saves Sudoku puzzles of any size"""
    
    # Built-in sample puzzles for different sizes
    SAMPLE_PUZZLES = {
        # 4x4 puzzles (2x2 boxes)
        4: {
            "easy": [
                [
                    [1, 0, 0, 4],
                    [0, 4, 1, 0],
                    [4, 0, 3, 0],
                    [0, 3, 0, 1]
                ],
                [
                    [0, 2, 0, 0],
                    [0, 0, 2, 1],
                    [2, 0, 0, 0],
                    [0, 0, 1, 2]
                ],
                [
                    [4, 0, 0, 0],
                    [0, 0, 4, 2],
                    [0, 4, 0, 0],
                    [0, 0, 0, 4]
                ],
                [
                    [0, 0, 3, 0],
                    [3, 0, 0, 2],
                    [0, 3, 0, 0],
                    [0, 1, 0, 0]
                ],
                [
                    [3, 0, 0, 0],
                    [0, 0, 3, 1],
                    [1, 0, 0, 0],
                    [0, 0, 0, 3]
                ]
            ],
            "medium": [
                [
                    [0, 0, 0, 4],
                    [0, 4, 0, 0],
                    [0, 0, 3, 0],
                    [2, 0, 0, 0]
                ],
                [
                    [0, 0, 0, 0],
                    [0, 4, 0, 1],
                    [2, 0, 0, 0],
                    [0, 0, 1, 0]
                ]
            ]
        },
        
        # 9x9 puzzles (3x3 boxes) - Classic Sudoku
        9: {
            "easy": [
                [
                    [5, 3, 0, 0, 7, 0, 0, 0, 0],
                    [6, 0, 0, 1, 9, 5, 0, 0, 0],
                    [0, 9, 8, 0, 0, 0, 0, 6, 0],
                    [8, 0, 0, 0, 6, 0, 0, 0, 3],
                    [4, 0, 0, 8, 0, 3, 0, 0, 1],
                    [7, 0, 0, 0, 2, 0, 0, 0, 6],
                    [0, 6, 0, 0, 0, 0, 2, 8, 0],
                    [0, 0, 0, 4, 1, 9, 0, 0, 5],
                    [0, 0, 0, 0, 8, 0, 0, 7, 9]
                ],
                [
                    [0, 2, 0, 6, 0, 8, 0, 0, 0],
                    [5, 8, 0, 0, 0, 9, 7, 0, 0],
                    [0, 0, 0, 0, 4, 0, 0, 0, 0],
                    [3, 7, 0, 0, 0, 0, 5, 0, 0],
                    [6, 0, 0, 0, 0, 0, 0, 0, 4],
                    [0, 0, 8, 0, 0, 0, 0, 1, 3],
                    [0, 0, 0, 0, 2, 0, 0, 0, 0],
                    [0, 0, 9, 8, 0, 0, 0, 3, 6],
                    [0, 0, 0, 3, 0, 6, 0, 9, 0]
                ],
                [
                    [2, 0, 0, 3, 0, 0, 0, 0, 0],
                    [8, 0, 4, 0, 6, 0, 0, 0, 3],
                    [0, 1, 3, 0, 0, 0, 2, 0, 0],
                    [0, 0, 0, 0, 0, 7, 0, 0, 0],
                    [0, 0, 0, 5, 0, 1, 0, 0, 0],
                    [0, 0, 0, 6, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 6, 7, 0],
                    [3, 0, 0, 0, 2, 0, 1, 0, 4],
                    [0, 0, 0, 0, 0, 6, 0, 0, 2]
                ]
            ],
            "medium": [
                [
                    [0, 0, 0, 6, 0, 0, 4, 0, 0],
                    [7, 0, 0, 0, 0, 3, 6, 0, 0],
                    [0, 0, 0, 0, 9, 1, 0, 8, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 5, 0, 1, 8, 0, 0, 0, 3],
                    [0, 0, 0, 3, 0, 6, 0, 4, 5],
                    [0, 4, 0, 2, 0, 0, 0, 6, 0],
                    [9, 0, 3, 0, 0, 0, 0, 0, 0],
                    [0, 2, 0, 0, 0, 0, 1, 0, 0]
                ]
            ],
            "hard": [
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 3, 0, 8, 5],
                    [0, 0, 1, 0, 2, 0, 0, 0, 0],
                    [0, 0, 0, 5, 0, 7, 0, 0, 0],
                    [0, 0, 4, 0, 0, 0, 1, 0, 0],
                    [0, 9, 0, 0, 0, 0, 0, 0, 0],
                    [5, 0, 0, 0, 0, 0, 0, 7, 3],
                    [0, 0, 2, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 4, 0, 0, 0, 9]
                ]
            ]
        }
    }
    
    def load_from_file(self, filename):
        """Load puzzle from text file - works with any size"""
        try:
            grid = []
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse numbers
                    row = []
                    for char in line.replace(',', ' ').split():
                        try:
                            num = int(char)
                            if num >= 0:
                                row.append(num)
                        except ValueError:
                            continue
                    
                    if row:
                        grid.append(row)
            
            if not grid:
                raise ValueError("No valid grid found in file")
            
            # Get size from grid
            size = len(grid)
            
            # Validate grid is square
            for row in grid:
                if len(row) != size:
                    raise ValueError(f"Grid is not square - inconsistent row lengths")
            
            # Validate size is perfect square
            box_size = int(size ** 0.5)
            if box_size * box_size != size:
                raise ValueError(f"Invalid size {size} - must be perfect square (4, 9, 16, 25, etc.)")
            
            return SudokuBoard(size, grid)
        
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found")
            return None
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    
    def load_sample(self, size=9, difficulty="easy"):
        """Load a built-in sample puzzle"""
        size = int(size)
        difficulty = difficulty.lower()
        
        # Check if we have samples for this size
        if size not in self.SAMPLE_PUZZLES:
            print(f"No built-in samples for {size}x{size}. Generating random puzzle.")
            return self.generate_puzzle(size, difficulty)
        
        # Check if difficulty exists
        if difficulty not in self.SAMPLE_PUZZLES[size]:
            print(f"No {difficulty} puzzles for {size}x{size}. Using easy.")
            difficulty = "easy"
        
        # Pick random puzzle from samples
        puzzles = self.SAMPLE_PUZZLES[size][difficulty]
        grid = random.choice(puzzles)
        
        # Make a copy so we don't modify the original
        grid = [row[:] for row in grid]
        
        return SudokuBoard(size, grid)
    
    def generate_puzzle(self, size=9, difficulty="medium"):
        """Generate a random puzzle of ANY size"""
        # Validate size
        box_size = int(size ** 0.5)
        if box_size * box_size != size:
            print(f"Invalid size {size}. Must be perfect square (4, 9, 16, 25, etc.)")
            return self.load_sample(9, difficulty)
        
        # Create empty board
        board = SudokuBoard(size)
        
        # Fill diagonal boxes first (they don't interfere with each other)
        # This gives us a valid partial solution to start from
        for box in range(0, size, box_size):
            nums = list(range(1, size + 1))
            random.shuffle(nums)
            idx = 0
            for i in range(box, box + box_size):
                for j in range(box, box + box_size):
                    board.grid[i][j] = nums[idx]
                    idx += 1
        
        # Determine how many cells to remove based on difficulty
        difficulty_map = {
            "easy": 0.3,      # Remove 30% of cells
            "medium": 0.5,    # Remove 50% of cells
            "hard": 0.7       # Remove 70% of cells
        }
        
        remove_ratio = difficulty_map.get(difficulty.lower(), 0.5)
        total_cells = size * size
        cells_to_remove = int(total_cells * remove_ratio)
        
        # Remove cells randomly
        all_cells = [(i, j) for i in range(size) for j in range(size)]
        random.shuffle(all_cells)
        
        for i, j in all_cells[:cells_to_remove]:
            board.grid[i][j] = 0
        
        return board
    
    def save_to_file(self, board, filename):
        """Save puzzle to text file"""
        try:
            with open(filename, 'w') as f:
                f.write(f"# Sudoku Puzzle {board.size}x{board.size}\n")
                f.write(f"# Box size: {board.box_size}x{board.box_size}\n")
                f.write(f"# 0 = empty cell\n")
                f.write("#\n")
                for row in board.grid:
                    f.write(" ".join(str(num) for num in row) + "\n")
            print(f"✓ Saved to {filename}")
        except Exception as e:
            print(f"Error saving file: {e}")
    
    def create_test_suite(self):
        """Create list of test puzzles"""
        suite = []
        
        # Add all sample puzzles
        for size in self.SAMPLE_PUZZLES:
            for difficulty in self.SAMPLE_PUZZLES[size]:
                for grid in self.SAMPLE_PUZZLES[size][difficulty]:
                    suite.append(SudokuBoard(size, [row[:] for row in grid]))
        
        return suite


# Test if run directly
if __name__ == "__main__":
    loader = PuzzleLoader()
    
    print("Testing 4x4 puzzle:")
    board4 = loader.load_sample(4, "easy")
    board4.display()
    
    print("\nTesting 9x9 puzzle:")
    board9 = loader.load_sample(9, "easy")
    board9.display()
    
    print("\nGenerating random 16x16 puzzle:")
    board16 = loader.generate_puzzle(16, "easy")
    board16.display()