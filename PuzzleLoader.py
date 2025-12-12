# PuzzleLoader.py - Simple Version
# Loads Sudoku puzzles from files or provides samples

from SudokuBoard import SudokuBoard

class PuzzleLoader:
    """Loads and saves Sudoku puzzles"""
    
    # Built-in sample puzzles
    SAMPLE_PUZZLES = {
        "easy": [
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
        "medium": [
            [0, 0, 0, 6, 0, 0, 4, 0, 0],
            [7, 0, 0, 0, 0, 3, 6, 0, 0],
            [0, 0, 0, 0, 9, 1, 0, 8, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 5, 0, 1, 8, 0, 0, 0, 3],
            [0, 0, 0, 3, 0, 6, 0, 4, 5],
            [0, 4, 0, 2, 0, 0, 0, 6, 0],
            [9, 0, 3, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 1, 0, 0]
        ],
        "hard": [
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
    }
    
    def load_from_file(self, filename):
        """Load puzzle from text file"""
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
                            if 0 <= num <= 9:
                                row.append(num)
                        except ValueError:
                            continue
                    
                    if len(row) == 9:
                        grid.append(row)
            
            if len(grid) != 9:
                raise ValueError(f"Expected 9 rows, got {len(grid)}")
            
            return SudokuBoard(9, grid)
        
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found")
            return None
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    
    def load_sample(self, difficulty="easy"):
        """Load a built-in sample puzzle"""
        difficulty = difficulty.lower()
        
        if difficulty not in self.SAMPLE_PUZZLES:
            print(f"Unknown difficulty. Using 'easy'")
            difficulty = "easy"
        
        grid = [row[:] for row in self.SAMPLE_PUZZLES[difficulty]]
        return SudokuBoard(9, grid)
    
    def generate_puzzle(self, difficulty="easy"):
        """Generate a random puzzle (uses samples for now)"""
        return self.load_sample(difficulty)
    
    def save_to_file(self, board, filename):
        """Save puzzle to text file"""
        try:
            with open(filename, 'w') as f:
                f.write("# Sudoku Puzzle (0 = empty cell)\n")
                for row in board.grid:
                    f.write(" ".join(str(num) for num in row) + "\n")
            print(f"✓ Saved to {filename}")
        except Exception as e:
            print(f"Error saving file: {e}")
    
    def create_test_suite(self):
        """Create list of test puzzles"""
        return [
            self.load_sample("easy"),
            self.load_sample("medium"),
            self.load_sample("hard")
        ]


# Test if run directly
if __name__ == "__main__":
    loader = PuzzleLoader()
    
    print("Easy Puzzle:")
    easy = loader.load_sample("easy")
    easy.display()
    
    print("\nSaving example...")
    loader.save_to_file(easy, "puzzle_example.txt")