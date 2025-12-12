# main.py - Simple Version
# Main program entry point

import sys
from SudokuBoard import SudokuBoard
from BacktrackingSolver import BacktrackingSolver
from CulturalAlgorithmSolver import CulturalAlgorithmSolver
from PuzzleLoader import PuzzleLoader
from Benchmarker import Benchmarker

class SudokuApp:
    """Main application"""
    
    def __init__(self):
        """Initialize app"""
        self.board = None
        self.solver = None
        self.loader = PuzzleLoader()
    
    def run_cli(self):
        """Run command-line interface"""
        print("=" * 50)
        print("   SUDOKU SOLVER")
        print("=" * 50)
        
        while True:
            print("\n[1] Load Puzzle from File")
            print("[2] Enter Puzzle Manually")
            print("[3] Load Sample Puzzle")
            print("[4] Solve with Backtracking")
            print("[5] Solve with Cultural Algorithm")
            print("[6] Compare Both Algorithms")
            print("[7] Run Full Benchmark")
            print("[0] Exit")
            
            choice = input("\nChoice: ").strip()
            
            if choice == "1":
                self._load_from_file()
            elif choice == "2":
                self._enter_manually()
            elif choice == "3":
                self._load_sample()
            elif choice == "4":
                self._solve_backtracking()
            elif choice == "5":
                self._solve_cultural()
            elif choice == "6":
                self._compare_algorithms()
            elif choice == "7":
                self._run_benchmark()
            elif choice == "0":
                print("Goodbye!")
                break
            else:
                print("Invalid choice!")
    
    def _load_from_file(self):
        """Load puzzle from file"""
        filename = input("Enter filename: ").strip()
        self.board = self.loader.load_from_file(filename)
        if self.board:
            print("✓ Loaded!")
            self.board.display()
    
    def _enter_manually(self):
        """Manual entry"""
        print("\nEnter 9 rows (use 0 for empty):")
        print("Example: 5 3 0 0 7 0 0 0 0")
        
        grid = []
        for i in range(9):
            while True:
                row_str = input(f"Row {i+1}: ").strip()
                try:
                    row = [int(x) for x in row_str.split()]
                    if len(row) != 9:
                        print("Need 9 numbers!")
                        continue
                    grid.append(row)
                    break
                except ValueError:
                    print("Invalid! Use numbers only.")
        
        self.board = SudokuBoard(9, grid)
        print("\n✓ Puzzle entered!")
        self.board.display()
    
    def _load_sample(self):
        """Load sample puzzle"""
        print("\nDifficulty:")
        print("[1] Easy")
        print("[2] Medium")
        print("[3] Hard")
        choice = input("Choice: ").strip()
        
        if choice == "1":
            difficulty = "easy"
        elif choice == "2":
            difficulty = "medium"
        else:
            difficulty = "hard"
        
        self.board = self.loader.load_sample(difficulty)
        print(f"\n✓ Loaded {difficulty} puzzle!")
        self.board.display()
    
    def _solve_backtracking(self):
        """Solve with backtracking"""
        if not self.board:
            print("✗ Load puzzle first!")
            return
        
        print("\n[1] Simple Backtracking")
        print("[2] Backtracking with Heuristic (faster)")
        choice = input("Choice: ").strip()
        
        use_heuristic = (choice == "2")
        solver = BacktrackingSolver(self.board.copy(), use_heuristic)
        
        print("\n🔍 Solving...")
        success = solver.solve()
        
        if success:
            print("✓ Solved!")
            solver.get_board().display()
            self._print_metrics(solver.get_metrics())
        else:
            print("✗ No solution!")
    
    def _solve_cultural(self):
        """Solve with Cultural Algorithm"""
        if not self.board:
            print("✗ Load puzzle first!")
            return
        
        print("\nSettings:")
        pop_size = int(input("Population size [150]: ") or "150")
        generations = int(input("Max generations [2000]: ") or "2000")
        
        solver = CulturalAlgorithmSolver(
            self.board.copy(),
            pop_size=pop_size,
            generations=generations
        )
        
        print("\n🧬 Evolving...")
        success, solution = solver.solve()
        
        if success and solution:
            print("✓ Solved!")
            solution.display()
            self._print_metrics(solver.get_metrics())
        else:
            print("✗ No perfect solution")
            if solver.population:
                print(f"Best: {solver.population[0].conflicts} conflicts")
    
    def _compare_algorithms(self):
        """Compare both algorithms"""
        if not self.board:
            print("✗ Load puzzle first!")
            return
        
        print("\n" + "="*60)
        print("  COMPARING ALGORITHMS")
        print("="*60)
        
        # Backtracking
        print("\n[1/2] Running Backtracking...")
        bt_solver = BacktrackingSolver(self.board.copy(), use_heuristic=True)
        bt_success = bt_solver.solve()
        bt_metrics = bt_solver.get_metrics()
        
        # Cultural Algorithm
        print("[2/2] Running Cultural Algorithm...")
        ca_solver = CulturalAlgorithmSolver(self.board.copy(), generations=1000)
        ca_success, ca_solution = ca_solver.solve()
        ca_metrics = ca_solver.get_metrics()
        
        # Display comparison
        print("\n" + "="*60)
        print(f"{'Metric':<25} {'Backtracking':>15} {'Cultural':>15}")
        print("-"*60)
        print(f"{'Success':<25} {str(bt_success):>15} {str(ca_success):>15}")
        print(f"{'Time (seconds)':<25} {bt_metrics['time']:>15.4f} {ca_metrics['time']:>15.4f}")
        
        if 'iterations' in bt_metrics:
            print(f"{'Iterations/Generations':<25} {bt_metrics['iterations']:>15} "
                  f"{ca_metrics.get('generations', 'N/A'):>15}")
        
        print("="*60)
        
        # Winner
        if bt_success and ca_success:
            winner = "Backtracking" if bt_metrics['time'] < ca_metrics['time'] else "Cultural"
            print(f"\n🏆 Fastest: {winner}")
    
    def _run_benchmark(self):
        """Run full benchmark"""
        print("\n🔬 Running Benchmark...")
        benchmarker = Benchmarker()
        results = benchmarker.run_full_benchmark()
        benchmarker.display_results(results)
        
        save = input("\nSave results? (y/n): ").strip().lower()
        if save == 'y':
            filename = input("Filename [benchmark_results.json]: ").strip()
            if not filename:
                filename = "benchmark_results.json"
            benchmarker.save_results(results, filename)
    
    def _print_metrics(self, metrics):
        """Print metrics nicely"""
        print("\n" + "-"*40)
        print("  Performance")
        print("-"*40)
        for key, value in metrics.items():
            key_formatted = key.replace('_', ' ').title()
            print(f"{key_formatted:<20}: {value}")
        print("-"*40)


# Main entry point
if __name__ == "__main__":
    app = SudokuApp()
    
    # Check command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--benchmark":
            benchmarker = Benchmarker()
            results = benchmarker.run_full_benchmark()
            benchmarker.display_results(results)
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python main.py [--benchmark]")
    else:
        # Default: CLI mode
        app.run_cli()