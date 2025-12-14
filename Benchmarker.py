# Benchmarker.py - Simple Version
# Compares solver performance

import json
import time
from statistics import mean, stdev
from SudokuBoard import SudokuBoard
from BacktrackingSolver import BacktrackingSolver
from CulturalAlgorithmSolver import CulturalAlgorithmSolver
from PuzzleLoader import PuzzleLoader

class Benchmarker:
    """Compares different solving algorithms"""
    
    def __init__(self):
        """Initialize benchmarker"""
        self.loader = PuzzleLoader()
        self.results = []
    
    def benchmark_single_puzzle(self, board, puzzle_name="Unknown"):
        """Test both algorithms on one puzzle"""
        print(f"\n{'='*60}")
        print(f"  Testing: {puzzle_name}")
        print(f"{'='*60}")
        
        original_board = board.copy()
        result = {
            'puzzle_name': puzzle_name,
            'algorithms': {}
        }
        
        # Test 1: Simple Backtracking
        print("[1/4] Simple Backtracking...")
        bt_simple = BacktrackingSolver(original_board.copy(), use_heuristic=False)
        start = time.time()
        bt_simple_success = bt_simple.solve()
        bt_simple_time = time.time() - start
        
        result['algorithms']['backtracking_simple'] = {
            'success': bt_simple_success,
            'metrics': bt_simple.get_metrics()
        }
        
        # Test 2: Backtracking with Heuristic
        print("[2/4] Backtracking with Heuristic...")
        bt_heuristic = BacktrackingSolver(original_board.copy(), use_heuristic=True)
        start = time.time()
        bt_heuristic_success = bt_heuristic.solve()
        bt_heuristic_time = time.time() - start
        
        result['algorithms']['backtracking_heuristic'] = {
            'success': bt_heuristic_success,
            'metrics': bt_heuristic.get_metrics()
        }
        
        # Test 3: Cultural Algorithm (small)
        print("[3/4] Cultural Algorithm (500 generations)...")
        ca_small = CulturalAlgorithmSolver(original_board.copy(), pop_size=100, generations=500)
        start = time.time()
        ca_small_success, ca_small_solution = ca_small.solve()
        ca_small_time = time.time() - start
        
        result['algorithms']['cultural_small'] = {
            'success': ca_small_success,
            'metrics': ca_small.get_metrics()
        }
        
        # Test 4: Cultural Algorithm (large)
        print("[4/4] Cultural Algorithm (2000 generations)...")
        ca_large = CulturalAlgorithmSolver(original_board.copy(), pop_size=150, generations=2000)
        start = time.time()
        ca_large_success, ca_large_solution = ca_large.solve()
        ca_large_time = time.time() - start
        
        result['algorithms']['cultural_large'] = {
            'success': ca_large_success,
            'metrics': ca_large.get_metrics()
        }
        
        # Print summary
        print(f"\n{'Results':<30}")
        print(f"{'-'*60}")
        print(f"{'Algorithm':<30} {'Success':<10} {'Time (s)':<10}")
        print(f"{'-'*60}")
        print(f"{'Backtracking (Simple)':<30} {str(bt_simple_success):<10} {bt_simple_time:<10.4f}")
        print(f"{'Backtracking (Heuristic)':<30} {str(bt_heuristic_success):<10} {bt_heuristic_time:<10.4f}")
        print(f"{'Cultural (500 gen)':<30} {str(ca_small_success):<10} {ca_small_time:<10.4f}")
        print(f"{'Cultural (2000 gen)':<30} {str(ca_large_success):<10} {ca_large_time:<10.4f}")
        print(f"{'-'*60}")
        
        return result
    
    def run_full_benchmark(self):
        """Run tests on all difficulty levels"""
        print("\n" + "="*60)
        print("  FULL BENCHMARK SUITE")
        print("="*60)
        
        difficulties = ["easy", "medium", "hard"]
        all_results = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'puzzles': []
        }
        
        for difficulty in difficulties:
            board = self.loader.load_sample(difficulty)
            result = self.benchmark_single_puzzle(board, f"{difficulty.title()} Puzzle")
            all_results['puzzles'].append(result)
        
        # Calculate overall statistics
        all_results['summary'] = self._calculate_summary(all_results['puzzles'])
        
        return all_results
    
    def _calculate_summary(self, puzzle_results):
        """Calculate average statistics"""
        summary = {
            'total_puzzles': len(puzzle_results),
            'algorithm_stats': {}
        }
        
        # Get algorithm names
        algo_names = []
        if puzzle_results:
            algo_names = list(puzzle_results[0]['algorithms'].keys())
        
        # Calculate stats for each algorithm
        for algo_name in algo_names:
            times = []
            successes = []
            
            for puzzle in puzzle_results:
                algo_data = puzzle['algorithms'][algo_name]
                successes.append(algo_data['success'])
                times.append(algo_data['metrics']['time'])
            
            summary['algorithm_stats'][algo_name] = {
                'success_rate': sum(successes) / len(successes) * 100,
                'avg_time': mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'std_time': stdev(times) if len(times) > 1 else 0
            }
        
        return summary
    
    def display_results(self, results):
        """Display results nicely"""
        print("\n" + "="*70)
        print("  BENCHMARK RESULTS")
        print("="*70)
        
        if 'summary' in results:
            summary = results['summary']
            
            print(f"\nTotal Puzzles: {summary['total_puzzles']}")
            print("\n" + "-"*70)
            print(f"{'Algorithm':<25} {'Success %':<12} {'Avg Time':<12} {'Std Dev':<12}")
            print("-"*70)
            
            for algo_name, stats in summary['algorithm_stats'].items():
                algo_display = algo_name.replace('_', ' ').title()
                print(f"{algo_display:<25} "
                      f"{stats['success_rate']:<12.1f} "
                      f"{stats['avg_time']:<12.4f} "
                      f"{stats['std_time']:<12.4f}")
            
            print("-"*70)
            
            # Winners
            print("\n🏆 WINNERS:")
            
            # Fastest
            fastest = min(summary['algorithm_stats'].items(), key=lambda x: x[1]['avg_time'])
            print(f"  Fastest:        {fastest[0].replace('_', ' ').title()}")
            
            # Most reliable
            best_success = max(summary['algorithm_stats'].items(), key=lambda x: x[1]['success_rate'])
            print(f"  Most Reliable:  {best_success[0].replace('_', ' ').title()}")
    
    def save_results(self, results, filename="benchmark_results.json"):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved to {filename}")
    
    def compare_two_algorithms(self, board):
        """Quick comparison of two algorithms"""
        print("\nComparing BACKTRACKING vs CULTURAL ALGORITHM")
        print("="*50)
        
        # Backtracking
        print("\n[1/2] Running Backtracking...")
        bt_solver = BacktrackingSolver(board.copy(), use_heuristic=True)
        bt_success = bt_solver.solve()
        bt_metrics = bt_solver.get_metrics()
        
        # Cultural Algorithm
        print("[2/2] Running Cultural Algorithm...")
        ca_solver = CulturalAlgorithmSolver(board.copy(), generations=1000)
        ca_success, _ = ca_solver.solve()
        ca_metrics = ca_solver.get_metrics()
        
        # Display
        print(f"\nBACKTRACKING:")
        print(f"  Success: {bt_success}")
        print(f"  Time: {bt_metrics['time']:.4f}s")
        
        print(f"\nCULTURAL ALGORITHM:")
        print(f"  Success: {ca_success}")
        print(f"  Time: {ca_metrics['time']:.4f}s")
        
        # Winner
        if bt_success and ca_success:
            winner = "BACKTRACKING" if bt_metrics['time'] < ca_metrics['time'] else "CULTURAL"
            print(f"\n🏆 Winner: {winner}")


# Test if run directly
if __name__ == "__main__":
    benchmarker = Benchmarker()
    results = benchmarker.run_full_benchmark()
    benchmarker.display_results(results)
    benchmarker.save_results(results)