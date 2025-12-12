from SudokoTestBoards import load_test_suite
import time
import csv

# ============================================
# Check solution (size-agnostic)
# ============================================
def check_valid(board):
    """Check if the board is valid (no duplicates in rows, columns, blocks)"""
    n = len(board.grid)

    # Determine block size (supports 4x4, 6x6, 9x9)
    if n == 6:
        block_rows, block_cols = 2, 3
    else:  # 4x4 or 9x9 (perfect square)
        block_rows = block_cols = int(n**0.5)

    # Check rows and columns
    for i in range(n):
        row = [x for x in board.grid[i] if x != 0]
        if len(row) != len(set(row)):
            return False

        col = [board.grid[r][i] for r in range(n) if board.grid[r][i] != 0]
        if len(col) != len(set(col)):
            return False

    # Check blocks
    for r in range(0, n, block_rows):
        for c in range(0, n, block_cols):
            block = []
            for i in range(block_rows):
                for j in range(block_cols):
                    val = board.grid[r+i][c+j]
                    if val != 0:
                        block.append(val)
            if len(block) != len(set(block)):
                return False

    return True


# ============================================
# Import Backtracking Solver
# ============================================
from BacktrackingSolver import BacktrackingSolver


# ============================================
# Board sizes to test
# ============================================
board_sizes = [4, 6, 9]

for size in board_sizes:
    print("=" * 60)
    print(f"   Running Backtracking Tests for {size}x{size} boards")
    print("=" * 60)
    
    # Load test boards for this size
    boards = load_test_suite(size=size)
    bt_results = []

    # Test each board
    for i, board in enumerate(boards):
        print(f"\n{'='*50}")
        print(f"Puzzle {i+1}/{len(boards)} ({size}x{size})")
        print(f"{'='*50}")
        
        # Check if board is valid before solving
        if not check_valid(board):
            print("❌ Invalid board (contains duplicates)")
            bt_results.append({
                "puzzle_number": i + 1,
                "size": size,
                "solved": False,
                "valid": False,
                "time": 0,
                "iterations": 0,
                "backtracks": 0,
                "solution": None
            })
            continue

        print("✅ Board is valid")
        print(f"Solving with Backtracking Algorithm...")
        
        # Make a copy of the board for solving
        board_copy = board.copy()
        solver = BacktrackingSolver(board_copy)
        
        # Solve with timing
        start_time = time.time()
        solved = solver.solve()
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        iterations = solver.metrics.iterations
        backtracks = solver.metrics.backtracks
        
        # Check solution validity
        if solved:
            print("✅ Solution found!")
            solver.board.display()
            valid = check_valid(solver.board)
            
            if valid:
                print("✅ Solution is valid")
            else:
                print("❌ Solution is INVALID (has duplicates)")
        else:
            print("❌ No solution found")
            valid = False

        # Print metrics
        print(f"\n📊 Metrics:")
        print(f"   Time: {elapsed_time:.4f} seconds")
        print(f"   Iterations: {iterations}")
        print(f"   Backtracks: {backtracks}")

        # Save results
        bt_results.append({
            "puzzle_number": i + 1,
            "size": size,
            "solved": solved,
            "valid": valid if solved else False,
            "time": elapsed_time,
            "iterations": iterations,
            "backtracks": backtracks,
            "solution": solver.board.grid if solved else None
        })

    # ============================================
    # Save results to CSV
    # ============================================
    filename = f"backtracking_results_{size}x{size}.csv"
    
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            "puzzle_number", 
            "size",
            "solved", 
            "valid", 
            "time", 
            "iterations",
            "backtracks",
            "solution"
        ])
        
        # Write data
        for r in bt_results:
            solution_str = str(r["solution"]) if r["solution"] else ""
            writer.writerow([
                r["puzzle_number"],
                r["size"],
                r["solved"],
                r["valid"],
                f"{r['time']:.4f}",
                r["iterations"],
                r["backtracks"],
                solution_str
            ])
    
    print(f"\n{'='*60}")
    print(f"✅ Results saved to: {filename}")
    print(f"{'='*60}\n")

# ============================================
# Print summary
# ============================================
print("\n" + "=" * 60)
print("   📊 TESTING COMPLETE")
print("=" * 60)
print("\nResults saved in:")
for size in board_sizes:
    print(f"  - backtracking_results_{size}x{size}.csv")
print()





# ============================================
# Check solution (size-agnostic)
# ============================================
def check_valid(board):
    """Check if the board is valid (no duplicates in rows, columns, blocks)"""
    n = len(board.grid)

    # Determine block size (supports 4x4, 6x6, 9x9)
    if n == 6:
        block_rows, block_cols = 2, 3
    else:  # 4x4 or 9x9 (perfect square)
        block_rows = block_cols = int(n**0.5)

    # Check rows and columns
    for i in range(n):
        row = [x for x in board.grid[i] if x != 0]
        if len(row) != len(set(row)):
            return False

        col = [board.grid[r][i] for r in range(n) if board.grid[r][i] != 0]
        if len(col) != len(set(col)):
            return False

    # Check blocks
    for r in range(0, n, block_rows):
        for c in range(0, n, block_cols):
            block = []
            for i in range(block_rows):
                for j in range(block_cols):
                    val = board.grid[r+i][c+j]
                    if val != 0:
                        block.append(val)
            if len(block) != len(set(block)):
                return False

    return True


# ============================================
# Import Cultural Algorithm Solver
# ============================================
from cultural_algorithm_solver import CulturalAlgorithmSolver


# ============================================
# Board sizes to test
# ============================================
board_sizes = [4, 6, 9]

for size in board_sizes:
    print("=" * 60)
    print(f"   Running Cultural Algorithm Tests for {size}x{size} boards")
    print("=" * 60)
    
    # Load test boards for this size
    boards = load_test_suite(size=size)
    ca_results = []

    # Test each board
    for i, board in enumerate(boards):
        print(f"\n{'='*50}")
        print(f"Puzzle {i+1}/{len(boards)} ({size}x{size})")
        print(f"{'='*50}")
        
        # Check if board is valid before solving
        if not check_valid(board):
            print("❌ Invalid board (contains duplicates)")
            ca_results.append({
                "puzzle_number": i + 1,
                "size": size,
                "solved": False,
                "valid": False,
                "time": 0,
                "generations": 0,
                "solution": None
            })
            continue

        print("✅ Board is valid")
        print(f"Solving with Cultural Algorithm...")
        
        # Make a copy of the board for solving
        board_copy = board.copy()
        solver = CulturalAlgorithmSolver(board_copy)
        
        # Solve with timing
        start_time = time.time()
        solved, solution_board = solver.solve()
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        generations = solver.current_generation
        
        # Check solution validity
        if solved and solution_board is not None:
            print("✅ Solution found!")
            solution_board.display()
            valid = check_valid(solution_board)
            
            if valid:
                print("✅ Solution is valid")
            else:
                print("❌ Solution is INVALID (has duplicates)")
        else:
            print("❌ No solution found")
            valid = False

        # Print metrics
        print(f"\n📊 Metrics:")
        print(f"   Time: {elapsed_time:.4f} seconds")
        print(f"   Generations: {generations}")

        # Save results
        ca_results.append({
            "puzzle_number": i + 1,
            "size": size,
            "solved": solved,
            "valid": valid if solved else False,
            "time": elapsed_time,
            "generations": generations,
            "solution": solution_board.grid if solved and solution_board else None
        })

    # ============================================
    # Save results to CSV
    # ============================================
    filename = f"cultural_results_{size}x{size}.csv"
    
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            "puzzle_number", 
            "size",
            "solved", 
            "valid", 
            "time", 
            "generations",
            "solution"
        ])
        
        # Write data
        for r in ca_results:
            solution_str = str(r["solution"]) if r["solution"] else ""
            writer.writerow([
                r["puzzle_number"],
                r["size"],
                r["solved"],
                r["valid"],
                f"{r['time']:.4f}",
                r["generations"],
                solution_str
            ])
    
    print(f"\n{'='*60}")
    print(f"✅ Results saved to: {filename}")
    print(f"{'='*60}\n")

# ============================================
# Print summary
# ============================================
print("\n" + "=" * 60)
print("   📊 TESTING COMPLETE")
print("=" * 60)
print("\nResults saved in:")
for size in board_sizes:
    print(f"  - cultural_results_{size}x{size}.csv")
