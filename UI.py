import tkinter as tk
from tkinter import messagebox
import time

from SudokoTestBoards import load_test_suite
from SudokuBoard import SudokuBoard
from BacktrackingSolver import BacktrackingSolver
from CulturalAlgorithmSolver import CulturalAlgorithmSolver


# ======================================================
# GUI CLASS
# ======================================================
class SudokuGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sudoku Solver - Backtracking & Cultural Algorithm")

        self.entries = [[None for _ in range(9)] for _ in range(9)]

        self.build_grid()
        self.build_buttons()

        # Load predefined puzzles
        self.boards = load_test_suite()
        self.current_index = 0

    # -------------------------------------------------------
    def build_grid(self):
        frame = tk.Frame(self.root)
        frame.grid(row=0, column=0, padx=10, pady=10)

        for r in range(9):
            for c in range(9):
                entry = tk.Entry(frame, width=3, font=("Arial", 20), justify="center")

                # Thicker borders for 3x3 blocks
                padx = (5, 1) if c % 3 == 0 and c != 0 else 1
                pady = (5, 1) if r % 3 == 0 and r != 0 else 1

                entry.grid(row=r, column=c, padx=padx, pady=pady)
                self.entries[r][c] = entry

    # -------------------------------------------------------
    def build_buttons(self):
        frame = tk.Frame(self.root)
        frame.grid(row=1, column=0, pady=10)

        tk.Button(frame, text="Load Puzzle", width=15,
                  command=self.load_puzzle).grid(row=0, column=0, padx=5)

        tk.Button(frame, text="Clear", width=15,
                  command=self.clear_board).grid(row=0, column=1, padx=5)

        tk.Button(frame, text="Solve (Backtracking)", width=20,
                  command=self.solve_backtracking).grid(row=1, column=0, padx=5, pady=5)

        tk.Button(frame, text="Solve (Cultural)", width=20,
                  command=self.solve_cultural).grid(row=1, column=1, padx=5, pady=5)

    # -------------------------------------------------------
    def load_puzzle(self):
        board = self.boards[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.boards)

        for r in range(9):
            for c in range(9):
                self.entries[r][c].delete(0, tk.END)
                val = board.grid[r][c]
                if val != 0:
                    self.entries[r][c].insert(0, str(val))

    # -------------------------------------------------------
    def clear_board(self):
        for r in range(9):
            for c in range(9):
                self.entries[r][c].delete(0, tk.END)

    # -------------------------------------------------------
    def read_grid(self):
        return [
            [
                int(self.entries[r][c].get()) if self.entries[r][c].get().isdigit() else 0
                for c in range(9)
            ]
            for r in range(9)
        ]

    # -------------------------------------------------------
    def write_grid(self, grid):
        for r in range(9):
            for c in range(9):
                self.entries[r][c].delete(0, tk.END)
                if grid[r][c] != 0:
                    self.entries[r][c].insert(0, str(grid[r][c]))

    # -------------------------------------------------------
    def solve_backtracking(self):
        grid = self.read_grid()

        # FIXED: SudokuBoard requires size + grid
        board = SudokuBoard(9, grid)

        start = time.time()
        solver = BacktrackingSolver(board)
        solved = solver.solve()
        end = time.time()

        if solved:
            self.write_grid(board.grid)
            messagebox.showinfo("Solved!",
                                    f"Backtracking solved the puzzle in {end - start:.4f} sec")
        else:
            messagebox.showwarning("Failed",
                                "Backtracking could not solve this puzzle")

    # -------------------------------------------------------
    def solve_cultural(self):
        grid = self.read_grid()

        # FIXED: SudokuBoard requires (size, grid)
        board = SudokuBoard(9, grid)

        start = time.time()
        solver = CulturalAlgorithmSolver(board)
        solved, sol_board = solver.solve()
        end = time.time()

        if solved:
            self.write_grid(sol_board.grid)
            messagebox.showinfo("Solved!",
                                f"Cultural Algorithm solved the puzzle in {end - start:.4f} sec")
        else:
            messagebox.showwarning("Failed",
                                   "Cultural Algorithm could not solve this puzzle")


# ======================================================
# RUN APP
# ======================================================
if __name__ == "__main__":
    root = tk.Tk()
    gui = SudokuGUI(root)
    root.mainloop()