# Visualizer.py - UPDATED VERSION
# Modern GUI with support for ANY Sudoku size
# COPY THIS ENTIRE FILE and replace your current Visualizer.py

import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
import time

from SudokuBoard import SudokuBoard
from BacktrackingSolver import BacktrackingSolver
from CulturalAlgorithmSolver import CulturalAlgorithmSolver
from PuzzleLoader import PuzzleLoader
from Benchmarker import Benchmarker


class ModernButton(tk.Button):
    """Custom button with hover effects"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.defaultBackground = self["background"]
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
    
    def on_enter(self, e):
        self['background'] = self['activebackground']
    
    def on_leave(self, e):
        self['background'] = self.defaultBackground


class SudokuGUI:
    """Modern GUI supporting ANY Sudoku size"""
    
    def __init__(self, root):
        """Initialize GUI"""
        self.root = root
        self.root.title("🧩 Universal Sudoku Solver - Any Size Support")
        self.root.geometry("1000x900")
        self.root.configure(bg="#1a1a2e")
        
        # Modern color scheme
        self.colors = {
            'bg_dark': '#1a1a2e',
            'bg_medium': '#16213e',
            'bg_light': '#0f3460',
            'accent': '#e94560',
            'accent_hover': '#ff6b88',
            'success': '#00d9ff',
            'warning': '#ffd700',
            'text': '#ffffff',
            'text_dim': '#b0b0b0',
            'cell_bg': '#ffffff',
            'cell_fixed': '#e8f4f8',
            'cell_solution': '#d4edda',
            'grid_line': '#533483'
        }
        
        self.loader = PuzzleLoader()
        self.board = None
        self.current_size = 9  # Start with 9x9
        self.entries = []
        
        # Build GUI components
        self.build_header()
        self.build_size_selector()
        self.build_grid_container()
        self.create_grid(9)  # Create initial 9x9 grid
        self.build_controls()
        self.build_status()
    
    def build_header(self):
        """Build header section"""
        header = tk.Frame(self.root, bg=self.colors['bg_medium'], height=100)
        header.pack(fill=tk.X, pady=(0, 10))
        header.pack_propagate(False)
        
        title = tk.Label(
            header,
            text="🧩 UNIVERSAL SUDOKU SOLVER",
            font=("Segoe UI", 24, "bold"),
            bg=self.colors['bg_medium'],
            fg=self.colors['success']
        )
        title.pack(pady=(15, 5))
        
        subtitle = tk.Label(
            header,
            text="Support for ANY board size: 4×4, 9×9, 16×16, 25×25, and more!",
            font=("Segoe UI", 11),
            bg=self.colors['bg_medium'],
            fg=self.colors['text_dim']
        )
        subtitle.pack()
    
    def build_size_selector(self):
        """Build size selector with custom input"""
        frame = tk.Frame(self.root, bg=self.colors['bg_dark'])
        frame.pack(pady=10)
        
        tk.Label(
            frame,
            text="Board Size:",
            font=("Segoe UI", 12, "bold"),
            bg=self.colors['bg_dark'],
            fg=self.colors['text']
        ).pack(side=tk.LEFT, padx=10)
        
        # Quick size buttons
        for size in [4, 9, 16, 25]:
            btn = ModernButton(
                frame,
                text=f"{size}×{size}",
                width=8,
                font=("Segoe UI", 10, "bold"),
                bg=self.colors['bg_light'],
                fg=self.colors['text'],
                activebackground=self.colors['accent'],
                relief=tk.FLAT,
                cursor="hand2",
                command=lambda s=size: self.change_board_size(s)
            )
            btn.pack(side=tk.LEFT, padx=3)
        
        # Custom size button
        custom_btn = ModernButton(
            frame,
            text="✏️ Custom Size",
            width=12,
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['warning'],
            fg=self.colors['bg_dark'],
            activebackground=self.colors['accent_hover'],
            relief=tk.FLAT,
            cursor="hand2",
            command=self.show_custom_size_dialog
        )
        custom_btn.pack(side=tk.LEFT, padx=5)
    
    def show_custom_size_dialog(self):
        """Show dialog for custom board size"""
        dialog_text = (
            "Enter board size (must be perfect square):\n\n"
            "Valid sizes:\n"
            "  • 4 = 2×2 boxes (4×4 board)\n"
            "  • 9 = 3×3 boxes (9×9 board)\n"
            "  • 16 = 4×4 boxes (16×16 board)\n"
            "  • 25 = 5×5 boxes (25×25 board)\n"
            "  • 36 = 6×6 boxes (36×36 board)\n"
            "  • 49, 64, 81, 100, etc.\n\n"
            "Recommended: 4-25 for best performance"
        )
        
        size_str = simpledialog.askstring(
            "Custom Board Size",
            dialog_text,
            parent=self.root
        )
        
        if size_str:
            try:
                size = int(size_str)
                box_size = int(size ** 0.5)
                
                # Validate perfect square
                if box_size * box_size != size:
                    messagebox.showerror(
                        "Invalid Size",
                        f"{size} is not a perfect square!\n\n"
                        "Valid sizes: 4, 9, 16, 25, 36, 49, 64, 81, 100, etc."
                    )
                    return
                
                # Warning for very large boards
                if size > 100:
                    response = messagebox.askyesno(
                        "Large Board Warning",
                        f"{size}×{size} is very large!\n\n"
                        "Solving may take a long time or fail.\n\n"
                        "Continue anyway?"
                    )
                    if not response:
                        return
                
                self.change_board_size(size)
                
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid number")
    
    def build_grid_container(self):
        """Build scrollable container for grid"""
        # Main container with scrollbars
        container = tk.Frame(self.root, bg=self.colors['bg_dark'])
        container.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Canvas for scrolling
        self.canvas = tk.Canvas(
            container,
            bg=self.colors['bg_dark'],
            highlightthickness=0,
            height=450
        )
        
        # Scrollbars
        v_scroll = tk.Scrollbar(container, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scroll = tk.Scrollbar(container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        # Configure canvas
        self.canvas.configure(
            yscrollcommand=v_scroll.set,
            xscrollcommand=h_scroll.set
        )
        
        # Pack scrollbars
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Frame inside canvas
        self.grid_container = tk.Frame(self.canvas, bg=self.colors['bg_dark'])
        self.canvas_window = self.canvas.create_window(
            (0, 0),
            window=self.grid_container,
            anchor='center'
        )
        
        # Bind events
        self.grid_container.bind('<Configure>', self.on_grid_configure)
        self.canvas.bind('<Configure>', self.on_canvas_configure)
    
    def on_grid_configure(self, event=None):
        """Update scroll region when grid changes"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def on_canvas_configure(self, event):
        """Center grid in canvas"""
        canvas_width = event.width
        canvas_height = event.height
        
        # Get grid size
        self.grid_container.update_idletasks()
        grid_width = self.grid_container.winfo_reqwidth()
        grid_height = self.grid_container.winfo_reqheight()
        
        # Center position
        x = max(0, (canvas_width - grid_width) // 2)
        y = max(0, (canvas_height - grid_height) // 2)
        
        self.canvas.coords(self.canvas_window, x, y)
    
    def create_grid(self, size):
        """Create Sudoku grid of specified size"""
        # Clear existing grid
        for widget in self.grid_container.winfo_children():
            widget.destroy()
        
        self.current_size = size
        self.entries = [[None for _ in range(size)] for _ in range(size)]
        
        # Grid frame with border
        grid_frame = tk.Frame(
            self.grid_container,
            bg=self.colors['grid_line'],
            padx=4,
            pady=4
        )
        grid_frame.pack()
        
        box_size = int(size ** 0.5)
        
        # Adjust cell size and font based on board size
        if size <= 4:
            cell_width = 4
            font_size = 18
            cell_pad = 10
        elif size <= 9:
            cell_width = 3
            font_size = 16
            cell_pad = 8
        elif size <= 16:
            cell_width = 2
            font_size = 12
            cell_pad = 6
        elif size <= 25:
            cell_width = 2
            font_size = 10
            cell_pad = 5
        else:
            cell_width = 1
            font_size = 8
            cell_pad = 4
        
        # Create grid cells
        for r in range(size):
            for c in range(size):
                entry = tk.Entry(
                    grid_frame,
                    width=cell_width,
                    font=("Segoe UI", font_size, "bold"),
                    justify="center",
                    bg=self.colors['cell_bg'],
                    fg=self.colors['bg_dark'],
                    relief=tk.FLAT,
                    borderwidth=1
                )
                
                # Add thicker borders for box boundaries
                padx = (3, 1) if c % box_size == 0 else (1, 1)
                pady = (3, 1) if r % box_size == 0 else (1, 1)
                
                if c == size - 1:
                    padx = (padx[0], 3)
                if r == size - 1:
                    pady = (pady[0], 3)
                
                entry.grid(row=r, column=c, padx=padx, pady=pady, ipady=cell_pad)
                self.entries[r][c] = entry
        
        # Update scroll region
        self.grid_container.update_idletasks()
        self.on_grid_configure()
    
    def change_board_size(self, size):
        """Change the board size"""
        if size != self.current_size:
            self.create_grid(size)
            box_size = int(size ** 0.5)
            self.update_status(
                f"Changed to {size}×{size} board (with {box_size}×{box_size} boxes)",
                "success"
            )
    
    def build_controls(self):
        """Build control buttons"""
        controls = tk.Frame(self.root, bg=self.colors['bg_dark'])
        controls.pack(pady=15)
        
        # Row 1: Load and Generate
        row1 = tk.Frame(controls, bg=self.colors['bg_dark'])
        row1.pack(pady=5)
        
        self.create_button(
            row1, "📂 Load Sample", self.load_sample_puzzle,
            self.colors['bg_light'], 15
        ).pack(side=tk.LEFT, padx=3)
        
        self.create_button(
            row1, "🎲 Generate Random", self.generate_random_puzzle,
            self.colors['bg_light'], 15
        ).pack(side=tk.LEFT, padx=3)
        
        self.create_button(
            row1, "📁 Load File", self.load_from_file,
            self.colors['bg_light'], 15
        ).pack(side=tk.LEFT, padx=3)
        
        self.create_button(
            row1, "🗑️ Clear", self.clear_board,
            self.colors['warning'], 12
        ).pack(side=tk.LEFT, padx=3)
        
        # Row 2: Solve
        row2 = tk.Frame(controls, bg=self.colors['bg_dark'])
        row2.pack(pady=5)
        
        self.create_button(
            row2, "⚡ Solve (Backtracking)", self.solve_backtracking,
            self.colors['success'], 22
        ).pack(side=tk.LEFT, padx=5)
        
        self.create_button(
            row2, "🧬 Solve (Cultural Algorithm)", self.solve_cultural,
            self.colors['accent'], 22
        ).pack(side=tk.LEFT, padx=5)
        
        # Row 3: Compare and Save
        row3 = tk.Frame(controls, bg=self.colors['bg_dark'])
        row3.pack(pady=5)
        
        self.create_button(
            row3, "🏆 Compare Both", self.compare_algorithms,
            "#9b59b6", 22
        ).pack(side=tk.LEFT, padx=5)
        
        self.create_button(
            row3, "💾 Save Puzzle", self.save_puzzle,
            "#e67e22", 22
        ).pack(side=tk.LEFT, padx=5)
    
    def create_button(self, parent, text, command, color, width):
        """Create styled button"""
        return ModernButton(
            parent,
            text=text,
            width=width,
            font=("Segoe UI", 10, "bold"),
            bg=color,
            fg=self.colors['text'],
            activebackground=self.colors['accent_hover'],
            relief=tk.FLAT,
            cursor="hand2",
            command=command,
            padx=8,
            pady=6
        )
    
    def build_status(self):
        """Build status bar"""
        self.status_frame = tk.Frame(
            self.root,
            bg=self.colors['bg_medium'],
            height=50
        )
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_frame.pack_propagate(False)
        
        container = tk.Frame(self.status_frame, bg=self.colors['bg_medium'])
        container.pack(expand=True)
        
        self.status_icon = tk.Label(
            container,
            text="●",
            font=("Segoe UI", 18),
            bg=self.colors['bg_medium'],
            fg=self.colors['success']
        )
        self.status_icon.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(
            container,
            text="Ready - Select board size and load a puzzle to start",
            font=("Segoe UI", 10),
            bg=self.colors['bg_medium'],
            fg=self.colors['text']
        )
        self.status_label.pack(side=tk.LEFT)
    
    def update_status(self, message, status_type="info"):
        """Update status bar"""
        icons = {
            'info': ('●', self.colors['success']),
            'working': ('◐', self.colors['warning']),
            'success': ('✓', self.colors['success']),
            'error': ('✗', self.colors['accent'])
        }
        
        icon, color = icons.get(status_type, icons['info'])
        self.status_icon.config(text=icon, fg=color)
        self.status_label.config(text=message)
        self.root.update()
    
    def load_sample_puzzle(self):
        """Load sample puzzle"""
        board = self.loader.load_sample(self.current_size, "easy")
        if board and board.size != self.current_size:
            self.change_board_size(board.size)
        if board:
            self.display_board(board)
            self.update_status(f"Loaded {self.current_size}×{self.current_size} sample puzzle", "success")
    
    def generate_random_puzzle(self):
        """Generate random puzzle"""
        # Ask difficulty
        diff = simpledialog.askstring(
            "Difficulty",
            "Choose difficulty:\n  • easy\n  • medium\n  • hard",
            parent=self.root
        )
        
        if diff and diff.lower() in ["easy", "medium", "hard"]:
            self.update_status("Generating puzzle...", "working")
            board = self.loader.generate_puzzle(self.current_size, diff.lower())
            self.display_board(board)
            self.update_status(
                f"Generated {diff} {self.current_size}×{self.current_size} puzzle",
                "success"
            )
    
    def load_from_file(self):
        """Load puzzle from file"""
        filename = filedialog.askopenfilename(
            title="Select Sudoku Puzzle",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            board = self.loader.load_from_file(filename)
            if board:
                if board.size != self.current_size:
                    self.change_board_size(board.size)
                self.display_board(board)
                self.update_status(f"Loaded {board.size}×{board.size} puzzle from file", "success")
            else:
                messagebox.showerror("Error", "Failed to load puzzle")
                self.update_status("Failed to load file", "error")
    
    def save_puzzle(self):
        """Save current puzzle"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            grid = self.read_grid()
            board = SudokuBoard(self.current_size, grid)
            self.loader.save_to_file(board, filename)
            self.update_status(f"Saved puzzle to {filename}", "success")
    
    def clear_board(self):
        """Clear all cells"""
        for r in range(self.current_size):
            for c in range(self.current_size):
                self.entries[r][c].delete(0, tk.END)
                self.entries[r][c].config(bg=self.colors['cell_bg'])
        self.update_status("Board cleared", "info")
    
    def read_grid(self):
        """Read grid from GUI"""
        grid = []
        for r in range(self.current_size):
            row = []
            for c in range(self.current_size):
                text = self.entries[r][c].get().strip()
                if text.isdigit() and 1 <= int(text) <= self.current_size:
                    row.append(int(text))
                else:
                    row.append(0)
            grid.append(row)
        return grid
    
    def display_board(self, board):
        """Display board on GUI"""
        for r in range(self.current_size):
            for c in range(self.current_size):
                self.entries[r][c].delete(0, tk.END)
                val = board.grid[r][c]
                if val != 0:
                    self.entries[r][c].insert(0, str(val))
                    self.entries[r][c].config(bg=self.colors['cell_fixed'])
    
    def highlight_solution(self, original_grid, solved_grid):
        """Highlight solution"""
        for r in range(self.current_size):
            for c in range(self.current_size):
                self.entries[r][c].delete(0, tk.END)
                if solved_grid[r][c] != 0:
                    self.entries[r][c].insert(0, str(solved_grid[r][c]))
                    if original_grid[r][c] == 0:
                        self.entries[r][c].config(bg=self.colors['cell_solution'])
                    else:
                        self.entries[r][c].config(bg=self.colors['cell_fixed'])
    
    def solve_backtracking(self):
        """Solve with backtracking"""
        grid = self.read_grid()
        original_grid = [row[:] for row in grid]
        
        board = SudokuBoard(self.current_size, grid)
        self.update_status("⚡ Solving with Backtracking...", "working")
        
        start_time = time.time()
        solver = BacktrackingSolver(board, use_heuristic=True)
        success = solver.solve()
        elapsed = time.time() - start_time
        
        if success:
            self.highlight_solution(original_grid, board.grid)
            metrics = solver.get_metrics()
            
            result = (
                f"✓ SOLVED with Backtracking!\n\n"
                f"Board: {self.current_size}×{self.current_size}\n"
                f"Time: {elapsed:.4f} seconds\n"
                f"Iterations: {metrics.get('iterations', 'N/A'):,}\n"
                f"Backtracks: {metrics.get('backtracks', 'N/A'):,}"
            )
            
            messagebox.showinfo("Success!", result)
            self.update_status(f"✓ Solved in {elapsed:.4f}s", "success")
        else:
            messagebox.showerror("Failed", "Could not solve this puzzle")
            self.update_status("✗ No solution found", "error")
    
    def solve_cultural(self):
        """Solve with Cultural Algorithm"""
        grid = self.read_grid()
        original_grid = [row[:] for row in grid]
        
        board = SudokuBoard(self.current_size, grid)
        self.update_status("🧬 Solving with Cultural Algorithm...", "working")
        
        # Adjust parameters based on size
        if self.current_size <= 4:
            pop_size, generations = 100, 500
        elif self.current_size <= 9:
            pop_size, generations = 150, 2000
        elif self.current_size <= 16:
            pop_size, generations = 300, 5000
        else:
            pop_size, generations = 500, 10000
        
        start_time = time.time()
        solver = CulturalAlgorithmSolver(board, pop_size=pop_size, generations=generations)
        success, solution = solver.solve()
        elapsed = time.time() - start_time
        
        if success and solution:
            self.highlight_solution(original_grid, solution.grid)
            metrics = solver.get_metrics()
            
            result = (
                f"✓ SOLVED with Cultural Algorithm!\n\n"
                f"Board: {self.current_size}×{self.current_size}\n"
                f"Time: {elapsed:.4f} seconds\n"
                f"Generations: {metrics.get('generations', 'N/A'):,}\n"
                f"Best Fitness: {metrics.get('best_fitness', 'N/A')}"
            )
            
            messagebox.showinfo("Success!", result)
            self.update_status(f"✓ Solved in {elapsed:.4f}s", "success")
        else:
            messagebox.showerror("Failed", "Could not find perfect solution")
            self.update_status("✗ No perfect solution", "error")
    
    def compare_algorithms(self):
        """Compare both algorithms"""
        grid = self.read_grid()
        board = SudokuBoard(self.current_size, grid)
        
        self.update_status("🏆 Comparing algorithms...", "working")
        
        # Backtracking
        bt_board = board.copy()
        bt_start = time.time()
        bt_solver = BacktrackingSolver(bt_board, use_heuristic=True)
        bt_success = bt_solver.solve()
        bt_time = time.time() - bt_start
        
        # Cultural Algorithm
        ca_board = board.copy()
        if self.current_size <= 4:
            pop, gen = 100, 500
        elif self.current_size <= 9:
            pop, gen = 100, 1000
        else:
            pop, gen = 200, 2000
        
        ca_start = time.time()
        ca_solver = CulturalAlgorithmSolver(ca_board, pop_size=pop, generations=gen)
        ca_success, _ = ca_solver.solve()
        ca_time = time.time() - ca_start
        
        # Results
        result = f"📊 COMPARISON ({self.current_size}×{self.current_size})\n\n"
        result += f"{'Algorithm':<25} {'Success':<10} {'Time (s)':<12}\n"
        result += f"{'-'*50}\n"
        result += f"{'⚡ Backtracking':<25} {str(bt_success):<10} {bt_time:<12.4f}\n"
        result += f"{'🧬 Cultural':<25} {str(ca_success):<10} {ca_time:<12.4f}\n"
        result += f"{'-'*50}\n\n"
        
        if bt_success and ca_success:
            winner = "⚡ Backtracking" if bt_time < ca_time else "🧬 Cultural"
            speedup = max(bt_time, ca_time) / min(bt_time, ca_time)
            result += f"🏆 Winner: {winner}\n"
            result += f"⚡ Speedup: {speedup:.2f}×"
        
        messagebox.showinfo("Comparison Results", result)
        self.update_status("Comparison complete", "success")


# For testing
if __name__ == "__main__":
    root = tk.Tk()
    app = SudokuGUI(root)
    root.mainloop()