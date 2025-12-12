# Visualizer.py - Simple Version
# Simple stub for visualization (GUI optional)

class Visualizer:
    """Handles visualization (GUI mode - optional)"""
    
    def __init__(self):
        """Initialize visualizer"""
        pass
    
    def show(self, board):
        """Show the board (simple text version)"""
        print("\nBoard Visualization:")
        print("="*30)
        board.display()
        print("="*30)
        print("Note: GUI visualization not implemented")
        print("This is a text-only display")
    
    def animate_solution(self, steps):
        """Animate solution steps (stub)"""
        print("Animation not implemented")
        print(f"Total steps: {len(steps)}")


# Test if run directly
if __name__ == "__main__":
    print("Visualizer module - GUI features not implemented")
    print("Use for text-based display only")