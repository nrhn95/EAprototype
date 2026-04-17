import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    if "--experiments" in sys.argv:
        # CLI experiment runner
        argv = [a for a in sys.argv[1:] if a != "--experiments"]
        from experiments.run_experiments import main as run_experiments
        run_experiments(argv)
    else:
        # GUI mode (default)
        from src.gui.app import launch
        launch()


if __name__ == "__main__":
    main()
