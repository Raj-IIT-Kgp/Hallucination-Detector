"""FreshQA 5-set evaluation; kept as a shortcut for `run_5_runs.py --dataset freshqa`."""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.run_5_runs import run_5_sets

if __name__ == "__main__":
    run_5_sets("freshqa", samples_per_set=20, num_sets=5, seed=42)

    # Exit immediately: interpreter shutdown joins every thread-pool worker, and a
    # DuckDuckGo/Wikipedia request stuck past its timeout would otherwise keep the process alive.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
