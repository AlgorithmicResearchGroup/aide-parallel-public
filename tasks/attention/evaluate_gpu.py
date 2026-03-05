"""GPU-aware evaluation harness for the Attention-Agent task.

This is a wrapper around evaluate.py that properly handles GPU device assignment
through environment variables set by Ray.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """
    Run the evaluation script with proper GPU configuration.
    The GPU assignment is handled through CUDA_VISIBLE_DEVICES environment variable
    set by Ray, so we use 'cuda' device if available.
    """

    # Get the directory of this script
    script_dir = Path(__file__).resolve().parent
    evaluate_script = script_dir / "evaluate.py"

    # Check if GPU is available via environment variable
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')

    # Determine device to use
    if cuda_visible and cuda_visible != '-1':
        device = 'cuda'
        print(f"Running evaluation on GPU (CUDA_VISIBLE_DEVICES={cuda_visible})")
    else:
        device = 'cpu'
        print("Running evaluation on CPU")

    # Get command line arguments (solution path is passed as first argument)
    args = sys.argv[1:]

    # Build the command to run the evaluation script
    cmd = [
        sys.executable,
        str(evaluate_script),
        '--device', device,
    ] + args

    # Run the evaluation script
    result = subprocess.run(cmd, capture_output=False, text=True)

    # Exit with the same code as the evaluation script
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()