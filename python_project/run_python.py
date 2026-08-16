import sys
import subprocess
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")

VARIANTS = {
    "pure_python": "pure_python.py",
    "numpy": "numpy_conv.py",
    "opencv": "opencv_conv.py",
    "numba": "numba_conv.py",
    "cupy": "cupy_conv.py",
    "pytorch": "pytorch_conv.py"
}

def main():
    if len(sys.argv) < 5:
        print("Usage: run_python.py <variant> <technique> <input_image> <output_image>")
        sys.exit(1)

    variant = sys.argv[1]
    technique = sys.argv[2]
    input_path = sys.argv[3]
    output_path = sys.argv[4]

    if variant not in VARIANTS:
        print(f"Unknown variant: {variant}")
        sys.exit(1)

    script_path = os.path.join(SRC_DIR, VARIANTS[variant])
    cmd = [sys.executable, script_path, technique, input_path, output_path]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {variant}: {result.stderr}")
        sys.exit(result.returncode)

    print(result.stdout.strip())

if __name__ == "__main__":
    main()
