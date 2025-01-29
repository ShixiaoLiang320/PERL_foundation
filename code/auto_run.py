import subprocess
import sys
import os

def run_scripts(n):
    try:
        n = int(n)
        if n <= 0:
            raise ValueError
    except ValueError:
        print("Please provide a valid positive integer as input.")
        return

    scripts = ["lstm_multi.py", "perl_multi.py"]  

    # List of script filenames with their absolute paths
    script_dir = "/Users/shixiaoliang/Library/Mobile Documents/com~apple~CloudDocs/Documents/PERL_foundation/code"
    scripts = [
        os.path.join(script_dir, "lstm_train.py"),
        os.path.join(script_dir, "perl_train.py")
    ]
    for i in range(n):
        print(f"\n--- Running iteration {i+1}/{n} ---\n")
        for script in scripts:
            print(f"Running {script}...")
            try:
                subprocess.run(["python", script])
            except Exception as e:
                print(f"Error while running {script}: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_scripts(sys.argv[1])
    else:
        n = input("Enter the number of iterations: ")
        run_scripts(n)
