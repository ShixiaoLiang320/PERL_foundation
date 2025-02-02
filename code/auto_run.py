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

    current_dir = os.path.dirname(os.path.abspath(__file__))

    scripts = [
        os.path.join(current_dir, "lstm_train.py"),
        os.path.join(current_dir, "perl_train.py")
        #os.path.join(current_dir, "lstm_multi.py"),
        #os.path.join(current_dir, "perl_multi.py")

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

