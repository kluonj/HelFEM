import subprocess
import re
import matplotlib.pyplot as plt
import argparse
import sys
import os

def parse_energy(output):
    """Parses the total energy from the atomic output."""
    for line in output.splitlines():
        if "Total                 energy" in line:
            # Line format: "Total energy is -7.4327265789"
            try:
                parts = line.split()
                return float(parts[-1])
            except ValueError:
                return None
    return None

def run_atomic(executable, rmax, nelem):
    """Runs the atomic executable with specified Rmax and nelem."""
    cmd = [
        executable,
        "--Z=3",
        f"--nelem={nelem}",
        "--nnodes=15",
        "--lmax=0",
        "--mmax=0",
        "--primbas=4",
        "--nela=2",
        "--nelb=1",
        f"--Rmax={rmax}"
    ]
    
    # print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return parse_energy(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(e.stdout)
        print(e.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(description="Study convergence of Lithium HF energy.")
    parser.add_argument("--atomic", default="./build/src/atomic", help="Path to atomic executable")
    args = parser.parse_args()
    
    executable = args.atomic
    if not os.path.exists(executable):
        print(f"Error: Executable not found at {executable}")
        sys.exit(1)

    rmax_values = list(range(20, 51, 10)) # 20, 30, 40, 50
    nelem_values = list(range(3, 11, 1))  # 3, 4, ..., 10

    results = {}

    print(f"{'Rmax':<10} | {'Nelem':<10} | {'Energy':<20}")
    print("-" * 46)

    for rmax in rmax_values:
        for nelem in nelem_values:
            energy = run_atomic(executable, rmax, nelem)
            results[(rmax, nelem)] = energy
            print(f"{rmax:<10} | {nelem:<10} | {energy}")

    # Plotting
    # 1. Energy vs Nelem (One line per Rmax)
    plt.figure(figsize=(10, 6))
    for rmax in rmax_values:
        energies = [results[(rmax, n)] for n in nelem_values]
        plt.plot(nelem_values, energies, marker='o', label=f'Rmax={rmax}')
    
    plt.xlabel('Number of Elements (nelem)')
    plt.ylabel('Total Energy (Ha)')
    plt.title('Lithium HF Convergence: Energy vs Nelem')
    plt.legend()
    plt.grid(True)
    plt.savefig('lithium_convergence_nelem.png')
    print("Saved plot to lithium_convergence_nelem.png")

    # 2. Energy vs Rmax (One line per Nelem)
    plt.figure(figsize=(10, 6))
    for nelem in nelem_values:
        energies = [results[(r, nelem)] for r in rmax_values]
        plt.plot(rmax_values, energies, marker='s', label=f'Nelem={nelem}')
    
    plt.xlabel('Box Size (Rmax)')
    plt.ylabel('Total Energy (Ha)')
    plt.title('Lithium HF Convergence: Energy vs Rmax')
    plt.legend()
    plt.grid(True)
    plt.savefig('lithium_convergence_rmax.png')
    print("Saved plot to lithium_convergence_rmax.png")

if __name__ == "__main__":
    main()
