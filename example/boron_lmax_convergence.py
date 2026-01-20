import subprocess
import argparse
import sys
import os
import matplotlib.pyplot as plt


def parse_energy(output):
    for line in output.splitlines():
        # Look for the total energy line (format may vary); search for 'Total energy' or 'Total energy is'
        if 'Total                 energy' in line:
            try:
                parts = line.split()
                return float(parts[-1])
            except ValueError:
                continue
    return None


def run_atomic(executable, lmax, nnodes=15, primbas=4, rmax=40.0):
    cmd = [
        executable,
        "--Z=5",
        "--nelem=4",
        f"--nnodes={nnodes}",
        f"--lmax={lmax}",
        f"--mmax={lmax}",
        f"--primbas={primbas}",
        "--nela=3",
        "--nelb=2",
        f"--Rmax={rmax}"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return parse_energy(res.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd)}")
        print(e.stdout)
        print(e.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description='Convergence study for Boron over lmax (and mmax=lmax)')
    parser.add_argument('--atomic', default='./build/src/atomic', help='Path to atomic executable')
    parser.add_argument('--lmin', type=int, default=0)
    parser.add_argument('--lmax', type=int, default=5)
    parser.add_argument('--rmax', type=float, default=30.0)
    parser.add_argument('--nnodes', type=int, default=15)
    parser.add_argument('--primbas', type=int, default=4)
    args = parser.parse_args()

    exe = args.atomic
    if not os.path.exists(exe):
        print(f"Executable {exe} not found.")
        sys.exit(1)

    lvals = list(range(args.lmin, args.lmax+1))
    energies = []

    print(f"lmax | Energy")
    print("----------------")
    for l in lvals:
        E = run_atomic(exe, l, nnodes=args.nnodes, primbas=args.primbas, rmax=args.rmax)
        energies.append(E)
        print(f"{l:4d} | {E}")

    # Plot
    plt.figure(figsize=(8,6))
    plt.plot(lvals, energies, marker='o')
    plt.xlabel('lmax (and mmax)')
    plt.ylabel('Total Energy (Ha)')
    plt.title('Boron HF Convergence vs lmax')
    plt.grid(True)
    outname = 'boron_lmax_convergence.png'
    plt.savefig(outname)
    print(f"Saved plot to {outname}")

if __name__ == '__main__':
    main()
