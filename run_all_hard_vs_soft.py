"""Runs the experiments to reproduce the table hard BC vs soft BC."""

import subprocess

LM = 1e-5
N_Omega, N_Gamma, N_init = 3000, 800, 500
ITERATIONS = 1000

"""
subprocess.run(
    f"python waveeq_exact_bc.py --iter {ITERATIONS} --LM {LM} --N_Omega {N_Omega} "
    f"--N_Gamma {N_Gamma} --N_init {N_init}",
    shell=True,
)

subprocess.run(
    f"python stokes_exact_bc.py --iter {ITERATIONS} --LM {LM} --N_Omega {N_Omega} "
    f"--N_Gamma {N_Gamma}",
    shell=True,
)

subprocess.run(
    f"python poisson_exact_bc.py --iter {ITERATIONS} --LM {LM} --N_Omega {N_Omega} "
    f"--N_Gamma {N_Gamma}",
    shell=True,
)

subprocess.run(
    f"python darcy_exact_bc.py --iter {ITERATIONS} --LM {LM} --N_Omega {N_Omega} "
    f"--N_Gamma {N_Gamma}",
    shell=True,
)"""

subprocess.run(
    f"python waveeq.py --iter {ITERATIONS} --LM {LM} --N_Omega {N_Omega} "
    f"--N_Gamma {N_Gamma} --N_init {N_init}",
    shell=True,
)

subprocess.run(
    f"python stokes.py --iter {ITERATIONS} --LM {LM} --N_Omega {N_Omega} "
    f"--N_Gamma {N_Gamma}",
    shell=True,
)

subprocess.run(
    f"python poisson.py --iter {ITERATIONS} --LM {LM} --N_Omega {N_Omega} "
    f"--N_Gamma {N_Gamma}",
    shell=True,
)

subprocess.run(
    f"python darcy.py --iter {ITERATIONS} --LM {LM} --N_Omega {N_Omega} "
    f"--N_Gamma {N_Gamma}",
    shell=True,
)
