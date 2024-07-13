"""Script to reproduce the results with the large computational budget."""

import subprocess

LM = 1e-5
N_Omega, N_Gamma, N_init, N_data = 3000, 800, 500, 40
ITERATIONS = 5000


# run all the equations
subprocess.run(
    f"python darcy.py --iter {ITERATIONS} --LM {LM} --N_Omega {N_Omega} "
    f"--N_Gamma {N_Gamma}",
    shell=True,
)

subprocess.run(
    f"python heat.py --iter {ITERATIONS} --LM {LM} --N_Omega {N_Omega} "
    f"--N_Gamma {N_Gamma}",
    shell=True,
)

subprocess.run(
    f"python poisson_inverse.py --iter {ITERATIONS} --LM {LM} --N_Omega {N_Omega} "
    f"--N_Gamma {N_Gamma} --N_data {N_data}",
    shell=True,
)

subprocess.run(
    f"python poisson.py --iter {ITERATIONS} --LM {LM} --N_Omega {N_Omega} "
    f"--N_Gamma {N_Gamma}",
    shell=True,
)

subprocess.run(
    f"python stokes_evo.py --iter {ITERATIONS} --LM {LM} --N_Omega {N_Omega} "
    f"--N_Gamma {N_Gamma}",
    shell=True,
)

subprocess.run(
    f"python stokes.py --iter {ITERATIONS} --LM {LM} --N_Omega {N_Omega} "
    f"--N_Gamma {N_Gamma}",
    shell=True,
)

subprocess.run(
    f"python waveeq.py --iter {ITERATIONS} --LM {LM} --N_Omega {N_Omega} "
    f"--N_Gamma {N_Gamma} --N_init {N_init}",
    shell=True,
)