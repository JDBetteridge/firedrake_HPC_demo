from hpc_poisson_tools import *
from time import time

import gc
import numpy as np

gc.disable()

baseN_list = [round(4*2**(ii/3)) for ii in range(10)]
nref = 2
degree = 2

csvfile = ResultsCSV('weak.csv')

baseN = baseN_list[int(np.log2(COMM_WORLD.size)) + 1]
problem, u_h, truth = make_problem(baseN, nref, degree)
for key, value in solver_dict.items():
    if (baseN > 13) and (key == 'LU'):
        parprint('BaseN:', baseN, 'too big for LU, skipping')
        continue
    if (baseN > 20) and (key == 'CG + GMG V-cycle'):
        parprint('BaseN:', baseN, 'too big for CG + MGV, skipping')
        continue
    u_h.assign(0.0)
    t = time()
    solver = LinearVariationalSolver(problem, solver_parameters=value)
    solver.solve()
    t = time() - t

    recerror = errornorm(truth, u_h)
    dofs = u_h.function_space().dim()
    csvfile.record_result(baseN, nref, degree, key, recerror, t, dofs)
    parprint(f'CPUs: {COMM_WORLD.size:3d} BaseN: {baseN:3d} Solver: {key:35} Error: {recerror:8.5e} Time: {t:8.5f} s')
