from hpc_poisson_tools import *
from time import time

import gc
gc.disable()

baseN_list = [4] + [round(4*2**(ii/3)) for ii in range(7)]
nref = 2
degree = 2

csvfile = ResultsCSV('singlecore.csv')

maxtime = 120
skip = []
for baseN in baseN_list:
    problem, u_h, truth = make_problem(baseN, nref, degree)
    for key, value in solver_dict.items():
        if key in skip:
            parprint(f'BaseN: {baseN} too big for {key}, skipping')
            continue

        u_h.assign(0)
        t = time()
        solver = LinearVariationalSolver(problem, solver_parameters=value)
        solver.solve()
        t = time() - t

        if t > maxtime:
            skip.append(key)

        recerror = errornorm(truth, u_h)
        dofs = u_h.function_space().dim()
        csvfile.record_result(baseN, nref, degree, key, recerror, t, dofs)
        parprint(f'BaseN: {baseN:3d} Solver: {key:35} Error: {recerror:8.5e} Time: {t:8.5f} s')
