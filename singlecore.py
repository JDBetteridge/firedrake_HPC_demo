from hpc_poisson_tools import *
from time import time

import gc
gc.disable()

baseN_list = [round(4*2**(ii/3)) for ii in range(7)]
nref = 2
degree = 2

baseN_list = [16]

csvfile = ResultsCSV('singlecore.csv')

for baseN in baseN_list:
    problem, u_h, truth = make_problem(baseN, nref, degree)
    for key, value in solver_dict.items():
        if (baseN > 13) and (key == 'LU'):
            parprint('BaseN:', baseN, 'too big for LU, skipping')
            continue

        u_h.assign(0)
        t = time()
        solver = LinearVariationalSolver(problem, solver_parameters=value)
        solver.solve()
        t = time() - t

        recerror = errornorm(truth, u_h)
        dofs = u_h.function_space().dim()
        csvfile.record_result(baseN, nref, degree, key, recerror, t, dofs)
        parprint(f'BaseN: {baseN:3d} Solver: {key:25} Error: {recerror:8.5f} Time: {t:8.5f} s')
