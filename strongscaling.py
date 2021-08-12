from hpc_poisson_tools import *
from time import time

import argparse
import gc

gc.disable()

parser = argparse.ArgumentParser()
parser.add_argument('--solver',
                    type=str,
                    default='Full MG',
                    choices=['Full MG', 'Matfree FMG', 'Telescoped matfree FMG'],
                    help='Solver key')
parser.add_argument('--telescope',
                    type=int,
                    default=1,
                    help='telescoping factor')
args, unknown = parser.parse_known_args()

baseN = 10
nref = 4
degree = 2

csvfile = ResultsCSV('strong.csv')

problem, u_h, truth = make_problem(baseN, nref, degree)

# We don't want to look at LU or CG + MGV strong scaling
del solver_dict['LU']
del solver_dict['CG + MGV']
solver_dict['Telescoped matfree FMG'] = {
    "mat_type": "matfree",
    "ksp_type": "preonly",
    "pc_type": "mg",
    "pc_mg_type": "full",
    "mg_levels_ksp_type": "chebyshev",
    "mg_levels_ksp_max_it": smooth_steps,
    "mg_levels_pc_type": "jacobi",
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled": {
        "mat_type": "aij",
        "pc_type": "telescope",
        "pc_telescope_reduction_factor": args.telescope,
        "pc_telescope_subcomm_type": "contiguous",
        "telescope_pc_type": "lu",
        "telescope_pc_factor_mat_solver_type": lu_solver
    }
}

key = args.solver
value = solver_dict[key]

u_h.assign(0.0)
t = time()
solver = LinearVariationalSolver(problem, solver_parameters=value)
solver.solve()
t = time() - t

recerror = errornorm(truth, u_h)
dofs = u_h.function_space().dim()
csvfile.record_result(baseN, nref, degree, key, recerror, t, dofs)
parprint(f'CPUs: {COMM_WORLD.size:4d} BaseN: {baseN:3d} Solver: {key:25} Error: {recerror:8.5f} Time: {t:8.5f} s')
