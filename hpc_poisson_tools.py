import csv
import os

from firedrake import *
from firedrake.petsc import PETSc

parprint = PETSc.Sys.Print


# Create class for generating CSVs of results
class ResultsCSV(object):

    fields = ['cores', 'baseN', 'nref', 'degree', 'solver name',
              'error', 'dofs', 'dofs_core', 'runtime']

    def __init__(self, filename, comm=COMM_WORLD):
        self.filename = filename
        self.comm = comm
        if self.comm.rank == 0:
            if not os.path.isfile(self.filename):
                with open(self.filename, 'w') as csvf:
                    writer = csv.DictWriter(csvf, fieldnames=self.fields)
                    writer.writeheader()

    def record_result(self, baseN, nref, degree, name, error, runtime, dofs):
        singleresult = {}
        singleresult['cores'] = self.comm.size
        singleresult['baseN'] = baseN
        singleresult['nref'] = nref
        singleresult['degree'] = degree
        singleresult['solver name'] = name
        singleresult['error'] = error
        singleresult['dofs'] = dofs
        singleresult['dofs_core'] = dofs/self.comm.size
        singleresult['runtime'] = runtime
        if self.comm.rank == 0:
            with open(self.filename, 'a') as csvf:
                writer = csv.DictWriter(csvf, fieldnames=self.fields)
                writer.writerow(singleresult)
                csvf.flush()


def make_problem(Nx, Nref, degree):
    # Create mesh and mesh hierarchy
    mesh = UnitCubeMesh(Nx, Nx, Nx)
    hierarchy = MeshHierarchy(mesh, Nref)
    mesh = hierarchy[-1]

    V = FunctionSpace(mesh, "CG", degree)
    dofs = V.dim()
    parprint('DOFs', dofs)

    u = TrialFunction(V)
    v = TestFunction(V)

    bcs = DirichletBC(V, zero(), (1, 2, 3, 4, 5, 6))

    x, y, z = SpatialCoordinate(mesh)

    a = Constant(1)
    b = Constant(2)
    exact = sin(pi*x)*tan(pi*x/4)*sin(a*pi*y)*sin(b*pi*z)
    truth = Function(V).interpolate(exact)
    f = -pi**2 / 2
    f *= 2*cos(pi*x) - cos(pi*x/2) - 2*(a**2 + b**2)*sin(pi*x)*tan(pi*x/4)
    f *= sin(a*pi*y)*sin(b*pi*z)

    a = dot(grad(u), grad(v))*dx
    L = f*v*dx
    u_h = Function(V)
    problem = LinearVariationalProblem(a, L, u_h, bcs=bcs)
    return problem, u_h, truth


lu_solver = "mumps"
smooth_steps = 10

# Define (quiet) solver options
lu_mumps = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": lu_solver
}

vmg = {
    "ksp_type": "cg",
    "pc_type": "mg",
}

fmg = {
    "ksp_type": "preonly",
    "pc_type": "mg",
    "pc_mg_type": "full",
    "mg_levels_ksp_type": "chebyshev",
    "mg_levels_ksp_max_it": smooth_steps,
    "mg_levels_pc_type": "jacobi",
    "mg_coarse_pc_type": "lu",
    "mg_coarse_pc_factor_mat_solver_type": lu_solver
}

fmg_matfree = {
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
        "pc_type": "lu",
        "pc_factor_mat_solver_type": lu_solver
    }
}

telescope_factor = 1  # Set to number of nodes!
fmg_matfree_telescope = {
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
        "pc_telescope_reduction_factor": telescope_factor,
        "pc_telescope_subcomm_type": "contiguous",
        "telescope_pc_type": "lu",
        "telescope_pc_factor_mat_solver_type": lu_solver
    }
}

solver_dict = {
    'LU': lu_mumps,
    'CG + MGV': vmg,
    'Full MG': fmg,
    'Matfree FMG': fmg_matfree,
    'Telescoped matfree FMG': fmg_matfree_telescope
}
