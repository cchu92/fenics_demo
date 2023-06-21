"""
Created on Mon Jun 19 10:05:58 2023

@author: chenchen.chu@itwm.fraunhofer.de

with given 2d microstructure, and applied pbcs, 
with applied macro graient; H=[a,b;c,d]; solve the average strain energy 
"""
import  dolfin as dl
import numpy as np
import fenics as fe

# import rve mesh 
path = '/Users/1321143263qq.com/Dropbox/ScientificProject/1PaperProject/2023/Cassini_oval/results/fem_solutions/fenics_x_python/bistable_fem_test/'
mesh = dl.Mesh(path + 'rve109.xml');
# dl.plot(mesh)

# measue the bounaryies 
coordinates = mesh.coordinates()
x_min, y_min = np.amin(coordinates, axis=0)[:2]
x_max, y_max = np.amax(coordinates, axis=0)[:2]
 
tol = 1.e-17;
def conner(x,on_bounary):
    return dl.on_bounary and abs(dl.x[0]-x_min)<tol and abs(dl.x[1] - y_min)<tol



class PeriodicBoundary(dl.SubDomain):
    def inside(self, x, on_boundary):
        return bool((dl.near(x[0], x_min) or dl.near(x[1], y_min)) and on_boundary)

    def map(self, x, y):
        if dl.near(x[0], x_max) and dl.near(x[1], y_max):
            y[0] = x[0] - (x_max - x_min)
            y[1] = x[1] - (y_max - y_min)
        elif dl.near(x[0], x_max):
            y[0] = x[0] - (x_max - x_min)
            y[1] = x[1]
        else:  # near(x[1], y_max)
            y[0] = x[0]
            y[1] = x[1] - (y_max - y_min)


# Define function space
V = dl.VectorFunctionSpace(mesh, "CG", 1, constrained_domain=PeriodicBoundary())
u = dl.TrialFunction(V) # displacement 

Fg = dl.as_matrix(((dl.Constant(0.0) , dl.Constant(0.0) ), (dl.Constant(0.0) , dl.Constant(-0.2) )))  # Macroscopic deformation gradient
H = np.array([[0.0, -0.1],
              [0.0, 0.1]])
Fg = dl.Constant(H)  # Macroscopic deformation gradient
# Create Dirichlet boundary condition(fixed one conner)
u0_fix= dl.Constant(0.0)
bc_0 = dl.DirichletBC(V, u0_fix, conner)


# ------------------------------
# Weak form, hyper ealastic model
# -------------------------------
# Compressible  material 
def Compressible_Neo_hookean(mu,lmbda,u,Fg):
# original model is:  psi = (mu / 2) * (Ic - 2) - mu * dl.ln(J) + (lmbda / 2) * (dl.ln(J))**2
    V = u.function_space();
    u_, du = dl.TestFunction(V), dl.TrialFunction(V)
    I = dl.Identity(u.geometric_dimension())
    Ft = I + dl.grad(u); F = Ft * dl.inv(Fg)
    J = dl.det(F)
    psi = (mu / 2) * (dl.tr(F.T*F) - 3) - mu * dl.ln(J) + (lmbda / 2) * (dl.ln(J))**2
	
    Pi = psi*dl.dx
    Res = dl.derivative(Pi, u, u_)
    Jac = dl.derivative(Res, u, du)
    return Res, Jac


# nearly incompressible material, kappa/mu > 10.0
def NearlyIncompElasticity(mu, u, Fg):
	dim = u.geometric_dimension()
	V = u.function_space()
	u_, du = dl.TestFunction(V), dl.TrialFunction(V)
	I = dl.Identity(dim)
	ln = dl.ln
	Ft = I + dl.grad(u); F = Ft * dl.inv(Fg)
	J = dl.det(F)
	psi = mu/2 * (dl.tr(F.T*F) + 1/J**2 - 3)
	Pi = psi*dl.dx
	Res = dl.derivative(Pi, u, u_)
	Jac = dl.derivative(Res, u, du)

	return Res, Jac

E,nu  = 1.e7, 0.3
mu, lmbda = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
F, J =  Compressible_Neo_hookean(mu,lmbda,u,Fg)
# F, J =  NearlyIncompElasticity(mu,u,Fg)
bc_0=[];
problem = dl.NonlinearVariationalProblem(F, u, bc_0, J)
solver = dl.NonlinearVariationalSolver(problem)
# solver.parameters['newton_solver']['absolute_tolerance'] = 1e-8
# solver.parameters['newton_solver']['relative_tolerance'] = 1e-8
# solver.parameters['newton_solver']['maximum_iterations'] = 50
# solver.parameters['newton_solver']['linear_solver'] = 'mumps'
solver.parameters["newton_solver"]["linear_solver"] = "lu"

solver.solve()













