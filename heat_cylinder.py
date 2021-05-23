"""
Heat eqn around a cylinder

adapted from ft01_heat.py tutorial
"""

from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np

T = 0.1            # final time
num_steps = 5      # number of time steps
dt = T / num_steps # time step size

square = Rectangle(Point(-1, -1), Point(1, 1))
cylinder = Circle(Point(0, 0), 0.01)

#domain = square - cylinder

domain = square

domain.set_subdomain(1, cylinder)
domain.set_subdomain(2, square-cylinder)

# create mesh and define function space
mesh = generate_mesh(domain, 64)

subdomains = MeshFunction("size_t", mesh, 2, mesh.domains())
boundary = MeshFunction("size_t", mesh, 1, mesh.domains())

for f in facets(mesh):
    domains = []
    for c in cells(f):
        domains.append(subdomains[c])

    domains = list(set(domains))
    if len(domains) > 1:
        boundary[f] = 2

V = FunctionSpace(mesh, 'P', 2)

# define boundary condition

u_D = Expression('1 - x[0]*x[0] - x[1]*x[1]',
                 degree=2)
"""
def boundary(x, on_boundary):
    return on_boundary

inner_bc = DirichletBC(V, u_D, boundary)
outer_bc = DirichletBC(V, 0.0, boundary)
bc = [outer_bc, inner_bc]
"""
inner_bc = DirichletBC(V, u_D, boundary, 2)

outer_bc = DirichletBC(V, u_D, 'on_boundary')
bc = [inner_bc, outer_bc]

# define initial value
u_n = project(u_D, V)

# define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)

F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
a, L = lhs(F), rhs(F)

u = Function(V)
t = 0

vtkfile_2 = File('heat_cylinder/time0.pvd')
vtkfile_2 << u

for n in range(num_steps):

    # Update current time
    t += dt
    u_D.t = t

    # Compute solution
    solve(a == L, u, bc)

    # Plot solution
    # plot(u)

    # Compute error at vertices
    u_e = interpolate(u_D, V)
    # error = np.abs(u_e.vector().array() - u.vector().array()).max()
    error = np.abs(u_e.vector() - u.vector()).max()
    print('t = %.2f: error = %.3g' % (t, error))

    # Update previous solution
    u_n.assign(u)

vtkfile_2 = File('heat_cylinder/time1.pvd')
vtkfile_2 << u
