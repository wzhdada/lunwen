#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:22:39 2020

@author: siobhanie
"""

from __future__ import print_function
from fenics import *
from dolfin import *
import matplotlib.pyplot as plt
import scipy.io
from scipy.io import savemat
from numpy import * 
from mshr import * 

parameters ['linear_algebra_backend'] = 'Eigen'
parameters ['reorder_dofs_serial'] = False


def helmholtz(mu,alpha,a,b,c,n,f1,extra1,extra2,h,mesh):  
    
    #Set up 
    V = FunctionSpace(mesh, 'Lagrange', 1)
    u, v = TrialFunction(V), TestFunction(V)
    
    #Boundart condition (Dirichlet on [a,c])
    u_D = Constant(0)
    def boundary(x, on_boundary):
        return on_boundary
    bc = DirichletBC(V, u_D, boundary)
    
    #Variational form
    L = h*v*dx
    
    a0 = (-(dot(grad(u), grad(v))))*dx
    A0, b = assemble_system(a0,L,bc);
    rows,cols,vals = as_backend_type(A0).data() 
    A0 = as_backend_type(A0).sparray()

    a1 = (extra1*f1*u*v)*dx
    A1, b = assemble_system(a1,L,bc);
    rows,cols,vals = as_backend_type(A1).data() 
    A1 = as_backend_type(A1).sparray()

    a2 = (extra2*f2*u*v)*dx
    A2, b = assemble_system(a2,L,bc);
    rows,cols,vals = as_backend_type(A2).data() 
    A2 = as_backend_type(A2).sparray()
    
    a = a0 + a1 + a2
    u = Function(V)
    solve(a == L,u,bc)

    # Plot solution
    plt.margins(0,0)
    plot(mesh) 
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    #plt.savefig("mesh_1.pdf",bbox_inches='tight',pad_inches=0)

    plt.show()    
    c=plot(u)
    plt.jet()
    plt.margins(0,0)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    #plt.savefig("2dsol_1.pdf",bbox_inches='tight',pad_inches=0)

    

if __name__ == "__main__":
    
    mu = Constant(.1)
    alpha = Constant(30)
    a = Constant(0)
    b = Constant(1)
    b_half = Constant(b/2)
    c = Constant(1.5)
    n = 5000 #grid pts 
    
    
    # Create mesh and define function space
    D1 = Rectangle(Point(0,0),Point(1.0,1.0))
    D2 = Rectangle(Point(0.1,0.1),Point(0.2,0.2))
    D3 = Circle(Point(0.5,0.5),0.05)
    D4 = Circle(Point(0.7,0.7),0.04)
    D5 = Circle(Point(0.1,0.7),0.04)
    domain = D1 - D2 - D3 - D4 - D5
    mesh = generate_mesh(domain,40)

    
    #Define the first part of the middle functions: (1+ mu k(x))^2 
    f1a = Expression('pow((1 + mu*(1 + x[0]*sin(30*pi*x[0]))),2)',degree=2,domain=mesh,mu=mu)
    f1b = Expression('pow((1 + mu*(1 + (1-x[0])*sin(30*pi*x[0]))),2)',degree=2,domain=mesh,mu=mu)
    f1c = Expression('pow((1+mu*1),2)',degree=2,domain=mesh,mu=mu)
    f1 = Expression('x[0] < b_half + DOLFIN_EPS ? f1a : (x[0] < b + DOLFIN_EPS ? f1b : x[0] <= c + DOLFIN_EPS ? f1c : 0)', \
                   f1a=f1a,f1b=f1b,f1c=f1c,b_half=b_half,b=b,c=c,degree=2)
        
    
    #Define the second part of the middle functions: beta(x)
    f2a = Expression('sin(x[0]*2*pi)',degree=2,domain=mesh,mu=mu)
    f2b = Expression('1',degree=0,domain=mesh,mu=mu)
    f2 = Expression('(x[0] <= b + DOLFIN_EPS ? f2a : x[0] <= c + DOLFIN_EPS ? f2b : 0)', \
                   f2a=f2a,f2b=f2b,b_half=b_half,b=b,c=c,degree=2)    
        
        
    #Define the right-hand side function (piecewise)
    p1 = Expression('exp(-alpha*pow(x[0],2))', degree=2, domain=mesh,alpha=alpha)
    p2 = Expression('0', degree=1, domain=mesh)
    h = Expression('x[0] <= b + DOLFIN_EPS ? p1 : (x[0] < c + DOLFIN_EPS ? p2 : 0)',
                   p1=p1, p2=p2, b=b, c=c, degree=2)
    
    
    extra1 = Expression('mu',mu=mu,degree=0)
    extra2 = Expression('sin(mu)',mu=mu,degree=0)

    helmholtz(mu,alpha,a,b,c,n,f1,extra1,extra2,h,mesh)

