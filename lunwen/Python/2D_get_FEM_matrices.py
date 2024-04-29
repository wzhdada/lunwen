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


def helmholtz(mu,alpha,a,b,c,n,f1,f2,h,mesh):  
    
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

    a1 = (u*v)*dx
    A1, b = assemble_system(a1,L,bc);
    rows,cols,vals = as_backend_type(A1).data() 
    A1 = as_backend_type(A1).sparray()

    a2 = (f1*u*v)*dx
    A2, b = assemble_system(a2,L,bc);
    rows,cols,vals = as_backend_type(A2).data() 
    A2 = as_backend_type(A2).sparray()
    
    a3 = (f1*f1*u*v)*dx
    A3, b = assemble_system(a3,L,bc);
    rows,cols,vals = as_backend_type(A3).data() 
    A3 = as_backend_type(A3).sparray()
    
    a4 = (f2*u*v)*dx
    A4, b = assemble_system(a4,L,bc);
    rows,cols,vals = as_backend_type(A4).data() 
    A4 = as_backend_type(A4).sparray()

    bvec = zeros(len(b),)
    for i in range(len(b)):
        bvec[i] = (b[i])
    b = bvec

    savemat('remove_mu.mat', {'A0':A0,'A1':A1,'A2':A2,'A3':A3,'A4':A4,'bvec_2d_':b})

    
    return 


if __name__ == "__main__":

    mu = None 
    alpha = Constant(30)
    a = Constant(0)
    b = Constant(1)
    b_half = Constant(b/2)
    c = Constant(1.5)
    n = 30000 #grid pts 
    
    # Create mesh and define function space
    D1 = Rectangle(Point(0,0),Point(1.0,1.0))
    D2 = Rectangle(Point(0.1,0.1),Point(0.2,0.2))
    D3 = Circle(Point(0.5,0.5),0.05)
    D4 = Circle(Point(0.7,0.7),0.04)
    D5 = Circle(Point(0.1,0.7),0.04)
    domain = D1 - D2 - D3 - D4 - D5
    mesh = generate_mesh(domain,700)
    
    
    #Define the first part of the middle functions: (1+ mu k(x))^2 
    f1a = Expression('(1 + x[0]*sin(30*pi*x[0]))',degree=2,domain=mesh)
    f1b = Expression('(1 + (1-x[0])*sin(30*pi*x[0]))',degree=2,domain=mesh)
    f1 = Expression('x[0] < b_half + DOLFIN_EPS ? f1a : x[0] <= b + DOLFIN_EPS ? f1b : 0', \
                   f1a=f1a,f1b=f1b,b_half=b_half,b=b,degree=2)
        
    
    #Define the second part of the middle functions: beta(x)
    f2a = Expression('sin(x[0]*2*pi)',degree=2,domain=mesh)
    f2b = Expression('1',degree=0,domain=mesh)
    f2 = Expression('x[0] <= b + DOLFIN_EPS ? f2a : x[0] <= c + DOLFIN_EPS ? f2b : 0', \
                   f2a=f2a,f2b=f2b,b_half=b_half,b=b,c=c,degree=2)    
        
        
    #Define the right-hand side function (piecewise)
    p1 = Expression('exp(-alpha*pow(x[0],2))', degree=2, domain=mesh,alpha=alpha)


    helmholtz(mu,alpha,a,b,c,n,f1,f2,p1,mesh)

