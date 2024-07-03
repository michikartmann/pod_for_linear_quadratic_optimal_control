#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:54:15 2023

@author: michaelkartmann
"""

import fenics as fenics
import numpy as np
from scipy.sparse import csr_matrix, diags, identity
from methods import collection
from model import model

def discretize(dx = 50, K = 100):
    
    ##### options
    options = collection()
    options.factorize = True
    
    ##### time discretization
    time = collection()
    time.t0 = 0
    time.T = 3
    time.K = K
    time.dt = (time.T-time.t0)/(time.K-1)
    time.t_v = np.linspace(time.t0, time.T, num=time.K )
    time.D = time.dt * np.ones(time.K)
    time.D[0] = 0
    time.D[-1] = time.dt
    time.cost_zero = time.dt
    time.D_diag = diags(time.D)
    
    # space discretization
    space = collection()
    L = 1
    x1 = 0; y1 = 0; x2 = L; y2 = L
    lower_left = fenics.Point(x1,y1)
    upper_right = fenics.Point(x2,y2)
    mesh = fenics.RectangleMesh(lower_left, upper_right, dx, dx)
    V = fenics.FunctionSpace(mesh, 'P', 1) 
    space.V = V
    space.mesh = mesh
    space.dx = dx
    space.DirichletBC = None
    
    ##### pde
    pde = collection()
    y = fenics.TrialFunction(V)
    v = fenics.TestFunction(V)
    
    # A
    reac_fun = fenics.Constant(1.0)
    A_reac_form = fenics.assemble(reac_fun * y * v * fenics.dx)
    A_reac = csr_matrix(fenics.as_backend_type(A_reac_form).mat().getValuesCSR()[::-1]) 
    diff_fun = fenics.Constant(1.0)
    A_diff_form = fenics.assemble(diff_fun* fenics.dot(fenics.nabla_grad(y),fenics.nabla_grad(v)) * fenics.dx)
    A_diff = csr_matrix(fenics.as_backend_type(A_diff_form).mat().getValuesCSR()[::-1]) 
    gamma = fenics.Constant(1.0)
    A_robin_form = fenics.assemble(gamma * y * v  * fenics.ds)
    A_robin = csr_matrix(fenics.as_backend_type(A_robin_form).mat().getValuesCSR()[::-1]) 
    pde.A = A_reac + A_diff + A_robin
    
    # B
    n_rhs = np.array([4,4])
    n_u = np.prod(n_rhs)     
    x_grid = np.linspace(0,1,n_rhs[0]+1)
    y_grid = np.linspace(0,1,n_rhs[1]+1)
    tol = 1e-14
    B = []
    for i in range(n_rhs[1]): # y
        for j in range(n_rhs[0]):# x
            char = fenics.Expression("x1<= x[0] &&  x[0] <= x2 + tol && y1 <=x[1] && x[1] <= y2  + tol ? k1 : k2", x1=x_grid[j],x2=x_grid[j+1],y1=y_grid[i],y2=y_grid[i+1],degree=0, tol=tol, k1=1, k2=0) 
            B.append( fenics.assemble(char*v*fenics.dx ).get_local())
            #plot(Function(V, assemble(char*v*dx )))
    B = np.array(B).T
    pde.B = B
    
    # M
    M_form = fenics.assemble(y * v * fenics.dx)   
    M = csr_matrix(fenics.as_backend_type(M_form).mat().getValuesCSR()[::-1])
    pde.M = M
    
    # products
    L2_form = fenics.assemble(y*v * fenics.dx)   
    L2 = csr_matrix(fenics.as_backend_type(L2_form).mat().getValuesCSR()[::-1])
    H10_form = fenics.assemble(fenics.dot(fenics.nabla_grad(y),fenics.nabla_grad(v)) * fenics.dx )
    H10 = csr_matrix(fenics.as_backend_type(H10_form).mat().getValuesCSR()[::-1])
    H1 = H10 + L2
    pde.state_products = {'H1': H1, 'L2': L2, 'H10': H10}
    pde.input_product = identity(n_u)
    
    # y0
    y0_fun= fenics.Constant(0.0)
    pde.y0 = fenics.interpolate(y0_fun, V).vector().get_local()
    space.dofs = len(pde.y0)
    
    # F
    f_fun = fenics.Expression('t * sin(2 * pi * x[0]) * cos(2 * pi * x[1])', degree = 1, t = time.t0)
    F = []
    for k in range(time.K):
        F.append(fenics.assemble(f_fun * v * fenics.dx).get_local())
        f_fun.t += time.dt
    pde.F = np.array(F).T
    
    # dims
    pde.input_dim = n_u
    pde.state_dim = space.dofs
    pde.type = 'FOM_with_symmetric_operator'
    
    ##### cost function
    cost = collection()
    # TODO generalize the code to time-dependent control bounds
    cost.ua = -2*np.ones(pde.input_dim * time.K) 
    cost.ub = 5*np.ones(pde.input_dim * time.K) 
    cost.regularization_parameter = 1e-3
    cost.terminal_parameter = 0
    cost.YT = 0
    cost.Ud = None
    cost.Yd = None
    
    ##### create model
    fom = model(pde, cost, time, space, options)
    return fom

def expression_to_vector(V, fenics_expression):
    y0 = fenics.interpolate(fenics_expression, V).vector().get_local()
    return y0

def time_expression_to_vector(V, fenics_expression):
    pass
