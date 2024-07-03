#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:52:47 2023
@author: michaelkartmann
"""

import numpy as np
from discretize import discretize

#%% create model and solve state

fom = discretize(dx = 50, K = 100)
fom.print_info()

# solve state to get Yd
U = 1*np.ones((fom.pde.input_dim, fom.time_disc.K))
Y, time = fom.solve_state(U, print_ = False)
if 0:
    fom.visualize_trajectory(Y)
    fom.plot_3d(Y[:,-1])
    print(f'L2(V)- state norm: {fom.space_time_norm(Y,"H1")}, control space norm: {fom.space_time_norm(U,"control")}')

#%% optimization - scenario 1: verification of the algorithm

if 0:
    
    # get YD, UD and do offline phase for optimization
    fom.offline_phase_optimization(Yd = Y, 
                                   YT = Y[:,-1],
                                   Ud = U)
    fom.cost_data.regularization_parameter = 1e-3
    
    # derivative check
    if 0:
        fom.derivative_check()
    
    # starting value and solver options
    U_0 = 5*np.ones((fom.input_dim, fom.time_disc.K)) 
    solver_options = fom.set_options(tol = 1e-12,
                                maxit = 300, 
                                save = False, 
                                plot = True,
                                print_info = True)
    
    # solve with Barzilai-Borwein gradient method
    u_BB, history_BB = fom.solve_ocp(U_0, 
                                    method = "GradientMethod",
                                    options = solver_options,
                                    linesearch = 'Barzilai-Borwein')
    print(f'Error between analytic solution Ud and u_BB {fom.space_time_norm(u_BB-fom.cost_data.Ud,"control")}')
    
    if 0:
        # solve wth Backtracking gradient method
        u_BT, history_BT = fom.solve_ocp(U_0, 
                                        method = "GradientMethod",
                                        options = solver_options,
                                        linesearch = 'Backtracking')
        print(f'Error between analytic solution Ud and u_BT {fom.space_time_norm(u_BT-fom.cost_data.Ud,"control")}')

#%% optimization - scenario 2: ud = 0 and study the impact of the regularization

if 1:
    
    # set ud to be zero
    Ud = np.zeros((fom.pde.input_dim, fom.time_disc.K))
    
    # get YD, UD and do offline phase for optimization
    fom.offline_phase_optimization(Yd = Y, 
                                   YT = Y[:,-1],
                                   Ud = Ud)
    fom.plot_3d(Y[:,-1], title = 'target state at end time')
    
    # starting value and solver options
    U_0 = 5*np.ones((fom.input_dim, fom.time_disc.K)) 
    solver_options = fom.set_options(tol = 1e-10,
                                maxit = 300, 
                                save = False, 
                                plot = True,
                                print_info = False)
    
    # set list of regularization parameters
    regularization_list = [1, 1e-3, 1e-5]
    
    for regularization in regularization_list:
        
        fom.cost_data.regularization_parameter = regularization
    
        # derivative check
        if 0:
            fom.derivative_check()
    
        # solve with Barzilai-Borwein gradient method
        u_BB, history_BB = fom.solve_ocp(U_0, 
                                        method = "GradientMethod",
                                        options = solver_options,
                                        linesearch = 'Barzilai-Borwein')
        
        # plot yd, plot adjoint state, state, control, norm von adjoint
        fom.plot_3d(history_BB['Y_opt'][:,-1], title = 'optimal state at end time')
        fom.plot_3d(history_BB['P_opt'][:,-1], title = 'optimal adjoint state at end time')
        print(f'Regularization parameter {regularization: 2.2e}: L2(V)- adjoint norm: {fom.space_time_norm(history_BB["P_opt"],"H1")}.')
