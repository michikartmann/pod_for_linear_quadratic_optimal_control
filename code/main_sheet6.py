#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:52:47 2023
@author: michaelkartmann
"""

import numpy as np
from discretize import discretize
from reduce import pod

#%% create FOM and solve for Yd

fom = discretize(dx = 50, K = 100)
fom.print_info()

# solve state to get Yd
U = 1*np.ones((fom.pde.input_dim, fom.time_disc.K))
Y, time = fom.solve_state(U, print_ = False)
if 0:
    fom.visualize_trajectory(Y)
    fom.plot_3d(Y[:,-1])
    print(f'L2(V)- state norm: {fom.space_time_norm(Y,"H1")}, control space norm: {fom.space_time_norm(U,"control")}')

# set ud to be zero
Ud = np.zeros((fom.pde.input_dim, fom.time_disc.K))

# get YD, UD and do offline phase for optimization
fom.offline_phase_optimization(Yd = Y, 
                               YT = Y[:,-1],
                               Ud = Ud)
fom.plot_3d(Y[:,-1], title = 'target state at end time')

#%% FOM optimization

# starting value and solver options
U_0 = 5*np.ones((fom.input_dim, fom.time_disc.K)) 
solver_options = fom.set_options(tol = 1e-10,
                            maxit = 300, 
                            save = False, 
                            plot = True,
                            print_info = True)

# FOM optimization
if 1:
    
    # set regularization
    fom.cost_data.regularization_parameter = 1e-3

    # derivative check
    if 0:
        fom.derivative_check()

    # solve with Barzilai-Borwein gradient method
    u_BB, history_BB = fom.solve_ocp(U_0, 
                                    method = "GradientMethod",
                                    options = solver_options,
                                    linesearch = 'Barzilai-Borwein')
    
    # plot yd, plot adjoint state, state, control, norm von adjoint
    fom.plot_3d(history_BB['Y_opt'][:,-1], title = 'FOM optimal state at end time')
    fom.plot_3d(history_BB['P_opt'][:,-1], title = 'FOM optimal adjoint state at end time')

#%% construct ROM out of optimal FOM snapshots

l = 20

if 1: # train with optimal snapshots
    Snapshots = [history_BB['Y_opt'], history_BB['P_opt']]
else: # train with initial snapshots
    Y, P = fom.get_snapshots(U_0)
    Snapshots = [Y, P]
    
pod_object = pod(model = fom, 
        space_product = fom.pde.state_products['H1'], 
        time_product = fom.time_disc.D_diag)

rom = pod_object.get_rom(l = l, 
                Snapshots = Snapshots, 
                PODmethod = 0,
                plot = True)
rom.print_info()

#%% ROM optimization

if 1:
    
    # derivative check
    if 0:
        rom.derivative_check()

    # starting value and solver options
    U_0 = 5*np.ones((fom.input_dim, fom.time_disc.K)) 

    # solve with Barzilai-Borwein gradient method
    u_BB_ROM, history_BB_ROM = rom.solve_ocp(U_0, 
                                    method = "GradientMethod",
                                    options = solver_options,
                                    linesearch = 'Barzilai-Borwein')
    
    # plot yd, plot adjoint state, state, control, norm von adjoint
    fom.plot_3d(pod_object.ROMtoFOM(history_BB_ROM['Y_opt'][:,-1]), title = 'ROM optimal state at end time')
    fom.plot_3d(pod_object.ROMtoFOM(history_BB_ROM['P_opt'][:,-1]), title = 'ROM optimal adjoint state at end time')
    
    a_posteriori_est, _, _ = fom.optimal_control_error_est(u_BB_ROM)
    true_error = fom.space_time_norm(u_BB_ROM-u_BB,"control")
    print(f'Error in optimal control {true_error}, a posteriori error estimate {a_posteriori_est}, effectivity {true_error/a_posteriori_est}')
    print(f'FOM time {history_BB["time"]}, ROM time {history_BB_ROM["time"]}, Speed-up {history_BB["time"]/history_BB_ROM["time"]}')