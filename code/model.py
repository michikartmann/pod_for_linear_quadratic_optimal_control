#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:54:15 2023

@author: michaelkartmann
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve, factorized
from time import time, perf_counter
from methods import collection
import fenics as fenics

class model():
    
    def __init__(self, pde, cost, time, space, options = None):
        self.pde = pde
        self.cost_data = cost
        self.options = options
        self.time_disc = time
        self.state_dim = pde.state_dim
        self.input_dim = pde.input_dim
        self.products = pde.state_products
        self.model_type = pde.type
        self.space_disc = space
        if self.options is not None:
            if options.factorize:
                self.factorize()
            else:
                self.pde.factorized_op = None
                self.pde.factorized_op_adjoint = None     
        else:
            options = collection()
            options.factorize = False
            self.pde.factorized_op = None
            self.pde.factorized_op_adjoint = None
            self.options = options
    
    def isFOM(self):
        return 'FOM' in self.pde.type
    
    def print_info(self):
        print(f'This models name is {self.model_type} with state dim {self.state_dim} and {self.time_disc.K} time steps, control dim is {self.input_dim}.')
        
    def factorize(self):
        self.pde.factorized_op = factorized(self.pde.M + self.time_disc.dt*self.pde.A)
        self.pde.factorized_op_adjoint = self.pde.factorized_op
        
    def offline_phase_optimization(self, Yd = None, YT = None, Ud = None):
        
        # TODO compute here particular solution y_hat corresponding to non-homogeneous data (f, y0) and save it
        
        if Yd is not None:
            self.cost_data.Yd = Yd
            self.cost_data.Mc_Yd = self.products['L2']@Yd
            self.cost_data.Yd_Mc_Yd = self.space_time_product(Yd, Yd, space_norm = 'L2', return_trajectory = True) 
        if YT is not None:
            self.cost_data.YT = YT
            self.cost_data.Mc_YT = self.products['L2']@YT
            self.cost_data.YT_Mc_YT = self.space_product(YT, YT, space_norm = 'L2')
        if Ud is not None:
            self.cost_data.Ud = Ud
        self.J10 = 0.5*self.time_disc.cost_zero*(self.pde.y0-self.cost_data.Yd[:,0]).T@self.products['L2'].dot(self.pde.y0-self.cost_data.Yd[:,0])
        print(f'Optimization offline phase done for {self.model_type} ...')

#%% state, adjoint methods

    def solve_linear_system(self, M, A, dt, rhs, factorized_op):
        if self.options.factorize:
           out = factorized_op(rhs)
        else:
            LHS = M + dt*A
            if self.isFOM():
                out = spsolve(LHS, rhs)
            else:
                out = np.linalg.solve(LHS, rhs)
        return out
        
    def solve_state(self, U = None, theta = 1, print_ = False, y0 = None):
        
        # init
        start_time = perf_counter()
        A = self.pde.A
        B = self.pde.B
        M = self.pde.M
        F = self.pde.F
        if y0 is None:
            y0 = self.pde.y0
        time_disc = self.time_disc
        dt = time_disc.dt
        K = time_disc.K
        y_current = y0.copy()
        Y = y_current.copy().reshape(-1,1)
        t = time_disc.t_v[0]
        if print_:
            print(f'k = {0}: t = {t},')
        
        # loop
        for k in range(1, K):
            
            # get current time
            t = time_disc.t_v[k]
            
            # build LHS and rhs
            RHS_mat = M
            rhs = RHS_mat.dot(y_current)
            if F is not None:
                rhs += dt*F[:,k]
            if U is not None:
                rhs += dt*B.dot(U[:,k])
            
            # solve system
            y_current = self.solve_linear_system(M, A, dt, rhs, self.pde.factorized_op)
            
            # save
            Y = np.concatenate((Y,y_current.copy().reshape(-1,1)), axis=1)
            if print_:
                print(f'k = {k}: t = {t}')
        
        time_ = perf_counter() - start_time
        if print_:
            print(f'Time stepping finished in time {time_}')
        
        return Y, time
    
    def solve_adjoint(self, Z, ZT):
        
        # init
        time_disc = self.time_disc
        dt = time_disc.dt
        K = time_disc.K
        A = self.pde.A
        M = self.pde.M
        B = self.pde.B
        
        # strategy 1 (OBD)
        if 0:
            p = np.zeros((A.shape[0],))
            p += self.cost_data.terminal_parameter*ZT
        
        # strategy 2 (DBO)
        else:
            rhs = time_disc.D[-1]*Z[:,-1] 
            rhs += self.cost_data.terminal_parameter*ZT 
            p = self.solve_linear_system(M, A.T, dt, rhs, self.pde.factorized_op_adjoint)
        P = p.copy().reshape(-1,1)
        B_listTP = [B.T.dot(p)]
        
        for k in range(K-2, -1, -1):
            
            # get t
            # t = time_disc.t_v[k]
            
            # assemble rhs and lhs
            F = dt*Z[:,k]
            rhs = M.dot(p) + F
            
            # solve system
            p = self.solve_linear_system(M, A.T, dt, rhs, self.pde.factorized_op_adjoint)

            # append and compute output
            P = np.concatenate((p.reshape(-1,1),P), axis=1)
            B_listTP.append(B.T.dot(p))
            
        # reverse time
        B_listTP.reverse()
        
        return P, np.array(B_listTP).T

#%% cost function and gradient

    def reduced_gradient(self, u, Y = None, P = None, B_listTP = None):  

        U = self.vector_to_matrix(u, self.input_dim) 
        if Y is None:
            Y, time = self.solve_state(U = U) 
        
        C = self.products['L2']
        if P is None or B_listTP is None:
            Z = C.dot(Y) - self.cost_data.Mc_Yd
            ZT = C.dot(Y[:,-1]) - self.cost_data.Mc_YT
            P, B_listTP = self.solve_adjoint(Z, ZT)
            
        dJ1 = B_listTP
        W = U - self.cost_data.Ud
        dJ2 = self.cost_data.regularization_parameter * self.pde.input_product.dot(W)
        dJ = dJ1 + dJ2
        return dJ.flatten(), Y, P, B_listTP
    
    def reduced_cost_fun(self, u, Y = None):
        
        U = self.vector_to_matrix(u, self.input_dim)
        if Y is None:
           Y, time = self.solve_state(U = U)
        
        # quadratic stuff trajectory
        J1_2 = 0.5 * self.space_time_product(Y, Y, 'L2')
        J1_2 -= self.space_time_product(Y, self.cost_data.Mc_Yd, 'identity') 
        J1_2 += 0.5* self.time_norm_scalar(self.cost_data.Yd_Mc_Yd)
        
        # quadratic stuff end time
        J3_2 = 0.5 * self.cost_data.terminal_parameter * self.space_product(Y[:,-1], Y[:,-1], space_norm = 'L2')
        J3_2 -= self.cost_data.terminal_parameter * self.space_product(Y[:,-1], self.cost_data.Mc_YT, space_norm = 'identity')
        J3_2 += 0.5 * self.cost_data.terminal_parameter * self.cost_data.YT_Mc_YT
        
        # quadratic stuff control
        W = U - self.cost_data.Ud
        J2_2 = 0.5 * self.cost_data.regularization_parameter * self.space_time_product(W, W, 'control')
        
        J = J1_2+J2_2+J3_2+self.J10
        return J
    
    def projectionUad(self, u):
        out = np.maximum( np.minimum(u, self.cost_data.ub), self.cost_data.ua)
        return out
    
    def get_true_function_value(self, u):
        return self.reduced_cost_fun(u)+self.J0
    
    def get_snapshots(self, u):
        U = self.vector_to_matrix(u, self.input_dim) 
        Y, time = self.solve_state(U = U) 
        C = self.products['L2']
        Z = C.dot(Y) - self.cost_data.Mc_Yd
        ZT = C.dot(Y[:,-1]) - self.cost_data.Mc_YT
        P, B_listTP = self.solve_adjoint(Z, ZT)
        return Y, P
    
#%% ocp solver

    def solve_ocp(self, u0, options = None, method = 'GradientMethod', linesearch = 'Barzilai-Borwein'):
        if options is None:
            options = self.set_options()  
        if  method == 'GradientMethod' and linesearch == 'Barzilai-Borwein':
            u_opt, history_opt = self.gradient_method(u0.flatten(), options, linesearch)
        elif method == 'GradientMethod' and linesearch == 'Backtracking':
            u_opt, history_opt = self.gradient_method(u0.flatten(), options, linesearch)
        return u_opt, history_opt

    def set_options(self, tol = 1e-6, maxit = 200, save = False, plot = True, print_info = True):
        options = {'print_info': print_info,
                   'print_final': True,
                   'plot': plot,
                   'save': save,
                   'path': None,
                   'tol': tol,
                   'maxit': maxit,
                   }
        return options
    
    def gradient_method(self, u_0, options, linesearch):
        
        if linesearch == 'Barzilai-Borwein':
            methodd = 'Projected Barzilai-Borwein Gradient method'
        elif linesearch == 'Backtracking':
            methodd = 'Projected Backtracking Gradient method'
        else:
            assert 0, 'Choose valid linesearch ...'
                
        if options['print_info']:
            print('#############################################################')
            print(f"Starting {methodd}")
         
        # init gradient, product and norms...
        control_norm = lambda x: self.space_time_norm(x, 'control')
        control_product = lambda x, y: self.space_time_product(x, y, 'control')
        fun = lambda x: self.reduced_cost_fun(x)
        gradient = lambda x: self.reduced_gradient(x)
        # projection_operator = lambda alpha, u, grad_u: grad_u
        projection_operator  = lambda alpha, u, grad_u: alpha*(u-self.projectionUad(u-1/alpha*grad_u))
        
        # initialize
        t = time()
        k = 0
        u_km1 = u_0
        grad_km1_uncons, Y, P, B_listTP = gradient(u_km1) 
        grad_km1 =  projection_operator(1, u_km1, grad_km1_uncons)
        u_k = grad_km1
        grad_k_uncons, Y, P, B_listTP = gradient(u_k)
        grad_k =  projection_operator(1, u_k, grad_k_uncons)
        grad_norm = control_norm(grad_k)
        grad_norm0 = grad_norm
        history = {'grad_norm': [grad_norm],
                   'time_stages': [],
                   'u_list': [u_k]
                    }
        
        if options['print_info']:
            print(f"k: {k:2}, grad_norm = {grad_norm: 2.4e}, rel grad_norm = {grad_norm/grad_norm0: 2.4e}")
            
        # loop
        while grad_norm > max(options['tol'],options['tol']*grad_norm0) and k<options['maxit']:
            
            
            if linesearch == 'Barzilai-Borwein':

                # compute BB steplength
                sk = u_k - u_km1
                dk = grad_k - grad_km1
                skdk = control_product(sk,dk)
                if k%2==0: 
                    alpha_k = control_product(dk,dk)/skdk
                else: 
                    alpha_k = skdk / control_product(sk,sk)
    
            elif linesearch == 'Backtracking':
                
                # linesearch parameters
                linesearch_para1 = 1e-3
                linesearch_para2 = 0.7
                max_iter_backtracking = 50
                
                # linesearch
                alpha_k = 0.5
                J_current = fun(u_k)
                u_new = self.projectionUad(u_k-alpha_k*grad_k_uncons)
                J_new = fun(u_new)
                count = 0
                norm_new = control_norm(u_k-u_new)
                while ((J_new - J_current)-(-linesearch_para1/alpha_k*norm_new**2)>0) and count <= max_iter_backtracking:
                        alpha_k *= linesearch_para2
                        u_new = self.projectionUad(u_k-alpha_k*grad_k_uncons)
                        J_new = fun(u_new)
                        norm_new = control_norm(u_k-u_new)
                        count += 1
                        
                if count == max_iter_backtracking:
                    print('Backtracking did not find a good steplength in {count} steps.')
               
            # get search direction depending on alpha
            grad_k =  projection_operator(alpha_k, u_k, grad_k_uncons)
            
            # update
            u_km1 = u_k
            u_k = u_k - grad_k/alpha_k
            grad_km1 = grad_k
            
            # compute new gradient and its norm
            grad_k_uncons, Y, P, B_listTP = gradient(u_k)
            grad_k = projection_operator(alpha_k, u_k, grad_k_uncons)
            grad_norm = control_norm(grad_k)
            
            # update history
            history['grad_norm'].append(grad_norm)
            history['u_list'].append(u_k)
            k += 1
            if options['print_info']:
                print(f"k: {k:2}, grad_norm = {grad_norm: 2.4e}, rel grad_norm = {grad_norm/grad_norm0: 2.4e}, alpha_k = {alpha_k}")
        
        # finalize
        history['Y_opt'] = Y
        history['P_opt'] = P
        history['B_listTP_opt'] = B_listTP
        history['time'] = time() - t 
        history['k'] = k
            
        if k == options['maxit']:
            history['flag'] = methodd + f' reached maxit of k = {k:2} iterations in {history["time"]: .3f} seconds with gradient norm of {grad_norm:2.4e}, rel grad_norm = {grad_norm/grad_norm0: 2.4e}.'
        else:
            history['flag'] = methodd + f' converged in k = {k:2} iterations in {history["time"]: .3f} seconds with gradient norm of {grad_norm:2.4e}, rel grad_norm = {grad_norm/grad_norm0: 2.4e}.'
        # history['flag'] += 
            
        if options['print_final']:
            print( history['flag'])
            print('#############################################################')            
        if options['plot']:
            plt.figure()
            plt.semilogy(history['grad_norm'])
            plt.title(rf'{methodd} convergence of $\|\nabla F(u_k)\|_U$')
            plt.xlabel(r'$k$')
            if options['save']:
                plt.savefig( options['path'] )
                
        U_opt = self.vector_to_matrix(u_k, self.input_dim)
        
        return U_opt, history
    
#%% products
    
    def time_norm_scalar(self, V, time_norm = None):
         if time_norm is None:
             time_norm = self.time_disc.D 
         return np.vdot(time_norm , V)
 
    
    def space_product(self, y1, y2, space_norm = 'L2'):
        if space_norm == 'L2':
            return y1.T.dot(self.products['L2'].dot(y2))
        elif space_norm == 'H1':
            return  y1.T.dot(self.products['H1'].dot(y2))
        elif space_norm == 'H10':
            return  y1.T.dot(self.products['H10'].dot(y2))
        elif space_norm == 'control': 
            return  y1.T.dot(self.pde.pde.dot(y2))
        elif space_norm == 'identity': 
            return  y1.T.dot(y2)
            
    def space_norm(self, y1, space_norm = 'L2'):
        return np.sqrt(self.space_product(y1, y1,space_norm))
    
    def space_time_product(self, v, w, space_norm = 'L2', time_norm = None, return_trajectory = False, space_mat = None):
        if time_norm is None:
            time_norm = self.time_disc.D
        if space_mat is not None:
            if len(v.shape)<2:
               v = self.vector_to_matrix(v, self.state_dim)
            if len(w.shape)<2:
               w = self.vector_to_matrix(w, self.state_dim)  
            if return_trajectory:
                return np.diag(v.T.dot(space_mat.dot(w)))
            else: 
                return np.vdot(time_norm , np.diag(v.T.dot(space_mat.dot(w))) )
        if space_norm == 'L2':
            if len(v.shape)<2:
               v = self.vector_to_matrix(v, self.state_dim)
            if len(w.shape)<2:
               w = self.vector_to_matrix(w, self.state_dim)  
            if return_trajectory:
                return np.diag(v.T.dot(self.products['L2'].dot(w)))
            else: 
                return np.vdot(time_norm , np.diag(v.T.dot(self.products['L2'].dot(w))) )
        elif space_norm == 'H1':
            if len(v.shape)<2:
               v = self.vector_to_matrix(v, self.state_dim)
            if len(w.shape)<2:
               w = self.vector_to_matrix(w, self.state_dim) 
            if return_trajectory: 
                return np.diag(v.T.dot(self.products['H1'].dot(w)))
            else: 
                return np.vdot(time_norm , np.diag(v.T.dot(self.products['H1'].dot(w))) )   
        elif space_norm == 'H10':
            if len(v.shape)<2:
               v = self.vector_to_matrix(v, self.state_dim)
            if len(w.shape)<2:
               w = self.vector_to_matrix(w, self.state_dim) 
            if return_trajectory: 
                return np.diag(v.T.dot(self.products['H10'].dot(w)))
            else:
                return np.vdot(time_norm , np.diag(v.T.dot(self.products['H10'].dot(w))) )
        elif space_norm == 'control': 
            if len(v.shape)<2:
               v = self.vector_to_matrix(v, self.input_dim)
            if len(w.shape)<2:
               w = self.vector_to_matrix(w, self.input_dim) 
            if return_trajectory: 
                return np.diag(v.T.dot(self.cost_data.input_product.dot(w)))
            else:
                return np.vdot(time_norm , np.diag(v.T.dot(self.pde.input_product.dot(w))) )
        elif space_norm == 'identity': 
            if len(v.shape)<2:
               v = self.vector_to_matrix(v, self.output_dim)
            if len(w.shape)<2:
               w = self.vector_to_matrix(w, self.output_dim) 
            if return_trajectory: 
                return np.diag(v.T.dot(w))
            else: 
                return np.vdot(time_norm , np.diag(v.T.dot(w)))
        
    def space_time_norm(self, v, space_norm = 'L2', time_norm = None):
        return np.sqrt(self.space_time_product(v, v, space_norm, time_norm = time_norm))
    
    def rel_error_norm(self, U1, U2, space_norm = 'L2'):
        return self.space_time_norm(U1-U2, space_norm = space_norm)/self.space_time_norm(U1, space_norm = space_norm)

#%% error estimation

    def optimal_control_error_est(self, Ur):
        dJ, Y, P, _ = self.reduced_gradient(Ur)
        xi = self.project_on_active_inactive_sets(dJ, Ur)
        norm = self.space_time_norm(xi, 'control')
        est = norm/self.cost_data.regularization_parameter
        return est, Y, P
    
    def project_on_active_inactive_sets(self, xi, u):
        u = u.flatten()
        lb = self.cost_data.ua
        ub = self.cost_data.ub
        out = -1*xi
        for i in range(len(u)):
            if u[i] == lb[i]:
                out[i] = -min(0, xi[i])
            elif u[i] == ub[i]:
                out[i] = -max(0, xi[i])
            else:
                pass
        return out

#%% plot
     
    def visualize_trajectory(self, Y):
         for k in range( 0, self.time_disc.K, int(self.time_disc.K/10) ):
             yy = Y[:,k]
             t = self.time_disc.t_v[k]
             self.plot_3d(yy, title = f"t = {t:2.3f}")
             
    def plot_3d(self, y, title=None, save_png=False, path=None, dpi='figure'):
        
        # read model
        dx = self.space_disc.dx
        mesh = self.space_disc.mesh
        V = self.space_disc.V
        
        # get plot data
        dims = ( dx+1 , dx+1 )
        X = np.reshape( mesh.coordinates()[:,0], dims )
        Y = np.reshape( mesh.coordinates()[:,1], dims )
        Z = np.reshape( y[fenics.vertex_to_dof_map(V)], dims )
        
        # plot
        fig = plt.figure()
        ax = fig.add_subplot( projection='3d' )
        if title is not None:
            ax.set_title(title)
        surf = ax.plot_surface( X, Y, Z, cmap=plt.cm.coolwarm )
        fig.colorbar(surf, shrink=0.5, aspect=5)
        # ax.view_init(elev=90, azim=-90, roll=0)
        if save_png:
            plt.savefig( path, dpi=dpi )
        plt.figure()
         
    def matrix_to_vector(self, V ):
         return V.flatten()
     
    def vector_to_matrix(self, v, dim ):
         return v.reshape(dim, self.time_disc.K)
     
#%% helpers

    def derivative_check(self, mode = 1):
        
        # init
        print('derivative check ...')
        f = self.reduced_cost_fun
        df = self.reduced_gradient
        Eps = np.array([1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6])
        u  = np.random.random((self.input_dim, self.time_disc.K))
        du = np.random.random((self.input_dim, self.time_disc.K))
        T = np.zeros(np.shape(Eps))
        T2 = T
        ff = f(u)
        
        # Compute right-side difference quotient
        for i in range(len(Eps)):
            #print(Eps[i])
            f_plus = f(u+Eps[i]*du)
            f_minus = f(u-Eps[i]*du)
            if mode == 1:
                ddd = self.space_time_product(df(u)[0], du, 'control')
                T[i] = abs( ( (f_plus - f_minus)/(2*Eps[i]) ) - ddd )
                T2[i] =  abs( ( (f_plus - ff)/(Eps[i]) ) - ddd )
            else:
                T[i] = abs( ( (f_plus - f_minus)/(2*Eps[i]) ) - df(u,du) )
                T2[i] =  abs( ( (f_plus - ff)/(Eps[i]) ) - df(u,du) )

        #Plot
        plt.figure()
        plt.xlabel('$eps$')
        plt.ylabel('$J$')
        plt.loglog(Eps, Eps, label='O(eps)')
        plt.loglog(Eps, T2, 'ro--',label='Test')
        plt.legend(loc='upper left')
        plt.grid()
        plt.title("Rightside difference quotient")