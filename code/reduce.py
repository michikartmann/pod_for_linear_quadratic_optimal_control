#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:54:15 2023

@author: michaelkartmann
"""
import scipy.sparse as sps
from scipy import linalg
import numpy as np
from model import model
from methods import collection
import matplotlib.pyplot as plt
from time import perf_counter

def petrov_galerkin_projection(model, U, V = None, product = None, H_prod = None):
    
    # init
    pde, cost = model.pde, model.cost_data
    if V is None: #then do Galerkin projection
        V = U
    
    # create projected pde
    projected_pde = collection()
    projected_pde.type = 'ROM_with_symmetric_operator'
    projected_pde.A = U.T@(pde.A.dot(V))
    projected_pde.M = U.T@(pde.M.dot(V))
    projected_pde.B = U.T@pde.B
    projected_pde.input_product = pde.input_product
    projected_pde.state_products = {}    
    for PP in pde.state_products.keys():
        MAT = pde.state_products[PP]
        projected_pde.state_products[PP] = (V.T@(MAT.dot(V)))  
    projected_pde.F = U.T@(pde.F)
    
    # TODO include different projections of the initial value
    projected_pde.y0 =  U.T@product@(pde.y0) 
    projected_pde.state_dim = np.shape(projected_pde.A[0])[0]
    projected_pde.input_dim = pde.input_dim
    
    # project cost
    projected_cost = collection()
    projected_cost.Ud = cost.Ud
    projected_cost.regularization_parameter = cost.regularization_parameter
    projected_cost.terminal_parameter = cost.terminal_parameter
    projected_cost.ua = cost.ua
    projected_cost.ub = cost.ub
    projected_cost.Mc_Yd = V.T@cost.Mc_Yd
    projected_cost.Mc_YT = V.T@cost.Mc_YT
    projected_cost.Yd_Mc_Yd = cost.Yd_Mc_Yd
    projected_cost.YT_Mc_YT = cost.YT_Mc_YT
    projected_cost.Ud = cost.Ud
    projected_cost.Yd = None
    projected_cost.YT = None
    
    return projected_pde, projected_cost

#%% POD

class pod():
     
    def __init__(self, model, model_toproject = None, H_prod = None, space_product = None, time_product = None):
        
        self.model = model
        if model_toproject is None:
            self.model_toproject = model
        else:
            self.model_toproject = model_toproject
        self.space_product = space_product
        self.truncation_tol = 1e-15
        self.time_product = time_product
        if space_product is not None:
            start_time = perf_counter()
            self.Wchol = linalg.cholesky(space_product.todense())
            end_time = perf_counter()
            print(f'Cholesky of space product done in {end_time-start_time}')
        else:
            self.Wchol = None 
        if H_prod is not None:
            self.H_prod = H_prod
    
    def ROMtoFOM(self, u):
        return self.V_right@u
            
    def FOMtoROM(self, u):
        if self.projection_product is None:
            return self.U_left.T@u
        else:
            return self.U_left.T@ self.projection_product@u
    
    def check_orthogonality(self):
        print(self.POD_Basis.T@self.space_product@self.POD_Basis)
        
    def get_rom(self, l, Snapshots, PODmethod, plot = True):
        
        print('ROM POD constructing ...')
        start_time = perf_counter()
        
        # 1. get POD basis    
        pod_basis, pod_values = self.pod_basis(Y = Snapshots, 
                                                l = l, 
                                                W = self.space_product, 
                                                D = self.time_product, 
                                                flag = PODmethod)
        if plot:
            self.plot_pod_values()
        self.U_left = pod_basis
        self.V_right = pod_basis
        self.projection_product = self.space_product
            
        # 2. project the model
        rom_pod = self.project(U = self.U_left, 
                               V = self.V_right, 
                               product = self.projection_product, 
                               H_prod = self.space_product, 
                               model_to_project = self.model_toproject)
        
        # 3. get error estimators
        rom_pod.error_est = self.get_error_estimator()
        
        # finalize
        self.rom_pod = rom_pod
        end_time = perf_counter()  
        print(f'ROM constructed in {end_time-start_time}')
        
        return rom_pod
    
    def project(self, U, V = None, product = None, H_prod = None, model_to_project = None):
        if model_to_project is None:
            model_to_project = self.model_toproject
        projected_pde, projected_cost = petrov_galerkin_projection(model_to_project, U, V, product , H_prod)
        rom = model(projected_pde, projected_cost, model_to_project.time_disc, model_to_project.space_disc, model_to_project.options)
        rom.type = projected_pde.type+'POD'
        return rom
        
    def pod_basis(self, Y, l, W = None, D = None, flag = 0, plot = False, energy_tolerance = None):
        """
        #     Compute POD basis

        #     Parameters
        #     ----------
        #     Y: list of/or ndarray shape (n_x,n_t),
        #         Matrix containing the vectors {y^k} (Y = [y^1,...,y^nt]),
        #         or a list containing different snapshot matrices
        #     l: int,
        #         Length of the POD-basis.
        #     W: ndarray, shape (n_x,n_x)
        #         Gramian of the Hilbert space X, that containts the snapshots.
        #     D: list of/or ndarray of shape (n_t,n_t)
        #         Matrix containing the weights of the time discretization.
        #     flag: int
        #         parameter deciding which method to use for computing the POD-basis
        #         (if flag==0 svd, flag == 1 eig of YY', flag == 2 eig of Y'Y (snapshot method).

        #     Returns
        #     -------
        #     POD_Basis: ndarray, shape (n_x,l)
        #                 matrix containing the POD-basis vectors
        #     POD_Values: ndarray, shape (l,)
        #            vector containing the eigenvalues of Yhat (see below)

        """
        
        ### INITIALIZATION
        # set truncation tol
        tol = self.truncation_tol
        truncate_normalized_POD_values = False
        
        if W is None: 
            # TODO set it to euclidian product
            pass 
        if D is None:
            pass
        
        # TODO do checks that the length of snapshots can vary
        if type(D) == list and 0:   
            Dsqrt = [d.sqrt() for d in D]
        else:
            Dsqrt = D.sqrt() 
            
        if type(Y) == list:
                
                if len(Y)==1:
                    Y = Y[0]
                else:
                    # at the moment all snapshots need to have the same length
                    K = len(Y)
                    Dsqrt = [Dsqrt]*K
                    Dsqrt = sps.block_diag(Dsqrt)
                    nx, nt = Y[0].shape
                    Y = np.concatenate( Y, axis=1 )      
        elif isinstance(Y, np.ndarray):
                nx, nt = Y.shape
        
        ### COMPUTE POD BASIS
        if flag == 0:
        # SVD
        
            # scale matrix
            Yhat = self.Wchol@Y@Dsqrt
            l_min = min(l,min(Yhat.shape)-1)
            print(f'Basissize dropped from {l} to {l_min} due to rank condition of snapshot matrix.')
            
            # perform svd
            U, S, V = sps.linalg.svds(Yhat, k=l_min)
            
            # get pod values
            POD_values = S**2
    
            # sort from biggest to lowest
            U = np.fliplr(U)
            POD_values = np.flipud(POD_values)
            
            # truncate w.r.t. the normalized singular values
            if truncate_normalized_POD_values:
                normalized_values = POD_values/POD_values[0]#abs(POD_values)/POD_values[0]
            else:
                normalized_values = POD_values
            print(f'Smallest singular value {normalized_values[-1]} and biggest {normalized_values[0]}.')
            indices = normalized_values > tol
            POD_values = POD_values[indices]
            U = U[:,indices]
            print(f'Basissize dropped from {l_min} to {U.shape[1]} due to truncation of small modes.')
            
            # get POD basis
            if 1:
                POD_Basis = linalg.solve_triangular(self.Wchol, U, lower = False)
            else:
                POD_Basis = U
        
        elif flag == 1: 
            # Compute eigenvalues of YY' with size (n_x, n_x):
           
            # scale matrix
            Yhat = self.Wchol@Y@Dsqrt
            Y_YT = Yhat@Yhat.T
            
            # compute eigenvalues
            POD_values, U = sps.linalg.eigsh(Y_YT, k = l, which = 'LM')
            
            # sort it from the biggest to the lowest
            U = np.fliplr(U)
            POD_values = np.flipud(POD_values)
            
            # truncate w.r.t. the normalized singular values
            if truncate_normalized_POD_values:
                normalized_values = POD_values/POD_values[0]
            else:
                normalized_values = POD_values   
            indices = POD_values > tol
            POD_values = POD_values[indices]
            U = U[:,indices]
            print(f'Basissize dropped from {l} to {U.shape[1]} due to truncation of small modes.')
            
            # get POD basis
            if 1:
                POD_Basis = linalg.solve_triangular(self.Wchol, U, lower = False)
            else:
                POD_Basis = U
            
        elif flag == 2: 
            # Method of snapshots: eigs of Y'Y with size (n_t,n_t)
           
            YT_Y = Dsqrt@Y.T@W@Y@Dsqrt

            if 1:
                POD_values, U = sps.linalg.eigsh(YT_Y, which = 'LM', k = l)
            else:
                pass
            
            # sort the computed eigenvalues and eigenvectors from biggest to lowest
            U = np.fliplr(U)
            POD_values = np.flipud(POD_values)
            
            # truncate w.r.t. the normalized singular values
            if truncate_normalized_POD_values:
                normalized_values = POD_values/POD_values[0]#abs(POD_values)/POD_values[0]
            else:
                normalized_values = POD_values
            indices = normalized_values > tol
            POD_values = POD_values[indices]
            U = U[:,indices]
            print(f'Basissize dropped from {l} to {U.shape[1]} due to truncation of small modes.')
            
            # get POD basis
            POD_Basis = Y@Dsqrt@U*1/(np.sqrt(POD_values))
            
        else:
            assert 0, 'wrong flag input ...'
            
        self.POD_Basis = POD_Basis
        self.POD_values = POD_values
        self.Singular_values = np.sqrt(POD_values)
        
        return POD_Basis, POD_values

    def get_error_estimator(self):
        # TODO
        return None
    
    def plot_pod_values(self):
        plt.figure()
        plt.title('POD Eigenvalues decay')
        plt.semilogy(self.POD_values)
        plt.show()
        