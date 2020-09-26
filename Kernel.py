# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 22:14:03 2020

@author: Stark
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from scipy import stats


#all notation follows Coupled_Diffusion.pdf
D = 1.

def P(j, expand = 0):
    PP = np.zeros((j,j))
    for i in range(j):
        for k in range(j):
            PP[i][k] = np.sin(np.pi / (j + 1) * (i + 1) * (k + 1) )
    
    inverse_PP = np.linalg.inv(PP)
    res = inverse_PP @ np.transpose(inverse_PP)
    m = res[0,0]
    inverse_PP = inverse_PP * np.sqrt(1 / m)
    if (expand == 0):
        return inverse_PP
    else:
        delta = expand - j
        B = np.zeros((j, delta))
        C = np.zeros((delta, j))
        D = np.eye(delta)
        return np.block([
        [inverse_PP, B],
        [C, D]
        ])

def Lambda(gamma, n, N):
    return (-2 * gamma * (1 - np.cos(np.pi * n / (N+1) )))

def M(N, gamma, t, expand = 0):
    A = np.asarray([np.exp(Lambda(gamma, i, N) * t) for i in range(1, N+1)])
    if (expand == 0):
        return np.diag(A)
    else:
        delta = expand - N
        B = np.zeros((N, delta))
        C = np.zeros((delta, N))
        D = np.eye(delta)
        return np.block([
        [np.diag(A), B],
        [C, D]
        ])
    
    
def inverse_sigma0(N, gamma, t, expand = 0):
    A = np.asarray([Lambda(gamma, i, N) / (np.exp(2* Lambda(gamma, i, N) * t) - 1) for i in range(1, N+1)])
    if (expand == 0):
        return np.diag(A)
    else:
        delta = expand - N
        B = np.zeros((N, delta))
        C = np.zeros((delta, N))
        D = np.diag(np.asarray([np.inf for i in range(delta)]))
        res = np.block([
        [np.diag(A), B],
        [C, D]
        ])
        return res
        
    
def K(At_inv, t, tau, N, gamma, expand = 0):
    #assumes that array sizes are the same, also that A is already computed at time t
    Mtaumt = M(N, gamma, tau - t, expand)
   
    invsigma = inverse_sigma0(N, gamma, tau - t, expand)
    
    return np.linalg.inv((At_inv + np.diag(np.diag(np.transpose(Mtaumt) @ np.diag(np.diag(invsigma @ Mtaumt))))))

def T(j, expand = 0):
    if(j == 1):
        return P(j, expand)
    else:
        return (np.linalg.inv(P(j-1, expand)) @ P(j, expand))
    
    
def initialize_A_inv(dt, D = 1., resolution = 5):
    Base = np.diag(np.asarray([np.inf for i in range(resolution)]))
    Base[0,0] = ((inverse_sigma0(1,1,dt, expand = 1))[0][0]) / D
    
    return Base 

def expand_A_inv(A, resolution):
    
    N = np.shape(A)[0]
    delta = resolution
    B = np.zeros((N, delta))
    C = np.zeros((delta, N))
    D = np.diag(np.asarray([np.inf for i in range(delta)]))
    return np.block([
    [A, B],
    [C, D]
    ])


def forward(At_inv, dt, steps, gamma, cur_size, laboratory_frame = False):
    
    MM = M(steps + 1, gamma, dt, expand = cur_size)
    KK = K(At_inv, 0, dt, steps + 1, gamma, expand = cur_size)
        
    SS = inverse_sigma0(steps + 1, gamma, dt, expand = cur_size)
        
    #this is clunky, may drastically decrease performance, if any problems arise consider looking here. 
    #the reason is treating cases like 0 * np.inf"
    inn = MM @ KK @ MM
    inn1 = inn @ SS
    inn1 = np.where(np.isnan(inn1), 0, inn1)
    inn2 = SS @ inn1
    inn2 = np.where(np.isnan(inn2), 0, inn2)
       
    inner = SS - inn2
    #this should go away
    #recheck k/k+1
    
    Ainv = inner
    
    if (laboratory_frame == False):
        return Ainv
    else:
        return ToLab(Ainv, steps, cur_size)
    

def ToLab(Ainv, steps, cur_size):
    TTi = np.linalg.inv(P(steps + 1, expand = cur_size))
    R1 = Ainv @ TTi
    R1 = np.where(np.isnan(R1), 0, R1)
    R2 = np.transpose(TTi) @ R1
    R2 = np.where(np.isnan(R2), 0, R2)
        
    return R2


def rough_compute(A_inv_initial, start, steps, delta_t, gamma, laboratory_frame = False, resolution = 5, D = 1.):
    """
    Returns inverse of covariance matrix after "steps - start" particles have been added, given starting A^{-1}
    
    steps is the number of particles to add (start = 0 with 2 particles including base means steps = 0)
    
    delta_t is essentially inverse velocity
    
    Size of matrices is expanded by 'resolution' every time they are filled. Continues until
    target number of steps has been reached. Higher resolution means lower accuracy but faster computation time.

    """
    
    Ainv = A_inv_initial
    cur_size = np.shape(A_inv_initial)[0]

    for k in range(1 + start, steps + start + 1):
        
        
        #further division for precise calculation
        
        if ((k+1) % resolution == 0):         
            cur_size += resolution
            Ainv = expand_A_inv(Ainv, resolution)
            
        inner = forward(Ainv, delta_t, k, gamma, cur_size, False)
        
        Ainv = transform_to_next(inner, k, cur_size)
        
    if (laboratory_frame == False):
        return Ainv
    else:
        return ToLab(Ainv, start + steps, cur_size)
    

def transform_to_next(Ainv_final, steps, cur_size):
    inner = Ainv_final
    TT = T(steps + 1, expand = cur_size)
    outer1 = inner @ TT
    outer1 = np.where(np.isnan(outer1), 0, outer1)
        
    outer2 = np.transpose(TT) @ outer1
    outer2 = np.where(np.isnan(outer2), 0, outer2)
        
    Ainv = outer2
    return Ainv
    
    
def optimal_precision(gamma, t):
    dt = 1 / gamma
    return max(1, np.int(t / dt))
    
def fixed_frame_compute(A_inv_initial, steps, delta_t, gamma, cur_size, precision = 1, laboratory_frame = False, D = 1.):
    """
    Returns precise covariance matrices in the current frame given starting A. 
    
    steps is the number of particles the have been inserted (steps = 0 means 2 particles including base)
    
    Output is time lattice and corresponding covariance matrices.
    
    delta_t is the time before the new addition.
    
    precision is the number of subdivisions of delta_t (if precision == 1, optimal value is calculated)
    """
    
    if (precision == 1):
        precision = optimal_precision(gamma, delta_t)
        
    lattice = np.linspace(start = delta_t / precision, stop = delta_t, num = precision)
    
    #stop when relative change becomes small to avoid singular matrices
    Covs = [forward(A_inv_initial, lat, steps, gamma, cur_size, False) for lat in lattice]
    
    return lattice, Covs

def relaxed(j, expand):
    R1 = inverse_sigma0(j, gamma, dt, expand) @ np.linalg.inv(P(j-1, expand))
    R1 = np.where(np.isnan(R1), 0, R1)
    R2 = P(j-1, expand) @ R1
    R2 = np.where(np.isnan(R2), 0, R2)
   
    return R2