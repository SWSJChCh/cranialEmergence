'''
VEGF.py - Samuel Johnson - 31/08/2025
'''

import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from scipy import signal
from numba import jit, prange

def createVEGFArray(width, length, c0, bcParam, xFactor=1):
    bcX = bcParam
    bcY = int(xFactor * bcParam)
    VEGFArray = np.full((width - 2 * bcX, length - 2 * bcY), c0)
    for i in prange(max(bcX, bcY) - 1, -1, -1):
        pad_x = (1 if i < bcX else 0)
        pad_y = (1 if i < bcY else 0)
        VEGFArray = np.pad(
            VEGFArray,
            pad_width=((pad_x, pad_x), (pad_y, pad_y)),
            constant_values=(i / max(bcX, bcY) * c0)
        )
    return VEGFArray

'''
Calculate and return the diffusion term of the PDE
'''
def diffusion(VEGFArray, D, L, meshScale):
    c = VEGFArray
    width, length = c.shape

    meshL = length
    dx = L / meshL
    dy = meshScale

    # periodic in y (rows, axis=0)
    c_up   = np.roll(c, -1, axis=0)
    c_down = np.roll(c,  1, axis=0)
    lap_y  = (c_up + c_down - 2.0 * c) / (dy**2)

    # Dirichlet in x: ghost columns = 0
    c_ext = np.zeros((width, length + 2))
    c_ext[:, 1:-1] = c

    c_left  = c_ext[:, 0:-2]
    c_right = c_ext[:, 2:  ]
    lap_x   = (c_left + c_right - 2.0 * c_ext[:, 1:-1]) / (dx**2)

    lap = lap_x + lap_y
    return D * lap


'''
Calculate and return the logistic term of the PDE
'''
@jit(nopython = True)
def logistic(VEGFArray, chi):
    #Lattice dimensions
    width = VEGFArray.shape[0]
    length = VEGFArray.shape[1]
    #(1 - c) Array
    logistArray = np.subtract(np.ones((width, length)), VEGFArray)
    #Return logistic term of the PDE
    return np.multiply(chi, np.multiply(VEGFArray, logistArray))

'''
Calculate and return the internalisation term of the PDE (Rescaled)
'''
def summation(VEGFArray, cellList, lmbd, R, searchRad, L, meshScale):
    #Lattice dimensions
    width, length = VEGFArray.shape
    #Mesh dimension
    meshL = length
    #Array to store cell positions
    cellPositions = np.array([[k.y / meshScale, k.x * meshL / L] \
                                for k in cellList])
    #Arrays to store the rows and columns for summation
    rows = np.arange(width)[:, np.newaxis]
    cols = np.arange(length)[np.newaxis, :]
    #Compute exponentials in the summation (with re-scaling)
    rowDiff = rows - cellPositions[:, 0][:, np.newaxis, np.newaxis]
    colDiff = cols - cellPositions[:, 1][:, np.newaxis, np.newaxis]
    summFinArray = np.exp(-(((rowDiff * meshScale)**2) + ((L / meshL)**2 * \
        colDiff**2)) / (3 * R**2))
    #Sum contributions from each cell
    summFinArray = np.sum(summFinArray, axis=0)
    #Multiply the final array by concentration and internalisation rate
    summFinArray = np.multiply(summFinArray, VEGFArray)
    summFinArray = np.multiply(summFinArray, lmbd / (2 * np.pi * R**2))
    #Return uptake term of the PDE
    return summFinArray

'''
Calculate and return the dilution term of the VEGF PDE (Rescaled)
'''
@jit(nopython = True)
def dilution(VEGFArray, L, Ldot):
    #Return dilution term of the PDE
    return np.multiply(Ldot / L, VEGFArray)

'''
Calculate and return an updated VEGF matrix from the VEGF PDE
'''
def updateVEGF(VEGFArray, D, chi, lmbd, R, posList, dt, subStep,
               searchRad, L, W, Ldot, meshScale):

    VEGFArray = VEGFArray + \
        (dt / subStep) * diffusion(VEGFArray, D, L, meshScale) - \
        (dt / subStep) * dilution(VEGFArray, L, Ldot) + \
        (dt / subStep) * logistic(VEGFArray, chi) - \
        (dt / subStep) * summation(VEGFArray, posList,
                                   lmbd, R, searchRad, L, meshScale)

    # Dirichlet in x: enforce zero concentration at boundaries
    VEGFArray[:, 0]  = 0.0   # left boundary (x = 0)
    VEGFArray[:, -1] = 0.0   # right boundary (x = L)

    print(np.sum(VEGFArray[:, 0]))
    return VEGFArray

