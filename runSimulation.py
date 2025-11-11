'''
runSimulation.py - Samuel Johnson - 31/08/2025
'''

import imageio.v2 as imageio
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patches as patches


import math
import os
import copy
import sys
import datetime
import imageio.v2 as imageio
import numpy as np
from scipy.ndimage import zoom
from numba import jit
from VEGF import *
from insertCell import *
from moveCell import *
from growthFunction import *

#Command line physical parameters
kSpring = float(sys.argv[1])
f0 = float(sys.argv[2])

#Command line VEGF parameter
lmda = float(sys.argv[3])

#New resizing function
def resize_array(arr, new_shape):
    zoom_factors = (new_shape[0] / arr.shape[0],
                    new_shape[1] / arr.shape[1])
    return zoom(arr, zoom_factors, order=1)  

#Memory efficient deep copy (no bonds)
def deepcopy_no_bonds(c):
    dup = copy.copy(c)                
    if hasattr(dup, "bonds"):
        dup.bonds = {}               
    return dup

#List for simulation data output
results = []

#Animation Boolean
animate = True
#Timestep for animation
animStep = 10

#Create output directory
if animate:
    date = datetime.datetime.now().strftime('%H-%M-%S')
    os.makedirs('Emergence' + date)

#Scale factor for mesh spacing (for computational speed-up)
meshScale = 1
#Simulation runtime
finTime = 24
#First cell insertion time
firstInsert = 6
#Final cell insertion time
lastInsert = 8
#Timestep (h)
dt = 1 / 60
#List of domain lengths for each timestep (μm)
lengthList = domainLengths(int(finTime + 1))
#Domain length (μm)
Len = int(lengthList[-1])
#Length of PDE mesh (for solver on unit-length mesh)
meshLen = int(lengthList[-1] / meshScale)
#Domain width (μm)
Wit = 120 
#Width of PDE mesh (for solver on unit-length mesh)
meshWit = int(Wit / meshScale)
#Boundary condition smoothing parameter
bcParam = 10
#Number of leader cells
leadNum = 5
#Proportion of domain for which zero-flux boundary conditions are expressed
spanLen = 0
#Timesteps per attempted cell insertion
insertStep = 1
#Repeats for data averaging
repeats = 1

#VEGF parameters
D = 0.001                      #Diffusion constant
subStep = 1                    #Solver steps per timestep
dx = 1                         #Spacestep (x / μm)
dy = 1                         #Spacestep (y / μm)
c0 = 1.0                       #Initial concentration of reactant
xi = 0.25                      #Sensing parameter

#Cell parameters
cellRad = 7.5                          #Cell radius (μm)
searchRad = 5 * cellRad                #Box size for internalisation (μm)
lenFilo = 3.5 * cellRad                #Filopodium length (μm)
lenFiloMax = 6 * cellRad               #Maximum detection length (μm)
filoNum = 5                            #Number of filopodia extended by cells
chi = 10**-4                           #Logistic production parameter

#Repeat simulations
for r in range(repeats):
    #Initialise time
    t = 0
    #Initialise VEGF Mesh (for solver)
    VEGFMesh = createVEGFArray(meshWit, meshLen, c0, bcParam)
    #Initialise VEGF Array (for cells)
    VEGFArray = resize_array(VEGFMesh, (Wit, Len))
    #List to store cell objects
    cellList = []
    #Data lists for visualisation
    cellMast = []
    filMast = []
    VEGFMast = []
    #Images for movie writer
    ims = []
    #Plot objects
    fig, ax = plt.subplots(1)
    #Counting variable
    counter = 0
    #Boolean for leader insertion
    leaderInsert = False
    #Leader ablation Boolean
    ablated = True
    #Follower reposition Boolean
    repositioned = True
    #Run main simulation loop
    while (t < finTime):
        #Increase counting variable
        counter += 1
        #Update time
        t += dt
        #Actual domain length (μm)
        Len = int(lengthList[counter])
        #Time derivative of domain length
        lenDot = (lengthList[counter] - lengthList[counter - 1]) / dt
        #Initial cells are leaders
        if not leaderInsert:
            #Create initial leader cells (evenly distributed line at LHS)
            initConfiguration(cellList, leadNum, Wit, cellRad, lenFilo)
            leaderInsert = True
        if t > firstInsert:
            #One-time deletion at 12h
            if not ablated and t >= 12.0:
                #Sort descending by x
                cellList.sort(key=lambda c: c.x, reverse=True)
                #Delete up to 15 cells
                del cellList[:15]
                ablated = True 
            #One-time reposition at 12h
            if not repositioned and t >= 12.0:
                repositionTrailingCells(cellList, 15, cellRad, domainLen=lengthList[counter], 
                                        gap=None, clearBonds=True)
                repositioned = True
            #List to track cell data at time t
            cellCopyList = []
            #Insert cells at constant time intervals
            if counter % insertStep == 0:
                #Insert cell
                cell = Cell(cellRad, lenFilo)
                insertCell(cell, cellList, Wit)
            #Update chemicals according to PDE
            for _ in range(subStep):
                #Update chemoattractant
                VEGFMesh = updateVEGF(VEGFMesh, D, chi, lmda, cellRad, \
                                      cellList, dt, subStep, searchRad, \
                                      Len, Wit, lenDot, meshScale)
            #Update VEGF Array
            VEGFArray = resize_array(VEGFMesh, (Wit, Len))

            filList = moveCells(VEGFArray, cellList, \
                            filoNum, lenFilo, lenFiloMax, xi, c0, cellRad, \
                            lengthList[counter - 1], lengthList[counter], kSpring, 
                            f0)
            
            #List of cell position and VEGF Array
            if counter % animStep == 0 and animate:
                for i in cellList:
                    cellCopyList.append(deepcopy_no_bonds(i))
                cellMast.append(cellCopyList)
                VEGFMast.append(VEGFArray.copy())
                filMast.append(filList.copy())
                
    #Output data
    for k, c in enumerate(cellList):
        theta = getattr(c, 'chainAngle', float('nan'))
        mag   = float(getattr(c, 'chemoCue', 0.0))
        results.append(
            f"{r}\t{k}\t{c.x:.3f}\t{c.y:.3f}\t{mag:.4f}\t{theta:.4f}\t{math.degrees(theta):.2f}"
        )

#Increase space between subplots
fig.tight_layout(pad=2.5)

#Produce .MP4 file of simulation
if animate:
    #Images
    ims = []
    for i in range(len(cellMast)):
        #Clear current axes
        ax.clear()
        #Set axes limits
        ax.set_xlim(0, Len)

        # Normalize polarisation magnitude to [0,1]
        norm = colors.Normalize(vmin=0, vmax=1)
        cmap = cm.get_cmap('OrRd')

        for j in cellMast[i]:
            # Define polarisation magnitude: e.g. j.cue or other scalar
            mag = getattr(j, 'chemoCue', 0.0)
            col = cmap(norm(mag))
            ax.add_patch(patches.Circle((j.x - 0.5, j.y - 0.5),
                                        cellRad,
                                        linewidth=0,
                                        facecolor=col,
                                edgecolor='k'))
                                
        # Polarisation arrows (one per cell)
        xs, ys, us, vs = [], [], [], []
        for j in cellMast[i]:
            ang = getattr(j, 'chainAngle', None)
            if ang is None:
                continue
            xs.append(j.x - 0.5)
            ys.append(j.y - 0.5)
            L = 1.5 * cellRad
            us.append(L * math.cos(ang))
            vs.append(L * math.sin(ang))

        mags = [getattr(j, 'chemoStrength', 1.0) for j in cellMast[i]]
        Q = ax.quiver(xs, ys, us, vs, mags,
                    angles='xy', scale_units='xy', scale=1,
                    pivot='mid', color='k', width=0.003, zorder=10)

        #Show VEGF Profile
        im0 = ax.imshow(VEGFMast[i], interpolation = 'none', vmin = 0, \
                           vmax = np.amax(VEGFMast[i]))

        #Title
        ax.set_title('Single Phenotype Simulation' \
        ' (VEGF) [24h]')

        #Colorbar
        cb0 = fig.colorbar(im0, shrink=0.75, aspect=3, ax=ax)

        #Save visualisation to folder
        plt.savefig('Emergence{}/image{}.png'.format(date, i), dpi=300)
        #Remove colorbar for visualisation
        cb0.remove()

    #Produce video from folder
    with imageio.get_writer('Emergence{}.mp4'.\
                            format(date), mode='I', fps=10) as writer:
        for i in range(len(cellMast)):
            filename = 'Emergence{}/image{}.png'.format(date, i)
            image = imageio.imread(filename)
            writer.append_data(image)
            os.remove(filename)

#Save simulation data
if not animate:
    with open("final_positions_kSpring={}_f0={}.txt".format(kSpring, f0), "a") as f:
        for line in results:
            f.write(line + "\n")
            
#Delete folder used to make MP4
else:
    os.rmdir('Emergence' + date)
