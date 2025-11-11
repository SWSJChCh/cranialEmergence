'''
insertCell.py - Samuel Johnson - 31/08/2025
'''

import math
import numpy as np

'''
Create Cell
'''
class Cell:

    def __init__(self, radius, filLength):
        self.radius = radius                         #Cell radius
        self.filLength = filLength                   #Filopodium length

'''
Initial lattice configuration
'''
def initConfiguration(cellList, leadNum, width, radius, filLength):
    #List of leader cells for initial conditions
    initList = []
    #Create leader cells
    for _ in range(leadNum):
        initList.append(Cell(radius, filLength))
    #Evently distributed y coordinates of leader cells
    yList = list(np.linspace(2 * radius, width - 2 * \
                 radius, len(initList)))
    #Update coordinates of leader cells and append cells to main cell list
    for i in range(len(initList)):
        initList[i].x = round(initList[i].radius)
        initList[i].y = yList[i]
        #Append created cells to main cell list
        cellList.append(initList[i])

'''
Insert cell into array
'''
def insertCell(cell, cellList, width):
    #Cells are inserted at LHS of domain
    xIns = round(cell.radius)
    #Cells are inserted into central migratory corridor
    yIns = np.random.uniform(round(cell.radius), \
           round(width - cell.radius))
    #Insertion Boolean
    insert = True
    #Determine if insertion causes overlap with other cell
    for i in cellList:
        if math.sqrt((i.x - xIns)**2 + (i.y - yIns)**2) < 2 * cell.radius:
            insert = False
    #Append cell to list of all cells
    if insert == True:
        cell.x = xIns
        cell.y = yIns
        cellList.append(cell)
    #Delete cell if overlap detected
    else:
        del cell
