'''
growthFunction.py - Samuel Johnson - 31/08/2025
'''

#Growth data source
#McLennan, R., Dyson, L., Prather, K.W., Morrison, J.A., Baker, R.E., Maini,
#P.K. and Kulesa, P.M., 2012. Multiscale mechanisms of cell migration during
#development: theory and experiment. Development, 139(16), pp.2935-2944.

import math
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#McLellan 2012 Domain Growth Measurements (t / h)
xData = [0.06082725060827251, 1.1557177615571776, 2.4330900243309004, \
3.588807785888078, 4.805352798053528, 5.96107055961070, 7.177615571776156, \
8.454987834549879, 9.549878345498783, 10.705596107055962, 12.10462287104623, \
13.199513381995134, 14.476885644768856, 15.632603406326034, 16.72749391727494, \
18.065693430656935, 19.16058394160584, 20.255474452554743, 21.654501216545015, \
22.99270072992701]

#McLellan 2012 Domain Growth Measurements (L(t) / μm)
yData = [297.98270893371756, 283.57348703170027, 321.0374639769452, \
323.9193083573487, 326.80115273775215, 372.9106628242075, 410.3746397694524, \
421.9020172910663, 430.54755043227664, 450.72046109510086, 505.47550432276654, \
646.685878962536, 695.6772334293948, 759.0778097982709, 862.8242074927954, \
891.64265129683, 931.9884726224784, 995.3890489913545, 1078.9625360230548, \
1058.7896253602305]

'''
Sigmoid function for domain length in time
'''
#Sigmoid function for data fitting
def sigmoid(t, L, t0, k, b):
    #Calculate domain length at specified time
    length = L / (1 + np.exp(-k * (t - t0))) + b
    #Return length at time t
    return length

'''
Generate list of domain lengths for each simulation timestep
'''
def domainLengths(finTime):
    #Lists to store domain length data
    lengthList = []
    #Find length for every minute before final time
    for i in range(finTime * 60):
        #L(t) calculated according to logistic function
        lengthList.append(round(sigmoid(i / 60, param[0], param[1], param[2], \
                          param[3])))
    #List of domain lengths for each minute before final time
    return lengthList

#Initial estimates for sigmoid parameters
p0 = [max(yData), np.median(xData), 1, min(yData)]

#Fit domain length data to sigmoid function
param, pcov = curve_fit(sigmoid, xData, yData, p0, method='dogbox')
