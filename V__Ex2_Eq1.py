"""
 'vol_1r_Eq1.py' - Volume computed by Eq(1)

  Script to compute the volume between two 2 flux surfaces using the
  the formula (1) in [M24]
  for the example field of an axisymmetric field perturbed with one hellical modes
  in modified toroidal coordinates (KMM23)


 Input: "ini_par_Ex2.txt" - Parameter values for the integration

"""

# MODULES used  # # # # # # # # # # # # # # # # # # # # # # # # # # #
import numpy as np

from scipy.integrate import odeint

import matplotlib.pyplot as plt
from V_Ex2_utilities import *   # Utilities

import time
import datetime
start = time.time()
# # # # # # # # # # #  # # # # # # # # # #  # # # # # # # # # #

# Global variabels and constants
at = 1e-10
rt= 1e-6


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# = = = Main Code = = = = = = = = = = = = = = = = = =
# = = = = = = = = = = = = = = = = 
#
# - = Parameters reading - = - = - = - = - = - = - = - =  
data = np.loadtxt("ini_par_Ex2.txt", dtype= float, comments="#", unpack=False)

tf = int(data[0]) # Integration time reference
yy1 = data[1]     # Crossing of Psi_2 on \tild{y}-axis

e1 = data[2]
m1 = int(data[3])
n1 = int(data[4])
w1 = data[5]
w2 = data[6]
R0 = data[7]

Ny = int(data[8]) # 2D grid
Nz = int(data[9]) # dimensions

arg = [e1,m1,n1,R0,w1,w2]

# - = Integration Parameters - = - = - = - = - = - = - =
mult = 2
Num= mult*3000  # Partition on time
tf1 = mult*tf  # To vary the time out input
t = np.linspace(0, tf1,Num)
######################################################
NPnt=2  #Number of points in the Poincare Section
######################################################

## Interval of \Psi to integrate # = = = = = = = = = = 
if yy1 < 0.334624348933092:
    yy0 = 0.005
    psi_bnd = 0.4728266 # Upper bound for the grid
    yl =  0.335
    zl = 0.47
elif yy1 < 0.6634467573:
    yy0 = 0.526 #0.52542128
    psi_bnd = 0
    yl = 0.9
    zl = 0.8
else:
    yy0 = 0.6635
    psi_bnd = 0.4728266 # Lower bound for the grid
    yl = 0.9
    zl = 0.8
    
 
pnt_list0 = [[yy0,0.0], [yy1,0.0]]

# - = Possible initial iterval data for eps = 0.007 - = - = - = - = - = - = - = - = - = 
# pnt_list0 = [[0.05,0.0], [0.334624348933092,0.0]] # Inner tori example
# pnt_list0 = [[0.005,0.0], [0.32,0.0]] # (Sugested - Inner region) 
# pnt_list0 = [[0.52542128,0.0], [0.6634467573,0.0]] # Magnetic island from the O-point
# pnt_list0 = [[0.526,0.0], [0.662,0.0]] # (Sugested - Island region)
# pnt_list0 = [[0.6635,0.0], [0.80,0.0]] # Outer tori example
# pnt_list0 = [[0.6635,0.0], [0.78,0.0]] # (Sugested - Island region)
# = = = = = = = = = = = = = = = = = = = = = = = = = =





pnt0 = pnt_list0[0]
E0 = energy(pnt0[0] + R0,pnt0[1], arg)
pnt1 = pnt_list0[1]
E1 = energy(pnt1[0] + R0,pnt1[1],arg)

sgn = 1  # Outer and Island regions
if E0 > E1: sgn = -1 # Inner region

print('    Ψ          [y,z] ')
print("{:.6f}".format(E0), ' ', pnt0)
print("{:.6f}".format(E1), ' ', pnt1)



## 2D Grid defition - = - = - = - = - = - = - = - = - = - = - = - = - = - =
Xgrid = []

ymax =  yl  # Window size
ymin = -yl
zmax =  zl  
zmin = -zl

yg = np.linspace(ymin, ymax, Ny)
zg = np.linspace(zmin, zmax, Nz)

deltay = (ymax-ymin)/(Ny)
deltaz = (zmax-zmin)/(Nz)
print('Dy =', deltay)
print('Dz =', deltaz)

for i in range(len(yg)):
    for j in range(len(zg)):
        Eg = energy(yg[i] + R0,zg[j],arg)
        psig = np.sqrt(yg[i]*yg[i] + zg[j]*zg[j])
        if sgn*(Eg-E1) < 0 and sgn*(Eg-E0) > 0 and sgn*(psig- psi_bnd ) > 0: 
            Xgrid.append([yg[i] ,zg[j]])

Xgrid = np.array(Xgrid)  
# - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = 

## Volume computation ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
Avt = 0 # Average of T*C
Vol = 0 # Colume computed with Eq (1)

for i1 in range(len(Xgrid)):
    X=[]
    Xcart=[]                

    x0, z0 =  Xgrid[i1] 
    y0=0.0
    
    X0 = cart2tor([x0 + R0,z0 , 0],R0)
    Xc = odeint(B, X0, t, args=(m1,n1,e1,R0,w1,w2),atol=1e-8, rtol=1e-6)

    N1 = mod2pi(Xc[-1][1])[1] # Crossings on the poloidal section

    while N1< NPnt: # Minimal number of crossings check
        Xc2 = odeint(B, Xc[-1], t, args=(m1,n1,e1,R0,w1,w2), atol=1e-8, rtol=1e-6)
        M1 = mod2pi(Xc2[-1][1])[1]
        N1 = mod2pi(Xc2[-1][1])[1]
        
        Xc= np.vstack([Xc,Xc2])
        X.append(X0)

    
    # Computation of return time T - - - 
    k=1
    i2 = 0
    for j in findcrossings(Xc,arg):
        tt=0    
        Yc , dt2 = refine_crossing(B,Xc[j],k,tf1/Num,arg)
        while j >= len(t):
            j = j-len(t)
            tt += tf1

        T = tt+t[j] +dt2

        i2 +=1
        if i2 == 1: 
            Avt = Avt + T/len(Xgrid)
            Vol = Vol + ( T)*deltay*deltaz
            break
        k += 1

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

print('Avt , Vol  [Ny]=  ', "{:.6f}".format(Avt), "{:.6f}".format(Vol), '[',Ny,']')


# - = - = - = - = - = - = - = - = - =  - = - = - = - = - = - = - = - = - = 


finish = time.time()
hours, rem = divmod(finish-start, 3600)
minutes, seconds = divmod(rem, 60)
print('time = {:0>2}:{:0>2}:{:02.0f}'.format(int(hours),int(minutes),seconds))
# = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = #
