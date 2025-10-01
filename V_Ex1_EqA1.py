"""
 'V_Ex1_EqA1.py' - Volume computed by (A1) formular in paper [MM25b] 
  Script to compute the volume enclosed by two flux surfaces from
  the axisymetric example field on the Sec 3.2 of the Burby et al [2023].
  in cilindrical coordinates.
  Volume V computed by formula (A1) derived from (1).

Input: 'ini_par_Ex1.txt'

"""

# MODULES used  # # # # # # # # # # # # # # # # # # # # # # # # # # #
import numpy as np

from scipy.integrate import odeint
# import simplejson
# import json
# import re
# import random
#import stringify
# import matplotlib.pyplot as plt

import time
import datetime
start = time.time()
# # # # # # # # # # #  # # # # # # # # # #  # # # # # # # # # #

# Global varialbes and constant
C = 1



# Mod funtion for angle variables - - - -
def mod2pi(angle):
    Nlap=0
    while angle >= 2*np.pi:
        angle -= 2*np.pi
        Nlap+=1
    while angle < 0:
        angle += 2*np.pi
        Nlap=+1
    return angle, Nlap


def atan3(y,x):
    res = np.arctan2(y,x)
    if x< 0 and y <0:
        res += 2*np.pi
    return res



# Search for points of the orbits close to the Poincare section on plane φ = 0
# and return array with the indexes in the orbit
def findcrossings(orb):
    prb = list()
    L=2*np.pi
    for ii in range(len(orb)-1):
        if (orb[ii][1] < L) and (orb[ii+1][1] > L):
            prb.append(ii)
            L += 2*np.pi
    return np.array(prb)


# Refine crossing on the Poincare Section
def refine_crossing(a,k,Dt):
    Dt2 = 0
    Dt1 = Dt
    b2 = a
    b = odeint(B, a, [Dt2,Dt1],atol=1e-10, rtol=1e-6)[-1]
    it=0
    while abs(b[1]-2*np.pi*k)>1e-6 and it<5:
        Dt0 = Dt1 - 0.5*(b[1]-2*np.pi*k)* (Dt1-Dt2) / (b[1]-b2[1]) # Secant method
        Dt2 = Dt1
        Dt1 = Dt0
        b2 = b
        if Dt2 > Dt1:
            break
        b = odeint(B, a, [0,Dt1],atol=1e-10, rtol=1e-6)[-1]
        it += 1
    return b, Dt1




# Coordinates coversion
def cart2cil(a):
    x0=a[0]
    y0=a[1]
    z0=a[2]
    r0= np.sqrt(x0**2 + y0**2)
    if x0 != 0:
        ph0 = np.arctan(y0/x0)
        if x0 < 0:
            ph0 += np.pi
    elif y0> 0:
        ph0=np.pi/2
    else:
        ph0=-np.pi/2    
    return [r0, ph0, z0]




# Magnetic field # B ==========
def B(x,t) : 
    R  = x[0]
    th = x[1]
    z = x[2]
    r = R-1
    BR  = -z/R
    Bph = C/R**2
    Bz = r/R
    return [BR, Bph, Bz]


# Level set of \psi
def energy(yc,zc):
    xc = 0
    Rc = np.sqrt(yc**2 + xc**2)
    rc = Rc-1
    E = (rc**2 + zc**2)/2
    return E

## Equi-partition in Psi between Psi(p0) and Psi(p1) | p0 < p1
def Psi_partition(p0,p1,N):
    Oini = []
    E2 = energy(p0[0],0)
    Ef = energy(p1[0],0)
    DE = (Ef -E2) /N
    for k in range(N+1):
        E2 = energy(p0[0],0)
        y2 = p0[0]
        y1 = p0[0] + (p1[0] - p0[0])/10
        E1 = energy(y1,0)
        it=0
        E_t = E2 + k*DE
        while abs(E1 - E_t)>1e-7 and it<10:
            y0 = y1 - (E1 - E_t) * (y1-y2) / (E1-E2) # Secant method
            y2 = y1
            y1 = y0
            E2 = E1
            E1 = energy(y1, 0.0)
            it += 1
        Oini.append([y0,0])
    return Oini

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# = = = Main Code = = = = = = = = = = = = = = = = = =
# = = = = = = = = = = = = = = = = 

# - = Parameters reading - = - = - = - = - = - = - = - =  
data = np.loadtxt("ini_par_Ex1.txt", dtype= float, comments="#", unpack=False)

tf = int(data[0]) # Integration time reference
rr1 = data[1]     # Crossing of Psi_2 on \tild{y}-axis

# - = Integration Parameters - = - = - = - = - = - = - =
NPsi = int(data[4])
# - = - = - = - = - = - = - = - = - = - = - = - = - = - =
NPnt= 1  #Number crossings on the poloidal section
# - = - = - = - = - = - = - = - = - = - = - = - = - = - =
Ng = int(data[5])
# - = - = - = - = - = - = - = - = - = - = - = - = - = - =

## Interval of \Psi to integrate # = = = = = = = = = = 
# pnt_list0 = [[1.001,0.0], [1.8,0.0]]
pnt_list0 = [[1.001,0.0], [1+rr1,0.0]]


## Psi partition / / / / / / / / / / / / / / / / / / 
E0 = energy(pnt_list0[0][0],pnt_list0[0][1])
E1 = energy(pnt_list0[1][0],pnt_list0[1][1])
print('E0 = ',E0, '[y,z] =  [', pnt_list0[0][0],pnt_list0[0][1],']')
print('E1 = ',E1, '[y,z] =  [', pnt_list0[1][0],pnt_list0[1][1],']')

# Initial conditions with equipartition on Energy
pnt_list = Psi_partition(pnt_list0[0],pnt_list0[1],NPsi-1)
## Psi partition / / / / / / / / / / / / / / / / / /


pnt0 = pnt_list0[0]
pnt1 = pnt_list0[1]


## - = - = - = - = - = - = - = - = - = - = - = - = - = - =

## Integration time - = - = - = - = - = - = - = - = - = - = 
mult = 10
Num=3000*mult   # Partition on time
tf1 = mult*10  # To vary the time out input
t = np.linspace(0, tf1,Num)


## Volume computation = = = = = = = = = = = = 
Vol=0

for i in range(len(pnt_list)):
    AvTl = 0
    X=[]
    Xcart=[]                
    x0, z0 =  pnt_list[i][0], pnt_list[i][1]
    y0=0.0
    
    ## Contour of level \Psi(x0,y0) ~ ~ ~
    Psi = energy(x0,z0)
#     if i==0: print('sqrt(2 Psi_0)',np.sqrt(2*Elvl))
    
    for j in range(Ng):
#         thetam = 2*np.pi*j/Ng
        Dtheta = 2*np.pi/Ng
        thetam = Dtheta*j
        
        # nu_m
        x1 , z1 =  1+(np.sqrt(2*Psi))*np.cos(thetam), np.sqrt(2*Psi) * np.sin(thetam)
        

        # i_e λ = |sqrt(g)| * e . (n x B)
        leB = (C *np.sin(Dtheta)/x1)

        ## integrate(T λ)  ~ sum ( T  i_e λ ) ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
        
        # Computation of return time T - - - 
        X1 = cart2cil([x1, 0, z1])

        # Integration of the orbit
        Xc = odeint(B, X1, t,atol=1e-10, rtol=1e-6)

        N1 = mod2pi(Xc[-1][1])[1]

        while N1< NPnt: # Minimal number of crossings check
            Xc2 = odeint(B, Xc[-1], t,atol=1e-10, rtol=1e-6)
            N1 = mod2pi(Xc2[-1][1])[1]
        
            Xc= np.vstack([Xc,Xc2])
            print('t was insuficient for 1 poloidal cross')

        
        k=1 
        i2 = 0 # Crossing counter
        for jj in findcrossings(Xc):
            tt=0    
            Yc , dt2 = refine_crossing(Xc[jj],k,tf1/Num)
            while j > len(t):
                jj = jj-len(t)
                tt += tf1
            T = tt+t[jj-1] +dt2

            i2 +=1
            if i2 ==2: 
                AvTl = AvTl + T*leB
                break

    Vol = Vol + AvTl
    


print('AvTL = ', Vol, '|| Vol = ', "{:.6f}".format((2/3)*Vol*(E1- E0)/NPsi), '(',NPsi,')', '[',Ng,']')
# print('Vol = ', "{:.6f}".format((2/3)*Vol*(E1- E0)/NPsi))



finish = time.time()
hours, rem = divmod(finish-start, 3600)
minutes, seconds = divmod(rem, 60)
print('time = {:0>2}:{:0>2}:{:02.0f}'.format(int(hours),int(minutes),seconds))
# = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = #
