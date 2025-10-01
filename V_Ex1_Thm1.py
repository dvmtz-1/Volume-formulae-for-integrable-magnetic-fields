"""
 'V_Ex1_Thm1.py' - Volume computed by Thm1 formular in paper [M24, MM25] 
  Script to compute the volume enclosed by two flux surfaces from
  the axisymetric example field on the Sec 3.2 of the Burby et al [2023].
  in cilindrical coordinates.

Input: 'ini_par_Ex1.txt'

"""
#
# MODULES used  # # # # # # # # # # # # # # # # # # # # # # # # # # #
import numpy as np

from scipy.integrate import odeint
import simplejson
import json
import re
import random
#import stringify
import matplotlib.pyplot as plt

import time
import datetime
start = time.time()
# # # # # # # # # # #  # # # # # # # # # #  # # # # # # # # # #

# Global varialbes and constant
C = 2




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

# For printing ~ ~ 3
def blspace(j):
    esp= ' '
    while len(esp) < len(str(j)):
        esp = " " + esp
    return esp

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

def findcrossings2(orb,th0,R0):
    jj=0
    for ii in range(2,len(orb)-1):
        c1 = atan3(orb[ii][2], orb[ii][0]-1) -th0 # +brch0
        c2 = atan3(orb[ii+1][2], orb[ii+1][0]-1) -th0 #+brch1
        if c1*c2 < 0 and abs(c1-c2)< 0.5 and (orb[ii][0]-1)*(R0-1) > 0:
            print('[',ii,'] c1, c2, c1*c2 = ', c1,c2,c1*c2)
            jj =ii
            break
    return jj

def findcrossings3(orb,th0,R0):
    prb = list()
    for ii in range(2,len(orb)-1):
        c1 = atan3(orb[ii][2], orb[ii][0]-1) -th0 
        c2 = atan3(orb[ii+1][2], orb[ii+1][0]-1) -th0 
        if c1*c2 < 0 and abs(c1-c2)< 0.5 and (orb[ii][0]-1)*(R0-1) > 0:
            prb.append(ii)
    return np.array(prb)



# Refine crossing on the Poincare Section
def refine_crossing(a,k,Dt):
    Dt2 = 0
    Dt1 = Dt
    b2 = a
    b = odeint(B, a, [Dt2,Dt1],atol=1e-8, rtol=1e-6)[-1]
    it=0
    while abs(b[1]-2*np.pi*k)>1e-7 and it<10:
        Dt0 = Dt1 - (b[1]-2*np.pi*k)* (Dt1-Dt2) / (b[1]-b2[1]) # Secant method
        Dt2 = Dt1
        Dt1 = Dt0
        b2 = b
        b = odeint(B, a, [0,Dt1],atol=1e-5, rtol=1e-6)[-1]
        it += 1
    return b, Dt1


# Refine crossing on the Poincare Section
def refine_crossing3(a,th0,Dt,R0):
    Dt2 = 0
    Dt1 = Dt
    b2 = a
    b = odeint(B, a, [0,Dt1],atol=1e-8, rtol=1e-6)[-1]
    #c1 = m*(b[1] - th0) - n*b[2]
    it=0
    while abs(np.arctan2(b[2], b[0]-1) -th0)>1e-7 and it<10 and (b[0]-1)*(R0-1) > 0:
        Dt0 = Dt1 - (np.arctan2(b[2], b[0]-1) -th0)* (Dt1-Dt2) / (np.arctan2(b[2], b[0]-1) - np.arctan2(b2[2], b2[0]-1)) # Secant method
        Dt2 = Dt1
        Dt1 = Dt0
        b2 = b
        b = odeint(B, a, [0,Dt1],atol=1e-5, rtol=1e-6)[-1]
        it += 1
    #print(m*(a[1] - th0-np.pi*k) - n*a[2], m*(b[1] - th0-np.pi*k) - n*b[2])
    return b, Dt1




# Coordinates coversion
def cart2cil(a):
    y0=a[1]
    z0=a[2]
    x0=a[0]
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


def cil2car(a):
    xx= a[0]*np.cos(a[1]) 
    yy= a[0] * np.sin(a[1])
    zz= a[2] 
    return [xx, zz]


# Magnetic field # B ===========
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
def energy(xc,zc):
    yc = 0
    Rc = np.sqrt(xc**2 + yc**2)
    rc = Rc-1
    E = (rc**2 + zc**2)/2
    return E


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
######################################################
NPnt=2  #Number of points in the Poincare Section
######################################################


## Interval of \Psi to integrate # = = = = = = = = = = 
pnt_list0 = [[1.001,0.0], [1+rr1,0.0]]


## Psi partition / / / / / / / / / / / / / / / / / / 
E0 = energy(pnt_list0[0][0],pnt_list0[0][1])
E1 = energy(pnt_list0[1][0],pnt_list0[1][1])
print('E0 = ',E0, '[y,z] =  [', pnt_list0[0][0],pnt_list0[0][1],']')
print('E1 = ',E1, '[y,z] =  [', pnt_list0[1][0],pnt_list0[1][1],']')

# Initial conditions with equipartition on Energy
pnt_list = Psi_partition(pnt_list0[0],pnt_list0[1],NPsi)
## Psi partition / / / / / / / / / / / / / / / / / / 

pnt0 = pnt_list0[0]
pnt1 = pnt_list0[1]



## Integration time - = - = - = - = - = - = - = - = - = - = 
mult = 2
Num=3000*mult   # Partition on time
tf1 = mult*10  # To vary the time out input
t = np.linspace(0, tf1,Num)



AvrT = 0

for i in range(len(pnt_list)):
    X=[]
    Xcart=[]                
    x0, z0 =  pnt_list[i][0], pnt_list[i][1]
    y0=0.0
    f0 =1

    X0 = cart2cil([x0,y0 , z0])

    # Integration of the orbit
    Xc = odeint(B, X0, t,atol=1e-8, rtol=1e-6)


    # POINCARE SECTION \\\\\\\\\\\\\\\\\\\\\\\\\ 
    N1 = mod2pi(Xc[-1][1])[1]


    while N1< NPnt: # Minimal number of crossings check
        Xc2 = odeint(B, Xc[-1], t,atol=1e-8, rtol=1e-6)
        N1 = mod2pi(Xc2[-1][1])[1]
        
        Xc= np.vstack([Xc,Xc2])
        

    X.append(X0)
    Xcart.append([y0, z0])
    X=np.array(X)
    Xcart=np.array(Xcart)
    k=1

    # Return time to the poloidal section
    t0 = 0
    for j in findcrossings(Xc):
        tt=0    
        Yc , dt2 = refine_crossing(Xc[j],k,tf1/Num)
        while j >= len(t):
            j = j-len(t)
            tt += tf1
        t2 = tt+t[j] +dt2
        t0 = t2
        
        X= np.vstack([X,Yc])  # X is in (psi, phi, z) coords
        Xcart= np.vstack([Xcart,cil2car(Yc)])
        k += 1
        
    esp=blspace(1)

    X3 =Xcart
    X4 = X3.transpose()
    # POINCARE SECTION ///////////////////////////// END //////

    ## Computation of the return time to the u-line
    th0 = atan3(X0[2], X0[0]-1)
    ind_int = findcrossings3(Xc,th0,X0[0])

    kk=0
    T0 = 0
    for j in ind_int:
        tt2 = 0
        Kc , dt3 = refine_crossing3(Xc[j],th0,tf1/Num,X0[0])
        j0 = j
        while j0 > len(t):
            j0 = j0-len(t)
            tt2 += tf1

        T0 = tt2 + t[j0] +dt3

        kk += 1
        if kk > 4:
            break
        if kk == 1:
            AvrT = AvrT +  (T0) / (NPsi+1)
            break

print('AvrT = ', "{:.5f}".format(AvrT))
print('V = int( τ T dΨ = 2π (E2 - E1) AvrT = ', "{:.6f}".format(AvrT*(E1- E0)*2*np.pi) ,' =' ,
      "{:.6f}".format(AvrT*(E1- E0)/(2*np.pi)),'4π^2')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = Figure 1 =
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# Integrable case reference level sets for first mode - = - = - = - = - = - = - = - = - = 
deltaX = 0.005
deltaZ= 0.005
# - = - = - = - = - = - = - = - = - = 
xp = np.arange(0, 2, deltaX)
zp = np.arange(-1, 1, deltaZ)
Xp, Zp = np.meshgrid(xp, zp)
# Hamiltonian - - - - - - - - - - - - 
H = ((np.abs(Xp)-1)**2 + Zp**2)/2

ilvl=[0.01,0.05,0.1,0.2,0.4,0.6]      #


#fig= plt.figure(figsize=(8,8))   # FULL poloidal plane
fig= plt.figure(figsize=(8,8)) # UPPER HALF poloidal plane
ax= fig.gca()
ax.set_xlim(0, 2)    # Upper half poloidal plane
ax.set_ylim(-1, 1)    # (as in cKAM conefield plots)
CS = ax.contour(Xp, Zp, H,levels = ilvl,colors=[(0.9,0.9,1)], alpha=0.9 , linestyles=['solid'])
# ax.clabel(CS, inline=1, fontsize=10)
CS3 = ax.contour(Xp, Zp, H,levels = [energy(pnt0[0],pnt0[1])], colors=['cyan'], alpha=0.9, linestyles=['solid'])
CS4 = ax.contour(Xp, Zp, H,levels = [E1], colors=['m'], alpha=0.9, linestyles=['solid'])


for i in range(len(pnt_list)):
    ax.contour(Xp, Zp, H,levels = [energy(pnt_list[i][0], pnt_list[i][1])], colors= [[0.8,0.5,0]],
                     alpha=0.9, linestyles=['solid'])

# for j in range(NPnt+1):
#     sc = ax.scatter(X3[j][:][0], X3[j][:][1], c='r',vmin=0, vmax=1, s=10, cmap = 'jet_r') # 'jet_r'
    
# p0 = ax.plot(pnt0[0],pnt0[1], 'or-', markersize = 3)
# p1 = ax.plot(pnt1[0],pnt1[1], 'ob-', markersize = 3)

# plt.savefig('volume_Thm1_50_4.png', dpi=600)
plt.show()
# - = - = - = - = - = - = - = - = - =  - = - = - = - = - = - = - = - = - = 


finish = time.time()
hours, rem = divmod(finish-start, 3600)
minutes, seconds = divmod(rem, 60)
print('time = {:0>2}:{:0>2}:{:02.0f}'.format(int(hours),int(minutes),seconds))
# = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = #
