"""
 'V_Ex2_Thm3p.py' - Volume computation by Thm 3'
 
  Script to compute the volume enclosed by using the new Theorem 3'[MM25b]
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

# Global variables and constants
at = 1e-10
rt= 1e-6


# Volume factor ~ \rho = R**2 / R0 B0#
def rho(f,x0,T1,T2,arg) :
    e1,m1,n1,R0,w1,w2 = arg[0],arg[1],arg[2],arg[3],arg[4],arg[5]
    x = odeint(f, x0, [0,T2],args=(m1,n1,e1,R0,w1,w2),atol=at, rtol=rt)[-1]
    a = (1 - x[0]/(R0*R0))**2
#     R = R0 * a / (1 - np.sqrt((1-a)) * np.cos(x[1]+T1))
    R = R0 * a / (1 - np.sqrt((1-a)) * np.cos(x[1]-0.5*x[2]+T1))
    ro = R*R/R0
    return ro


def Psi_partition(p0,p1,N,arg):
    e1,m1,n1,R0,w1,w2 = arg[0],arg[1],arg[2],arg[3],arg[4],arg[5]
    Oini = []
    E2 = energy(p0[0] + R0,0,arg)
    Ef = energy(p1[0] + R0,0,arg)
    DE = (Ef -E2) /N
    for k in range(N+1):
        E2 = energy(p0[0] + R0,0,arg)
        y2 = p0[0]
        y1 = p0[0] + (p1[0] - p0[0])/5
        E1 = energy(y1 + R0,0,arg)
        it=0
        E_t = E2 + k*DE
        while abs(E1 - E_t)>1e-9 and it<40:
            y0 = y1 - 0.3*(E1 - E_t) * (y1-y2) / (E1-E2) # Secant method
            y2 = y1
            y1 = y0
            E2 = E1
            E1 = energy(y1 + R0, 0.0,arg)
            it += 1
        Oini.append([y0,0])
    return Oini


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# = = = Main Code = = = = = = = = = = = = = = = = = =
# = = = = = = = = = = = = = = = = 
#
# - = Parameters reading - = - = - = - = - = - = - = - =  
data = np.loadtxt("ini_par_Ex2.txt", dtype= float, comments="#", unpack=False)

tf = int(data[0]) # Integration time reference
yy1 = data[1]     # Crossing of Psi_2 on \tild{y}-axis
m1 = int(data[3])
n1 = int(data[4])
w1 = data[5]
w2 = data[6]
R0 = data[7]
e1 = data[2]
ep1 = epstr2(e1) # epsilon 1 string name for file
arg = [e1,m1,n1,R0,w1,w2]
######################################################
NPsi = int(data[10]) # Size of the partition on Psi
# ######################################################
Ng = int(data[11]) # q - Harmonic mean  (Thm3')
######################################################

# - = Integration Parameters - = - = - = - = - = - = - =
# NPsi = 100
# - = - - = - = - = - = - = - = - = - = - = - = - =
######################################################
# NPnt=10  #Number of points in the Poincare Section
######################################################
Nc = 1  # Crossings for T average (Thm4)
######################################################
# Ng = 6  # q - Harmonic mean  (Thm3')
######################################################


## Interval of \Psi to integrate # = = = = = = = = = = 
if yy1 < 0.334624348933092:
    yy0 = 0.005
    Nc=1 # First crossing over the (phi,vtheta)-planel
elif yy1 < 0.6634467573:
    yy0 = 0.526 #0.52542128
    Nc = 2 # Second crossing over (phi,vtheta)-plane 
else:
    yy0 = 0.6635
    Nc = 1 # First crossing over the (phi,vtheta)-plane
    
 
pnt_list0 = [[yy0,0.0], [yy1,0.0]]

# - = Possible initial iterval data for eps = 0.007 - = - = - = - = - = - = - = - = - = 
# pnt_list0 = [[0.05,0.0], [0.334624348933092,0.0]] # Inner tori example
# pnt_list0 = [[0.005,0.0], [0.32,0.0]] # (Sugested - Inner region) 
# pnt_list0 = [[0.52542128,0.0], [0.6634467573,0.0]] # Magnetic island from the O-point
# pnt_list0 = [[0.526,0.0], [0.662,0.0]] # (Sugested - Island region)
# pnt_list0 = [[0.6635,0.0], [0.80,0.0]] # Outer tori example
# pnt_list0 = [[0.6635,0.0], [0.78,0.0]] # (Sugested - Island region)
# = = = = = = = = = = = = = = = = = = = = = = = = = =


## Psi partition / / / / / / / / / / / / / / / / / / 
pnt0 = pnt_list0[0]
Psi0 = energy(pnt0[0] + R0,pnt0[1],arg)
pnt1 = pnt_list0[1]
Psi1 = energy(pnt1[0] + R0,pnt1[1],arg)

print('     Ψ        [y,z]  ')
print("{:.6f}".format(Psi0), pnt0)
print("{:.6f}".format(Psi1), pnt1)

pnt_list = Psi_partition(pnt0,pnt1,NPsi-1,arg)
# / / / / / / / / / / / / / / / / / / / / / / / / / /


## Integration time and resolution - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = 
mult = 10        # Resolution factor
Num= mult*3000   # Partition on time
tf1 = mult*tf    # Integration time
t = np.linspace(0, tf1,Num) # Time partition



AvDρinv = 0

for i in range(len(pnt_list)):
    X=[]
    Xcart=[]                
    y0, z0 =  pnt_list[i][0] + R0, pnt_list[i][1]
    x0=0.0
    f0 =1

    X0 = cart2tor([y0 , z0 , x0],R0)
    
    # Integration of first point
    Xc = odeint(v, X0, t, args=(m1,n1,e1,R0,w1,w2),atol=at, rtol=rt)

    ind_int, cls = findcrossings4(Xc,X0[1],arg) # cls=1 (In) | -1 (Out) | 0 (Island)
    Ncross = len(ind_int)
    
    
    while Ncross< Nc:   # Minimal number of crossings check
        Xc2 = odeint(v, Xc[-1], t, args=(m1,n1,e1,R0,w1,w2),atol=at, rtol=rt)
        Xc= np.vstack([Xc,Xc2])
        
        ind_int, cls = findcrossings4(Xc,X0[1],arg)

        Ncross = len(ind_int)
    

    X.append(X0)   # Add initial conditions to the list of crossings
    Xcart.append([y0, z0]) # Same in [y,z]-coords
    X=np.array(X)
    Xcart=np.array(Xcart)
    k=1
    
    if len(findcrossings(Xc,arg)) ==0: print('No crossing found at ', y0,z0)
    for j in findcrossings(Xc,arg):
        tt=0
        
        Yc , dt2 = refine_crossing(v,Xc[j],k,tf1/Num,arg)
        while j > len(t):
            j = j-len(t)
            tt += tf1
        
        X= np.vstack([X,Yc])    # X ~collection of crossings in (psi,varthera, phi) coords
        Yc[0]= np.sqrt(2*Yc[0]) # Yc[0] is now: psi --> r = sqrt(2 psi)
        Xcart= np.vstack([Xcart,tor2car(Yc,R0)]) # X ~collection of crossings in [y,z] coords
        k += 1
        


    ## Computation of return time T to the u-line - - - 
    cls1 = 0
    tc_av = 0

        
    ind_int, cls = findcrossings4(Xc,X0[1],arg)
    kk=1
    tc_old = 0

    for j in ind_int:
        if cls !=0 or kk%2==0:
            tt2 = 0
            cls1 = cls*kk
            b , dt3 , it = refine_crossing3(v,Xc[j],X0[1],tf1/Num,cls1,arg)
            j0 = j
            while j0 > len(t):
                j0 = j0-len(t)
                tt2 += tf1

            T = tt2 + t[j0] +dt3
        if kk >= Nc: break #
        kk += 1

    
    glat=[] # Harmonic lattice points
    Avp = 0 # average of 1/\rho
    
    ## Harmonic mean points computation
    for j in range(Ng):
        for k in range(Ng):
            glat.append([j*2*np.pi/Ng, k*T/Ng])

    for j in range(len(glat)):  
        p = rho(v,X0,glat[j][0],glat[j][1], arg)
#         Avp += 1/(p*len(glat))
        Avp += p/(len(glat))

    Delta = 2*np.pi * T
    AvDρinv += Avp*Delta / NPsi
##################################################
        



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

print('V = int( Δ/<ρ> ) dΨ = |Ψ2 - Ψ1| Av(Δ/Avρ) = ', "{:.8f}".format(AvDρinv*np.abs(Psi1- Psi0)))
print(' <Δ/<ρ>>   Vthm3p     (NΨ)    [Nc]  {mult}  {{Ng}}')
print("{:.5f}".format(AvDρinv), "{:.6f}".format(AvDρinv*np.abs(Psi1- Psi0)),
      ' (',NPsi,')', ' [',Nc,']','{',mult,'} ',' {{',Ng,'}}')




finish = time.time()
hours, rem = divmod(finish-start, 3600)
minutes, seconds = divmod(rem, 60)
print('time = {:0>2}:{:0>2}:{:02.0f}'.format(int(hours),int(minutes),seconds))
# = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = #
