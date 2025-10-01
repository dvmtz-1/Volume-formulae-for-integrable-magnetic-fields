"""
 'V_Ex2_EqA1.py' - Volume computation by Eq(A1)
 
  Script to compute the volume enclosed by using Eq(A1) [MM25b]
  for the example field of an axisymmetric field perturbed with one hellical modes
  in modified toroidal coordinates (KMM23)


 Input: "ini_par_Ex2.txt" - Parameter values for the integration

"""
#
# MODULES used  # # # # # # # # # # # # # # # # # # # # # # # # # # #
import numpy as np

from scipy.integrate import odeint
from scipy import interpolate

import matplotlib.pyplot as plt
from V_Ex2_utilities import *   # Utilities

import time
import datetime
start = time.time()
# # # # # # # # # # #  # # # # # # # # # #  # # # # # # # # # #

# Global variables and constants
at = 1e-10
rt= 1e-6
deltaR = 0.001 # Grid resolution 
deltaQ= 0.001  # for contour function


# Volume factor ~ \rho = R**2 / R0 B0#
def rho(f,x0,T1,T2,arg) :
    e1,m1,n1,R0,w1,w2 = arg[0],arg[1],arg[2],arg[3],arg[4],arg[5]
    x = odeint(f, x0, [0,T2],args=(m1,n1,e1,R0,w1,w2),atol=at, rtol=rt)[-1]
    a = (1 - x[0]/(R0*R0))**2
    R = R0 * a / (1 - np.sqrt((1-a)) * np.cos(x[1]-0.5*x[2]+T1))
    ro = R*R/R0
    return ro

## Equi-partition in Psi between Psi(p0) and Psi(p1) | p0 < p1
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


## Symplectic coordinates (\tilde{y},\tilde{z}) to (R,z) coordinates
def sympl2Rz(y,z,R0):
    ψ, Q ,phi = cart2tor([y +R0, z , 0],R0)
    r =  R0*np.sqrt(1-(1-(ψ/R0**2))**2)
    alpha =  1-(1-(ψ/R0**2))**2
    R = R0*(1-alpha)/(1-np.sqrt(alpha)*np.cos(Q))
    z= r* np.sqrt((R0+r)/(R0-r))*np.tan(Q/2)* (1 + (R-R0)/r)
    return R , z



# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# = = = Main Code = = = = = = = = = = = = = = = = = =
# = = = = = = = = = = = = = = = = 
#
# - = Parameters reading - = - = - = - = - = - = - = - =  
data = np.loadtxt("ini_par_Ex2.txt", dtype= float, comments="#", unpack=False)

tf = int(data[0]) # Integration time reference
yy1 = data[1]     # Crossing of Psi_2 on \tild{y}-axis

e1 = data[2]
ep1 = epstr2(e1) # epsilon 1 string name for file
m1 = int(data[3])
n1 = int(data[4])
w1 = data[5]
w2 = data[6]
R0 = data[7]
######################################################
NPsi = int(data[10]) # Size of the partition on Psi
######################################################
Nth = int(data[12]) # Number of points on the contour 
######################################################
arg = [e1,m1,n1,R0,w1,w2]

# - = Integration Parameters - = - = - = - = - = - = - =
# NPsi = 100
# - = - - = - = - = - = - = - = - = - = - = - = - =
######################################################
NPnt=1  #Number of points in the Poincare Section
######################################################
# Nth = 200  # Number of points on the contour 
######################################################

## Interval of \Psi to integrate # = = = = = = = = = = 
if yy1 < 0.334624348933092:
    yy0 = 0.005
    Nc=1 # First crossing over the (phi,vtheta)-planel
    r = np.arange(0.001, 0.4728266, deltaR) # Core
    q = np.arange(0, 2*np.pi, deltaQ)
elif yy1 < 0.6634467573:
    yy0 = 0.526 #0.52542128
    Nc = 2 # Second crossing over (phi,vtheta)-plane
    r = np.arange(0.01, R0, deltaR)  # Mag island
    q = np.arange(-np.pi, np.pi, deltaQ)
else:
    yy0 = 0.6635
    Nc = 1 # First crossing over the (phi,vtheta)-plane
    r = np.arange(0.4728266, R0, deltaR) # Out
    q = np.arange(0, 2*np.pi, deltaQ)
    
 
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
mult = 10
Num= mult*3000   # Partition on time
tf1 = mult*tf  # To vary the time out input
t = np.linspace(0, tf1,Num)

## Level sets ~ Flux surfaces grid
RR, Q = np.meshgrid(r, q)
fi = 0
# Level sets of Psi - - - - - - - - - - - - 
ψ = R0*(R0 - np.sqrt(R0**2-RR**2))     # Out and mag island
H = ψ/4 + ψ**2 + e1*(ψ**(m1/2))*(ψ-R0**2)*np.cos(m1*Q-n1*fi)
Ht=m1*H-n1*ψ
##
fig= plt.figure(figsize=(8,8)) # to use in the contour function
ax= fig.gca()
##

Vol = 0  # Volumen  

for i in range(len(pnt_list)):
    AvTl = 0 # int(Tλ)(Ψ0)
    
    X=[]
    Xcart=[]                
    y0, z0 =  pnt_list[i][0] + R0, pnt_list[i][1]
    x0=0.0
    f0 =1

    X0 = cart2tor([y0 , z0 , x0],R0)
    
#   # BEGIN contour GRID + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
    # Contour discretisation # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    E_lv = energy(y0,z0,arg)

    CS = ax.contour(np.sqrt(2*ψ) *np.cos(Q) , np.sqrt(2*ψ) *np.sin(Q) , Ht, levels=[E_lv]) # Sypml coords

    # Rewriting the contour in parametic form
    X=[]
    n0=[]      # Num of 'elements'
    m0 = len(CS.allsegs[0])  # Num of vertices of element [1] = Num of components

    for ii in range(m0):
        X.append([ii])
        n0.append([ii])
    for ii in range(m0):
        n0[ii] = len(CS.allsegs[0][ii]) # number of points in the component ii
        for jj in range(n0[ii]):
            X[ii].append(CS.allsegs[0][ii][jj])
        X[ii].pop(0)
        

    for ii in range(m0):
        XX=[]
        YY=[]
        for jj in range(n0[ii]):
            XX.append(X[ii][jj][0])
            YY.append(X[ii][jj][1])
            

    # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
    # is needed in order to force the spline fit to pass through all the input points.
    tck, u = interpolate.splprep([XX, YY], s=0, per=True)

    # evaluate the spline fits for Nth+1 evenly spaced distance values
    xi, yi = interpolate.splev(np.linspace(0, 1, Nth+1), tck) #1000,100,50
    
#   # END of GRID + + + + + + + + + + + + + + + + + + + + + + + +
    
    # integrate(T λ)  # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    for j in range(len(yi)-1):
        X0 = cart2tor([xi[j] +R0, yi[j] , 0],R0) # Point to evaluate θ_n
        
        
        ## Semi-displacement ε_n ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
        if j<len(yi)-1: 
            Yf1, Yf2 = xi[j+1] , yi[j+1]  # Next point θ_{n+1}
        else:
            Yf1, Yf2 = xi[1]   , yi[1]
        if j > 0:
            Yb1, Yb2 = xi[j-1] , yi[j-1]  # Previous point θ_{n-1}
        else:
            Yb1, Yb2 = xi[-2]  , yi[-2]


        Em1, Em2 = (Yf1-Yb1)/2 , (Yf2-Yb2)/2   # Semi-displacement | Symplectic coords
        em1 = ( Em1 * np.cos(X0[1]) + Em2 * np.sin(X0[1])) * np.sqrt(2*X0[0]) # Symp to mTor: # Vector (Em1 j + Em2 k)
        em2 = (-Em1 * np.sin(X0[1]) + Em2 * np.cos(X0[1])) / np.sqrt(2*X0[0]) # Symp to mTor: # to (em1 e_ψ + em2 e_ϑ)

        #  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~         
        b = B(X0, 0 ,m1,n1,e1,R0,w1,w2) # Magnetic field
        v0 = v(X0,0 ,m1,n1,e1,R0,w1,w2)  # Scaled magentic field (∂_ϑ A_ϕ, -∂_ψ A_ϕ, 1) 
#         if i==0: print('v',v)

        ## Adapted toroidal coords metric ~ ~ ~
        a = (1 - X0[0]/(R0*R0))**2    #
        r = R0 * (1-a)**(0.5)
        R = R0 * a / (1 - (1-a)**(0.5) * np.cos(X0[1]))

        ## Adapted diagonal metric
        g11 = 1/(2*X0[0])
        g12 = 0
        g22 = 2*X0[0]
        g33 = R0*R0
        
        ## Computation of vector n = ∇Ψ/|∇Ψ|^2
        dPsi1, dPsi2, dPsi3 = m1*v0[1] -n1, -m1*v0[0], n1*v0[0] # dΨ

        gPsi1, gPsi2, gPsi3 = dPsi1/g11, dPsi2/g22, dPsi3/g33   # ∇Ψ

        ngPsi = g11 * (gPsi1**2) + g22 * (gPsi2**2) + g33 * (gPsi3**2) + 2 * g12 * gPsi1 * gPsi2 #|∇Ψ|^2
        
        nn1, nn2, nn3 = gPsi1/ngPsi, gPsi2/ngPsi, gPsi3/ngPsi   # n = ∇Ψ/|∇Ψ|^2
        
        
        ## Triple product  e . (n x B)  =  i_e λ 
        iel = em1 * (nn2*b[2] - nn3*b[1]) - em2* (nn1*b[2] - nn3*b[0]) #  e . (n x B)
        iel = (iel / (b[2])) #* (R0)**2 # i_e λ = |sqrt(g)| * e . (n x B)

        
        ## Computation of return time T
        Xc = odeint(B, X0, t, args=(m1,n1,e1,R0,w1,w2),atol=at, rtol=rt)
        N1 = mod2pi(Xc[-1][1])[1]
        
        while N1< NPnt:  # Minimal number of crossings check
            Xc2 = odeint(B, Xc[-1], t, args=(m1,n1,e1,R0,w1,w2), atol=1e-8, rtol=1e-6)
            N1 = mod2pi(Xc2[-1][1])[1]     
            Xc= np.vstack([Xc,Xc2])

     

        i2 = 0 # Crossing counter
        for k in findcrossings(Xc,arg):
            tt=0    
            Yc , dt2 = refine_crossing(B,Xc[k],Nc,tf1/Num,arg)
            while k >= len(t):
                k = k-len(t)
                tt += tf1
            T = tt+t[k] +dt2 # Return time T to the poloidal section

            i2 +=1
            if i2 == 1: # and i2% C ==0:
                AvTl = AvTl + T *(iel) # 
                break

    Vol = Vol + AvTl

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

print('V = int( int(Tλ) ) dΨ = (|Ψ2 - Ψ1|/NPsi) Av(int(Tλ)) = ', "{:.8f}".format(Vol*np.abs(Psi1- Psi0)/NPsi))
print('Av(int(Tλ)  2VEq2      (NΨ)    [Nth] ')
print("{:.5f}".format(Vol), "{:.6f}".format(Vol*np.abs(Psi1- Psi0)/NPsi),
      ' (',NPsi,')', ' [',Nth,']')




finish = time.time()
hours, rem = divmod(finish-start, 3600)
minutes, seconds = divmod(rem, 60)
print('time = {:0>2}:{:0>2}:{:02.0f}'.format(int(hours),int(minutes),seconds))
# = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = #
