"""
'vol_utilities.py' ~ Utility functions for the "volume" codes to implement
                    results from MacKay's' VolumeTori2-1.pdf' [9May24]
                    and new paper DM[25]
"""
# MODULES used  # # # # # # # # # # # # # # # # # # # # # # # # # # #
import numpy as np
from scipy.integrate import odeint

# Global varialbes and constant
at = 1e-14
rt= 1e-12

## Volume codes common functions = = = = = = = = = = = = = = = = = = = = = = 

# Magnetic field #
def B(x,t,m1,n1,e1,R0,w1,w2) : 
    ψ  = x[0]
    vth = x[1]
    ph = x[2]
    a = (1 - ψ/(R0*R0))**2    #
    R = R0 * a / (1 - (1-a)**(0.5) * np.cos(vth))
    Bψ  =   m1*e1*(ψ**(m1/2))*(ψ-4)*np.sin(m1*vth-n1*ph)  *R0/R**2 # /(ψ*R) 
    Bth = (w1 + 2*w2*ψ + e1*(ψ**(m1/2))*(((ψ-4)*m1/(2*ψ)) + 1)*np.cos(m1*vth-n1*ph))  *R0/R**2 # /(ψ*R) 
    Bph = 1  *R0/R**2 # /(ψ*R) 
    return [Bψ, Bth, Bph]


# Magnetic field #
def v(x,t,m1,n1,e1,R0,w1,w2) : 
    ψ  = x[0]
    vth = x[1]
    ph = x[2]
    Bψ  =   m1*e1*(ψ**(m1/2))*(ψ-4)*np.sin(m1*vth-n1*ph)  # *R0/R**2                            #  ∂_ϑ A_ϕ
    Bth = (w1 + 2*w2*ψ + e1*(ψ**(m1/2))*(((ψ-4)*m1/(2*ψ)) + 1)*np.cos(m1*vth-n1*ph)) # *R0/R**2 # -∂_ψ A_ϕ
    Bph = 1  #*R0/R**2  
    return [Bψ, Bth, Bph]


# Level set of \Psi
def energy(yc,zc, arg):
    e1,m1,n1,R0,w1,w2 = arg[0],arg[1],arg[2],arg[3],arg[4],arg[5]
    ψe , the , phie = cart2tor([yc , zc , 0],R0)
    A = w1*ψe + w2*ψe**2 + e1*(ψe**(m1/2))*(ψe-R0**2)*np.cos(m1*the)
    E = m1*A - n1*ψe
    return E


# Search for points of the orbits close to the Poincare section on plane φ = 0
# and return array with the indexes in the orbit
def findcrossings(orb, arg):
    e1,m1,n1,R0,w1,w2 = arg[0],arg[1],arg[2],arg[3],arg[4],arg[5]
    prb = list()
    L=2*np.pi
    for ii in range(len(orb)-1):
        if (orb[ii][2] < L) and (orb[ii+1][2] > L):
            prb.append(ii)
            L += 2*np.pi
    return np.array(prb)

# Refine crossing on the Poincare Section
def refine_crossing(f,a,k,Dt,arg):
    e1,m1,n1,R0,w1,w2 = arg[0],arg[1],arg[2],arg[3],arg[4],arg[5]
    Dt2 = 0
    Dt1 = Dt
    b2 = a
    b = odeint(f, a, [Dt2,Dt1],args=(m1,n1,e1,R0,w1,w2),atol=at, rtol=rt)[-1]
    it=0
    while abs(b[2]-2*np.pi*k)>1e-7 and it<10:
        Dt0 = Dt1 - (b[2]-2*np.pi*k)* (Dt1-Dt2) / (b[2]-b2[2]) # Secant method
        Dt2 = Dt1
        Dt1 = Dt0
        b2 = b
        b = odeint(f, a, [0,Dt1],args=(m1,n1,e1,R0,w1,w2),atol=at, rtol=rt)[-1]
        it += 1
    return b, Dt1

# Find crossing2 on with the u-line and return indexes and class of orbit
def findcrossings4(orb,th0,arg):
    e1,m,n,R0,w1,w2 = arg[0],arg[1],arg[2],arg[3],arg[4],arg[5]
    prb = list()
    cls=0
    # Main u-line
    for ii in range(3,len(orb)-1):
        c1 = m*(orb[ii][1] - th0) - n*orb[ii][2]
        c2 = m*(orb[ii+1][1] - th0) - n*orb[ii+1][2]
        if c1*c2 < 0:
            prb.append(ii+1)
    # u-line lower images 
    if len(prb)==0:
        cls = -1
        for ii in range(3,len(orb)-1):
            c1 = m*(orb[ii][1] - th0 -cls* 2*np.pi/m) - n*orb[ii][2]
            c2 = m*(orb[ii+1][1] - th0 -cls* 2*np.pi/m) - n*orb[ii+1][2]
            if c1*c2 < 0:
                prb.append(ii+1)
                cls -=1
                #print('c1 , c2 = ',c1,c2,cls)
    if len(prb) == 0:
        cls = 1
        for ii in range(3,len(orb)-1):
            c1 = m*(orb[ii][1] - th0 -cls* 2*np.pi/m) - n*orb[ii][2]
            c2 = m*(orb[ii+1][1] - th0 -cls* 2*np.pi/m) - n*orb[ii+1][2]
            if c1*c2 < 0:
                prb.append(ii+1)
                #print('c1 , c2 = ',c1,c2, cls)
                cls +=1
                
    if cls > 0: cls = 1
    if cls < 0: cls = -1
    return np.array(prb) , cls

# Refine crossing on the u-line
def refine_crossing3(f,a,th0,Dt,k,arg):
    e1,m,n,R0,w1,w2 = arg[0],arg[1],arg[2],arg[3],arg[4],arg[5]
    Dt2 = 0
    Dt1 = Dt
    b2 = a
    b = odeint(f, a, [0,Dt1],args=(m,n,e1,R0,w1,w2),atol=at, rtol=rt)[-1]
    #c1 = m*(b[1] - th0) - n*b[2]
    it=0
    while abs(m*(b[1] - th0-np.pi*k) - n*b[2])>1e-8 and it<10:
        Dt0 = Dt1 - (m*(b[1] - th0-np.pi*k) - n*b[2])* (Dt1-Dt2) / (m*b[1] - m*b2[1] - n*b[2] + n*b2[2]) # Secant method
        Dt2 = Dt1
        Dt1 = Dt0
        b2 = b
        if Dt1 < 0: break
        b = odeint(f, a, [0,Dt1],args=(m,n,e1,R0,w1,w2),atol=at, rtol=rt)[-1]
        it += 1
    #print(m*(a[1] - th0-np.pi*k) - n*a[2], m*(b[1] - th0-np.pi*k) - n*b[2])
    return b, Dt1, it

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

def mod2pi2(angle):
    Nlap=0
    while angle >= 2*np.pi:
        angle -= 2*np.pi
    while angle < 0:
        angle += 2*np.pi
    return angle

# Coordinates coversion - - - - - -
# Cartesian (x=0,y,z) to adapted toroidal (psi,vtheta,phi) coordinates
def cart2tor(a,R0):
    y0=a[0]-R0
    z0=a[1]
    x0=a[2]
    ph0=0
    #r0= np.sqrt(y0**2 + z0**2)
    r0= (y0**2 + z0**2)/2
    if y0 != 0:
        th0 = np.arctan(z0/y0)
        if y0 < 0:
            th0 += np.pi
    elif z0> 0:
        th0=np.pi/2
    else:
        th0=-np.pi/2    
    return [r0, th0,ph0]

# Adapted toroidal (psi,vtheta,phi) to Cartesian (x=0,y,z) coordinates
def tor2car(a,R0):
    rho= R0 + a[0]*np.cos(a[1]) 
    xx= rho * np.sin(a[2])
    yy= rho * np.cos(a[2]) 
    zz= a[0] * np.sin(a[1])
    return [yy, zz]


## Screen-printing utilities ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
# Blank space of lenght len(str(j)) ~ ~ 
def blspace(j):
    esp= ' '
    while len(esp) < len(str(j)):
        esp = " " + esp
    return esp

## Epsilon string for file name # #
def epstr2(e):
    if e < 0.1 and e >= 0.01:
        ep = '0'+str(int(e*1000))
    elif e >= 0.001:
        ep = '00'+str(int(e*10000))
    else:
        ep = '000'+str(int(e*100000))
    return ep