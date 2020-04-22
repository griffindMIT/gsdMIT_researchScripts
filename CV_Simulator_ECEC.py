# -*- coding: utf-8 -*-

#Import packages
import numpy as np 
from scipy import optimize as spopt
import math
import matplotlib.pyplot as MP
import CV_Simulator_Fcns_ECEC as fx
import time

#Start timing
tic = time.time()

#Close all open figures
MP.close('all')

#Physical Constants
F = 96485.332 #C/mol - Faraday constant
T = 298.15 #K - Temperature - tech. an experiment constant but roll w/ it
R = 8.314 #J/mol-K - Gas const.

#Mechanism 
#X = Y + e (kXY)
#Y -> W + U (kY)
#W = U + e (kWU)
#U + (H2O) -> Z (kcU) 

#Reaction Constants
D = 10 ** (-9) #m^2/s - Diffusion coeff (mult by 10^4 to get cm^2/s)
ksXY = 10 ** (-6) #m/s - Surface rate constant (max value ~ 10^2 m/s or 10^4 cm/s)
kcY = 0 #1/s - Homog. rate constant, first order
ksWU = 0.00 #m/s - Surface rate constant
kcU = 0.00 #1/(sM) - Homog. rate constant, second order
#Set toggles to 1 to enforce ~equilibrium~ (infinitely fast ks for XY, WU)
EqTog = np.array([1,0])
ConcertTog = 0

E0XY = 2.5 #V - True reversible potl for X=Y
BsXY = 0.5 #Symmetry factor for X = Y
nXY = 1 # # of electrons transferred for X = Y
VTXY = BsXY*nXY*F/(R*T)
E0WU = 2.0 #V - True reversible potl for W=U
BsWU = 0.5 #Symmetry factor for W = U
nWU = 1 # # of electrons transferred for W = U
VTWU = BsWU*nWU*F/(R*T)

#Experiment constants
nu = 0.1 #V/s- Sweep rate
Ei = 1.5 #V - Initial voltage
Emax = 4.0 #V - Max voltage (end for forwards sweep
Emin = 1.0 #V - Min voltage (end for backwards sweep)

#Set initial bulk concentrations - MUST have a decimal!
Xb = 0.01 #M - bulk conc. of X
Yb = 0.0 #M - bulk conc. of Y
Wb = 0.0 #M - bulk conc. of W
Ub = 0.0 #M - bulk conc. of U
Zb = 0.0 #M - bulk conc of Z

#Computational toggles
concplots = 1 #toggle for concentration plot displays (1 = yes, 0 = no)
plotDens = 50; #Approximate # of plots desired to appear
dx = 1e-10 #m - smallest spatial resolution - set for convergence
dE = 0.001  #V - potential step - set for convergence
BT = 0.25 #grid expansion factor 0 < BT < 1, higher is faster & less accurate

############################ COMPUTATION BEGINS #######################

tf = (Emax-Emin)/nu #s, timescale of experiment operation
dt = dE/nu #s, timestep 
Ds = D*dt/(dx*dx) #dimensionless diffusion coeff
xmax = 6.0*np.sqrt(D*tf) #maximum diffusion distance
N = 1 + math.ceil(np.log(1 + (xmax*(np.exp(BT) - 1)/dx) )/BT)  #+1 for ghost pt

#Defines points 0, 1, 2, ... N in spatial grid
spcgrid = np.array(range(0,N+1)) #Get grid indices, reshape vector
spcgrid = spcgrid.reshape((len(spcgrid),1))
PF = (dx/(np.exp(BT)-1)) #"Prefactor" for grid calculation
xgrid = PF*(np.exp(BT*spcgrid) - 1) #Compute xgrid positions
#Spacing for electrode - critical parameter for current determination
dx1 = dx*(np.exp(BT/2)-1)/(np.exp(BT) - 1)
#Allocate initial concentration vectors
X0 = np.array((N+1)*[Xb]) #X is present at points 0,1,...N
Y0 = np.array((N+1)*[Yb]) #Y is absent at points 0,1,...N
W0 = np.array((N+1)*[Wb])
U0 = np.array((N+1)*[Ub])
Z0 = np.array((N+1)*[Zb])
C = np.concatenate([X0,Y0,W0,U0,Z0]) #C = [X;Y;W;U;Z] #Total states = 5N + 5
C = C.reshape((len(C),1)) #Convert to col vector
Cb = np.array([Xb,Yb,Wb,Ub,Zb])

#Compute diffusion vectors
D1i = Ds*np.exp(2*BT*((5/4) - spcgrid))
D2i = Ds*np.exp(2*BT*((3/4) - spcgrid))
#Corrections for special cases
D1i[0] = 0
D2i[0] = 0
D1i[1] = Ds*((np.exp(BT) - 1)/(np.exp(BT*0.5) - 1))
#Allocate sum
D3i = 1 + D1i + D2i
#Create DAm matrix for function use
#For DAji, col = j, row = i. 
Dm = np.concatenate([D1i,D2i,D3i],axis=1)

#Initialize potential vector
#For CV analysis
Evt = np.concatenate([np.arange(Ei+dE,(Emax+dE),dE),np.arange(Emax,(Emin-dE),-dE),np.arange(Emin,Ei+2*dE,dE)])
#For Linear Sweep analysis
#Evt = np.arange(Ei+dE,Emax+dE,dE)
#Either way, need to preallocate
Evt = Evt.reshape(len(Evt),1)
dispFreq = (Emax-Emin)/(dE*plotDens); #Frequency of concentration plot displays
Istor = 0.0*Evt
count = 0


for E in Evt:
     #Compute electrochemically dependent rates
     kfXY = ksXY*np.exp((E-E0XY)*VTXY)
     kbXY = ksXY*np.exp(-(E-E0XY)*VTXY)
     kfWU = ksWU*np.exp((E-E0WU)*VTWU)
     kbWU = ksWU*np.exp(-(E-E0WU)*VTWU)
     #Package rates for sending to function
     kVect = np.array([kfXY,kbXY,kcY,kfWU,kbWU,kcU])
     #Call linear solver function
     Cnew = fx.MatSolve(C,EqTog,kVect,N,Dm,D,Cb,dx1,dt)
     #Compute current (sum of X and W flux)
     Istor[count,:] = nXY*F*D*(Cnew[1] - Cnew[0])/dx1 + nWU*F*D*(Cnew[2*N+3] - Cnew[2*N+2])/dx1
     if count % dispFreq == 0:
         #Generate concentration plot every dispFreq pts
         print(count,E)
         if concplots == 1:
             MP.figure(2)
             #Overindex for 'landing' point
             #Plot concentration profiles
             MP.plot(xgrid,Cnew[0:(N+1)],'-b') #X
             MP.plot(xgrid,Cnew[(N+1):(2*N+2)],'-r') #Y
             MP.plot(xgrid,Cnew[(2*N+2):(3*N+3)],'-g') #W
             MP.plot(xgrid,Cnew[(3*N+3):(4*N+4)],'-k') #U
             MP.plot(xgrid,Cnew[(4*N+4)::],'-y') #Z
             #Plot aesthetics
             MP.title(['Voltage = ',np.round(E[0],4),' V'])
             MP.ylabel('Conc. (mM)')
             MP.xlabel('Distance from electrode (m)')
             MP.show()
             MP.close(2)
     count = count + 1
     C = Cnew
     
#Plot current-voltage relationship
MP.figure(3)    
MP.plot(Evt,Istor)
MP.xlabel('Voltage (V)')
MP.ylabel('Current (A)')
MP.show()

dex = np.where(Istor == max(Istor))
#Analytical solutions from Saveant for kinetic & equilibrium control
TPP = E0XY + 0.78*R*T/(BsXY*F) - R*T*np.log(ksXY*np.sqrt(R*T/(BsXY*F*nu*D)))/(BsXY*F)
TPPeq = E0XY + 1.11*R*T/F
TPPC = 0.496*F*Cb[0]*np.sqrt(D)*np.sqrt(BsXY*F*nu/(R*T))
TPPCeq = 0.446*F*Cb[0]*np.sqrt(D)*np.sqrt(BsXY*F*nu/(R*T))
print('Peak Potential/True/Eq // / Peak Current/True/Eq')
print(Evt[dex],[np.round(TPP,4)],[np.round(TPPeq,4)],\
      [np.format_float_scientific(Istor[dex][0],precision=5)],\
      [np.format_float_scientific(TPPC,precision=5)],\
      [np.format_float_scientific(TPPCeq,precision=5)])

#End timing
toc = time.time()
print('Time Elapsed',toc-tic)