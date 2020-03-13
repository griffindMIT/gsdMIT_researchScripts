# -*- coding: utf-8 -*-

#Import packages
import numpy as np 
from scipy import optimize as spopt
#from scipy.optimize import Bounds
import math
import matplotlib.pyplot as MP
import CV_Simulator_Fcns as fx
import time

#Start timing
tic = time.time()

#Rxn of interest: A - e = B, reversible as kf/kb. 
#Experiment of interest: Linear sweep from Ei to Ef

#Physical Constants
F = 96485.332 #C/mol - Faraday constant
T = 298 #K - Temperature
R = 8.314 #J/mol-K - Gas const.
BT = 0.25 #grid expansion factor 0 < BT < 1

#Reaction Constants
DA = 10 ** (-9) #m^2/s - Diffusion coeff of species A (mult by 10^4 to get cm^2/s)
DB = 10 ** (-9) #m^2/s - Diffusion coeff of species B
ks = 10 ** -5 #m^2/s - Surface rate constant (max value ~ 1 m^2/s or 10^4 cm^2/s)
E0 = 2.5 #V - True reversible potential
AP = 0.5 #Transfer coefficient
n = 1 # # of electrons transferred
VT = AP*n*F/(R*T) #V -mod. thermal voltage for n = 1

#Experiment constants
nu = 0.1 #V/s- Sweep rate
Ei = 2 #V - Initial voltage
Emax = 4.0 #V - Max voltage (end for forwards sweep
Emin = 1.0 #V - Min voltage (end for backwards sweep)
dE = 0.01 #V - potential step
Ab = 0.01 #M - bulk conc. of A - MUST have a decimal!
Bb = 0 #M - bulk conc. of B

############################ COMPUTATION BEGINS #######################

tf = (Emax-Emin)/nu #s, timescale of experiment operation
dt = dE/nu #s, timestep
kValues = np.array([ks*np.exp(VT*(Emin-E0)),ks*np.exp(VT*(Emax-E0))])
Ds = R*T*max(kValues)/(F*nu) #Target dimensionless diffusion coeff
print('DS = ',Ds)
rat = Ds/max([DA,DB]) #ratio of dt/(dx)^2
dx = np.sqrt(dt/rat) #Necesary 'base' x-spacing for accuracy

xmax = 6.0*np.sqrt(max([DA,DB])*tf) #maximum diffusion distance
N = 1 + math.ceil(np.log(1 + (xmax*(np.exp(BT) - 1)/dx) )/BT)  #+1 for ghost pt
print('N =',N)
#Defines points 0, 1, 2, ... N in spatial grid
spcgrid = np.array(range(0,N+1)) #Get grid indices
spcgrid = spcgrid.reshape((len(spcgrid),1))
PF = (dx/(np.exp(BT)-1))
xgrid = PF*(np.exp(BT*spcgrid) - 1)
#Spacing for electrode - important!
dx1 = dx*(np.exp(BT/2)-1)/(np.exp(BT) - 1)
#MP.figure(1)
#for x in spcgrid:
#    MP.plot([xgrid[x][0],xgrid[x][0]],[0,1],'-b')  
#MP.xlabel('Distance (m)')
#MP.title('Expanding Grid Spacing')
#MP.ylim([0,1])
#MP.xlim([0,1.05*max(xgrid)])
#MP.show()

#Allocate initial concentration vectors
A0 = np.array((N+1)*[Ab]) #A is present at points 0,1,...N
B0 = np.array((N+1)*[Bb]) #B is absent at points 0,1,...N
C = np.concatenate([A0,B0]) #C = [A;B] #Total states = 2N + 2
C = C.reshape((len(C),1)) #Convert to col vector
Cb = np.array([Ab,Bb])
Nst = 2*(N+1)

#Compute diffusion vectors
DA1i = Ds*np.exp(2*BT*((5/4) - spcgrid))
DA2i = Ds*np.exp(2*BT*((3/4) - spcgrid))
#Corrections for special cases
DA1i[0] = 0
DA2i[0] = 0
DA1i[1] = Ds*((np.exp(BT) - 1)/(np.exp(BT*0.5) - 1))
#Allocate sum
DA3i = 1 + DA1i + DA2i
#Create DAm matrix for function use
#For DAji, col = j, row = i. 
DAm = np.concatenate([DA1i,DA2i,DA3i],axis=1)
#Equivalent matrix for DB is DAm/(DA/DB)
#Drat = DA/DB

#Attempt #1 - nonlinear constrained optimization approach
#Need to feed in guess of function - current conc values
#Function creates system of nonlinear equations

#Create reduced evaluator function...
fcn1 = lambda cc,cco,kayf,kayb : fx.Eval(cc,cco,kayf,kayb,N,DAm,DA,DB,Cb,dx1)
# lb = 0*np.arange(0,len(C))
# ub = np.inf*np.arange(1,len(C)+1)
# bnds = Bounds(lb, ub)

#Initialize potential vector
Evt = np.concatenate([np.arange(Ei+dE,(Emax+dE),dE),np.arange(Emax,(Emin-dE),-dE),np.arange(Emin,Ei+2*dE,dE)])
Evt = Evt.reshape(len(Evt),1)
Istor = 0.0*Evt
count = 0


for E in Evt:
     #Compute rates
     #En = Evt[np.where(Evt == E)[0][0] + 1]
     kf = ks*np.exp((E-E0)*VT)
     kb = ks*np.exp(-(E-E0)*VT)  
     #Set initial guess (current concs)
     Cg = C
     #Create secondary reduced evaluator function
     fcn2 = lambda cc: fcn1(cc,C,kf,kb)
     Cnew = spopt.root(fcn2,Cg,method='hybr',tol=1e-14).x
     #Cnew = spopt.fsolve(fcn2,Cg)
     #Cnew = spopt.newton_krylov(fcn2,Cg)
     #Cnew = spopt.minimize(fcn2,C.flatten(),method='hybrd',options={'xatol': 1e-8,'fatol': 1e-8})
     Cnew = Cnew.reshape(len(Cnew),1) 
     Istor[count,:] = DA*(Cnew[1] - Cnew[0])/dx1 #+ (E-1)/(8000) 
     #Istor[count,:] = Cnew[0]
     #Istor[count,:] = Cnew[N+1]
     print(E,kf,kb,Cnew[0],Istor[count,:])
     if count % 100 == 0:
         MP.figure(count)
         MP.plot(xgrid,Cnew[0:(N+1)],'-b')
         MP.plot(xgrid,Cnew[(N+1)::],'-r')
         MP.title(['Voltage = ',np.floor(E[0]),' V'])
         MP.ylabel('Conc. (mM)')
         MP.xlabel('Distance from electrode (m)')
         MP.legend((1,2),('Reactant','Product'),loc=5)
         MP.show()
         print(count,E)
     count = count + 1
     C = Cnew
     
MP.figure(2)    
MP.plot(Evt,Istor)
MP.xlabel('Voltage (V)')
MP.ylabel('Current (A)')
MP.show()

dex = np.where(Istor == max(Istor))
print(Evt[dex],Istor[dex])

#End timing
toc = time.time()
print('Time Elapsed',toc-tic)






    


