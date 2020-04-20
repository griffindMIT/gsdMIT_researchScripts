# -*- coding: utf-8 -*-

#Import packages
import numpy as np 
from scipy import optimize as spopt
import math
import matplotlib.pyplot as MP
import CV_Simulator_Fcns as fx
import time

#Start timing
tic = time.time()

#Rxn of interest: A - e = B, reversible as kf/kb. 
#Experiment of interest: Linear sweep from Ei to Ef

#Close all open figures
MP.close('all')

#Physical Constants
F = 96485.332 #C/mol - Faraday constant
T = 298.15 #K - Temperature
R = 8.314 #J/mol-K - Gas const.

#Reaction Constants
DA = 10 ** (-9) #m^2/s - Diffusion coeff of species A (mult by 10^4 to get cm^2/s)
#DB = 10 ** (-9) #m^2/s - Diffusion coeff of species B
DB = 10**(-9)
ks = 10 ** (-6) #m/s - Surface rate constant (max value ~ 10^2 m/s or 10^4 cm/s)
E0 = 2.5 #V - True reversible potential
AP = 0.5 #Transfer coefficient
n = 1 # # of electrons transferred
VT = AP*n*F/(R*T) #modified thermal voltage for n = 1

#Experiment constants
nu = 0.1 #V/s- Sweep rate
Ei = 2 #V - Initial voltage
Emax = 4.0 #V - Max voltage (end for forwards sweep
Emin = 1.0 #V - Min voltage (end for backwards sweep)

Ab = 0.01 #M - bulk conc. of A - MUST have a decimal!
Bb = 0 #M - bulk conc. of B

#Computational toggles
gridplots = 0 #toggle for expanding grid display(1 = yes, 0 = no)
concplots = 1 #toggle for concentration plot displays (1 = yes, 0 = no)
plotDens = 10; #Approximate # of plots desired to appear
dx = 1e-10 #m - smallest spatial resolution - set for convergence
dE = 0.0001  #V - potential step - set for convergence
BT = 0.25 #grid expansion factor 0 < BT < 1, higher is faster & less accurate
XX = (Emax-Emin)/(dE*plotDens); #Frequency of concentration plot displays
LinSolv = 1 #Linear solution toggle (1 = yes, 0 = no)
CstSolv = 0 #Constrained solution toggle (1 = yes, 0 = no)


############################ COMPUTATION BEGINS #######################

tf = (Emax-Emin)/nu #s, timescale of experiment operation
dt = dE/nu #s, timestep 
kValues = np.array([ks*np.exp(VT*(Emin-E0)),ks*np.exp(VT*(Emax-E0))])

Ds = DA*dt/(dx*dx) #dimensionless diffusion coeff
xmax = 6.0*np.sqrt(max([DA,DB])*tf) #maximum diffusion distance
N = 1 + math.ceil(np.log(1 + (xmax*(np.exp(BT) - 1)/dx) )/BT)  #+1 for ghost pt

#Defines points 0, 1, 2, ... N in spatial grid
spcgrid = np.array(range(0,N+1)) #Get grid indices
spcgrid = spcgrid.reshape((len(spcgrid),1))
PF = (dx/(np.exp(BT)-1))
xgrid = PF*(np.exp(BT*spcgrid) - 1)
#Spacing for electrode - important!
dx1 = dx*(np.exp(BT/2)-1)/(np.exp(BT) - 1)
if gridplots == 1:
    MP.figure(1)
    for x in spcgrid:
        MP.plot([xgrid[x][0],xgrid[x][0]],[0,1],'-b')  
        MP.xlabel('Distance (m)')
        MP.title('Expanding Grid Spacing')
        MP.ylim([0,1])
        MP.xlim([0,1.05*max(xgrid)])
        MP.show()

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

#Create reduced evaluator function...

if CstSolv == 0:
    #Direct solution
    fcn1 = lambda cc,cco,kayf,kayb : fx.Eval(cc,cco,kayf,kayb,N,DAm,DA,DB,Cb,dx1)
else:
    #Cost function return
    fcn1 = lambda cc,cco,kayf,kayb : fx.OptEval(cc,cco,kayf,kayb,N,DAm,DA,DB,Cb,dx1)

#Initialize potential vector
#For CV analysis
#Evt = np.concatenate([np.arange(Ei+dE,(Emax+dE),dE),np.arange(Emax,(Emin-dE),-dE),np.arange(Emin,Ei+2*dE,dE)])
#For Linear Sweep analysis
Evt = np.arange(Ei+dE,Emax+dE,dE)
#Either way
Evt = Evt.reshape(len(Evt),1)
Istor = 0.0*Evt
count = 0


for E in Evt:
     #Compute rates
     kf = ks*np.exp((E-E0)*VT)
     kb = ks*np.exp(-(E-E0)*VT)  
     #Set initial guess (current concs)
     Cg = C
     #Create secondary reduced evaluator function
     fcn2 = lambda cc: fcn1(cc,C,kf,kb)
     #Linear solver
     if LinSolv == 1:
         Cnew = fx.MatLin(C,kf,kb,N,DAm,DA,DB,Cb,dx1,CstSolv)
     else:
         #Nonlinear solver
        if CstSolv == 0:
            Cnew = spopt.fsolve(fcn2,Cg)
        else:
            Cnew = spopt.minimize(fcn2,Cg)
            Cnew = Cnew.x
            #stuff     
     Istor[count,:] = n*F*DA*(Cnew[1] - Cnew[0])/dx1
     if count % XX == 0:
         #Generate concentration plot every XX pts
         print(count,E)
         if concplots == 1:
             MP.figure(2)
             MP.plot(xgrid,Cnew[0:(N+1)],'-b')
             MP.plot(xgrid,Cnew[(N+1)::],'-r')
             MP.title(['Voltage = ',np.round(E[0],4),' V'])
             MP.ylabel('Conc. (mM)')
             MP.xlabel('Distance from electrode (m)')
             MP.legend(('Reactant','Product'))
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
TPP = E0 + 0.78*R*T/(AP*F) - R*T*np.log(ks*np.sqrt(R*T/(AP*F*nu*DA)))/(AP*F)
TPPeq = E0 + 1.11*R*T/F
TPPC = 0.496*F*Cb[0]*np.sqrt(DA)*np.sqrt(AP*F*nu/(R*T))
TPPCeq = 0.446*F*Cb[0]*np.sqrt(DA)*np.sqrt(AP*F*nu/(R*T))
print('Peak Potential/True/Eq // / Peak Current/True/Eq')
print(Evt[dex],[np.round(TPP,4)],[np.round(TPPeq,4)],\
      [np.format_float_scientific(Istor[dex][0],precision=5)],\
      [np.format_float_scientific(TPPC,precision=5)],\
      [np.format_float_scientific(TPPCeq,precision=5)])

#End timing
toc = time.time()
print('Time Elapsed',toc-tic)






    


