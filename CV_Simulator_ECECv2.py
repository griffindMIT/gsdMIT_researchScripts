# -*- coding: utf-8 -*-

#Import packages
import numpy as np 
#from scipy import optimize as spopt
import math
import matplotlib.pyplot as MP
import CV_Simulator_Fcns_ECECv2 as fx
import time

def simCV(value):
    #Generic CV simulation w/many hardcoded values. See bottom of code for 
    #running sim/running scenarios
        
    #Start timing
    tic = time.time()

    #Close all open figures
    MP.close('all')

    #Physical Constants
    F = 96485.332 #C/mol - Faraday constant
    T = 298.15 #K - Temperature - tech. an experiment constant but roll w/ it
    R = 8.314 #J/mol-K - Gas const.

    #Mechanism - Reversibility is built-in but can be turned off
    #X = Y + e (k1), Electrical
    #Y = W + U (k2) Chemical
    #W = U + e (k3) Electrical
    #U + H2O = Z (k4) Chemical
    #X = W + U + e (k5) Electro-chemical (concerted reaction)

    D = 10 ** (-9) #m^2/s - Diffusion coefficient (mult. by 10^4 to get cm^2/s).
    #Asymmetric diffusion is still a bear...

    #Reaction Constants
    k1s = 10 ** (0) #m/s - Surface rate constant (max value ~ 10^2 m/s or 10^4 cm/s)
    k2f = 10 ** (3) #1/s - Homog. rate constant, first order. 
    DtG2 = 15 #J/mol, Estimate of delta G for rxn 2
    K2 = np.exp(-DtG2*4184/(R*T)) #Equilibrium constant for rxn 2
    K2 = 3
    k2b = k2f/K2 #1/s - Reverse homog reaction rate of rxn 2, estimated via equilibrium constant
    k3s = 0.0 #10 ** (-6) #m/s - Surface rate constant
    k4f  = 0.0 #0.001 #1/(sM) - Homog. rate constant, second order
    DtG4 = 10 #kcal/mol, Estimate of delta G for rxn 2
    K4 = np.exp(-DtG4)/(R*T) #Equilibrium constant for rxn 4
    k4b = k4f/K4 #Reverse homog reaction rate of rxn 2, estimated via equilibrium constant
    k5fs = 0 #10 ** (-6) #Surface rate constant for concerted reaction (rxn 5)
    DtG5 = 45 #kcal/mol, Estimate of delta G for rxn 5
    K5 = np.exp(-DtG5/(R*T)) #Equilib. constant for rxn 5
    k5bs = k5fs/K5 #Reverse rate constant for rxn 5

    #Irreversible toggles - uncomment if you wish to force irreversible reactions. 
    #k2b = 0.0
    k4b = 0.0
    k5bs = 0.0

    #Set toggles to 1 to enforce ~equilibrium~ everywhere
    #Equilibrium = (infinitely fast ks for rxn 1, rxn 3)
    EqTog = np.array([0,0])
    #Set toggle to 1 to enable 'equilibrium detection'
    #The exact implementation of this is... unclear at the moment
    AutoEq = 0

    #Establish useful electrochemical constants
    E01 = 2.5 #V - Reversible potl for X=Y (rxn 1)
    Bs1 = 0.5 #Symmetry factor for X = Y (rxn 1)
    #Butler-Volmer equation considered here - Marcus not yet.
    n1 = 1 # number of electrons transferred for X = Y (rxn 1)
    E03 = 1.5 #V - Reversible potl for W=U (rxn 3)
    Bs3 = 0.5 #Symmetry factor for W = U (rxn 3)
    n3 = 1 # # of electrons transferred for W = U (rxn 3)
    E05 = 3 #V - Reversible potl for reaction 5
    Bs5 = 0.5 #Symmetry factor for X = W + U + e (rxn 5)
    n5 = 1 #number of electrons transfered for rxn 5
    VT = F/(R*T) #Thermal voltage - useful constant

    #Experiment constants
    nu = 10 ** (0) #V/s- Sweep rate
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
    plotDens = 25; #Approximate # of plots desired to appear during experiment
    #dx = 10 ** (-16)   #m - smallest spatial resolution - set for convergence (test w/analytic)
    dE = 0.001  #V - potential step - set for convergence
    BT = 0.25 #grid expansion factor 0 < BT < 1, higher is faster & less accurate

    ############################ COMPUTATION BEGINS #######################

    #Interpreting some computational inputs and defining variables
    tf = 5*(Emax-Emin)/nu #s, timescale of experiment operation
    dt = dE/nu #s, timestep 
    #Begin generating space grid - get total distance and point count
    Nmin = 40 #Set minimum number of grid points (arbitrary but OK)
    xmax = 6.0*np.sqrt(D*tf) #maximum diffusion distance
    dxNmin =  0.9*xmax*(np.exp(BT)-1)/(np.exp(BT*(Nmin-2))) #dx needed so that N = 35 w/ 0.9 safety
    dx = min(10**(-10),dxNmin) #dx is no larger than 10^(-10)...
    N = 1 + math.ceil(np.log(1 + (xmax*(np.exp(BT) - 1)/dx) )/BT)  #+1 for ghost pt
    Ds = D*dt/(dx*dx) #dimensionless diffusion coeff
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
    #EvtCyc = np.concatenate([np.arange(Ei+dE,(Emax+dE),dE),np.arange(Emax,(Emin-dE),-dE),np.arange(Emin,Ei+2*dE,dE)])
    #Add cycles to the CV
    #Evt = np.concatenate([EvtCyc,EvtCyc])
    #For Linear Sweep analysis
    Evt = np.arange(Ei+dE,Emax+dE,dE)
    #Either way, need to preallocate
    Evt = Evt.reshape(len(Evt),1)
    dispFreq = (Emax-Emin)/(dE*plotDens); #Frequency of concentration plot displays
    Istor = 0.0*Evt
    count = 0
    wmax = 0

    for E in Evt:
        #Compute electrochemically dependent rates
        k1f = k1s*np.exp((E-E01)*VT*(1 - Bs1))
        k1b = k1s*np.exp(-(E-E01)*VT*Bs1)
        k3f = k3s*np.exp((E-E03)*VT*(1 - Bs3))
        k3b = k3s*np.exp(-(E-E03)*VT*Bs3)
        k5f = k5fs*np.exp((E-E05)*VT*(1 - Bs5))
        k5b = k5bs*np.exp(-(E-E05)*VT*())
        #Package rates for sending to function
        kVect = np.array([k1f,k1b,k2f,k2b,k3f,k3b,k4f,k4b,k5f,k5b])
        #Call linear solver function
        Cnew = fx.MatSolve(C,EqTog,kVect,N,Dm,D,Cb,dx1,dt)
        #Check solution with 'half incremented soln.'
        #Compute echem. dependent rates at halfway between E(t-1) and E(t)...
        Ehalf = E - 0.5*dE
        k1fH = k1s*np.exp((E-E01)*VT*(1 - Bs1))
        k1bH = k1s*np.exp(-(E-E01)*VT*Bs1)
        k3fH = k3s*np.exp((E-E03)*VT*(1 - Bs3))
        k3bH = k3s*np.exp(-(E-E03)*VT*Bs3)
        k5fH = k5fs*np.exp((E-E05)*VT*(1 - Bs5))
        k5bH = k5bs*np.exp(-(E-E05)*VT*(Bs5))
        #kVectHalf = np.array([k1fH,k1bH,k2fH,k2bH,k3fH,k3bH,k4fH,k4bH,k5fH,k5bH])
        
        #Compute current (sum of X and W flux)
        Istor[count,:] = n1*F*D*(Cnew[1] - Cnew[0])/dx1 #+ n3*F*D*(Cnew[2*N+3] - Cnew[2*N+2])/dx1
        wnew = Cnew[2*N+2]
        if wnew > wmax:
            wmax = wnew
        if abs(E-1.55) < dE:
            print(E,value,Cnew[0],Cnew[N+1],Cnew[2*N+3],Cnew[3*N+4])

        if count % dispFreq == 0:
            #Generate concentration plot every dispFreq pts
            if concplots == 1:
                #print(count,E)
                MP.figure(2)
                #Overindex for 'landing' point
                #Plot concentration profiles
                MP.plot(xgrid,Cnew[0:(N+1)],'-b') #X
                MP.plot(xgrid,Cnew[(N+1):(2*N+2)],'-r') #Y
                MP.plot(xgrid,Cnew[(2*N+2):(3*N+3)],'-k') #W
                MP.plot(xgrid,Cnew[(3*N+3):(4*N+4)],'-g') #U
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

    #Peak potential from first sweep
    Istor1 = Istor
    #Istor1 = Istor[0:len(EvtCyc)]
    #Istor2 = Istor[len(EvtCyc)::]
    dex1 = np.where(Istor1 == max(Istor1))
    #dex2 = np.where(Istor2 == max(Istor2))
    #Peak potential from second sweep
    #Analytical solutions from Saveant for kinetic & equilibrium control
    TPP = E01 + 0.78*R*T/((1 - Bs1)*F) - R*T*np.log(k1s*np.sqrt(R*T/((1 - Bs1)*F*nu*D)))/((1 - Bs1)*F)
    TPPeq = E01 + 1.11*R*T/F
    if k2f > 0:
        TPP_EC = E01 + 0.78*R*T/(F) - (R*T/(2*F))*np.log(R*T*k2f/(F*nu))
    else:
        TPP_EC = 0
    TPPC = 0.496*F*Cb[0]*np.sqrt(D)*np.sqrt(Bs1*F*nu/(R*T))
    TPPCeq = 0.446*F*Cb[0]*np.sqrt(D)*np.sqrt(2*Bs1*F*nu/(R*T))
    #print('Peak Potential / E Kinetic / E Eq / EC') #' // / Peak Current/True/Eq')
    #print(Evt[dex],'/',[np.round(TPP,4)],'/',[np.round(TPPeq,4)],'/',[np.round(TPP_EC,4)])#,\
    #print(k2b*dt*C[2*N+2])
    #print([np.round(wmax*1000,3)])
    #print('Delta=',np.round(Evt[dex]-TPP_EC,4))
      #[np.format_float_scientific(Istor[dex][0],precision=5)],\
      #[np.format_float_scientific(TPPC,precision=5)],\
      #[np.format_float_scientific(TPPCeq,precision=5)])
    
   #Compute second derivative of Istor through easy methods
   #nl = len(Istor)
   #I2deriv = Istor[0:nl-2] + -2*Istor[1:nl-1] + Istor[2:nl]
   #redEvt = Evt[1:nl-1]
   #I2deriv[0:100] = 0 #screening for low alpha values
   #dexlow = np.where(I2deriv == min(I2deriv))
   #dexlow = dexlow[0][0]
   #dexhi = np.where(I2deriv == max(I2deriv))
   #dexhi = dexhi[0][0]
   #nearzerI2 = I2deriv[dexhi:dexlow]
   #nearzerEvt = redEvt[dexhi:dexlow]
   #Einf = nearzerEvt[np.where(abs(nearzerI2) == min(abs(nearzerI2)))];
   #Einf = Einf[0]
   #MP.figure(4)
   #MP.plot(redEvt,I2deriv)
   #MP.plot(np.array([redEvt[dexlow],redEvt[dexhi]]),np.array([0,0]))
   #MP.plot(Einf,0,'ok')
   #MP.show()
   #print(Evt[dex],[np.round(Einf,4)],Evt[dex]-Einf)
    
    #End timing
    toc = time.time()
    elapsed = np.round(toc-tic,1)
    wmax2 = [np.round(wmax*1000,3)]
    
    #print('Value / PP1 / PP2 / E Anal. / EC Anal. / Wmax / t ')
    #print(value,np.round(Evt[dex1][0],4),np.round(Evt[dex2][0],4),np.round(TPP,4),np.round(TPP_EC,4),wmax2[0],elapsed)
    
    #print('Time Elapsed:',np.round(toc-tic,1),' s')
    
#Set up experiments down here
#Run a single CV with one input - this line
simCV(-15.0)

#Run over a range of inputs - set values in valVect
#valVect = np.arange(15.0,-16.0,-1.0)
#valVect = np.array([0.001,0.0005,0.0001,0.00005])
#for vv in valVect:
#    simCV(vv)
