# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 13:15:03 2020

@author: ChemeGrad2019
"""

#Import packages
import numpy as np 
#from scipy import optimize as spopt
import math
import matplotlib.pyplot as MP
import CV_SimulatorFcnsv4 as fx
from scipy import optimize as spopt
import time

#Simulation function. Takes one input value - the rest hardcoded. 
#Value used for parameter sweep experiments. 
def simCV(value):
    ######################### HEADER ########################################
    
    #Establish mechanism and species:
    #X = Y + e (k1), Electrical
    #Y = W + U (k2) Chemical
    #W = U + e (k3) Electrical
    #U + (H2O )= Z + (H3O+) (k4) Chemical
    #Assumed constant and high concentration of (H2O) and (H3O+) throughout - pseudo first order
    #X = W + U + e (k5) Electro-chemical (concerted reaction)
    #Y = U (k6) Alternative Chemical step (unimolecular), for CE support.  
    
    #Timing, clear figures, import physical constants
    tic = time.time()
    MP.close('all')
    F = 96485.332 #C/mol - Faraday constant
    T = 298.15 #K - Temperature
    R = 8.314 #J/mol-K - Gas constant 
    VT = F/(R*T) #Thermal voltage (~29 mV @ RT), useful constant
    
    ############### PHYSICAL PROPERTY INFORMATION ###########################
    #Stored in the 'getConstants' function. Each 'option' (last input parameter) gives a different set.
    kConst = getConstants(F,R,T,0)
    E0v = getConstants(F,R,T,1)
    Bsv = getConstants(F,R,T,2)
    Dv = getConstants(F,R,T,3)
    FcEq = getConstants(F,R,T,4)
    quickRate = lambda ee: computeRates(ee,VT,E0v,Bsv,kConst)

    ####################### EXPERIMENT PARAMETERS ##########################
    #Establish parameters for the CV
    #CV is run as Ei -> Emax -> Emin -> Ei for # cycles at rate of nu.
    Ei = 1.5 #V - Initial voltage
    Emax = 3.0 #V - Max voltage 
    Emin = 1.0 #V - Min voltage 
    nu = 10 ** (value) #V/s- Sweep rate
    cycles = 1 #Number of CV cycles (1+)
    LSweep = 1 #Toggle. If (1), Experiment is Ei -> Emax, if (0) use full CV cycle.
    #Set bulk/initial concentrations of species. Only X is present for E, E', EC, only Y for CE. 
    Cb = np.array([0.01,0.0,0.0,0.0,0.0]) #M - concs. for species X, Y, W, U, Z
    
    ######################## COMPUTATIONAL TOGGLES ##########################
    #Toggles - changes mode of solver/computation or the information displayed
    showPlots = 1 #(1) Show concentration plots with voltage titles as CV is being run, (0) do not. 
    showCV = 1 #(1) Show CV (current vs. voltage) at end of run, (0) do not. 
    secDeriv = 1 #(1) Perform second derivative analysis on linear sweep (required), (0) do not. 
    stepCheck = 1 #(1) Perform step resolution check to ensure good dE convergence, (0) do not.
    stepPlot = 1 #(1) Plot stored (final) resolution deviations at end of computation, (0) do not.
    freeGrid = 1 #(1) Use dynamically-generated voltage grid, (0) do not. 
    adaptStep = 1 #(1) Use adaptive timestepping with dynamic grid, (0) do not. 
    nonLin = 0 #(1) Always use nonlinear solver, (0) do not. 
    nonLinCorr = 0 #(1) Use nonlinear solver in case of high resErr / solver failure, (0) do not. 
    #No support for nonlinear solver in dynamic grid - yet. 
    #No support for second derivative in dynamic grid - yet. 
    
    #Constraints - inputs for solver tolerances, etc. 
    plotDensity = 10 #Approximate # of plots per cycle. May increase w/ adaptive stepping. 
    secDerivBurn = 100 # Number of initial points to ignore for second derivative analysis (helps avoid false zeroes). Set to 1 for no burn.
    dx_max = 10 ** (-10) #m - maximum allowable 'dx' spacing (Empirical, may change)
    Nmin = 40 #Minimum allowable # of points in space grid
    dE_i = 0.001 #V, initial voltage grid spacing. Without free grid, fixes all voltage points. 
    dE_max = 0.005 #V, maximum allowable grid spacing in freeGrid
    dE_min = 10 ** (-4) #V, minimum allowable grid spacing in freeGrid
    stepErrTol = 0.001 # -, minimum allowable fractional deviation
    minDev = (10 ** (-6))*Cb[0] #M, absolute threshold for concentration deviations to be considered 'relevant' in resCheck
    BT = 0.25 #(Beta) Spatial grid exp. coeff, 0 < BT < 1. 0 = linear, 1 = max. Lower = more accurate, 0.25-0.5 typ. 
    
    ############################ AUTOMATIC TOGGLE CORRECTIONS ################
    #Certain combinations of toggles are incompatable - this fixes that. 
    #Linear sweep means only one cycle happens
    if LSweep == 1:
        cycles = 1
    #Turns off second derivative analysis, since it requires a linear sweep & fixed step sizes.
    if LSweep == 0 or freeGrid == 1:
        secDeriv = 0
    #Turn s off adaptive timestepping if freeGrid or resCheck is disabled (default is do not do it!)
    if freeGrid == 0 or stepCheck == 0:
        adaptStep = 0
        #print('Adaptive timestepping disabled')
    #Turns off resPlot if resCheck is not enabled since the plot will be all zeros. 
    if stepCheck == 0:
        stepPlot = 0
    if BT > 1:
        BT = 1
        print('BT Set to 1 for stability.')
        
    ########################## PRE-COMPUTATION CALCULATIONS #################
    #Establish spatial gridding
    t_diff = 3*cycles*(Emax-Emin)/nu #s, timescale of diffusion action over experiment length. 
    xmax = 6.0*np.sqrt(max(Dv)*t_diff) #m, spacescale of diffusion action over experiment length.
    dxNmin = 0.9*xmax*(np.exp(BT)-1)/(np.exp(BT*(Nmin-2))) #m, dx-spacing needed for Nmin points (0.9 safety factor)
    dx = min(dxNmin,dx_max) #m, pick smallest of two dx options. 
    N = 1 + math.ceil(np.log(1 + (xmax*(np.exp(BT) - 1)/dx) )/BT)  # # of spatial points needed +1 for ghost pt (bulk)
    #Spacing between surface and first conc point - Very important parameter!
    dx1 = dx*(np.exp(BT/2)-1)/(np.exp(BT) - 1) # m
    #Defines points 0, 1, 2, ... N in spatial grid
    spcGridInd = np.arange(0,N+1).reshape(len(np.arange(0,N+1)),1) #Get grid indices 
    gridPF = (dx/(np.exp(BT)-1)) #"Prefactor" for grid calculation as defined by Rudolph.
    xgrid = gridPF*(np.exp(BT*spcGridInd) - 1) #Compute xgrid positions (m)
    
    #Combine all concentrations into vector C. C will be updated at each time point. 
    #C = [X;Y;W;U;Z] Total states = 5N + 5. Reshaped into column vector. 
    #Allocate initial concentration vector using 'blank' concentration C0i
    C0i = np.array((N+1)*[1.0]) #All species present at points 0,1,...N
    #Multiplying by bulk concentrations gives the initial condition C, package into one vector.
    C = np.concatenate([Cb[0]*C0i,Cb[1]*C0i,Cb[2]*C0i,Cb[3]*C0i,Cb[4]*C0i]).reshape(5*N+5,1) 
    
    #Compute coefficients for diffusion terms
    Dsu = Dv[0]/(dx*dx) #1/s, dimensionless diffusion coeff for species. X w/0 time scaling applied. 
    #Compute (mostly scaled) diffusion vectors at desired grid points
    D1i = Dsu*np.exp(2*BT*((5/4) - spcGridInd)) #1/s
    D2i = Dsu*np.exp(2*BT*((3/4) - spcGridInd)) #1/s
    #Corrections for 3 near-surface points
    D1i[0] = 0
    D2i[0] = 0
    D1i[1] = Dsu*((np.exp(BT) - 1)/(np.exp(BT*0.5) - 1))
    #Create Dm matrix for function use, each column = +/- from "center point."
    Dmu = np.concatenate([D1i,D2i],axis=1) #1/s
    ChemProdMax = 0.0 #Allocate space for this value which is tracked. 
    dE = dE_i #Allocate dE variable
    count = 0 #Allocate counter variable
    
    ############################ SOLUTION OF EQUATIONS ######################
    if freeGrid == 0:
    ############################# FIXED-GRID SOLUTION ########################
        #Indentation for all-Fixed-Grid solution. 
        #Scale diffusion coefficients, dt, accordingly. dE will not change.
        dt = dE/nu
        Dm = Dmu*dt
        #Create voltage vector as described previously. 
        if LSweep == 1:
            #For linear sweep, voltage vector is simply increasing range of value. 
            Evt = np.arange(Ei+dE,Emax+dE,dE)
            Evt = Evt.reshape(len(Evt),1)
            #The 'cycle' simply has 1 ending index - the end of the sweep.
            cyc_inds = np.array([len(Evt)])
        else:
            #Otherwise - concatenate forward/reverse/forward sweeps into a single 'cycle'...
            Evt_cyc = np.concatenate([np.arange(Ei+dE,(Emax+dE),dE),np.arange(Emax,(Emin-dE),-dE),np.arange(Emin,Ei+2*dE,dE)])
            #...then rescale to column vector, concatenate again to # of cycles specified. 
            Evt_cyc.reshape(len(Evt_cyc),1)
            Evt = np.concatenate(cycles*[Evt_cyc])
            #Note the indices where the cycles end! (+/- 2 dE)
            cyc_inds = len(Evt_cyc)*np.arange(0,cycles,1)
        #Each Evt point will be one calculation - compute the display frequency for desired # of plots.
        dispFreq = np.round(len(Evt)/plotDensity,0)
        #Preallocate storage for current, res-dev variable. 
        Istor = 0.0*Evt
        stepErrStor = 0.0*Evt
        debugStore = 0.0*Evt
        for E in Evt: #E represents the next concentration point's
            #Indentation for computation loop.
            #Set error / failure toggle
            stepErr = 0
            failed = 0
            #Compute electrochemically dependent rates
            kVectxx = quickRate(E[0])
            #Compute rates at half-step position
            Ex = E[0] - (dE/2)
            kVectx = quickRate(Ex)
            #If NOT using all-nonlinear solver...
            if nonLin == 0:
                #Try computing concentrations @ E.
                #Any references to 'C' at this point are no longer valid!
                try:
                    Cxx = fx.MatSolve(C,kVectxx,Dm,dt,FcEq,N,Dv,dx1,Cb)
                    #If the resolution check is on, try computing @ Ex. 
                    if stepCheck == 1:
                        Cx1 = fx.MatSolve(C,kVectx,0.5*Dm,0.5*dt,FcEq,N,Dv,dx1,Cb)
                        Cx2 = fx.MatSolve(Cx1,kVectxx,0.5*Dm,0.5*dt,FcEq,N,Dv,dx1,Cb)
                except:
                    #If any of them have failed, the thing fails. 
                    failed = 1
                #Compute step-error metric. Why here? so that we can trigger nonlin if desired. 
                if failed == 0 and stepCheck == 1:
                    #Didn't fail, do want to check res.
                    stepErr = compResError(Cxx, Cx2, N, minDev)
                elif failed == 1 and stepCheck == 1:
                    #This is the failure condition - resErr will be above tol. 
                    stepErr = 10
            if nonLin == 1 or ((failed == 1 or stepErr > stepErrTol) and nonLinCorr == 1):
                #If nonLin is on, OR if it's failed/above tolerance (something wrong) and nonlinCorr is on. 
                #Always use the starting conc as the guess to avoid NAN issues with failed == 1
                fcnxx = lambda cc: fx.NonLinEval(cc,C,kVectxx,Dm,dt,FcEq,N,Dv,dx1,Cb)
                Cxx = spopt.fsolve(fcnxx,C)
                if nonLin ==1 and stepCheck == 1:
                    #This means that only nonlinear is computed and step-check is desired. 
                    fcnx1 = lambda cc: fx.NonLinEval(cc,C,kVectx,0.5*Dm,0.5*dt,FcEq,N,Dv,dx1,Cb)
                    Cx1 = spopt.fsolve(fcnx1,C)
                    fcnx2 = lambda cc: fx.NonLinEval(cc,Cx1,kVectxx,0.5*Dm,0.5*dt,FcEq,N,Dv,dx1,Cb)
                    Cx2 = spopt.fsolve(fcnx2,Cx1)
                    stepErr = compResError(Cxx,Cx2,N,minDev)
            #Store the resErr
            stepErrStor[count] = stepErr
            #Store current
            Istor[count] = F*Dv[0]*(Cxx[1] - Cxx[0])/dx1 #+ more terms...
            #Generate plot
            if count %dispFreq == 0 and showPlots == 1:
                makePlot(Cxx,E,N,xgrid)
            #Track maximum of chemical-step product (useful convergence metric)
            ChemProd = C[2*N+2]
            if ChemProd > ChemProdMax:
                ChemProdMax = ChemProd
            #Update concentration, count
            C = Cxx
            count += 1
            #if abs(E - 2.5) < 0.0001:
            #   Debug stuff
            #    g = 2 + 2
            #    #print('boy howdy')
        #No post-run correction needed, since Evt/Istor already in desired format
    else:
    ############################# FREE-GRID SOLUTION ########################
        #Indentation level for varible-grid solution. 
        #SEt the current voltage
        E = Ei
        #Estimate the max # of points required... 
        pointsReq = np.round(3*cycles*(Emax-Emin)/dE_min,0)
        #Preallocate space for voltages, currents, and resErrs
        Evt_tmp = 0.0*np.arange(0.0,pointsReq,1.0).reshape(int(pointsReq),1) #Huge overestimate of space
        Istor_tmp = 0.0*Evt_tmp
        stepErrStor_tmp = 0.0*Evt_tmp
        #Compute the display frequency in an ad-hoc manner...
        dispFreq = np.round((Emax-Emin)/(dE*plotDensity),0)
        #Allocate cyc_inds array
        if LSweep == 1:
            cyc_inds = np.array([0])
        else:
            cyc_inds = np.arange(0,cycles,1)
        #Begin for-loop that regulates cycles...
        for jj in cyc_inds:
            #Indentation level for cycle-level solving
            #Outer loop - is the cycle done? has it passed the minimum? 
            cyc_done = 0
            min_done = 0
            while cyc_done == 0:
                #Indentation level for in-cycle solving...
                converged = 0 #dE and dE/2 have similar convergence
                rescaled = 0 #dE has not been adjusted
                failed = 0 #All solver have passed well
                failcount = 0 #Counter of # of failures
                while converged == 0:
                    #Indentation level for single-point resolution
                    #Get two relevant voltages, E + dE (Exx) and E + dE/2 (Ex)
                    Exx = E + dE
                    #Apply corrective bounding. 
                    #Why the 0.9999? To prevent a step of 0.0001, basically. 
                    if Exx >= 0.99*Emax and dE > 0:
                        #If the step would take it over the top - stop at the top. 
                        Exx = Emax
                    elif 1.01*Emin >= Exx and dE < 0:
                        #If the step would take it below the minimum - stop at the minimum
                        Exx = Emin
                        min_done = 1
                    elif min_done == 1 and Exx >= 0.99*Ei:
                        #Make sure that we return the voltage back to the Ei between cycles. 
                        Exx = Ei
                    #Compute rates at Exx
                    kVectxx = quickRate(Exx)
                    #Compute time resolution of step
                    dt = abs(Exx - E)/nu
                    #Rescale diffusion coefficient matrix accordingly
                    Dm = Dmu*dt
                    #Need to fix this function input...
                    try:
                        Cxx = fx.MatSolve(C,kVectxx,Dm,dt,FcEq,N,Dv,dx1,Cb)
                    except:
                        failed = 1
                        failcount += 1
                    stepErr = 0
                    if stepCheck == 1:
                        #Indentation level for res check
                        #Get the voltage for the half-step. 
                        Ex = 0.5*(Exx - E) + E
                        #Also compute rates @Ex
                        kVectx = quickRate(Ex)
                        #Compute the two-step
                        try:
                            Cx1 = fx.MatSolve(C,kVectx,0.5*Dm,0.5*dt,FcEq,N,Dv,dx1,Cb)
                            Cx2 = fx.MatSolve(C,kVectxx,0.5*Dm,0.5*dt,FcEq,N,Dv,dx1,Cb)
                        except:
                            failed = 1
                            failcount += 1
                        #Compute the deviations here
                        stepErr = compResError(Cxx, Cx2, N, minDev)
                    #Returning to out of res-check indentation - resolve loop.
                    if (stepErr > stepErrTol and failed == 0) or failcount > 5:
                        #There is some sort of problem here. 
                        if adaptStep == 0:
                            #Don't fix it. 
                            converged = 1
                        else: 
                            #Has it already failed?
                            if rescaled == 1 and abs(dE) == dE_min:
                                #Conditions that means it has failed before but already at minimum.
                                #print('Failed at voltage: ',str(np.round(E,5)))
                                converged = 1
                            else:
                                #Attempt to fix it by shrinking step size
                                dE = (dE/abs(dE))*(abs(dE) ** 1.25)
                                if abs(dE) < dE_min:
                                    dE = (dE/abs(dE))*dE_min
                                rescaled = 1
                    elif (stepErr < stepErrTol and failed == 0):
                        #If the error is low and it has not failed - OK to converge.
                        converged = 1
                        if rescaled == 0 and adaptStep == 1:
                            #If it was not rescaled and adaptive stepping is on, expand step (keep sign)
                            dE = (dE/abs(dE))*(abs(dE) ** 0.8)
                        #Prevent dE from going over the maximum while retaining sign
                        if abs(dE) > dE_max:
                                dE = (dE/abs(dE))*dE_max
                    else: 
                        #This means that it has failed but not >5 times - try restarting.
                        failed = 0
                #This level of indentation is outside the while loop - the result is converged. 
                #Store the voltage, current, and error of this last computation
                Evt_tmp[count] = Exx
                Istor_tmp[count] = F*Dv[0]*(Cxx[1] - Cxx[0])/dx1 # + more stuff
                stepErrStor_tmp[count] = stepErr
                #Store the maximum chem. prod
                ChemProd = C[2*N+2]
                if ChemProd > ChemProdMax:
                    ChemProdMax = ChemProd
                #if abs(E - 2.5) < 0.0001:
                #   Tool for debugging @ specific potential
                #    g = 2 + 2
                #
                #Create plots, if appropriate
                if count %dispFreq == 0 and showPlots == 1:
                    makePlot(Cxx,E,N,xgrid)
                #Update concentrations, count variable
                C = Cxx
                count += 1
                
                #Update dE/end condition, if necessary. 
                if Exx == Emax:
                #If at maximum - linear sweep is done, reverse direction
                    dE = -1*dE
                    if LSweep  == 1:
                        cyc_done = 1
                elif Exx == Emin:
                    #If at minimum - next Ei will be starting voltage, reverse direction
                    dE = -1*dE
                    min_done = 1
                elif Exx == Ei and min_done == 1:
                    #If minimum is done and back to start - cycle is complete
                    cyc_done = 1
                #Update voltage to the next timepoint
                E = Exx

            #Indentation level for 'cyc_done' loop. 
            #When each cycle complete - record final count. 
            cyc_inds[jj] = count
        #Indentation level for whole flexible-grid computation
        Evt = Evt_tmp[0:count]
        Istor = Istor_tmp[0:count] #
        stepErrStor = stepErrStor_tmp[0:count]
            
    ############################# ANALYSIS OF SOLUTION #######################
    #Generate CV plot
    if showCV == 1:
        MP.figure(2)    
        MP.plot(Evt,Istor)
        MP.xlabel('Voltage (V)')
        MP.ylabel('Current (A)')
        MP.show()
    #Generate resErr plot
    if stepPlot == 1:
        MP.figure(3)
        MP.plot(np.arange(0,count,1),stepErrStor)
        MP.show()
    #Compute second derivative of Istor through easy finite difference methods
    if secDeriv ==1:
       secondDerivativeAnalysis(Evt,Istor,secDerivBurn)
    #Geat peak potential (1-peak CV)
    PP = getPeakPotl(Evt,Istor,cyc_inds,cycles)    
    #Get analytical solutions from saveant
    savExact = saveant(E0v,Bsv,F,R,T,nu,Dv,kConst,Cb)
    #End timing, compute job runtime
    toc = time.time()
    elapsed = np.round(toc-tic,1)
    #Round and present max chemical product as mM
    SChemProdMax = np.round(ChemProdMax*1000,3)
    #Round and present max res-error
    maxStepErr = np.round(max(stepErrStor),6)
    #Print output string: input value, peak potentials, exact E-kinetic, exact EC, max chem prod, max error, timing 
    print(value,PP,np.round(savExact[0],5),np.round(savExact[2],5),SChemProdMax,maxStepErr[0],elapsed)
 
def getPeakPotl(Evt,Istor,cyc_inds,cycles):
    #Get peak potentials (assumes 1 peak)
    #From first pass - maximum potential 
    Istor1 = Istor[0:cyc_inds[0]+1]
    dex1 = np.where(Istor1 == max(Istor1))
    PP = str(np.round(Evt[dex1][0],5))
    #From 2 -> nth passes
    if cycles > 1:
        for jj in np.arange(0,cycles-1,1):
            cyc_start = cyc_inds[jj] 
            cyc_end = cyc_inds[jj+1]+1
            Istorjj = Istor[cyc_start:cyc_end]
            Evtjj = Evt[cyc_start:cyc_end]
            dexjj = np.where(Istorjj == max(Istorjj))
            #Simply adds to the peak potential string. 
            PP = PP + ' ' + str(np.round(Evtjj[dexjj][0],5))
    return PP
    
def saveant(E0v,Bsv,F,R,T,nu,Dv,kConst,Cb):
    #Analytical solutions from Saveant for kinetic & equilibrium control
    #Peak potentials - kinetic, equilibrium, "EC" mechanism
    APP_ekin = E0v[0] + 0.78*R*T/((1 - Bsv[0])*F) - R*T*np.log(kConst[0]*np.sqrt(R*T/((1 - Bsv[0])*F*nu*Dv[0])))/((1 - Bsv[0])*F)
    APP_eeq = E0v[0] + 1.11*R*T/F
    APP_ec = 0
    if kConst[2] > 0:
        APP_ec = E0v[0] + 0.78*R*T/(F) - (R*T/(2*F))*np.log(R*T*kConst[2]/(F*nu))
    #Peak currents - kinetic, equilibrium (check these)
    APC_ekin = 0.496*F*Cb[0]*np.sqrt(Dv[0])*np.sqrt((1-Bsv[0])*F*nu/(R*T))
    APC_eq = 0.446*F*Cb[0]*np.sqrt(Dv[0])*np.sqrt((1-Bsv[0])*F*nu/(R*T))
    return np.array([APP_ekin,APP_eeq,APP_ec,APC_ekin,APC_eq])
    
def secondDerivativeAnalysis(Evt,Istor,secDerivBurn):
    nl = len(Istor)
    I2deriv = Istor[0:nl-2] + -2*Istor[1:nl-1] + Istor[2:nl]
    redEvt = Evt[1:nl-1]
    I2deriv[0:secDerivBurn] = 0 #screening for low alpha values
    dexlow = np.where(I2deriv == min(I2deriv))
    dexlow = dexlow[0][0]
    dexhi = np.where(I2deriv == max(I2deriv))
    dexhi = dexhi[0][0]
    nearzerI2 = I2deriv[dexhi:dexlow]
    nearzerEvt = redEvt[dexhi:dexlow]
    Einf = nearzerEvt[np.where(abs(nearzerI2) == min(abs(nearzerI2)))];
    Einf = Einf[0]
    MP.figure(4)
    MP.plot(redEvt,I2deriv)
    MP.plot(np.array([redEvt[dexlow],redEvt[dexhi]]),np.array([0,0]))
    MP.plot(Einf,0,'ok')
    MP.show()
    #Need to add a print statement

def compResError(C1,C2,N,minDev):
    deviations = C1[[0,N+1,2*N+2,3*N+3,4*N+4]] - C2[[0,N+1,2*N+2,3*N+3,4*N+4]]
    mdev = max(abs(deviations))
    if mdev > minDev:
        #Deviation is not tiny, record the res err. 
        ddv = np.where(abs(deviations) == mdev)
        ddv = ddv[0][0]
        return mdev/abs(C2[ddv*N+ddv])
    else:
        return 0
    
def getConstants(F,R,T,optn):
    DtGv = np.array([10.0,15.0,12.0,10.0,45.0]) # kcal/mol, Delta G for rxns 1,2,3,4,5
    if optn == 0: #rate constants
        Kv = np.exp(-DtGv/(R*T)) # Eq. constant, rxns 1->5
        #Manually reset Eq. constants if desired
        Kv[1] = 3
        #For surface reactions - maximum value is 10^2 m/s or 10^4 cm/s.
        #For kinetic reactions - forward/backward split determined by equilibrium info. 
        #For concerted reaction - forward/backward split influenced in symmetry coefficient. 
        #Set rxns with a decimal value or a power (recommended)
        k1s = 10 ** (0) #m/s - Surf., rxn 1
        k2f = 10 ** (0) #1/s - Homog., first order, rxn 2 
        k2b = k2f/Kv[1] # 1/(sM) - Reverse homog., second order, rxn 2
        k3s = 0.0 #m/s - Surf., rxn 3
        k4f  = 0.0 #(1/(sM))*(M) - Homog., Pseudo-first order w/ const. (H2O), rxn 4
        k4b = k4f/Kv[3] #Reverse homog., Pseudo-first order w/ const. 
        k5s = 0.0 #m/s Surf., concerted rxn 5
        #Manually reset reverse rxns if desired
        #k2b = 0
        k4b = 0
        return np.array([k1s,k2f,k2b,k3s,k4f,k4b,k5s])
    elif optn == 1: #Potentials
        #Import electrochemical information. All reactions assumed to have single electron transfer. 
        #Reversible (oxidation) potentials for the 3 echem reactions - compute from thermochemistry. 
        E01 = DtGv[0]*4184/F #V
        E03 = DtGv[2]*4184/F #V
        E05 = DtGv[4]*4184/F #V
        #Manually reset reversible potentials
        E01 = 2.5 #V
        E03 = 1.0 #V
        E05 = 3.0 #V
        return np.array([E01,E03,E05])
    elif optn == 2: #Symmetry factors
        #Import symmetry factors - currently static symmetry factors only. 
        Bs1 = 0.5
        Bs3 = 0.5
        Bs5 = 0.25
        return np.array([Bs1,Bs3,Bs5])
    elif optn == 3: #Diffusion coefficients
        #Import diffusion coefficients for species i (as Di)
        Dx = 10 ** (-9) #m^2/s - Diffusion coefficient. (Mult. by 10^4 to get cm^2/s).
        Dy = 10 ** (-9) #Input diffusion coeffs for other species. 
        Dw = 10 ** (-9)
        Du = 10 ** (-9)
        Dz = 10 ** (-9)
        return np.array([Dx,Dy,Dw,Du,Dz])
    else: #Equilibrium array
        #Each toggle is a reaction ri: [(r1),(r2),(r3),(r4)]. (1) uses eq., (0) uses kinetic equations
        return np.array([0,0,0,0])

#def genGrid(,option):
    

def computeRates(E,VT,E0v,Bsv,kconv):
    k1f = kconv[0]*np.exp((E-E0v[0])*VT*(1 - Bsv[0]))
    k1b = kconv[0]*np.exp(-(E-E0v[0])*VT*Bsv[0])
    k3f = kconv[3]*np.exp((E-E0v[1])*VT*(1 - Bsv[1]))
    k3b = kconv[3]*np.exp(-(E-E0v[1])*VT*Bsv[1])
    k5f = kconv[6]*np.exp((E-E0v[2])*VT*(1 - Bsv[2]))
    k5b = kconv[6]*np.exp(-(E-E0v[2])*VT*Bsv[2])
    return np.array([k1f,k1b,kconv[1],kconv[2],k3f,k3b,kconv[4],kconv[5],k5f,k5b])

def makePlot(C,E,N,xgrid):
    MP.figure(1)
    #Overindex for 'landing' point
    #Plot concentration profiles
    MP.plot(xgrid,C[0:(N+1)],'-b') #X
    MP.plot(xgrid,C[(N+1):(2*N+2)],'-r') #Y
    MP.plot(xgrid,C[(2*N+2):(3*N+3)],'-k') #W
    MP.plot(xgrid,C[(3*N+3):(4*N+4)],'-g') #U
    MP.plot(xgrid,C[(4*N+4)::],'-y') #Z
    #Plot aesthetics
    MP.title(['Voltage = ',np.round(E,4),' V'])
    MP.ylabel('Conc. (mM)')
    MP.xlabel('Distance from electrode (m)')
    MP.show()
    MP.close(1)  

#Run single-value experiment
simCV(0.0)
#Establish value vector of specified #s
#valVect = np.array([0.005,0.0025,0.001,0.0005,0.00025])
#valVect = np.array([0.0025,0.001,0.0005,0.00025,0.0001,0.00005,0.000025,0. ])
#Establish value vector of range of #s
#valVect = np.arange(-4.0,-6.0,-1.0)
#Run experiments over many values
#for vv in valVect:
#    simCV(vv)