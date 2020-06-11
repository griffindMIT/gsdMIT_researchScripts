# -*- coding: utf-8 -*-

#Import packages
import numpy as np 
import matplotlib.pyplot as MPL
import scipy.sparse as spar
from scipy.sparse import linalg as sparL
#from scipy import optimize as spopt

def MatSolve(C0,EqTog,kVect,N,Dm,D,Cb,dx1,dt):
    #Generate matrix
    Mat = makeMat(EqTog,kVect,N,Dm,D,Cb,dx1,dt,C0)
    #Correct the "RHS" - the initial state vector
    evals,evects = sparL.eigs(Mat)
    
    
    Ceql = modC(C0,Cb,N,kVect,dx1,dt,D,EqTog)
    #Solve sparse matrix for new C
    C = sparL.spsolve(Mat,Ceql)
    return C

def modC(C0,Cb,N,kVect,dx1,dt,D,EqTog):
    Ceql = C0.reshape(len(C0),1);
    #All "last entries" = bulk concentrations.
    Ceql[[N,2*N+1,3*N+2,4*N+3,5*N+4]] = np.array([Cb[0],Cb[1],Cb[2],Cb[3],Cb[4]]).reshape(5,1)
    #Extract & nondimensionalize rate constants for boundary condition assignments
    k1f = kVect[0]*dx1/D
    k1b = kVect[1]*dx1/D
    k2f = kVect[2]
    k2b = kVect[3]
    k3f = kVect[4]*dx1/D
    k3b = kVect[5]*dx1/D
    k4f = kVect[6]
    k4b = kVect[7]
    k5f = kVect[8]*dx1/D
    k5b = kVect[9]*dx1/D
    #Extract previous timepoint data for U0 (u0) and W0 (w0)
    w0 = C0[2*N+2]
    u0 = C0[3*N+3]
    #Assessment of equilibrium happens here (coming soon)
    #Turn off equilibrated rxns if the rate constants are identically zero
    if k1f == 0:
        EqTog[0] = 0
    if k3f == 0:
        EqTog[1] = 0
    #Begin modification of surface BCs. 
    if k5f == 0:
        #No concerted reaction - all surface BCs are = 0. 
        Ceql[[0,N+1,2*N+2,3*N+3,4*N+4]] = np.array([0,0,0,0,0]).reshape(5,1)
    else:
        #Concerted mechanism is present. Variable surface BCs. 
        #X boundary condition has 3 options: 
        if k1f == 0 or EqTog[0] == 1:
            if abs(k5b) >0:
                #k5b is present, kinetic or equilibrium control (k5b scaling)
                Ceql[0] = -w0*u0
            else:
                #k5b is zero, no bimolecular back reaction. 
                Ceql[0] = 0
        else:
            #Presence of reaction 1 necessitates k1b scaling.  
            Ceql[0] = -(k5f/k1b)*(w0*u0)
        #Y boundary condition always equal to zero
        Ceql[N+1] = 0
        #W boundary condition has three options
        if k3f == 0:
            #No interchange reaction, scaling for bimolecular back rxn only
            Ceql[2*N+2] = (k5b/k5f)*(u0*w0)
        elif EqTog[1] == 1:
            #"Doubled" back reaction to account for eq. W & U flux. 
            Ceql[2*N+2] = 2*(k5b/k5f)*(u0*w0)
        else:
            #Presence of kinetic IC reaction introduces k3b scaling
            Ceql[2*N+2] = k5b/k3b*(u0*w0)
        #U boundary condition always equal to zero
        Ceql[3*N+3] = 0
        #Z boundary condition always equal to zero
        Ceql[4*N+4] = 0   
    #Start modifying central points. 
    #Grab u, w data
    upv = C0[2*N+3:3*N+2].reshape((N-1),1)
    wpv = C0[3*N+4:4*N+3].reshape((N-1),1)
    #Indexes for start/end of center points
    specLow = np.array([1,N+2,2*N+3,3*N+4,4*N+5])
    specHigh = np.array([N-1,2*N,3*N+1,4*N+2,5*N+3])
    #Modification #1: Add -k2b*w(i)*u(i)*dt to central Y points. 
    modifY = -1*k2b*dt*np.multiply(upv,wpv)
    Ceql[specLow[1]:(specHigh[1]+1)] = Ceql[specLow[1]:(specHigh[1]+1)] + modifY
    #Modification #2: Multiply W central points by (1 + k2b*dt*u(i))
    liner = np.arange(1,N)/np.arange(1,N)
    liner = liner.reshape(N-1,1)
    modifW = (liner + k2b*dt*upv)
    Ceql[specLow[2]:(specHigh[2]+1)] = np.multiply(Ceql[specLow[2]:(specHigh[2]+1)],modifW)
    #Modification #3: Multiply U central points by (1 + k2b*dt*w(i))
    modifU = (liner + k2b*dt*wpv)
    Ceql[specLow[3]:(specHigh[3]+1)] = np.multiply(Ceql[specLow[2]:(specHigh[2]+1)],modifU)
    return Ceql

def makeMat(EqTog,kVect,N,Dm,D,Cb,dx1,dt,C0):
    #Breakout and non-dimensionalize rate constants (tedious... but useful)
    k1f = kVect[0]*dx1/D
    k1b = kVect[1]*dx1/D
    k2f = kVect[2]
    k2b = kVect[3]*dx1/D
    k3f = kVect[4]*dx1/D
    k3b = kVect[5]*dx1/D
    k4f = kVect[6]
    k4b = kVect[7]
    k5f = kVect[8]*dx1/D
    k5b = kVect[9]*dx1/D
    #Extract previous timepoint data for U0 (u0) and W0 (w0)
    w0 = C0[2*N+2]
    u0 = C0[3*N+3]
    
    #Assign # of species in the mechanism
    Nspec = 5 
    #Assessment of equilibrium happens here (coming soon)
    #Turn off equilibrated rxns if the rate constants are identically zero
    if k1f == 0:
        EqTog[0] = 0
    if k3f == 0:
        EqTog[1] = 0
    
    #Compute total # of entries... accounting going on below...
    ############################ ACCOUNTING FOR BC ENTRIES ###################
    #Initialize 'total entry' count tracker, entTot. 
    entTot = 0
    #Add 1 entry for each 'fixed concentration' BC
    entTot = entTot + 5
    #Compute BC entries
    if k5f == 0:
        #Assumes that if k5f = 0, k1f =/= 0
        if EqTog[0] == 1:
            #Use Nernstian equation for X
            entTot = entTot + 2
        else:
            #Use kinetic eqn for X
            entTot = entTot + 3
        #Regardless of equilibrium, use equal-flux condition for Y
        entTot = entTot + 4
        if EqTog[1] == 1:
            #Use Nernstian equation for W
            entTot = entTot + 2
            #Use flux equality condition for U
            entTot = entTot + 4
        elif k3f == 0: 
            #No reaction 3 occurs, use a no-flux condition for W
            entTot = entTot + 2
            #Use a no-flux condition for U
            entTot = entTot + 2
        else:
            #Use kinetic eqn for W
            entTot = entTot + 3
            #Use flux equality condition for U
            entTot = entTot + 4
            
    else:
        #Since k5f =/= 0, k1f may be zero, equilibriated, or kinetic
        if k1f == 0:
            if abs(k5b) > 0:
                #Use kinetic control for concerted only (X BC)
                entTot = entTot + 4
            else:
                #Use concerted kinetic for X that doesn't explode
                entTot = entTot + 2
            #Use a no-flux condition for Y
            entTot = entTot + 2            
        elif EqTog[0] == 1:
            if abs(k5b) > 0:
                #Use derived partial-equilibrium expression (X BC)
                entTot = entTot + 6
            else:
                #Use the version that doesn't explode
                entTot = entTot + 4
            #Use a Nernst relation for Y
            entTot = entTot + 2
        else: 
            #Use kinetic control for both reactions 1 and 2 (X BC)
            entTot = entTot + 5
            #Use kinetic control for Y reaction as well
            entTot = entTot + 3
            
        if k3f == 0:
            #Use only concerted kinetic relationship for W
            entTot = entTot + 4
            #Use flux equality condition for U
            entTot = entTot + 4
        elif EqTog[1] == 1:
            #Use derived kinetic relationship for W
            entTot = entTot + 5
            #Use Nernst relationship for U
            entTot = entTot + 2
        else: 
            #Use kinetic control for all reactions for W
            entTot = entTot + 4
            #Use... flux balance for U
            entTot = entTot + 8
    #Regardless of all else, use a no-flux condition for Z
    entTot = entTot + 2
    
    ##################### ACCOUNTING FOR CENTER POINT ENTRIES ###############
    #Number of center points that are actually present
    ctct = (N-1) #???
    #Every species needs at least 3 entries over (N-1) points 
    centerDiffs = 3*Nspec*ctct
    #X requires no additional entries. 
    #Y has a self-catalytic term, plus two back-reaction terms, need +2etr/pt
    centerYPts = 2*ctct
    #W has a self-cat term, plus one gen and one back-rxn term, need +2etr/pt
    centerWPts = 2*ctct
    #U has a self-cat term, plus two gen and one back-rxn term, need +3etr/pt
    centerUPts = 3*ctct
    #Z has a self-cat term, plus one gen term, need +1etr/pt
    centerZPts = ctct
    #Add all center points, add to total # of entries
    centerTots = centerDiffs + centerYPts + centerWPts + centerUPts + centerZPts   
    entTot = entTot + centerTots
    
    ##################### ASSIGNING BC TO MATRIX ###########################
    #Generate storage space fo entry value, row, column index
    entVal = 0.0*np.arange(0,entTot).reshape(entTot,1)
    rowID = 0.0*np.arange(0,entTot).reshape(entTot,1)
    colID = 0.0*np.arange(0,entTot).reshape(entTot,1)
    #entVal is the multiplication factor for the matrix entry (-1, D, 1/x, etc.)
    #"Row ID" refers to the position of the equation. All eqns share a row ID.
    #"Col ID" refers to the position of the concentration in the C vector. 
    #Col IDs for X0, Y0, W0, U0, Z0 = 0, N+1, 2*N+2, 3*N+3, 4*N+4
    #Col IDs for XN, YN, WN, UN, ZN = N, 2*N+1, 3*N+2, 4*N+3, 5*N+4
    #Bimolecular reactions may depend on the previous timepoint's concentration data
    #This is denoted as lowercase (e.g., u0 is U0 in the input conc (C0)). 
    
    #Assign toggled BCs
    #Eqns 0 through 4: Bulk concentration BCs
    #AN = Ab, where A = {X,Y,W,U,Z}, Ab = bulk, row@AN
    #Note carefully that vector[index1:index2] retrieves entries index1 to (index2-1).
    entVal[0:5] = np.array([1,1,1,1,1]).reshape(Nspec,1)
    rowID[0:5] =np.array([N,2*N+1,3*N+2,4*N+3,5*N+4]).reshape(Nspec,1)
    colID[0:5] = np.array([N,2*N+1,3*N+2,4*N+3,5*N+4]).reshape(Nspec,1)
    #Assign variable BCs
    #Start incrementer - keeps track of eqn # for the addition of more elements.
    inc = 5
    #Following through same logic path as above, now implementing equations. 
    if k5f ==0:
        if EqTog[0] ==1:
            #Nernst for X: (k1f/k1b)*X0 - Y0 = 0, row@X0
            entVal[inc:(inc+2)] = np.array([k1f/k1b,-1]).reshape(2,1)
            rowID[inc:(inc+2)] = np.array([0,0]).reshape(2,1)
            colID[inc:(inc+2)] = np.array([0,N+1]).reshape(2,1)
            inc = inc + 2
        else:
            #Kinetic for X: (1/k1b + k1f/k1b)X0 - (1/k1b)X1 - Y0 = 0, row@X0
            entVal[inc:(inc+3)] = np.array([((1/k1b) + (k1f/k1b)),(-1/k1b),-1]).reshape(3,1)
            rowID[inc:(inc+3)] = np.array([0,0,0]).reshape(3,1)
            colID[inc:(inc+3)] = np.array([0,1,N+1]).reshape(3,1)
            inc = inc + 3            
        #Equal-flux for Y: X1 - X0 + Y1 - Y0 = 0, row@Y0
        entVal[inc:(inc+4)] = np.array([1,-1,1,-1]).reshape(4,1)
        rowID[inc:(inc+4)] = np.array([N+1,N+1,N+1,N+1]).reshape(4,1)
        colID[inc:(inc+4)] = np.array([1,0,N+2,N+1]).reshape(4,1)
        inc = inc + 4
        if EqTog[1] ==1:
            #Nernst for W: (k3f/k3b)W0 - U0 = 0, row@W0
            entVal[inc:(inc+2)] = np.array([(k3f/k3b),-1]).reshape(2,1)
            rowID[inc:(inc+2)] = np.array([2*N+2,2*N+2]).reshape(2,1)
            colID[inc:(inc+2)] = np.array([2*N+2,3*N+3]).reshape(2,1)
            inc = inc + 2
            #Equal-flux for U: W1 - W0 + U1 - U0 = 0, row@U0
            entVal[inc:(inc+4)] = np.array([1,-1,1,-1]).reshape(4,1) 
            rowID[inc:(inc+4)] = np.array([(3*N+3),(3*N+3),(3*N+3),(3*N+3)]).reshape(4,1) 
            colID[inc:(inc+4)] = np.array([2*N+3,2*N+2,3*N+4,3*N+3]).reshape(4,1)
            inc = inc + 4
        elif k3f ==0:
            #No flux for W: W1 - W0 = 0, row@W0
            entVal[inc:(inc+2)] = np.array([1,-1]).reshape(2,1)
            rowID[inc:(inc+2)] = np.array([2*N+2,2*N+2]).reshape(2,1)
            colID[inc:(inc+2)] = np.array([2*N+3,2*N+2]).reshape(2,1)
            inc = inc + 2
            #No flux for U: W1 - W0 = 0, row@U0
            entVal[inc:(inc+2)] = np.array([1,-1]).reshape(2,1)
            rowID[inc:(inc+2)] = np.array([3*N+3,3*N+3]).reshape(2,1)
            colID[inc:(inc+2)] = np.array([3*N+4,3*N+3]).reshape(2,1)
            inc = inc + 2
        else:
            #Kinetic for W: (1/k3b + k3f/k3b)W0 - (1/k3b)W1 - U0 = 0, row@W0
            entVal[inc:(inc+3)] = np.array([((1/k3b)+(k3f/k3b)),(-1/k3b),-1]).reshape(3,1)
            rowID[inc:(inc+3)] = np.array([2*N+2,2*N+2,2*N+2]).reshape(3,1)
            colID[inc:(inc+3)] = np.array([2*N+2,2*N+3,3*N+3]).reshape(3,1)
            inc = inc + 3
            #Equal-flux for U: W1- W0 + U1 - U0 = 0, row@U0
            entVal[inc:(inc+4)] = np.array([1,-1,1,-1]).reshape(4,1) 
            rowID[inc:(inc+4)] = np.array([(3*N+3),(3*N+3),(3*N+3),(3*N+3)]).reshape(4,1) 
            colID[inc:(inc+4)] = np.array([2*N+3,2*N+2,3*N+4,3*N+3]).reshape(4,1)
            inc = inc + 4
    else:
        if k1f ==0:
            if abs(k5b) > 0:
                #kinetic for X: (1/k5b + k5f/k5b)X0 - (1/k5b)X1 - (u0)W0 - (w0)U0 = - (w0u0), row@X0
                entVal[inc:(inc+4)] = np.array([((1/k5b) + (k5f/k5b)),(-1/k5b),(-u0),(-w0)]).reshape(4,1)
                rowID[inc:(inc+4)] = np.array([0,0,0,0]).reshape(4,1)
                colID[inc:(inc+4)] = np.array([0,1,2*N+2,3*N+3]).reshape(4,1)
                inc = inc + 4
            else:
                #Non-exploding scaling: (1 + k5f)X0 - X1 = 0
                entVal[inc:(inc+2)] = np.array([(1+k5f),-1]).reshape(2,1)
                rowID[inc:(inc+2)] = np.array([0,0]).reshape(2,1)
                colID[inc:(inc+2)] = np.array([0,1]).reshape(2,1)
                inc = inc + 2
            #No flux for Y: Y1 - Y0 = 0 , row@Y0
            entVal[inc:(inc+2)] = np.array([1,-1]).reshape(2,1)
            rowID[inc:(inc+2)] = np.array([N+1,N+1]).reshape(2,1)
            colID[inc:(inc+2)] = np.array([N+2,N+1]).reshape(2,1)
            inc = inc + 2
        elif EqTog[0] ==1:
            if abs(k5b) > 0:
                #Partial kin. for X: (1/k5b + k5f/k5b)X0 - (1/k5b)X1 - (1/k5b)Y1 + (1/k5b)Y0 ... 
                #...  - (u0)W0 - (w0)U0 = -(w0u0), row@X0
                entVal[inc:(inc+3)] = np.array([((1/k5b)+(k5f/k5b)),(-1/k5b),(-1/k5b)]).reshape(3,1)
                entVal[(inc+3):(inc+6)] = np.array([(1/k5b),(-u0),(-w0)]).reshape(3,1)
                rowID[inc:(inc+6)] = np.array([0,0,0,0,0,0]).reshape(6,1)
                colID[inc:(inc+6)] = np.array([0,1,N+2,N+1,2*N+2,3*N+3]).reshape(6,1)
                inc = inc + 6
            else:
                #Non-exploding scaling: (1 + k5f)X0 - X1 - Y1 + Y0 = 0
                entVal[inc:(inc+4)] = np.array([(1 + k5f),-1,-1,1]).reshape(4,1)
                rowID[inc:(inc+4)] = np.array([0,0,0,0]).reshape(4,1)
                colID[inc:(inc+4)] = np.array([0,1,N+2,N+1]).reshape(4,1)
                inc = inc + 4
            #Nernst for Y: (k1f/k1b)X0 - Y0 = 0 row@Y0
            entVal[inc:(inc+2)] = np.array([(k1f/k1b),-1]).reshape(2,1)
            rowID[inc:(inc+2)] = np.array([N+1,N+1]).reshape(2,1)
            colID[inc:(inc+2)] = np.array([0,N+1]).reshape(2,1)
            inc = inc + 2
        else:
            #kinetic for X: (1/k1b + k1f/k1b + k5f/k1b)X0 - (1/k1b)X1 - Y0 ...
            #... - (u0*k5b/kb1)W0 - (w0*k5b/kb1)*U0 = -(k5b/kb1)(u0w0), row@X0
            entVal[inc:(inc+2)] = np.array([((1/k1b)+(k1f/k1b)+(k5f/k1b)),(-1/k1b)]).reshape(2,1)
            entVal[(inc+2):(inc+5)] = np.array([-1,(-u0*k5b/k1b),-(w0*k5b/k1b)]).reshape(3,1)
            rowID[inc:(inc+5)] = np.array([0,0,0,0,0]).reshape(5,1)
            colID[inc:(inc+5)] = np.array([0,1,N+1,2*N+2,3*N+3]).reshape(5,1)
            inc = inc + 5
            #kinetic for Y: (1/k1f + k1b/k1f)Y0 - (1/k1f)Y1 - X0 = 0, row@Y0
            entVal[inc:(inc+3)] = np.array([((1/k1f)+(k1b/k1f)),(-1/k1f),-1]).reshape(3,1)
            rowID[inc:(inc+3)] = np.array([N+1,N+1,N+1]).reshape(3,1)
            colID[inc:(inc+3)] = np.array([N+1,N+2,0]).reshape(3,1)
            inc = inc + 3
        if k3f ==0:
            #Kinetic for W: (1/k5f + u0*k5b/k5f)W0 - (1/k5f)W1 - X0 + (w0*k5b/k5f)U0 = (k5b/k5f)(u0w0), row@W0
            entVal[inc:(inc+2)] = np.array([(1/k5f + u0*k5b/k5f),(-1/k5f)]).reshape(2,1)
            entVal[(inc+2):(inc+4)] = np.array([-1,(w0*k5b/k5f)]).reshape(2,1)
            rowID[inc:(inc+4)] = np.array([2*N+2,2*N+2,2*N+2,2*N+2]).reshape(4,1)
            colID[inc:(inc+4)] = np.array([2*N+2,2*N+3,0,3*N+3]).reshape(4,1)
            inc = inc + 4
            #Equal flux for U: W1 - W0 + U0 - U1 = 0  row@U0
            entVal[inc:(inc+4)]= np.array([1,-1,1,-1]).reshape(4,1)
            rowID[inc:(inc+4)]= np.array([3*N+3,3*N+3,3*N+3,3*N+3]).reshape(4,1)
            colID[inc:(inc+4)]= np.array([2*N+3,2*N+2,3*N+3,3*N+4]).reshape(4,1)
            inc = inc + 4
        elif EqTog[1] == 1:
            #Kinetic for W: (1/k5f + 2*u0*k5b/k5f)W0 - (1/k5f)W1 - 2X0 + ...
            #... +(1/k5f + 2*w0*k5b/k5f)U0 - (1/k5f)U1 = 2k5b/k5f(u0w0) row@W0
            entVal[inc:(inc+3)] = np.array([((1/k5f)+(2*u0*k5b/k5f)),(-1/k5f),-2]).reshape(3,1)
            entVal[(inc+3):(inc+5)] = np.array([((1/k5f)+(2*w0*k5b/k5f)),(-1/k5f)]).reshape(2,1) 
            rowID[inc:(inc+5)] = np.array([2*N+2,2*N+2,2*N+2,2*N+2,2*N+2]).reshape(5,1)
            colID[inc:(inc+5)] = np.array([2*N+2,2*N+3,0,3*N+3,3*N+4]).reshape(5,1)
            inc = inc + 5
            #Nernst for U: (k3f/k3b)W0 - U0 = 0, row@U0
            entVal[inc:(inc+2)] = np.array([(k3f/k3b),-1]).reshape(2,1) 
            rowID[inc:(inc+2)] = np.array([3*N+3,3*N+3]).reshape(2,1)
            colID[inc:(inc+2)] = np.array([2*N+2,3*N+3]).reshape(2,1)
            inc = inc + 2
        else:
            #Kinetic for W: (1/k3b + u0*k5b/k3b + k3f/k3b)W0 - (1/k3b)W1 ...
            #... - (k5f/k3b)X0 + (w0*k5b/k3b - 1)U0 = k5b/k3b(u0w0) row@W0
            entVal[inc:(inc+2)] = np.array([((1/k3b)+(u0*k5b/k3b)+(k3f/k3b)),(-1/k3b)]).reshape(2,1)
            entVal[(inc+2):(inc+4)] = np.array([(-k5f/k3b),((w0*k5b/k3b) - 1)]).reshape(2,1)
            rowID[inc:(inc+4)] = np.array([2*N+2,2*N+2,2*N+2,2*N+2]).reshape(4,1)
            colID[inc:(inc+4)] = np.array([2*N+2,2*N+3,0,3*N+3]).reshape(4,1)
            inc = inc + 4
            #Overall flux balance for U: X1-X0+Y1-Y0-0.5U1+0.5U0-0.5W1+0.5W0=0 row@U0
            entVal[inc:(inc+8)] = np.array([1,-1,1,-1,-0.5,0.5,-0.5,0.5]).reshape(8,1)
            rowID[inc:(inc+8)] = np.array([3*N+3,3*N+3,3*N+3,3*N+3,3*N+3,3*N+3,3*N+3,3*N+3]).reshape(8,1)
            colID[inc:(inc+8)] = np.array([1,0,N+2,N+1,3*N+4,3*N+3,2*N+3,2*N+2]).reshape(8,1)
            inc = inc + 8
    
    #Assign no-flux for Z: Z1 - Z0 = 0, row@Z0
    entVal[inc:(inc+2)] = np.array([1,-1]).reshape(2,1)
    rowID[inc:(inc+2)] = np.array([4*N+4,4*N+4]).reshape(2,1)
    colID[inc:(inc+2)] = np.array([4*N+5,4*N+4]).reshape(2,1)
    inc = inc + 2
          
    #For peeping at boundary conditions      
    #entVal = entVal.flatten()
    #rowID = rowID.flatten()
    #colID = colID.flatten()
    #mat = spar.coo_matrix((entVal,(rowID,colID)),shape=((5*N+5),(5*N+5)))
    #mat = mat.tocsr()
    #MPL.spy(mat,markersize=3)
    #MPL.show()
    
    #Allocate 'central points'    
    #Limits for species: (after surface BC, before bulk BC). lowercase = previous timepoint concs
    #(The bracketing on the D's refers to math notation not coding... subtract one for code)
        #X, ind = 1 to N-1
            #Dm[3]X(i) - Dm[2]X(i+1) - Dm[1]X(i-1) = x(i)
        #Y, ind =  N+2 to 2*N
            #(Dm[3] + k2f*dt)Y(i) - Dm[2]Y(i+1) - Dm[1]Y(i-1) - k2b*dt*w(i)U(i) - k2b*dt*u(i)W(i)= ...
            #... = y(i) - k2b*dt*u(i)w(i)
        #W, ind = 2*N+3 to 3*N+1
            #(Dm[3] + k2b*dt*u(i))W(i) - Dm[2]W(i+1) - Dm[1]W(i-1) - k2f*dt*Y(i) + k2b*dt*w(i)U(i) = ...
            # ... = w(i)(1 + k2b*dt*u(i))
        #U, ind = 3*N+4 to 4*N+2
            #(Dm[3] + k2b*dt*w(i) + k4f*dt)U(i) - Dm[2]U(i+1) - Dm[1]U(i-1) - k2f*dt*Y(i) ...
            # ... + k2b*dt*u(i)*W(i) - k4b*dt*Z(i) = u(i)(1 + k2b*dt*w(i))
        #Z, ind = 4*N+5 to 5*N+3
            #(Dm[3] + k4b*dt)Z(i) - Dm[2]Z(i+1) - Dm[1]Z(i-1) - k1f*dt*u(i) = z(i)
    
    #Begin assignment according to the above equations (hardcoded switches)
    #Indexer for what species is currently being treated
    specCounter = np.array([0,1,2,3,4])
    #Indexes for start/end of center points
    specLow = np.array([1,N+2,2*N+3,3*N+4,4*N+5])
    specHigh = np.array([N-1,2*N,3*N+1,4*N+2,5*N+3])
    #linerCount = # of additional terms by species
    linerCountVect = [0,1,1,2,0]
    #Get the previous timepoint data for U, W (useful in many rxns)
    upv = C0[2*N+3:3*N+2].reshape((N-1),1)
    wpv = C0[3*N+4:4*N+3].reshape((N-1),1)
    for ss in specCounter:
        #get bounds
        hi = specHigh[ss]
        lo = specLow[ss]
        #Create vector of "i" values
        rowVector = np.arange(lo,hi+1).reshape((hi+1-lo),1)
        #Add (i+1) term
        entVal[inc:(inc+(N-1))] = -Dm[1:N,1].reshape(N-1,1)
        rowID[inc:(inc+(N-1))] = rowVector
        colID[inc:(inc+(N-1))] = rowVector + 1
        inc = inc + (N-1)
        #Add (i-1) term
        entVal[inc:(inc+(N-1))] = -Dm[1:N,0].reshape(N-1,1)
        rowID[inc:(inc+(N-1))] = rowVector 
        colID[inc:(inc+(N-1))] = rowVector - 1
        inc = inc + (N-1)
        #Compute catalytic term by assigning ACF (autocatalytic factor)
        if ss == 1: #Species Y
            ACF = k2f*dt
        elif ss == 2: #Species W
            ACF = k2b*dt*upv
        elif ss == 3: #Species U
            ACF = k2b*dt*wpv + k4f*dt
        elif ss == 4: #Species Z
            ACF = k4b*dt
        else: #Species X
            ACF = 0
        #Add (i) term
        entVal[inc:(inc+(N-1))] = (Dm[1:N,2].reshape(N-1,1) + ACF)
        rowID[inc:(inc+(N-1))] = rowVector
        colID[inc:(inc+(N-1))] = rowVector
        inc = inc + (N-1)
        #Add additional terms as required by the species (v. hardcoded)
        if ss != 0:
            #Add simple indexer for adding liners
            linerCount = np.arange(0,linerCountVect[ss]+1)
            for L in linerCount:
                liner = np.arange(1,N)/np.arange(1,N)
                liner = liner.reshape(N-1,1)
                rowID[inc:(inc+(N-1))] = rowVector
                if ss == 1: #Y, 2 terms
                    if L == 0:
                        #First term addition: -k2b*dt*w(i), on U(i) cols. 
                        #Liner use is not necessary for ones using old data
                        entVal[inc:(inc+(N-1))] = -k2b*dt*wpv
                        colID[inc:(inc+(N-1))] = np.arange(specLow[3],(specHigh[3]+1)).reshape(N-1,1)
                        inc = inc + (N-1)
                    else:
                        #Second term addition: -k2b*dt*u(i) on W(i) cols. 
                        entVal[inc:(inc+(N-1))] = -k2b*dt*upv
                        colID[inc:(inc+(N-1))] = np.arange(specLow[2],(specHigh[2]+1)).reshape(N-1,1)
                        inc = inc + (N-1)
                elif ss == 2: #W, 2 terms
                    if L == 0:
                        #First term addition: -k2f*dt, on Y(i) cols. 
                        entVal[inc:(inc+(N-1))] = -k2f*dt*liner
                        colID[inc:(inc+(N-1))] = np.arange(specLow[1],(specHigh[1]+1)).reshape(N-1,1)
                        inc = inc + (N-1)
                    else:
                        #Second term addition: k2b*dt*w(i), on U(i) cols. 
                        entVal[inc:(inc+(N-1))] = k2b*dt*wpv
                        colID[inc:(inc+(N-1))] = np.arange(specLow[3],(specHigh[3]+1)).reshape(N-1,1)
                        inc = inc + (N-1)
                elif ss == 3: #U, 3 terms
                    if L == 0:
                        #First term addition: -k2f*dt, on Y(i) cols.
                        entVal[inc:(inc+(N-1))] =  -k2f*dt*liner
                        colID[inc:(inc+(N-1))] = np.arange(specLow[1],(specHigh[1]+1)).reshape(N-1,1)
                        inc = inc + (N-1)
                    elif L == 1:
                        #Second term addition: k2b*dt*u(i), on W(i) cols
                        entVal[inc:(inc+(N-1))] = k2b*dt*upv 
                        colID[inc:(inc+(N-1))] = np.arange(specLow[2],(specHigh[2]+1)).reshape(N-1,1)
                        inc = inc + (N-1)
                    else: 
                        #Third term addition: -k4b*dt, on Z(i) cols
                        entVal[inc:(inc+(N-1))] = -k4b*dt*liner
                        colID[inc:(inc+(N-1))] = np.arange(specLow[4],(specHigh[4]+1)).reshape(N-1,1)
                        inc = inc + (N-1)
                else: #Z, 1 term
                    #Term addition: -k4f*dt, on U(i) cols
                    entVal[inc:(inc+(N-1))] = -k4f*dt*liner 
                    colID[inc:(inc+(N-1))] = np.arange(specLow[3],(specHigh[3]+1)).reshape(N-1,1)
                    inc = inc + (N-1)
                    
    #Generate matrix
    entVal = entVal.flatten()
    rowID = rowID.flatten()
    colID = colID.flatten()
    mat = spar.coo_matrix((entVal,(rowID,colID)),shape=((5*N+5),(5*N+5)))
    mat = mat.tocsr()
    #For peeping at central point + boundary condition allocation
    #MPL.spy(mat,markersize=3)
    #MPL.show()
    return mat