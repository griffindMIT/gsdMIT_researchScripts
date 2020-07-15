# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 15:44:31 2020

@author: ChemeGrad2019
"""

#Import packages
import numpy as np 
import matplotlib.pyplot as MPL
import scipy.sparse as spar
from scipy.sparse import linalg as sparL
from scipy import optimize as spopt

def NonLinEval(C,C0,kVect,Dm,dt,FcEq,N,Dv,dx1,Cb):
    #Initialize function vector...
    f = 0.0*np.arange(0,len(C0))
    f = f.reshape(len(f),1)
    #Reshape concentrations
    C = C.reshape(len(C),1)
    C0 = C0.reshape(len(C0),1)
    #For convenience, unpack all species, making sure to over index. 
    #Col IDs for X0, Y0, W0, U0, Z0 = 0, N+1, 2*N+2, 3*N+3, 4*N+4
    #Col IDs for XN, YN, WN, UN, ZN = N, 2*N+1, 3*N+2, 4*N+3, 5*N+4
    #From previous timepoint:
    X0 = C0[0:N+1]
    Y0 = C0[N+1:2*N+2]
    W0 = C0[2*N+2:3*N+3]
    U0 = C0[3*N+3:4*N+4]
    Z0 = C0[4*N+4::]
    #Concentration being evaluated:
    X = C[0:N+1]
    Y = C[N+1:2*N+2]
    W = C[2*N+2:3*N+3]
    U = C[3*N+3:4*N+4]
    Z = C[4*N+4::]
    #Unpack and scale rates..
    k1f = kVect[0]*dx1/Dv[0]
    k1b = kVect[1]*dx1/Dv[0]
    k2f = kVect[2]
    k2b = kVect[3]
    k3f = kVect[4]*dx1/Dv[0]
    k3b = kVect[5]*dx1/Dv[0]
    k4f = kVect[6]
    k4b = kVect[7]
    k5f = kVect[8]*dx1/Dv[0]
    k5b = kVect[9]*dx1/Dv[0]
    
    #Assign bulk constraints 
    f[N] = X[-1] - Cb[0]
    f[2*N+1] = Y[-1] - Cb[1]
    f[3*N+2] = W[-1] - Cb[2]
    f[4*N+3] = U[-1] - Cb[3]
    f[5*N+4] = Z[-1] - Cb[4]
    
    
    #Compute BC entries...
    if k5f ==0:
        if FcEq[0] ==1:
            #Nernst for X: (k1f/k1b)*X0 - Y0 = 0, row@X0
            #rowID[inc:(inc+2)] = np.array([0,0]).reshape(2,1)
            f[0] = (k1f/k1b)*X[0] - Y[0]
        else:
            #Kinetic for X: (1/k1b + k1f/k1b)X0 - (1/k1b)X1 - Y0 = 0, row@X0
            #rowID[inc:(inc+3)] = np.array([0,0,0]).reshape(3,1)          
            f[0] = (1/k1b + k1f/k1b)*X[0] - (1/k1b)*X[1] - Y[0]
        #Equal-flux for Y: X1 - X0 + Y1 - Y0 = 0, row@Y0
        #rowID[inc:(inc+4)] = np.array([N+1,N+1,N+1,N+1]).reshape(4,1)
        f[N+1] = X[1] - X[0] + Y[1] - Y[0]
        if FcEq[2] ==1:
            #Nernst for W: (k3f/k3b)W0 - U0 = 0, row@W0
            #rowID[inc:(inc+2)] = np.array([2*N+2,2*N+2]).reshape(2,1)
            f[2*N+2] = (k3f/k3b)*W[0] - U[0]
            #Equal-flux for U: W1 - W0 + U1 - U0 = 0, row@U0
            #rowID[inc:(inc+4)] = np.array([(3*N+3),(3*N+3),(3*N+3),(3*N+3)]).reshape(4,1) 
            f[3*N+3] = W[1] - W[0] + U[1] - U[0]
        elif k3f ==0:
            #No flux for W: W1 - W0 = 0, row@W0
            #rowID[inc:(inc+2)] = np.array([2*N+2,2*N+2]).reshape(2,1)
            f[2*N+2] = W[1] - W[0]
            #No flux for U: U1 - U0 = 0, row@U0
            #rowID[inc:(inc+2)] = np.array([3*N+3,3*N+3]).reshape(2,1)
            f[3*N+3] = U[1] - U[0]
        else:
            #Kinetic for W: (1/k3b + k3f/k3b)W0 - (1/k3b)W1 - U0 = 0, row@W0
            #rowID[inc:(inc+3)] = np.array([2*N+2,2*N+2,2*N+2]).reshape(3,1)
            f[2*N+2] = (1/k3b + k3f/k3b)*W[0] - (1/k3b)*W[1] - U[0] 
            #Equal-flux for U: W1- W0 + U1 - U0 = 0, row@U0
            #rowID[inc:(inc+4)] = np.array([(3*N+3),(3*N+3),(3*N+3),(3*N+3)]).reshape(4,1) 
            f[3*N+3] = W[1] - W[0] + U[1] - U[0]
    else:
                #if k1f ==0:
        #    if abs(k5b) > 0:
        #        #kinetic for X: (1/k5b + k5f/k5b)X0 - (1/k5b)X1 - (u0)W0 - (w0)U0 = - (w0u0), row@X0
        #        rowID[inc:(inc+4)] = np.array([0,0,0,0]).reshape(4,1)
        #    else:
        #        #Non-exploding scaling: (1 + k5f)X0 - X1 = 0
        #        rowID[inc:(inc+2)] = np.array([0,0]).reshape(2,1)
        #    #No flux for Y: Y1 - Y0 = 0 , row@Y0
        #    rowID[inc:(inc+2)] = np.array([N+1,N+1]).reshape(2,1)
        #elif FcEq[0] ==1:
        #    if abs(k5b) > 0:
        #        #Partial kin. for X: (1/k5b + k5f/k5b)X0 - (1/k5b)X1 - (1/k5b)Y1 + (1/k5b)Y0 ... 
        #        #...  - (u0)W0 - (w0)U0 = -(w0u0), row@X0
        #        rowID[inc:(inc+6)] = np.array([0,0,0,0,0,0]).reshape(6,1)
        #    else:
        #        #Non-exploding scaling: (1 + k5f)X0 - X1 - Y1 + Y0 = 0
        #        rowID[inc:(inc+4)] = np.array([0,0,0,0]).reshape(4,1)
        #    #Nernst for Y: (k1f/k1b)X0 - Y0 = 0 row@Y0
        #    rowID[inc:(inc+2)] = np.array([N+1,N+1]).reshape(2,1)
        #else:
        #    #kinetic for X: (1/k1b + k1f/k1b + k5f/k1b)X0 - (1/k1b)X1 - Y0 ...
        #    #... - (u0*k5b/kb1)W0 - (w0*k5b/kb1)*U0 = -(k5b/kb1)(u0w0), row@X0
        #    rowID[inc:(inc+5)] = np.array([0,0,0,0,0]).reshape(5,1)
        #    #kinetic for Y: (1/k1f + k1b/k1f)Y0 - (1/k1f)Y1 - X0 = 0, row@Y0
        #    rowID[inc:(inc+3)] = np.array([N+1,N+1,N+1]).reshape(3,1)
        #if k3f ==0:
        #    #Kinetic for W: (1/k5f + u0*k5b/k5f)W0 - (1/k5f)W1 - X0 + (w0*k5b/k5f)U0 = (k5b/k5f)(u0w0), row@W0
        #    rowID[inc:(inc+4)] = np.array([2*N+2,2*N+2,2*N+2,2*N+2]).reshape(4,1)
        #    #Equal flux for U: W1 - W0 + U0 - U1 = 0  row@U0
        #    rowID[inc:(inc+4)]= np.array([3*N+3,3*N+3,3*N+3,3*N+3]).reshape(4,1)
        #elif FcEq[2] == 1:
        #    #Kinetic for W: (1/k5f + 2*u0*k5b/k5f)W0 - (1/k5f)W1 - 2X0 + ...
        #    #... +(1/k5f + 2*w0*k5b/k5f)U0 - (1/k5f)U1 = 2k5b/k5f(u0w0) row@W0
        #    rowID[inc:(inc+5)] = np.array([2*N+2,2*N+2,2*N+2,2*N+2,2*N+2]).reshape(5,1)
        #    #Nernst for U: (k3f/k3b)W0 - U0 = 0, row@U0
        #    rowID[inc:(inc+2)] = np.array([3*N+3,3*N+3]).reshape(2,1)
        #else:
        #    #Kinetic for W: (1/k3b + u0*k5b/k3b + k3f/k3b)W0 - (1/k3b)W1 ...
        #    #... - (k5f/k3b)X0 + (w0*k5b/k3b - 1)U0 = k5b/k3b(u0w0) row@W0
        #    rowID[inc:(inc+4)] = np.array([2*N+2,2*N+2,2*N+2,2*N+2]).reshape(4,1)
        #    #Overall flux balance for U: X1-X0+Y1-Y0-0.5U1+0.5U0-0.5W1+0.5W0=0 row@U0S
        #    rowID[inc:(inc+8)] = np.array([3*N+3,3*N+3,3*N+3,3*N+3,3*N+3,3*N+3,3*N+3,3*N+3]).reshape(8,1)
        print('Not currently supported')
    #BC for Z. 
    f[4*N+4] = Z[1] - Z[0]
    
    #Compute central entries...
    Dm1 = Dm[1:N,0].reshape(len(Dm[1:N,0]),1)
    Dm2 = Dm[1:N,1].reshape(len(Dm[1:N,1]),1)
    Dm3 = 1 + Dm1 + Dm2
    
    #X
    f[1:N] = -Dm1*X[0:(N-1)] + Dm3*X[1:N] - Dm2*X[2:N+1] - X0[1:N]
    #Y
    f[N+2:2*N+1] = -Dm1*Y[0:(N-1)] + (Dm3 + k2f*dt)*Y[1:N] - Dm2*Y[2:N+1] - k2b*dt*W[1:N]*U[1:N] - Y0[1:N]
    #W
    f[2*N+3:3*N+2]= -Dm1*W[0:(N-1)] + (Dm3 + k2b*dt*U[1:N])*W[1:N] - Dm2*W[2:N+1] -k2f*dt*Y[1:N] - W0[1:N]
    #U
    f[3*N+4:4*N+3] = -Dm1*U[0:(N-1)] + (Dm3 + k2b*dt*W[1:N] + k4f*dt)*U[1:N] - Dm2*U[2:N+1] -k2f*dt*Y[1:N] -k4b*dt*Z[1:N] - U0[1:N]
    #Z
    f[4*N+5:5*N+4] = -Dm1*Z[0:(N-1)] + (Dm3 + k4b*dt)*Z[1:N] - Dm2*Z[2:N+1] - k4f*dt*U[1:N]   - Z0[1:N]
    f = f.flatten()
    return f


def MatSolve(C0,kVect,Dm,dt,FcEq,N,Dv,dx1,Cb):
    #Generate matrix
    Mat = makeMat(C0,kVect,Dm,dt,FcEq,N,Dv,dx1,Cb)
    #Correct the "RHS" - the initial state vector
    #evals,evects = sparL.eigs(Mat)
    Ceql = modC(C0,kVect,Dm,dt,FcEq,N,Dv,dx1,Cb)
    #Solve sparse matrix for new C
    return sparL.spsolve(Mat,Ceql)

def modC(C0,kVect,Dm,dt,FcEq,N,Dv,dx1,Cb):
    Ceql = (1.0*C0).reshape(len(C0),1);
    #All "last entries" = bulk concentrations.
    Ceql[[N,2*N+1,3*N+2,4*N+3,5*N+4]] = np.array([Cb[0],Cb[1],Cb[2],Cb[3],Cb[4]]).reshape(5,1)
    #Extract & nondimensionalize rate constants for boundary condition assignments
    k1f = kVect[0]*dx1/Dv[0]
    k1b = kVect[1]*dx1/Dv[0]
    k2f = kVect[2]
    k2b = kVect[3]
    k3f = kVect[4]*dx1/Dv[0]
    k3b = kVect[5]*dx1/Dv[0]
    k4f = kVect[6]
    k4b = kVect[7]
    k5f = kVect[8]*dx1/Dv[0]
    k5b = kVect[9]*dx1/Dv[0]
    #Extract previous timepoint data for U0 (u0) and W0 (w0)
    w0 = C0[2*N+2]
    u0 = C0[3*N+3]
    #Assessment of equilibrium happens here (coming soon)
    #Turn off equilibrated rxns if the rate constants are identically zero
    if k1f == 0:
        FcEq[0] = 0
    if k3f == 0:
        FcEq[2] = 0
    #Begin modification of surface BCs. 
    if k5f == 0:
        #No concerted reaction - all surface BCs are = 0. 
        Ceql[[0,N+1,2*N+2,3*N+3,4*N+4]] = np.array([0,0,0,0,0]).reshape(5,1)
    else:
        #Concerted mechanism is present. Variable surface BCs. 
        #X boundary condition has 3 options: 
        if k1f == 0 or FcEq[0] == 1:
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
        elif FcEq[2] == 1:
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
    modifW = (liner + k4b*dt*upv)
    Ceql[specLow[2]:(specHigh[2]+1)] = np.multiply(Ceql[specLow[2]:(specHigh[2]+1)],modifW)
    #Modification #3: Multiply U central points by (1 + k2b*dt*w(i))
    modifU = (liner + k4b*dt*wpv)
    Ceql[specLow[3]:(specHigh[3]+1)] = np.multiply(Ceql[specLow[2]:(specHigh[2]+1)],modifU)
    return Ceql

def makeMat(C0,kVect,Dm,dt,FcEq,N,Dv,dx1,Cb):
    #Breakout and non-dimensionalize rate constants (tedious... but useful)
    k1f = kVect[0]*dx1/Dv[0]
    k1b = kVect[1]*dx1/Dv[0]
    k2f = kVect[2]
    k2b = kVect[3]*dx1/Dv[0]
    k3f = kVect[4]*dx1/Dv[0]
    k3b = kVect[5]*dx1/Dv[0]
    k4f = kVect[6]
    k4b = kVect[7]
    k5f = kVect[8]*dx1/Dv[0]
    k5b = kVect[9]*dx1/Dv[0]
    k6f = kVect[10]
    k6b = kVect[11]
    #Extract previous timepoint data for U0 (u0) and W0 (w0)
    w0 = C0[2*N+2][0]
    u0 = C0[3*N+3][0]
    
    #Turn off equilibrated rxns if the rate constants are identically zero
    if k1f == 0:
        FcEq[0] = 0
    if k3f == 0:
        FcEq[2] = 0
    
    #Compute total # of entries... accounting going on below... 
    #5 for fixed-conc BCs, 25 for maximum # of variable BCs (see equations)
    #(N-1) center points so 3 * # of species times that for diffusion stuff. 
    #Add additional catalytic requirements (2 for Y, 2 for W, 3 for U, 1 for Z).
    entTot = 5 + 25 + (3*5 + 2 + 2 + 3 + 1)*(N-1)
    inc = 0
    
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
    #Begin incrementer - this essentially defines the 'scheme' for introducing new entries
    entVal[inc:(inc+5)] = np.array([1,1,1,1,1]).reshape(5,1)
    rowID[inc:(inc+5)] =np.array([N,2*N+1,3*N+2,4*N+3,5*N+4]).reshape(5,1)
    colID[inc:(inc+5)] = np.array([N,2*N+1,3*N+2,4*N+3,5*N+4]).reshape(5,1)
    inc += 5
    #Following through same logic path as above, now implementing variable equations. 
    if k5f ==0: #NO concerted reaction (simpler equations)
        #X BCs. Nernst equation: (k1f/k1b)X0 - Y0 = 0, 2 pts @ row X0
        opt_Nernst_r1 = np.array([[k1f/k1b,-1.0],2*[0],[0,N+1]]) #Contains entries/rowIDs/colIDs for Nernst BC.
        #Kinetic equation: 
        opt_Kinetic_r1 = np.array([[(1.0+k1f),-1.0,-k1b],3*[0],[0,1,N+1]]) #Contains "" for kinetic BC. length = # of points. 
        choose = {True:opt_Nernst_r1,False:opt_Kinetic_r1}
        XBC = choose.get(FcEq[0]==1,'default')
        #Y BCs. Opposite-flux: X1 - X0 + Y1 - Y0 = 0, 4 pts @ row Y0
        YBC = np.array([[1.0,-1.0,1.0,-1.0],4*[N+1],[1,0,N+2,N+1]])
        #W BCs. Nernst equation: (k3f/k3b)W0 - U0 = 0, 2 pts @ row @W0
        opt_Nernst_r3 = np.array([[k3f/k3b,-1.0],2*[2*N+2],[2*N+2,3*N+3]])
        #Kinetic equation:  (1 + k3f)W0 - W1 - k3b*U0 = 0, 3 pts # row W0
        opt_Kinetic_r3 = np.array([[(1.0 + k3f),-1.0,-k3b],3*[2*N+2],[2*N+2,2*N+3,3*N+3]])    
        #Simplifies to no-flux if k3f = 0 - no need for a k3f == 0 option.
        choose = {True:opt_Nernst_r3,False:opt_Kinetic_r3}
        WBC = choose.get(FcEq[2]==1,'default')
        #U BCs. Equal-flux to W equation: W1 - W0 + U1 - U0 = 0, 4 pts @ U0
        UBC = np.array([[1.0,-1.0,1.0,-1.0],4*[3*N+3],[2*N+3,2*N+2,3*N+4,3*N+3]])
    else: #Concerted reaction - more complicated equations. 
        #X BCs: rxn1 Nernst w/concert: (1 + k5f)X0 - X1 - Y1 + Y0 - k5b*(u0)W0 - k5b(w0)U0 = -(w0u0), 6 pts @ row X0
        opt_Nernst_r1Xct = np.array([[(1.0 + k5f),-1.0,-1.0,1.0,-k5b*u0,-k5b*w0],6*[0],[0,1,N+2,N+1,2*N+2,3*N+3]])
        #rxn 1 kinet w/ concert: (1 + k1f + k5f)X0 - X1 - k1bY0 -(u0*k5b)W0 - (w0*k5b)U0 = -k5b(u0w), 5 pts @ row X0
        opt_Kinetic_r1Xct = np.array([[(1.0 + k1f + k5f),-1.0,-k1b,-k5b*u0,-k5b*w0],5*[0],[0,1,N+1,2*N+2,3*N+3]])
        #Y BCs: Nernst constraint: (k1f/k1b)X0 - Y0 = 0, 2 pts @ row Y0
        opt_Nernst_r1Yct = np.array([[(k1f/k1b),-1.0],2*[N+1],[0,N+1]])
        #Full kinetic: (1 + k1b)Y0 - Y1 - k1fX0 = 0, 3 pts @ row Y0, goes to no-flux if k1f = 0. 
        opt_Kinetic_r1Yct = np.array([[(1.0 + k1b),-1.0,-k1f],3*[N+1],[N+1,N+2,0]])
        choose = {True:[opt_Nernst_r1Xct,opt_Nernst_r1Yct],False:[opt_Kinetic_r1Xct,opt_Kinetic_r1Yct]}
        XYBC = choose.get(FcEq[0]==1,'default')
        XBC = XYBC[0]
        YBC = XYBC[1]
        #W BCs: rxn3 Nernst w/concert: (1 + 2u0*k5b)W0 - W1 -2*k5f*X0 + (1 + 2*w0*k5b)*U0 - U1 = 2k5b*u0*w0, 5 pts @ row W0
        opt_Nernst_r3Wct = np.array([[(1.0 + 2*u0*k5b),-1.0,-2*k5f,(1.0 + 2*w0*k5b),-1.0],5*[2*N+2],[2*N+2,2*N+3,0,3*N+3,3*N+4]])
        #W BCs: rxn3 Kinetic w/concert: (1 + u0*k5b + k3f)W0 - W1 - k5f*X0 + (w0*k5b - k3b)U0 = k5b(u0w0), 4 pts @ row W0
        opt_Kinetic_r3Wct = np.array([[(1 + u0*k5b + k3f),-1.0,-k5f,(w0*k5b - k3b)],4*[2*N+2],[2*N+2,2*N+3,0,3*N+3]])
        #U BCs: rxn3 Nernst w/ concert: k3f/k3b * W0 - U0 = 0, 2 pts @ row U0
        opt_Nernst_r3Uct = np.array([[k3f,-k3b],2*[3*N+3],[2*N+2,3*N+3]])
        #U BCs: rxn3 Kinetic w/concert: X1 - X0 + Y1 - Y0 - 0.5U1 + 0.5 U0 - 0.5 W1 + 0.5 W0 = 0, 8 pts @ row W0
        tg = {True:0,False:1}.get(k3f==0,'default') #Toggle that enables 'equal flux' if k3f = 0
        opt_Kinetic_r3Uct = np.array([[tg,-tg,tg,-tg,-0.5,0.5,-0.5,0.5],8*[3*N+3],[1,0,N+2,N+1,3*N+4,3*N+3,2*N+3,2*N+2]])
        choose = {True:[opt_Nernst_r3Wct,opt_Nernst_r3Uct],False:[opt_Kinetic_r3Wct,opt_Kinetic_r3Uct]}
        WUBC = choose.get(FcEq[2]==1,'default')
        WBC = WUBC[0]
        UBC = WUBC[1]
    #ZBC: No flux: Z1 - Z0 = 0, 2 points @ row Z0
    ZBC = np.array([[1.0,-1.0],2*[4*N+4],[4*N+5,4*N+4]])
    #Combine all results...
    results = [XBC,YBC,WBC,UBC,ZBC]
    for result in results:
        pts = np.shape(result)[1]
        entVal[inc:(inc+pts)] = result[0,::].reshape(pts,1)
        rowID[inc:(inc+pts)] = result[1,::].reshape(pts,1)
        colID[inc:(inc+pts)] = result[2,::].reshape(pts,1)
        inc += pts
    
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
        inc += (N-1)
        #Add (i-1) term
        entVal[inc:(inc+(N-1))] = -Dm[1:N,0].reshape(N-1,1)
        rowID[inc:(inc+(N-1))] = rowVector 
        colID[inc:(inc+(N-1))] = rowVector - 1
        inc += (N-1)
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
        entVal[inc:(inc+(N-1))] = (1 + Dm[1:N,0].reshape(N-1,1) + Dm[1:N,1].reshape(N-1,1) + ACF)
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