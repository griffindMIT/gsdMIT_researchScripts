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
    k6f = kVect[10]
    k6b = kVect[11]
    
    #Assign bulk constraints 
    f[N] = X[-1] - Cb[0]
    f[2*N+1] = Y[-1] - Cb[1]
    f[3*N+2] = W[-1] - Cb[2]
    f[4*N+3] = U[-1] - Cb[3]
    f[5*N+4] = Z[-1] - Cb[4]
    
    #Compute BC entries...
    if k5f ==0: #NO concerted reaction (simpler equations)
        #X BCs. Nernst equation: k1fX0 - k1bY0 = 0
        opt_Nernst_r1 = k1f*X[0] - k1b*Y[0] 
        #Kinetic equation: (1+k1f)X0 - X1 - k1bY = 0
        opt_Kinetic_r1 = (1 + k1f)*X[0] - X[1] - k1b*Y[0]  
        f[0] = {True:opt_Nernst_r1,False:opt_Kinetic_r1}.get(FcEq[0]==1)
        #Y BCs. Opposite-flux: X1 - X0 + Y1 - Y0 = 0
        f[N+1] = X[1] - X[0] + Y[1] - Y[0]
        #W BCs. Nernst equation: k3fW0 -k3bU0 = 0
        opt_Nernst_r3 = (k3f)*W[0] - k3b*U[0]
        #Kinetic equation:  (1 + k3f)W0 - W1 - k3b*U0 = 0
        opt_Kinetic_r3 = (1 + k3f)*W[0] - W[1] - k3b*U[0]
        #Simplifies to no-flux if k3f = 0 - no need for a k3f == 0 option.
        f[2*N+2] = {True:opt_Nernst_r3,False:opt_Kinetic_r3}.get(FcEq[2]==1)
        #U BCs. Equal-flux to W equation: W1 - W0 + U1 - U0 = 0
        f[3*N+3] = W[1] - W[0] + U[1] - U[0]
    else: #Concerted reaction - more complicated equations. 
        #X BCs: rxn1 Nernst w/concert: (1 + k5f)X0 - X1 - Y1 + Y0 - k5b*U0*W0 = 0
        opt_Nernst_r1Xct = (1 + k5f)*X[0] - X[1] - Y[1] + Y[0] - k5b*U[0]*W[0]
        #rxn 1 kinet w/ concert: (1 + k1f + k5f)X0 - X1 - k1bY0 - k5b*U0*W0 = 0
        opt_Kinetic_r1Xct = (1 + k1f + k5f)*X[0] - X[1] - k1b*Y[0] - k5b*U[0]*W[0]
        #Y BCs: Nernst constraint: (k1f/k1b)X0 - Y0 = 0
        opt_Nernst_r1Yct = k1f*X[0] - k1b*Y[0]
        #Full kinetic: (1 + k1b)Y0 - Y1 - k1fX0 = 0, goes to no-flux if k1f = 0. 
        opt_Kinetic_r1Yct = (1 + k1b)*Y[0] - Y[1] - k1f*X[0]
        XYBC = {True:[opt_Nernst_r1Xct,opt_Nernst_r1Yct],False:[opt_Kinetic_r1Xct,opt_Kinetic_r1Yct]}.get(FcEq[0]==1)
        f[0] = XYBC[0]
        f[N+1] = XYBC[1]
        #W BCs: rxn3 Nernst w/concert: 2k5bU0W0 - 2k5fX0 - U1 - W1 + U0 + W0 = 0
        opt_Nernst_r3Wct = 2*k5b*U[0]*W[0] - 2*k5f*X[0] - U[1] - W[1] + U[0] + W[0]
        #W BCs: rxn3 Kinetic w/concert: k5bU0W0 - k5fX0 + (1 + k3f)W0 - W1 - k3bU0 = 0
        opt_Kinetic_r3Wct = k5b*U[0]*W[0] - k5f*X[0]  + (1 + k3f)*W[0] - W[1] - k3b*U[0]
        #U BCs: rxn3 Nernst w/ concert: k3f*W0 - k3b*U0 = 0, 2 pts @ row U0
        opt_Nernst_r3Uct = k3f*W[0] - k3b*U[0]
        #U BCs: rxn3 Kinetic w/concert: X1 - X0 + Y1 - Y0 - 0.5U1 + 0.5 U0 - 0.5 W1 + 0.5 W0 = 0, 8 pts @ row W0
        tg = {True:0,False:1}.get(k3f==0) #Toggle that enables 'equal flux' if k3f = 0
        opt_Kinetic_r3Uct = -tg*X[1] + tg*X[0] - tg*Y[1] + tg*Y[0] - 0.5*U[1] + 0.5*U[0] - 0.5*W[1] + 0.5*W[0]
        WUBC = {True:[opt_Nernst_r3Wct,opt_Nernst_r3Uct],False:[opt_Kinetic_r3Wct,opt_Kinetic_r3Uct]}.get(FcEq[2]==1)
        f[2*N+2] = WUBC[0]
        f[3*N+3] = WUBC[1]
    #ZBC: No flux: Z1 - Z0 = 0, 2 points @ row Z0
    f[4*N+4] = Z[1] - Z[0]
    
    #Compute central entries...
    Dm1 = Dm[1:N,0].reshape(len(Dm[1:N,0]),1)
    Dm2 = Dm[1:N,1].reshape(len(Dm[1:N,1]),1)
    Dm3 = 1 + Dm1 + Dm2
    
    #X, Y, W, U, Z. All equations are (i-1),(i+1) diffusion terms, (i) diffusion terms w/ SCF, and added rxn terms as described in makeMat
    f[1:N] = -Dm1*X[0:(N-1)] + Dm3*X[1:N] - Dm2*X[2:N+1] - X0[1:N]
    f[N+2:2*N+1] = -Dm1*Y[0:(N-1)] + (Dm3 + (k2f+ k6f)*dt)*Y[1:N] - Dm2*Y[2:N+1] - k2b*dt*W[1:N]*U[1:N] -k6b*dt*W[1:N] - Y0[1:N]
    f[2*N+3:3*N+2]= -Dm1*W[0:(N-1)] + (Dm3 + k6b*dt + k2b*dt*U[1:N])*W[1:N] - Dm2*W[2:N+1] -(k2f+k6f)*dt*Y[1:N] - W0[1:N]
    f[3*N+4:4*N+3] = -Dm1*U[0:(N-1)] + (Dm3 + k2b*dt*W[1:N] + k4f*dt)*U[1:N] - Dm2*U[2:N+1] -k2f*dt*Y[1:N] -k4b*dt*Z[1:N] - U0[1:N]
    f[4*N+5:5*N+4] = -Dm1*Z[0:(N-1)] + (Dm3 + k4b*dt)*Z[1:N] - Dm2*Z[2:N+1] - k4f*dt*U[1:N]   - Z0[1:N]
    return f.flatten()


def MatSolve(C0,kVect,Dm,dt,FcEq,N,Dv,dx1,Cb):
    #Generate matrix
    Mat = makeMat(C0,kVect,Dm,dt,FcEq,N,Dv,dx1,Cb)
    #Correct the "RHS" - the initial state vector
    Ceql = modC(C0,kVect,Dm,dt,FcEq,N,Dv,dx1,Cb)
    #Solve sparse matrix for new C
    return sparL.spsolve(Mat,Ceql)

def modC(C0,kVect,Dm,dt,FcEq,N,Dv,dx1,Cb):
    #Establish modified C variable - prevents 'backflow' contamination of C0
    Ceql = (1.0*C0).reshape(len(C0),1)
    #All "last entries" = bulk concentrations.
    Ceql[[N,2*N+1,3*N+2,4*N+3,5*N+4]] = np.array([Cb[0],Cb[1],Cb[2],Cb[3],Cb[4]]).reshape(5,1)
    #Extract & nondimensionalize rate constants for boundary condition assignments
    k2b = kVect[3]
    k5f = kVect[8]*dx1/Dv[0]
    k5b = kVect[9]*dx1/Dv[0]
    #Extract previous timepoint data for U0 (u0) and W0 (w0)
    w0 = C0[2*N+2][0]
    u0 = C0[3*N+3][0]
    #Assessment of equilibrium happens here (coming soon)
    #Turn off equilibrated rxns if the rate constants are identically zero
    if kVect[0] == 0:
        FcEq[0] = 0
    if kVect[4] == 0:
        FcEq[2] = 0
    #Begin modification of surface BCs. 
    #Y, U, Z BCs always have eq = 0
    Ceql[[N+1,3*N+3,4*N+4]] = np.array([0,0,0]).reshape(3,1)
    if k5f == 0: #No concerted reaction, X, W BCs == 0
        Ceql[[0,2*N+2]] = np.array([0,0]).reshape(2,1)
    else: #Concerted reaction, XBC = -k5b*u0*w0, WBC = 2*k5b*u0*w0 if eq., k5b*u0*w0 if kinetic.
        scl = {True:2.0,False:1.0}.get(FcEq[2]==1) #Toggle for correcting WBC
        Ceql[[0,2*N+2]] = np.array([-k5b*u0*w0,k5b*w0*u0*scl]).reshape(2,1)
    #Start modifying central points. Access u, w data. 
    upv = C0[2*N+3:3*N+2].reshape((N-1),1)
    wpv = C0[3*N+4:4*N+3].reshape((N-1),1)
    #Indexes for start/end of center points (easy reference)
    specLow = np.array([1,N+2,2*N+3,3*N+4,4*N+5])
    specHigh = np.array([N-1,2*N,3*N+1,4*N+2,5*N+3])
    #Modification #1: Add -k2b*w(i)*u(i)*dt to central Y points. 
    Ceql[specLow[1]:(specHigh[1]+1)] += -k2b*dt*np.multiply(upv,wpv)
    #Modification #2: Multiply W central points by (1 + k2b*dt*u(i))
    Ceql[specLow[2]:(specHigh[2]+1)] = np.multiply(Ceql[specLow[2]:(specHigh[2]+1)],(np.ones([N-1,1])+k2b*dt*upv))
    #Modification #3: Multiply U central points by (1 + k2b*dt*w(i))
    Ceql[specLow[3]:(specHigh[3]+1)] = np.multiply(Ceql[specLow[3]:(specHigh[3]+1)],(np.ones([N-1,1])+k2b*dt*wpv))
    return Ceql

def makeMat(C0,kVect,Dm,dt,FcEq,N,Dv,dx1,Cb):
    #Breakout and non-dimensionalize rate constants (tedious... but useful)
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
    
    #Total # of entries...5 for fixed-conc BCs, 25 for maximum # of variable BCs
    #(N-1) center points, so 3 * # of species times that for diffusion terms. 
    #Add additional rxn requirements to center as well (2 for Y, 2 for W, 3 for U, 1 for Z).
    entTot = 5 + 25 + (3*5 + 2 + 2 + 3 + 1)*(N-1)
    
    ##################### ASSIGNING BC TO MATRIX ###########################
    #Generate storage object for matrix assignment. Contains 3 x enttot matrix (entries, row indexes, columns) + incrementer
    entryInc = entryStorage(entTot)
    #Col IDs for X0, Y0, W0, U0, Z0 = 0, N+1, 2*N+2, 3*N+3, 4*N+4
    #Col IDs for XN, YN, WN, UN, ZN = N, 2*N+1, 3*N+2, 4*N+3, 5*N+4
    #Bimolecular reactions may depend on the previous timepoint's concentration data
    #This is denoted as lowercase (e.g., u0 is U0 in the input conc (C0)). 
    
    #Bulk conc BCs: AN = Ab, where A = {X,Y,W,U,Z}, Ab = bulk, row@AN
    #Syntax of input to entry update fcn is a 3 x pts array of [0]:entries, [1]:rowIDs, [2]: colIDs.
    entryInc.update(np.array([[1,1,1,1,1],[N,2*N+1,3*N+2,4*N+3,5*N+4],[N,2*N+1,3*N+2,4*N+3,5*N+4]]))

    
    #Following through same logic path as above, now implementing variable equations. 
    if k5f ==0: #NO concerted reaction (simpler equations)
        #X BCs. Nernst equation: (k1f/k1b)X0 - Y0 = 0, 2 pts @ row X0
        opt_Nernst_r1 = np.array([[k1f,-k1b],2*[0],[0,N+1]]) #Contains entries/rowIDs/colIDs for Nernst BC.
        #Kinetic equation: (1+k1f)X0 - X1 - k1bY = 0, 3 pts # row X0
        opt_Kinetic_r1 = np.array([[(1.0+k1f),-1.0,-k1b],3*[0],[0,1,N+1]]) #Contains "" for kinetic BC. length = # of points. 
        XBC = {True:opt_Nernst_r1,False:opt_Kinetic_r1}.get(FcEq[0]==1)
        #Y BCs. Opposite-flux: X1 - X0 + Y1 - Y0 = 0, 4 pts @ row Y0
        YBC = np.array([[1.0,-1.0,1.0,-1.0],4*[N+1],[1,0,N+2,N+1]])
        #W BCs. Nernst equation: (k3f/k3b)W0 - U0 = 0, 2 pts @ row @W0
        opt_Nernst_r3 = np.array([[k3f,-k3b],2*[2*N+2],[2*N+2,3*N+3]])
        #Kinetic equation:  (1 + k3f)W0 - W1 - k3b*U0 = 0, 3 pts # row W0
        opt_Kinetic_r3 = np.array([[(1.0 + k3f),-1.0,-k3b],3*[2*N+2],[2*N+2,2*N+3,3*N+3]])    
        #Simplifies to no-flux if k3f = 0 - no need for a k3f == 0 option.
        WBC = {True:opt_Nernst_r3,False:opt_Kinetic_r3}.get(FcEq[2]==1)
        #U BCs. Equal-flux to W if k3f = 0, opposite flux of k3f =/= 0: (+/-)(W1 - W0) - U1 + U0 = 0, 4 pts @ U0
        tg = {True:1,False:-1}.get(k3f==0)
        UBC = np.array([[tg*1.0,-1.0*tg,-1.0,1.0],4*[3*N+3],[2*N+3,2*N+2,3*N+4,3*N+3]])
    else: #Concerted reaction - more complicated equations. 
        #X BCs: rxn1 Nernst w/concert: (1 + k5f)X0 - X1 - Y1 + Y0 - k5b*(u0)W0 - k5b(w0)U0 = -k5b(w0u0), 6 pts @ row X0
        opt_Nernst_r1Xct = np.array([[(1.0 + k5f),-1.0,-1.0,1.0,-k5b*u0,-k5b*w0],6*[0],[0,1,N+2,N+1,2*N+2,3*N+3]])
        #rxn 1 kinet w/ concert: (1 + k1f + k5f)X0 - X1 - k1bY0 -(u0*k5b)W0 - (w0*k5b)U0 = -k5b(u0w0), 5 pts @ row X0
        opt_Kinetic_r1Xct = np.array([[(1.0 + k1f + k5f),-1.0,-k1b,-k5b*u0,-k5b*w0],5*[0],[0,1,N+1,2*N+2,3*N+3]])
        #Y BCs: Nernst constraint: (k1f/k1b)X0 - Y0 = 0, 2 pts @ row Y0
        opt_Nernst_r1Yct = np.array([[(k1f),-k1b],2*[N+1],[0,N+1]])
        #Full kinetic: (1 + k1b)Y0 - Y1 - k1fX0 = 0, 3 pts @ row Y0, goes to no-flux if k1f = 0. 
        opt_Kinetic_r1Yct = np.array([[(1.0 + k1b),-1.0,-k1f],3*[N+1],[N+1,N+2,0]])
        XYBC = {True:[opt_Nernst_r1Xct,opt_Nernst_r1Yct],False:[opt_Kinetic_r1Xct,opt_Kinetic_r1Yct]}.get(FcEq[0]==1)
        XBC = XYBC[0]
        YBC = XYBC[1]
        #W BCs: rxn3 Nernst w/concert: (1 + 2u0*k5b)W0 - W1 -2*k5f*X0 + (1 + 2*w0*k5b)*U0 - U1 = 2k5b*u0*w0, 5 pts @ row W0
        opt_Nernst_r3Wct = np.array([[(1.0 + 2*u0*k5b),-1.0,-2*k5f,(1.0 + 2*w0*k5b),-1.0],5*[2*N+2],[2*N+2,2*N+3,0,3*N+3,3*N+4]])
        #W BCs: rxn3 Kinetic w/concert: (1 + u0*k5b + k3f)W0 - W1 - k5f*X0 + (w0*k5b - k3b)U0 = k5b(u0w0), 4 pts @ row W0
        opt_Kinetic_r3Wct = np.array([[(1 + u0*k5b + k3f),-1.0,-k5f,(w0*k5b - k3b)],4*[2*N+2],[2*N+2,2*N+3,0,3*N+3]])
        #U BCs: rxn3 Nernst w/ concert: k3f*W0 - k3b*U0 = 0, 2 pts @ row U0
        opt_Nernst_r3Uct = np.array([[k3f,-k3b],2*[3*N+3],[2*N+2,3*N+3]])
        #U BCs: rxn3 Kinetic w/concert: -X1 + X0 - Y1 + Y0 - 0.5U1 + 0.5 U0 - 0.5 W1 + 0.5 W0 = 0, 8 pts @ row W0
        tg = {True:[0,-1],False:[1,1]}.get(k3f==0) #Toggle that enables 'equal flux' if k3f = 0
        opt_Kinetic_r3Uct = np.array([[-tg[0],tg[0],-tg[0],tg[0],-0.5,0.5,-0.5*tg[1],0.5*tg[1]],8*[3*N+3],[1,0,N+2,N+1,3*N+4,3*N+3,2*N+3,2*N+2]])
        WUBC = {True:[opt_Nernst_r3Wct,opt_Nernst_r3Uct],False:[opt_Kinetic_r3Wct,opt_Kinetic_r3Uct]}.get(FcEq[2]==1)
        WBC = WUBC[0]
        UBC = WUBC[1]
    #ZBC: No flux: Z1 - Z0 = 0, 2 points @ row Z0
    ZBC = np.array([[1.0,-1.0],2*[4*N+4],[4*N+5,4*N+4]])
    #Combine all results...
    results = [XBC,YBC,WBC,UBC,ZBC]
    for result in results:
        entryInc.update(result)

    #For viewing boundary condition placement. Comment for production.
    #MPL.spy(spar.coo_matrix((entryInc.entries[0,::],(entryInc.entries[1,::],entryInc.entries[2,::])),shape=((5*N+5),(5*N+5))).tocsr(),markersize=3)
    #MPL.show()
    
    #Allocate central points (after surface BC, before bulk BC). Eqns follow.
    #(The bracketing on the D's refers to math notation not coding... subtract one for code!)
    #X, ind = 1 to N-1 -> #Dm[3]X(i) - Dm[2]X(i+1) - Dm[1]X(i-1) = x(i)
    #Y, ind =  N+2 to 2*N -> #(Dm[3] + (k2f + k6f)*dt)Y(i) - Dm[2]Y(i+1) - Dm[1]Y(i-1) - k2b*dt*w(i)U(i) - ((k2b*u(i) + k6b)*dt)W(i) = y(i) - k2b*dt*u(i)w(i)
    #W, ind = 2*N+3 to 3*N+1 -> #(Dm[3] + (k2b*u(i) + k6b)*dt)W(i) - Dm[2]W(i+1) - Dm[1]W(i-1) - (k2f+k6f)*dt*Y(i) + k2b*dt*w(i)U(i) = w(i)(1 + k2b*dt*u(i))
    #U, ind = 3*N+4 to 4*N+2 -> #(Dm[3] + k2b*dt*w(i) + k4f*dt)U(i) - Dm[2]U(i+1) - Dm[1]U(i-1) - k2f*dt*Y(i) + k2b*dt*u(i)*W(i) - k4b*dt*Z(i) = u(i)(1 + k2b*dt*w(i))
    #Z, ind = 4*N+5 to 5*N+3 -> #(Dm[3] + k4b*dt)Z(i) - Dm[2]Z(i+1) - Dm[1]Z(i-1) - k1f*dt*u(i) = z(i)
    
    #Begin assignment according to the above equations (hardcoded switches and indexes)
    #Indexer for what species is currently being treated
    specCounter = np.array([0,1,2,3,4])
    #Indexes for start/end of center points A1 -> A(N-1)
    specLow = np.array([1,N+2,2*N+3,3*N+4,4*N+5])
    specHigh = np.array([N-1,2*N,3*N+1,4*N+2,5*N+3])
    #Get the previous timepoint data for U, W (useful in many rxns)
    upv = C0[2*N+3:3*N+2].flatten()
    wpv = C0[3*N+4:4*N+3].flatten()
    for ss in specCounter:
        #Create vector of "i" values
        rowVector = np.arange(specLow[ss],specHigh[ss]+1)
        #Add (i+1) diffusion term
        entryInc.update(np.array([-Dm[1:N,1],rowVector,(rowVector+1)]))
        #Add (i-1) diffusion term
        entryInc.update(np.array([-Dm[1:N,0],rowVector,(rowVector-1)]))
        #Assign SCF (self-cat. factor) for species: SCF for X, Y, W, U, Z.
        SCF = {0:0, 1:(k2f + k6f)*dt, 2:(k2b*upv + k6b)*dt, 3:(k2b*wpv + k4f)*dt, 4:k4b*dt}.get(ss)
        #Add (i) self-diffusion term with SCF to account for reactions:
        entryInc.update(np.array([(1 + Dm[1:N,0] + Dm[1:N,1] + SCF),rowVector,rowVector]))
        #Add additional terms as required by the species (also v. hardcoded)
        if ss == 1: #Species Y, 2 terms
            #Term #1: -k2b*w(i)*dt on U(i) cols.
            entryInc.update(np.array([-k2b*dt*wpv,rowVector,np.arange(specLow[3],(specHigh[3]+1))]))
            #Term #2: -(k2b*u(i) + k6b)*dt on W(i) cols
            entryInc.update(np.array([-(k2b*upv + k6b)*dt,rowVector,np.arange(specLow[2],(specHigh[2]+1))]))
        elif ss == 2: #Species W, 2 terms
            #Term #1: -(k2f+k6f)*dt, on Y(i) cols
            entryInc.update(np.array([-(k2f+k6f)*dt*np.ones([N-1]),rowVector,np.arange(specLow[1],(specHigh[1]+1))]))
            #Term #2: k2b*dt*w(i), on U(i) cols. 
            entryInc.update(np.array([k2b*dt*wpv,rowVector,np.arange(specLow[3],(specHigh[3]+1))]))
        elif ss == 3: #Species U, 3 terms
            #Term #1: -k2f*dt, on Y(i) cols. 
            entryInc.update(np.array([-k2f*dt*np.ones([N-1]),rowVector,np.arange(specLow[1],(specHigh[1]+1))]))
            #Term #2: k2b*dt*u(i) on W(i) cols.
            entryInc.update(np.array([k2b*dt*upv,rowVector,np.arange(specLow[2],(specHigh[2]+1))]))
            #Term #3: -k4b*dt, on Z(i) cols
            entryInc.update(np.array([-k4b*dt*np.ones([N-1]),rowVector,np.arange(specLow[4],(specHigh[4]+1))]))
        elif ss == 4: #Species Z, 1 term
            #Term addition: -k4f*dt, on U(i) cols 
            entryInc.update(np.array([-k4f*dt*np.ones([N-1]),rowVector,np.arange(specLow[3],(specHigh[3]+1))]))              

    #Generate and return matrix
    return spar.coo_matrix((entryInc.entries[0,::],(entryInc.entries[1,::],entryInc.entries[2,::])),shape=((5*N+5),(5*N+5))).tocsr()

class entryStorage:
    def __init__(obj,entTot):
        #Generate 'entries' array, 3 x entTot. rows = {entries, row IDs, col IDs}.
        obj.entries = 0.0*np.arange(0,3*entTot).reshape(3,entTot)
        #Incrementer variable keeps track of 'position' along the columns of entries. 
        obj.inc = 0
    def update(self,values):
        #To update with variable point-count inputs, the point count is determined...
        pts = np.shape(values[0,::])[0]
        #The value input (same row structure as storage) is assigned to the appropriate 'slice' of entries
        self.entries[0::,self.inc:(self.inc + pts)] = values
        #Incrementer is increased so that newer entries won't overwrite
        self.inc += pts
        
        