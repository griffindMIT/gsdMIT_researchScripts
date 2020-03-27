# -*- coding: utf-8 -*-

#Import packages
import numpy as np 
import matplotlib.pyplot as MPL
from scipy import sparse as spar

def MatLin(C0,kf,kb,N,DAm,DA,DB,Cb,dx1):
    #Unpack species A, B. Need to overindex.
    C0 = C0.reshape(len(C0),1)
    Drat = DA/DB
    #Compute total # of entries - currently hardcoded...
    entTot = (3 + 4 + 2) + 3*(2*N)
    #Generate storage space for entry value, row, column index
    entVal = 0.0*np.arange(0,entTot)
    entVal = entVal.reshape(len(entVal),1)
    rowID = 0.0*np.arange(0,entTot)
    rowID = rowID.reshape(len(rowID),1)
    colID = 0.0*np.arange(0,entTot)
    colID = colID.reshape(len(colID),1)
    #Hardcoded BCs coming up
    #Eqn f[0] - BC at surface: Reaction condition
    entVal[0] = (1 + kf*dx1/DA)
    rowID[0] = 0
    colID[0] = 0
    entVal[1] = -1
    rowID[1] = 0
    colID[1] = 1
    entVal[2] = -kb*dx1/DB
    rowID[2] = 0
    colID[2] = N+1
    #Eqn  f[N] - BC at surface: Flux equality
    entVal[3] = Drat
    rowID[3] = N+1
    colID[3] = 1
    entVal[4] = -Drat
    rowID[4] = N+1
    colID[4] = 0
    entVal[5] = 1
    rowID[5] = N+1
    colID[5] = N+2
    entVal[6] = -1
    rowID[6] = N+1
    colID[6] = N+1
    #Eqn f[N+1] - BC at bulk: fixed concentration
    entVal[7] = 1
    rowID[7] = N
    colID[7] = N
    #Eqn f[2N+1] - BC at bulk: fixed concentration
    entVal[8] = 1
    rowID[8] = 2*N + 1
    colID[8] = 2*N + 1
    #The rest of the equations
    inc = 9 
    pos = np.array([0,1,2,3,4,5])
    sign = np.array([-1.0,1.0,-1.0])
    dex = np.array([0,2,1])
    for p in pos:
        if p > 2:
            Drat = DA/DB
            rowVector = np.arange((N+2),(2*N +1)) #overindex - 'central rows' of B
            p = p - 3
        else:
            Drat = 1
            rowVector = np.arange(1,N) #overindex- 'central rows' of A
        rowVector = rowVector.reshape(len(rowVector),1)
        DAcurr = Drat*DAm[1:N,dex[p]].reshape(len(DAm[1:N,dex[p]]),1)
        entVal[inc:(inc+(N-1))] = sign[p]*DAcurr
        rowID[inc:(inc+(N-1))] =  rowVector #no sign dependence
        colID[inc:(inc+(N-1))] =  rowVector + (p-1)
        #print(p)
        inc = inc + N
        
    #Generate matrix
    entVal = entVal.flatten()
    rowID = rowID.flatten()
    colID = colID.flatten()
    mat = spar.coo_matrix((entVal,(rowID,colID)),shape=((2*N+2),(2*N+2)))
    mat = mat.tocsr()
    #MPL.spy(mat)
    #MPL.show()
    #Temporary examinement
    #C = spar.linalg.spsolve(mat,C0)
    mat = mat.toarray()
    C0[0] = 0
    C0[N] = Cb[0]
    C0[N+1] = 0
    C0[len(C0)-1] = Cb[1]
    C = np.linalg.solve(mat,C0)
    return C
    
def Eval(C,C0,kf,kb,N,DAm,DA,DB,Cb,dx1):
    #Initialize function vector - length
    f = 0.0*np.arange(0,len(C))
    f = f.reshape(len(f),1)
    #Unpack species A, B. Need to overindex.
    C = C.reshape(len(C),1)
    C0 = C0.reshape(len(C0),1)
    A = C[0:(N+1)] #A = concentration at new time point
    B = C[(N+1)::]
    A0 = C0[0:(N+1)] #A0 = concentration at old time point
    B0 = C0[(N+1)::]
    #Eqn 0 - BC at surface: Reaction condition
    f[0] = (((1 + kf*dx1/DA)*A[0] - A[1] - kb*dx1*B[0]/DB))*1000
    #Eqn 1-> (N-1) - Diffusion equations for species A    
    #Eval = (-DA1i*A(i-1)) + (DA3i*Ai) - (DA2i*Ai+1) - A0(i)
    DA1 = DAm[1:N,0].reshape(len(DAm[1:N,0]),1)
    DA3 = DAm[1:N,2].reshape(len(DAm[1:N,2]),1)
    DA2 = DAm[1:N,1].reshape(len(DAm[1:N,1]),1)
    Drat = DA/DB
    f[1:(N)] = -DA1*A[0:(N-1)] + DA3*A[1:(N)] - DA2*A[2:(N+1)] -A0[1:N]
    #Eqn N- Fixed concentration
    f[N] = (A[-1] - Cb[0])
    #Repeat for second species...
    #Eqn N+1 - BC at surface: Flux equality 
    f[N+1] = ((Drat*(A[1] - A[0]) + (B[1] - B[0])))*1000
    #Eqn N+2 -> 2N+2 - Diffusion equations for species B
    f[(N+2):(2*N+1)] = -DA1*B[0:(N-1)]/Drat + DA3*B[1:N]/Drat - DA2*B[2:(N+1)]/Drat - B0[1:N]
    f[(2*N+1)] = B[-1] - Cb[1]
    #f[N::] = f[N::]*Drat
    f = f.flatten()
    return f

def OptEval(C,C0,kf,kb,N,DAm,Drat,Cb):
    #Initialize function vector - length
    f = 0*np.arange(0,len(C))
    f = f.reshape(len(f),1)
    #Unpack species A, B. Need to overindex.
    A = C[0:(N+1)].reshape(len(C[0:(N+1)]),1) #A = concentration at new time point
    B = C[(N+1)::].reshape(len(C[(N+1)::]),1) 
    A0 = C0[0:(N+1)] #A0 = concentration at old time point
    B0 = C0[(N+1)::]
    #Eqn 0 - BC at surface: Reaction condition
    f[0] = (1 + kf)*A[0] - A[1] - kb*B[0]
    #Eqn 1-> (N-1) - Diffusion equations for species A    
    #Eval = (-DA1i*A(i-1)) + (DA3i*Ai) - (DA2i*Ai+1) - A0(i)
    DA1 = DAm[1:N,0].reshape(len(DAm[1:N,0]),1)
    DA3 = DAm[1:N,2].reshape(len(DAm[1:N,2]),1)
    DA2 = DAm[1:N,1].reshape(len(DAm[1:N,1]),1)
    f[1:(N)] = -DA1*A[0:(N-1)] + DA3*A[1:(N)] - DA2*A[2:(N+1)] -A0[1:N]
    #Eqn N- Fixed concentration
    f[N] = A[-1] - Cb[1]
    #Repeat for second species...
    #Eqn N+1 - BC at surface: Flux equality 
    f[N+1] = Drat*(A[1] - A[0]) + (B[1] - B[0]) 
    #Eqn N+2 -> 2N+2 - Diffusion equations for species B
    f[(N+2):(2*N+1)] = -DA1*B[0:(N-1)]/Drat + DA3*B[1:N]/Drat - DA2*B[2:(N+1)]/Drat - B0[1:N]
    f[(2*N+1)] = B[-1] - Cb[1]
    f = np.sum(np.square(f))
    return f