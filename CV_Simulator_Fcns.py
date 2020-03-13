# -*- coding: utf-8 -*-

#Import packages
import numpy as np 

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