# -*- coding: utf-8 -*-

#Import packages
import numpy as np 
#import matplotlib.pyplot as MPL
import scipy.sparse as spar
from scipy.sparse import linalg as sparL
#from scipy import optimize as spopt

def MatSolve(C0,EqTog,kVect,N,Dm,D,Cb,dx1,dt):
    #Generate matrix
    Mat = makeMat(EqTog,kVect,N,Dm,D,Cb,dx1,dt)
    #Correct the "RHS" - the initial state vector
    #All "first entries" = 0, all "last entries" = bulk.
    C0 = C0.reshape(len(C0),1);
    C0[[0,N+1,2*N+2,3*N+3,4*N+4]] = np.array([0,0,0,0,0]).reshape(5,1)
    C0[[N,2*N+1,3*N+2,4*N+3,5*N+4]] = np.array([Cb[0],Cb[1],Cb[2],Cb[3],Cb[4]]).reshape(5,1)
    #Solve sparse matrix for new C
    C = sparL.spsolve(Mat,C0)
    return C
    
def makeMat(EqTog,kVect,N,Dm,D,Cb,dx1,dt):
    Nspec = 5 # # of species ### HARDCODED ###    
    #Compute total # of entries... accounting going on below...
    #Need 3 entries/spec for every center point. Point 1 to (N-1) is N entries. 
    centerEntrs = 3*Nspec*(N)
    #Need 1 entries/spec for every non-self catalytic center point (W, U, Z)
    catylEntrs = 3*N
    #Need 1 entry/spec for every fixed conc point. 
    fixedEntrs = Nspec
    #Need 16 flux/rxn entries ### HARDCODED ### - goes down by 1 per reversilbe
    surfbcEntr = 16 - sum(EqTog)
    #Need 4 more entries for the rxn condition
    entTot = centerEntrs + fixedEntrs + surfbcEntr + catylEntrs
    #Generate storage space fo entry value, row, column index
    entVal = 0.0*np.arange(0,entTot).reshape(entTot,1)
    #entVal is the multiplication factor for the matrix entry (-1, D, 1/x, etc.)
    #"Row ID" refers to the position of the equation. All eqns share a row ID.
    #"Col ID" refers to the position of the concentration in the C vector. 
    #Col IDs for X0, Y0, W0, U0, Z0 = 0, N+1, 2*N+2, 3*N+3, 4*N+4
    #Col IDs for XN, YN, WN, UN, ZN = N, 2*N+1, 3*N+2, 4*N+3, 5*N+4
    rowID = 0.0*np.arange(0,entTot).reshape(entTot,1)
    colID = 0.0*np.arange(0,entTot).reshape(entTot,1)
    #Assign hardcoded BCs
    #Eqns 0 through 4: Bulk concentration BCs
    entVal[[0,1,2,3,4]] = np.array([1,1,1,1,1]).reshape(Nspec,1)
    rowID[[0,1,2,3,4]] =np.array([N,2*N+1,3*N+2,4*N+3,5*N+4]).reshape(Nspec,1)
    colID[[0,1,2,3,4]] = np.array([N,2*N+1,3*N+2,4*N+3,5*N+4]).reshape(Nspec,1)
    #Eqn 5: Y flux constraint - 4 entries
    #Eqn: X1 - X0 + Y1 - Y0 = 0
    entVal[[5,6,7,8]] = np.array([1,-1,1,-1]).reshape(4,1)
    rowID[[5,6,7,8]] = np.array([N+1,N+1,N+1,N+1]).reshape(4,1)
    colID[[5,6,7,8]] = np.array([1,0,N+2,N+1]).reshape(4,1)
    #Eqn 6: U flux constraint
    #Eqn: W1 - W0 + U1 - U0 = 0
    entVal[[9,10,11,12]] = np.array([1,-1,1,-1]).reshape(4,1) 
    rowID[[9,10,11,12]] = np.array([3*N+3,3*N+3,3*N+3,3*N+3]).reshape(4,1)
    colID[[9,10,11,12]] = np.array([2*N+3,2*N+2,3*N+4,3*N+3]).reshape(4,1)
    #Eqn 7: Z flux constraint - 2 entries
    #Eqn: Z1 - Z0 = 0
    entVal[[13,14]] = np.array([1,-1]).reshape(2,1)
    rowID[[13,14]] = np.array([4*N+4,4*N+4]).reshape(2,1)
    colID[[13,14]] = np.array([4*N+5,4*N+4]).reshape(2,1)
    #Why are X and W done last? to allow for equilib toggling with minimal
    #interference with 'hardcoded' nature of BCs...
    inc = 15
    #Definition of kVect = np.array([kfXY,kbXY,kcY,kfWU,kbWU,kcU])
    #Eqn 8: X reaction constraint
    if EqTog[0] == 1:
        #Equilibrium reaction constraint - 2 entries
        #Eqn: (kf/kb)*X0 - Y0 = 0
        entVal[[inc,inc+1]] = np.array([(kVect[0]/kVect[1]), -1]).reshape(2,1) 
        rowID[[inc,inc+1]] = np.array([0,0]).reshape(2,1)
        colID[[inc,inc+1]] = np.array([0,N+1]).reshape(2,1)
        inc = inc + 2
    else:
        #Kinetic reaction constraint - 3 entries
        #Eqn: (1 + kf*dx1/D)*X0 - X1 - (kb*dx1/D)*Y0 = 0
        entVal[[inc,inc+1,inc+2]] = np.array([(1 + (kVect[0]*dx1/D)),-1,-1*(kVect[1]*dx1/D)]).reshape(3,1)
        rowID[[inc,inc+1,inc+2]] = np.array([0,0,0]).reshape(3,1)
        colID[[inc,inc+1,inc+2]] = np.array([0,1,N+1]).reshape(3,1)
        inc = inc + 3
    #Eqn 9: W reaction constraint
    if EqTog[1] == 1:
        #Equilibrium reaction constraint - 2 entries
        #Eqn: (kf/kb)*W0 - U0 = 0
        entVal[[inc,inc+1]] = np.array([(kVect[3]/kVect[4]), -1]).reshape(2,1) 
        rowID[[inc,inc+1]] = np.array([2*N+2,2*N+2]).reshape(2,1)
        colID[[inc,inc+1]] = np.array([2*N+2,3*N+3]).reshape(2,1)
        inc = inc + 2
    else:
        #Kinetic reaction constraint - 3 entries
        #Eqn: (1 + kf*dx1/D)*W0 - W1 - (kb*dx1/D)*U0 = 0
        entVal[[inc,inc+1,inc+2]] = np.array([(1 + (kVect[3]*dx1/D)),-1,-1*(kVect[4]*dx1/D)]).reshape(3,1)
        rowID[[inc,inc+1,inc+2]] = np.array([2*N+2,2*N+2,2*N+2]).reshape(3,1)
        colID[[inc,inc+1,inc+2]] = np.array([2*N+2,2*N+3,3*N+3]).reshape(3,1)
        inc = inc + 3
    
    #Allocate 'central points'    
    #Limits for species: (after surface BC, before bulk BC). cc = 'corresponding conc'
        #X -> 1 to N-1
            #Eqn: -Dm[col 1]*X(i-1) + Dm[col 3]*X(i) - Dm[col2]X(i+1) = cc
        #Y -> N+2 to 2*N
            #Eqn: -Dm[col 1]*Y(i-1) + (kc*dt + Dm[col 3])*Y(i) - Dm[col2]Y(i+1) = cc
        #W -> 2*N+3 to 3*N+1
            #Eqn: -Dm[col 1]*W(i-1) + (-1*kc*dt + Dm[col 3])*W(i) - Dm[col2]W(i+1) = cc
        #U -> 3*N+4 to 4*N+2
            #Eqn: -Dm[col 1]*U(i-1) + (-1*kc*dt + kc*dt + Dm[col 3])*U(i) - Dm[col2]U(i) = cc 
        #Z -> 4*N+5 to 5*N+3
            #Eqn: -Dm[col 1]*Z(i-1) + Dm[col 3]*Z(i) - Dm[col 2]*Z(i+1) = cc
    
    #Begin some hardcoded nonsense...
    specCounter = np.array([0,1,2,3,4])
    specLow = np.array([1,N+2,2*N+3,3*N+4,4*N+5])
    specHigh = np.array([N-1,2*N,3*N+1,4*N+2,5*N+3])
    pos = np.array([0,1,2])
    sign = np.array([-1.0,1.0,-1.0])
    dex = np.array([0,2,1])
    for ss in specCounter:
        high = specHigh[ss]
        low = specLow[ss]
        rowVector = np.arange(low,high+1).reshape((high+1-low),1)
        for p in pos:
            Dmcc = Dm[1:N,dex[p]].reshape(len(Dm[1:N,dex[p]]),1)
            ACF = 0 #Autocatalytic factor
            if p == 1: 
                if ss == 1: #Species Y - autocat factor
                    ACF = kVect[2]*dt    
                if ss == 3: #Species U
                    ACF = kVect[5]*dt
                if ss == 2 or ss == 3 or ss == 4: #Catalytic production of W,U,Z
                    liner = np.arange(1,N)/np.arange(1,N) 
                    liner = liner.reshape(len(liner),1)
                    rowID[inc:(inc+(N-1))] = rowVector
                    if ss == 2 or ss == 3: #If producing W,U from Y
                        CF = -kVect[2]*dt #Catalytic factor
                        entVal[inc:(inc+(N-1))] = CF*liner
                        colID[inc:(inc+(N-1))] = np.arange(N+2,2*N+1).reshape(N-1,1) #Y positions
                    else: #If producing Z from U
                        CF = -kVect[5]*dt
                        entVal[inc:(inc+(N-1))] = CF*liner
                        colID[inc:(inc+(N-1))] = np.arange(N+2,2*N+1).reshape(N-1,1) #U positions
                    inc = inc + N    
            entVal[inc:(inc+(N-1))] = sign[p]*(ACF + Dmcc) ##missing the overindex?
            rowID[inc:(inc+(N-1))] = rowVector
            colID[inc:(inc+(N-1))] = rowVector + (p-1)
            inc = inc + N
                
                
    #Generate matrix
    entVal = entVal.flatten()
    rowID = rowID.flatten()
    colID = colID.flatten()
    mat = spar.coo_matrix((entVal,(rowID,colID)),shape=((5*N+5),(5*N+5)))
    mat = mat.tocsr()
    #MPL.spy(mat,markersize=3)
    #MPL.show()
    return mat