import numpy as np
import time

def Map2K(B,Q): #B = <b_n^+ b_m>  || Qkn =Matriz cambio de base mapping  ----> return <a_k^+ a_k'>
    Nk=len(B)
    N=Q.shape[0]
    M=Q.shape[1]
    Qd=Q.transpose().conjugate()
    Pout=np.zeros([Nk,Nk])
    
    for k in range(Nk):
        for kp in range(Nk):
            Pout[k][kp]=np.sum([[B[n][m]*Qd[k][m]*Q[kp][n] for m in range(M)] for n in range(N)]) 
            
    return Pout
    
def K2X(B):
    N=B.shape[0]
    Pout=np.zeros([N])
    for n in range(N):
        Pout[n]=np.sum([[ B[k][kp]*np.exp(-1j*(k-kp)*n) for kp in range(N)]for k in range(N)])
        
    return Pout

def Map2X(B,Q): #B = <b_n^+ b_m>  || Qkn = Matriz cambio de base mapping  ----> return <a_n^+ a_n>)
    return K2X(Map2K(B,Q))
    

def Map2Xcontraction(C,Q): #C[b][bp] = <b_n^+ b_m>  || Qd = Matriz cambio de base mapping  ----> return <a_n^+ a_n>)
    '''
    DESARROLLO EN LAS NOTAS... Con esto reducimos la complejidad de O(N^5) hasta O(3N^3)...se podria hacer con np.tensordot()
    C[b][bp] = <b_n^+ b_m>  (Nm x Nm)
    Q[kp][b] = Matriz cambio de base mapping (N x Nm)
    K[bp][k] = Q[k][bp]^+ Cambio de base traspuesta (Nm x N)
    U[n][k][kp] = 3-tensor de exp(-i(k-kp)n)  ----> Implementado directamente para ahorrar RAM
    '''
    print('Map ---> X \n')
    
    N=Q.shape[0]
    Nm=Q.shape[1]
    #U=np.array([[[np.exp(-1j*(k-kp)*n)/N for kp in range(N)]for k in range(N)]for n in range(N)])
    K=Q.T.conjugate() 
    '''1º Contraccion: C[b][bp]-Q[kp][b] ---> QC[bp][kp]'''
    QC=np.zeros([Nm,N], dtype=np.complex128)
    for bp in range(Nm):
        for kp in range(N):
            QC[bp][kp]= np.sum([C[b][bp]*Q[kp][b] for b in range(Nm)])
    
    '''2º Contraccion: QC[bp][kp]-K[bp][k] ---> QCK[kp][k]'''
    QCK=np.zeros([N,N], dtype=np.complex128)
    for kp in range(N):
        for k in range(N):
            QCK[kp][k]= np.sum([QC[bp][kp]*K[bp][k] for bp in range(Nm)])
    
    '''3º Contraccion: QCK[kp][k]-U[n][k][kp] ---> QCKU[k][n][k]'''
    QCKU=np.zeros([N,N,N], dtype=np.complex128)
    cte=-1j*2*np.pi/N  #Para que k y kp vaya de 0 a N
    n0=N//2 # Para que salga centrado en 0
    for k in range(N):
        for n in range(N):
            QCKU[k][n][k]=np.sum([QCK[kp][k]*np.exp(cte*(k-kp)*(n-n0)) for kp in range(N)])  # ESTE ES EL MAS GORDO
    QCKU/=N
    
    '''4º Contraccion: QCKU[k][n][k]---> B[n]'''    
    B=np.zeros([N], dtype=np.complex128)
    for n in range(N):
        B[n]=np.sum([QCKU[k][n][k] for k in range(N)])
    
    
    return B
    
