import Modules.Models
import numpy as np
from numba import njit
from numba.experimental import jitclass
from numba import int8, int16, float64

# TODO (Numba JIT):
# 1. Adicionar o decorador @jitclass(spec) antes da definição da classe.
# 2. Definir o dicionário 'spec' mapeando os tipos dos atributos (idxM, coeff, tauVet, nIdx).

spec=[('idxM',int16[:,:]),('coeff',float64[:]),('tauVet',float64[:]),('nIdx',int8),]
@jitclass(spec)
class MatrixIndividual:
    def __init__(self,
                indexingMatrix,
                coeffVet ,
                tauVet,
                nIdx #obs: a ordem de coeffs será n,k,minus . Assim , minusIdx=kIdx+1=nIdx+2
                ):

        self.idxM=indexingMatrix
        self.coeff=coeffVet
        self.tauVet=tauVet
        self.nIdx=nIdx
        
        return None
    def getPos(self,key):
        origIdx=key[0]
        targetIdx=key[1]
        varIdx=key[2]
        return self.idxM[origIdx,targetIdx]+(varIdx-self.nIdx)
    
    def __getitem__(self,key):

        origIdx=key[0]
        targetIdx=key[1]
        varIdx=key[2]

        return self.coeff[ self.idxM[origIdx,targetIdx]+(varIdx-self.nIdx) ]

    def tau(self,idx):
        return self.tauVet[idx]
    def replaceTau(self,alt_tau):
        for i in range(len(self.tauVet)):
            self.tauVet[i]=alt_tau[i]
    
    def replaceCoeffVet(self,newCoeffVet):
        aux_old=0
        aux_new=0
        threshold=len(self.coeff)
        while aux_old<threshold:
            self.coeff[aux_old]=newCoeffVet[aux_new]
            self.coeff[aux_old+1]=newCoeffVet[aux_new+1]
            aux_old+=3
            aux_new+=2
    
