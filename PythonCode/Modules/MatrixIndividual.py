import Modules.Models
import numpy as np
from numba import njit
from numba.experimental import jitclass
from numba import int8, int16, float64

# TODO (Numba JIT):
# 1. Adicionar o decorador @jitclass(spec) antes da definição da classe.
# 2. Definir o dicionário 'spec' mapeando os tipos dos atributos (idxM, coef, tauVet, nIdx).

spec=[('idxM',int16[:,:]),('coef',float64[:]),('tauVet',float64[:]),('nIdx',int8),]
@jitclass(spec)
class MatrixIndividual:
    def __init__(self,
                indexingMatrix,
                coefVet ,
                tauVet,
                nIdx #obs: a ordem de coeffs será n,k,minus . Assim , minusIdx=kIdx+1=nIdx+2
                ):

        self.idxM=indexingMatrix
        self.coef=coefVet
        self.tauVet=tauVet
        self.nIdx=nIdx
        
        return None

    def __getitem__(self,key):

        origIdx=key[0]
        targetIdx=key[1]
        varIdx=key[2]

        return self.coef[ self.idxM[origIdx,targetIdx]+(varIdx-self.nIdx) ]

    def tau(self,idx):
        return self.tauVet[idx]
    
    def replaceCoeffVet(self,newCoeffVet):
        self.coef=newCoeffVet
        
        
    
