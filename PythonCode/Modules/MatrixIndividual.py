import Models
import numpy as np

class MatrixIndividual:
    def __init__(self,
                indexingMatrix : np.ndarray,
                coefVet : np.ndarray,
                tauVet : np.ndarray,
                nIdx: np.int8, #obs: a ordem de coeffs será n,k,minus . Assim , minusIdx=kIdx+1=nIdx+2
                ):

        self.idxM=indexingMatrix
        self.coef=coefVet
        self.tauVet=tauVet
        self.nIdx=nIdx
        
        return None

    def __getitem__(self,key):

        origIdx,targetIdx,varIdx=key

        return self.coef[ np.int16(self.idxM[origIdx,targetIdx]+(varIdx-self.nIdx) )]

    def tau(self,idx:np.int8):
        return self.tauVet[idx]
        
        
    
