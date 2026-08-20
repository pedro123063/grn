import numpy as np
from numba import njit


@njit
def main_eq(val, nval, kval, minusone):

    result = (val**nval / (val**nval + kval**nval))
    
    if minusone==-1.0:
        return  1 - result
    return result

@njit
def full_eq(vals, label1, label2, matrixInd,simple_result = False):
    nIdx=matrixInd.nIdx
    val1 = vals[label1]
    val2 = vals[label2]
    nval = matrixInd[label1,label2,nIdx] # 'n'
    kval = matrixInd[label1,label2,nIdx+1] # 'k'
    tau = matrixInd.tau(label1)
    minusone = matrixInd[label1,label2,nIdx+2]# '-'
    
    if simple_result:
        return main_eq(val2, nval, kval, minusone)
    else:
        return (main_eq(val2, nval, kval, minusone) - val1)/tau

@njit
def complex_eqs(vals, label1, secondary_labels,matrixInd):
    val1 = vals[label1]
    
    tau =matrixInd.tau(label1)
    nIdx=matrixInd.nIdx

    
    total = 0
    for i in range(len(secondary_labels)):
        group=secondary_labels[i]     
        subtotal = 0
        for j in range(len(group)):
            sign_label=group[j]
            
            label2 = sign_label
            minusone = matrixInd[label1,label2,nIdx+2]# '-'
            
            
            val2 = vals[label2]
            nval = matrixInd[label1,label2,nIdx]
            kval = matrixInd[label1,label2,nIdx+1]# 'k'
            
            result = main_eq(val2, nval, kval, minusone)
            
            if j == 0:
                subtotal = result
            else:
                subtotal *= result
        
        if i == 0:
            total = subtotal
        else:
            total += subtotal
            
    return 1/tau * (total - val1)