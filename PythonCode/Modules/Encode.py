import numpy as np

def encodeLabels_and_encodingDict(labels): #retorna um dicionário com tanto o par (cod,simbolo) quanto (simbolo,cod) 
    labels=sorted(labels)
    encodingDict={}
    encodedLabels=np.ndarray((len(labels)),dtype=np.int8)
    for i,label in enumerate(labels):
        encodingDict[i]=label
        encodingDict[label]=i
        encodedLabels[i]=np.int8(i)
    number=len(labels)

    aux=('n','k','-','tau')
    vAux=len(labels)
    
    for i,symb in enumerate(aux):
        encodingDict[vAux+i]=symb
        encodingDict[symb]=vAux+i

    return encodedLabels,encodingDict

def encodeDict(encodingDict,toBeEncoded): #substitui todas as chaves textuais por chaves do encoding
    encoded={}
    for key,value in toBeEncoded.items():
        if isinstance(value,dict):
            aux=encodeDict(encodingDict,value)
            encoded[encodingDict[key]]=aux
        else:
            encoded[encodingDict[key]]=value

    return encoded

def encodeBounds(encodingDict,toBeEncodedBoundsDict): #substitui todas as chaves textuais por chaves do encoding
    encoded={}
    for key,value in toBeEncodedBoundsDict.items():
        encoded[encodingDict[key]]=value
    return encoded

def generateCoefVet(encodingDict,encodedDict):
    aux=0
    for key1,value1 in encodedDict.items():
        for key2,value2 in value1.items():
            if key2 !=encodingDict['tau']:
                aux+=3
    return np.zeros(aux,dtype=np.double)

def generateTauVet(labels):
    return np.zeros(len(labels),dtype=np.double)

def generateIdxMatrix(labels):
    matrix=np.full((len(labels),len(labels)),-1,dtype=np.int16)
    return matrix

def populateMIdx(matrix,encodingDict,encodedDict):
    aux=np.int16(0)
    for key1,value1 in encodedDict.items():
        for key2,_ in value1.items():
            if key2 != encodingDict["tau"]:
                matrix[key1,key2]=np.int16(aux)
                aux+=3
def populateTauVet(tauVet,encodedBounds,encodingDict):
    tauVet[:]=np.double(encodedBounds[encodingDict['tau']][0])

def populateCoefVet(coefVet,encodedBounds,encodingDict):
    coefVet[0::3]=np.double(encodedBounds[encodingDict['n']][0])
    coefVet[1::3]=np.double(encodedBounds[encodingDict['k']][0])
    coefVet[2::3]=np.double(1.0)

def encode(labels,coeff,bounds)->dict :
    encodedLabels,encodingDict = encodeLabels_and_encodingDict(labels)
    encodedDict=encodeDict(encodingDict,coeff)
    encodedBounds=encodeBounds(encodingDict,bounds)
    coeffVet=generateCoefVet(encodingDict,encodedDict)
    tauVet=generateTauVet(labels)
    idxMatrix=generateIdxMatrix(labels)
    populateMIdx(idxMatrix,encodingDict,encodedDict)
    populateTauVet(tauVet,encodedBounds,encodingDict)
    populateCoefVet(coeffVet,encodedBounds,encodingDict)
    
    return (encodedLabels,encodingDict,encodedBounds,idxMatrix,coeffVet,tauVet)