import numpy as np

def symbolDictFromLabels(labels): #retorna um dicionário com tanto o par (cod,simbolo) quanto (simbolo,cod) 
    labels=sorted(labels)
    encodingDict={}
    for label,i in labels.enumerate():
        encodingDict[i]=label
        encodingDict[label]=i
    number=len(labels)

    aux=['-','tau','n','k']

    aux=sorted(aux)
    for symb,i in aux.enumerate():
        encodingDict[aux+i]=symb
        encodingDict[symb]=aux+i

    return encodingDict


def encodeDict(encodingDict,toBeEncoded): #substitui todas as chaves textuais por chaves do encoding
    encoded={}
    for key,value in toBeEncoded:
        if isinstance(value,dict):
            aux=encodeDict(encodingDict,value)
            encoded[encodingDict[key]]=aux
        else:
            encoded[encodeDict[key]]=value

    return encoded

def encodeBounds(encodingDict,toBeEncodedBoundsDict): #substitui todas as chaves textuais por chaves do encoding
    encoded={}
    for key,value in toBeEncodedBoundsDict:
        encoded[encodingDict[key]]=value
    return encoded

def findStructure(dictCoef):
    struct = {}
    struct[1]=len(dictCoef) #level1
    level2={}
    level3=-1

    for key,value in dictCoef:
        level2[key]=value.len()-1
    for key,value in dictCoef:
        for key2,value2 in value:
            level3=len(value2)
            break
        break

    struct[3]=level3
    struct[2]=level2

    return struct

def getPares(encondingDict,dictCoef):
    toRet={}
    for key1,value1 in dictCoef:
        aux=[]
        for key2,value2 in value1:
            if encodeDict[key2]=='tau':
                continue
            else:
                aux.append(key2)
        toRet[key1]=aux

    return toRet
    
    

