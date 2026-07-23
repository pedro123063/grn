import numpy as np

def searchSymbolSet(dictionary,symbols=None): #primeira invocação deve ser sem passar nada pro symbols
    symbolSet = None
    if isinstance(symbols,list):
        symbols = set(symbols)
    elif isinstance(symbols,set):
        symbols=symbols
    elif symbols is None:
        symbols=set()

    for key,value in dictionary.items():
        symbolSet.add(key)

        if isinstance(value,dict):
            searchSymbolSet(value,symbolSet)


    return symbolSet
        
        

def createSymbolsArray(symbolSet):
    symbolsArraySorted = np.array(sorted(symbolSet))

    return symbolsArraySorted
    

def encodeSymbolsArray(symbolsArraySorted): # verificar se isso garante estabilidade e determinismo no acesso
    sz=len(symbolsArraySorted)
    encodedSymbolsArray = np.zeros(sz,dtype=int)
    
    for i in range(sz):
        encodedSymbolsArray[i]=i
    return encodedSymbolsArray

def encodeDict(symbolsArray,encodedSymbolsArray,dictionary):

    encodedDict = {}

    for key,value in dictionary.items():
        if key in symbolsArray:
            position = np.searchsorted(symbolsArray,key)
            newKey = encodedSymbolsArray[position]
            newValue=None

            if isinstance(value,dict):
                newValue=encodeDict(symbolsArray,encodedSymbolsArray,value)

            else:
                newValue=encodedSymbolsArray[np.searchsorted(symbolsArray,value)]
                
            encodedDict[newKey]=newValue

    return encodedDict

def transformEncodedDictToMatrix(encodedDict):
