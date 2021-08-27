import numpy as np
import pandas as pd

# Functions for sampling data in a way that reduces label redundancy 
# From Marcos Lopez de Prado's "Advances in Financial Machine Learning" Chapter 4
# The same chapter also showcases techniques to create sample weights based 
# on returns,label class frecuency, or time-decay
# Chapter 2 showcases another sampling technique

def getIndMatrix(tm, barIx):
    """
    Matrix used by getAvgUniqueness() and seqBootstrap()
    tm : pandas series with the timestamps of the features as index and the timestamps of the labels as values
    barIx : index that goes from the timestamp of the first feature to the timestamp of the last label
    :return:
    """
    indM = pd.DataFrame(0, index=barIx, columns=range(tm.shape[0]))
    for i, (t0, t1) in enumerate(tm.iteritems()):
        indM.loc[t0:t1, i] = 1.0
    return indM

def getAvgUniqueness(indM):
    # Average uniqueness of the labels from the indicator matrix
    c = indM.sum(axis=1)  # Concurrency
    u = indM.div(c, axis=0)  # Uniqueness
    avgU = u[u > 0].mean()  # Average uniqueness
    return avgU

def seqBootstrap(indM,sLength = None):
    # Sample based on the labels uniqueness
    if sLength is None:sLength = indM.shape[1]
    phi = []
    while len(phi) < sLength:
        avgU = pd.Series(dtype='float64')
        for i in indM:
            indM_ = indM[phi + [i]] 
            avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1]
        prob = avgU/avgU.sum()
        phi += [np.random.choice(indM.columns, p=prob)]
    return phi

if __name__ == '__main__' : 
    # Standard uniqueness vs Sequential uniqueness demo from the book
    t1=pd.Series([2,3,5,6,9],index=[0,2,4,5,7]) # t1,t0: The timestamps of the labels and of the features 
    barIx=range(t1.max()+1) 
    print(t1)
    print(barIx)
    print('*********')
    indM=getIndMatrix(t1,barIx) 
    print(indM)
    print('*********')
    phi=np.random.choice(indM.columns,size=indM.shape[1])
    print(phi)
    print('Standard uniqueness:',getAvgUniqueness(indM[phi]).mean())
    print('*********')
    phi=seqBootstrap(indM)
    print(phi)
    print('Sequential uniqueness:',getAvgUniqueness(indM[phi]).mean())