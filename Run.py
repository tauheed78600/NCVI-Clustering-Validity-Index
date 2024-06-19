import numpy as np
import pandas as pd

import Kmeans,fcm,dfc
import flicm


def callmain(dts,k):
    NCEI,ARI,FM,JI,MI,HI=[],[],[],[],[],[]
    if dts =='db1':

        data = pd.read_csv(r"Dataset_processed/data.csv", header=None)
        Label = np.array(pd.read_csv(r"Dataset_processed/Label.csv", header=None))
    elif dts == 'db2':
        data = np.array(pd.read_csv(r"Dataset_processed/heart_clever.csv", header=None))
        # Label = np.array(pd.read_csv(r"Dataset_processed/Label.csv", header=None))
        data = data[:,0:-1]
        Label = data[0:,-1]
    elif dts == 'db3':
        data = pd.read_csv(r"Dataset_processed/Process_iris.csv", header=None)
        Label = np.array(pd.read_csv(r"Dataset_processed/Label_iris.csv", header=None))

    #########comp methods
    # Apply DFC
    dfc.main(data, Label, k, NCEI, ARI, FM, JI, MI, HI)
    # Apply FliCm
    flicm.main(data, Label, k, NCEI, ARI, FM, JI, MI, HI)
    #Apply FCm
    fcm.main(data,Label,k,NCEI,ARI,FM,JI,MI,HI)
    # Apply K-means
    NCEI,ARI,FM,JI,MI,HI= Kmeans.main(data,Label,k,NCEI,ARI,FM,JI,MI,HI)
    return NCEI,ARI,FM,JI,MI,HI


# callmain('db1',3)