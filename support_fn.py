import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics

def split_data_per(data,label,per):
    sz=data.shape
    limit=int(sz[0]*per)
    tr_data=data[1:limit]
    tst_data=data[limit:]
    tr_lab=label[1:limit]
    tst_lab=label[limit:]
    return (tr_data,tr_lab,tst_data,tst_lab)

def split_data_per_uniquely(data,label,per):
    sz=data.shape
    uq=np.unique(label)
    n_uq=len(uq)
    for i in range(n_uq):
        D1=data[label==uq[i],:]
        L1=label[label==uq[i]]
        le=len(L1)
        limit = int(le * float(per))
        if limit==0:
            limit=1
        tr_data = D1[0:limit,:]
        tst_data = D1[limit:,:]
        tr_lab = L1[0:limit]
        tst_lab = L1[limit:]

        if i==0:
            tr_data_all=tr_data
            tst_data_all=tst_data
            tr_lab_all=tr_lab
            tst_lab_all=tst_lab

        else:
            tr_data_all=np.concatenate((tr_data_all,tr_data))
            tst_data_all=np.concatenate((tst_data_all,tst_data))
            tr_lab_all=np.concatenate((tr_lab_all,tr_lab))
            tst_lab_all=np.concatenate((tst_lab_all,tst_lab))

    return (tr_data_all,tr_lab_all,tst_data_all,tst_lab_all)

def split_kfold_uniquely(label,kvalue):
    # #--------example---------
    # kvalue=5+int(i)
    # M=support_fn.split_kfold_uniquely(labels,kvalue)
    # logi = M[0, :].flatten()
    # tr_data = data[logi==0]
    # tr_lab = labels[logi==0]
    # tst_data = data[logi==1]
    # tst_lab = labels[logi==1]
    # sz = tr_data.shape

    uq=np.unique(label)
    n_uq = len(uq)
    sz = label.shape
    outstack=np.zeros([kvalue,sz[0]])
    for i in range(kvalue):
        out=np.zeros(sz)
        for j in range(n_uq):
            logi=label==uq[j]
            NZ_logi=np.count_nonzero(logi)
            limit=int(np.ceil(NZ_logi/kvalue))
            rn=np.random.permutation(NZ_logi)
            val=np.zeros(NZ_logi)
            val[rn[1:limit]]=1
            out[logi]=val
            # out=np.resize(out,(outstack.shape[0],outstack.shape[1]))
        outstack[i,:]=out
    return  outstack


def normalize_data(data,choice = 1):
    if type(data)==type(pd.DataFrame()):
        # input data is the dataframe object
        data=data.astype(float)
        sz=data.shape
        mn=data.min().min()
        mx=data.max().max()
        if choice==1:
            denom = (mx - mn)
            if denom == 0:
                denom = 1
            d=(data-mn)/denom
            return d
        if choice==2:
            # normalize the data through each field
            for i in range(sz[1]):
                d2=data.iloc[:, i]
                mn2=d2.min()
                mx2=d2.max()
                denom=(mx2-mn2)
                if denom==0:
                    denom=1
                d3=(d2-mn2)/denom
                data.iloc[:, i]=d3
            return data
    elif type(data)==type(np.array([])):
        # input data is the dataframe object
        data = data.astype(float)
        sz = data.shape
        mn = data.min().min()
        mx = data.max().max()
        if choice == 1:
            denom = (mx - mn)
            if denom == 0:
                denom = 1
            d = (data - mn) / denom
            return d
        if choice == 2:
            # normalize the data through each field
            for i in range(sz[1]):
                d2 = data[:, i]
                mn2 = d2.min()
                mx2 = d2.max()
                denom = (mx2 - mn2)
                if denom == 0:
                    denom = 1
                d3 = (d2 - mn2) / denom
                data[:, i] = d3
            return data



def write_to_file(data, filename, filehead,xi,xlabel_name):
    # data = is in the numpy array type
    # filename = file name with csv or txt extension
    # filehead = char array  initial field heads for each column of data

    # example:
    # data = np.random.random([10, 3])
    # sp.write_to_file(data, 'aaa.csv', ['data_1', 'data_2'])

    out = open(filename, 'w')
    out.write('%s,' % xlabel_name)
    for m in filehead:
        out.write('%s,' % m)
    out.write('\n')
    sz = data.shape
    for row in range(sz[0]):
        out.write('%f,' % xi[row])
        for column in range(sz[1]):
            out.write('%f,' % data[row][column])
        out.write('\n')
    out.close()

def write_to_file2(data, filename, filehead):
    # data = is in the numpy array type
    # filename = file name with csv or txt extension
    # filehead = char array  initial field heads for each column of data

    # example:
    # data = np.random.random([10, 3])
    # sp.write_to_file(data, 'aaa.csv', ['data_1', 'data_2'])

    out = open(filename, 'w')
    for m in filehead:
        out.write('%s,' % m)
    out.write('\n')
    sz = data.shape
    for row in range(sz[0]):
        for column in range(sz[1]):
            out.write('%f,' % data[row][column])
        out.write('\n')
    out.close()


def get_metrics(tst_lab,y_pred):

    uq=np.unique(tst_lab)
    n_uq = len(uq)

    acc_all=[]
    sen_all=[]
    spe_all=[]
    pre_all=[]
    f1m_all=[]
    fpr_all=[]
    fnr_all=[]
    for i in range(n_uq):
        tst_lab_now=tst_lab==uq[i]
        y_pred_now=y_pred==uq[i]
        confMat = sklearn.metrics.confusion_matrix(tst_lab_now, y_pred_now)
        confMatList = confMat.tolist()
        TN = confMatList[0][0]
        TP = confMatList[1][1]
        FN = confMatList[1][0]
        FP = confMatList[0][1]
        acc_now = (TP + TN) / (TN + TP + FN + FP)  # accuracy
        if (TP + FN)==0:
            sen_now=0
            fnr_now=0
        else:
            sen_now = (TP) / (TP + FN)  # sensitivity
            fnr_now = FN / (FN + TP)
        if (TN + FP)==0:
            spe_now=0
            fpr_now=0
        else:
            spe_now = (TN) / (TN + FP)  # specificity
            fpr_now = FP / (FP + TN)
        if (TP + FP)==0:
            pre_now=0
        else:
            pre_now = (TP) / (TP + FP)  # precision
        f1m_now=TP/(TP+0.5*(FP+FN))

        acc_all=acc_all+[acc_now]
        sen_all=sen_all+[sen_now]
        spe_all=spe_all+[spe_now]
        pre_all = pre_all + [pre_now]
        f1m_all = f1m_all + [f1m_now]
        fpr_all = fpr_all + [fpr_now]
        fnr_all = fnr_all + [fnr_now]

    out={}
    out['acc']=np.array(acc_all)
    out['sen']=np.array(sen_all)
    out['spe']=np.array(spe_all)
    out['pre'] = np.array(pre_all)
    out['f1m'] = np.array(f1m_all)
    out['fpr'] = np.array(fpr_all)
    out['fnr'] = np.array(fnr_all)
    out['far'] = out['fpr']
    out['frr'] = out['fnr']

    return out



def fill_trail_data(data):
    #----fill the zeros at the end of stacked numpy array(wsn code output) to make it all has same length
    len_all = []
    aa=len(data)
    for x in range(aa):
        len_all = len_all + [len(data[x])]
    max_len=max(len_all)
    data_new=np.array([])
    for x in range(aa):
        # data_new=[data_new, data[x]+[[0]*(max_len-len_all[x])]]
        dnow=np.array(data[x])
        dfilled=np.concatenate((dnow,np.zeros((max_len-len_all[x]))))
        if x==0:
            data_new=dfilled
        else:
            data_new=np.concatenate((data_new,dfilled),axis=0)
    data_new=np.reshape(data_new,[aa,max_len])
    data_new=data_new.transpose()
    return data_new


def entropy(labels, base=None):
  """ Computes entropy of label distribution. """
  from math import log, e
  n_labels = len(labels)

  if n_labels <= 1:
    return 0

  value,counts = np.unique(labels, return_counts=True)
  probs = counts / n_labels
  n_classes = np.count_nonzero(probs)

  if n_classes <= 1:
    return 0

  ent = 0.

  # Compute entropy
  base = e if base is None else base
  for i in probs:
    ent -= i * log(i, base)

  return ent


def correlation_vector(x,y):
    x1=x.mean()
    y1=y.mean()
    x2=x-x1
    y2=y-y1
    out=(x2*y2)/sum(x2**2)*sum(y2**2)
    return out

def mat2gray(input):
      mn=input.min()
      mx=input.max()
      out=(input-mn)/(mx-mn)
      return out


def data_augmentation(data,labels):
    rn=np.random.random(data.shape[0])
    limit_1=int(0.01*data.shape[0])
    limit_2 = int(0.02 * data.shape[0])
    limit_3 = int(0.03 * data.shape[0])
    d_aug1=np.fliplr(data[rn[0:limit_1],:])
    d_aug2 = np.round(data[rn[limit_1:limit_2], :])
    d_aug3 = np.roll(data[rn[limit_2:limit_3], :],1)
    l_aug1=labels[rn[0:limit_1]]
    l_aug2 = labels[rn[limit_1:limit_2]]
    l_aug3 = labels[rn[limit_2:limit_3]]
    out1=np.vstack((d_aug1,d_aug2,d_aug3))
    out2=np.hstack((l_aug1,l_aug2,l_aug3))
    return out1,out2


def DD_metric(in1,in2):
    out=(np.linalg.norm(in1.flatten()-in2.flatten()))/np.prod(in1.shape)
    return out


def get_metrics2(tst_lab,y_pred):
    uq=np.unique(tst_lab)
    n_uq = len(uq)
    n_elements=int(np.prod(tst_lab.shape))
    pre_all=np.count_nonzero(tst_lab==y_pred)/n_elements
    sen_all=np.sum((tst_lab==y_pred)*np.arange(n_elements,0,-1))/(np.sum(np.arange(n_elements,0,-1)))
    f1m_all=2*(pre_all*sen_all)/(pre_all+sen_all)
    out={}
    out['pre'] = np.array(pre_all)
    out['sen'] = np.array(sen_all)
    out['f1m'] = np.array(f1m_all)
    return out


