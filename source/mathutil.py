"""
Python package for NCA utility
Developer: Harry He
Algorithm:  Takuya Isomura, Taro Toyoizumi
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import collections

def cal_entropy(data,log_flag=False,byte_flag=False, torch_flag=False):
    """
    Cal entropy of a vector
    :param data:
    :param log_flag: if input data is log probability
    :return:
    """
    adj = 1
    if byte_flag:
        adj = np.log(2)
    if not torch_flag:
        data=np.array(data)
        if log_flag:
            data=np.exp(data)
        # assert len(data.shape) == 1
        data_adj=np.zeros(data.shape)+data
        data_adj = data_adj / np.sum(data_adj, axis=-1,keepdims=True)
        ent=-np.sum(data_adj*np.log(data_adj + 1e-9)/adj,axis=-1)
    else:
        if log_flag:
            data=torch.exp(data)
        data = data / torch.sum(data, dim=-1, keepdim=True)
        ent = -torch.sum(data * torch.log(data + 1e-9) / adj, dim=-1)
    return ent

def cal_entropy_raw(data,data_discrete=True,data_bins=None):
    """
    Calculate entropy of raw data
    :param data: [Ndata of value]
    :param data_discrete: if data is discrete
    :return:
    """
    data=np.array(data)
    assert len(data.shape) == 1
    if data_discrete:
        if len(set(data))<=1:
            return 0
        np_data_bins=np.array(list(set(data)))
        data_bins_s = np.sort(np_data_bins)
        prec=data_bins_s[1]-data_bins_s[0]
        data_bins = np.concatenate((data_bins_s.reshape(-1,1),np.array(data_bins_s[-1]+prec).reshape(1,1)),axis=0)
        data_bins=data_bins.reshape(-1)
        data_bins=data_bins-prec/2
    pdata, _ = np.histogram(data, bins=data_bins)
    pdata=pdata/np.sum(pdata)
    return cal_entropy(pdata)

def cal_entropy_raw_ND_discrete(data):
    """
    Calculate entropy of raw data
    N dimensional discrete data
    :param data: [Ndata of D-dim value]
    :param data_discrete: if data is discrete
    :return:
    """
    datanp=np.array(data)
    if len(datanp.shape) == 1:
        datanp=datanp.reshape(-1,1)
    assert len(datanp.shape) == 2
    assert datanp.shape[0]>datanp.shape[1]

    # datatup = []
    # for iin in range(len(datanp)):
    #     datatup.append(tuple(datanp[iin]))
    # datatup= [tuple(datanp[iin]) for iin in range(len(datanp))] # Slow!!!

    projV=np.random.random(datanp.shape[1])
    datatup=datanp.dot(projV)

    itemcnter = collections.Counter(datatup)

    pvec = np.zeros(len(itemcnter))
    for ii, val in enumerate(itemcnter.values()):
        pvec[ii] = val
    pvec = pvec / np.sum(pvec)

    return cal_entropy(pvec)

def cal_entropy_raw_ND_discrete_torch(datanp):
    """
    Calculate entropy of raw data
    N dimensional discrete data
    :param data: [Ndata of D-dim value]
    :param data_discrete: if data is discrete
    :return:
    """
    datanp=datanp.type(torch.FloatTensor)
    device = datanp.device
    if len(datanp.shape) == 1:
        datanp=datanp.view(-1,1)
    assert len(datanp.shape) == 2
    assert datanp.shape[0]>datanp.shape[1]

    projV=torch.rand(datanp.shape[1]).to(device)
    datatup=datanp.matmul(projV)

    itemcnter = collections.Counter(datatup.cpu().numpy())

    pvec = np.zeros(len(itemcnter))
    for ii, val in enumerate(itemcnter.values()):
        pvec[ii] = val
    pvec = pvec / np.sum(pvec)

    return cal_entropy(pvec)

def cal_muinfo(p,q,pq):
    """
    Calculate multual information
    :param p: marginal p
    :param q: marginal q
    :param pq: joint pq
    :return:
    """
    assert len(p)*len(q) == pq.shape[0]*pq.shape[1]
    ptq=p.reshape(-1,1)*q.reshape(1,-1)
    return cal_kldiv(pq,ptq)

def cal_muinfo_raw(x,y,x_discrete=True,y_discrete=True,x_bins=None,y_bins=None):
    """
    Calculation of mutual information between x,y from raw data (May have problem)!!!
    :param x: [Ndata of value]
    :param y: [Ndata of value]
    :param x_discrete: if x is discrete
    :param y_discrete: if y is discrete
    :param x_res: x resolution (None if discrete)
    :param y_res: y resolution (None if discrete)
    :return:
    """
    x=np.array(x)
    y = np.array(y)

    assert len(x) == len(y)
    assert len(x.shape) == 1
    assert len(y.shape) == 1

    if x_discrete:
        x_bins=len(set(x))
    px,_ = np.histogram(x, bins=x_bins)
    px = px/np.sum(px)

    if y_discrete:
        y_bins=len(set(y))
    py, _ = np.histogram(y, bins=y_bins)
    py = py / np.sum(py)

    pxy,_,_= np.histogram2d(x,y,bins=[x_bins,y_bins])
    pxy=pxy/np.sum(pxy)

    return cal_muinfo(px,py,pxy)

def cal_muinfo_raw_ND_discrete(X,Z):
    """
    Calculate multual information of N dimensional discrete data X and Z
    I(X;Z) = H(X) + H(Z) - H(X,Z)
    :param X: [Ndata of D-dim value]
    :param Z: [Ndata of D-dim value]
    :return:
    """
    Xnp = np.array(X)
    if len(Xnp.shape)==1:
        Xnp=Xnp.reshape(-1,1)
    assert len(Xnp.shape) == 2
    assert Xnp.shape[0] > Xnp.shape[1]

    Znp = np.array(Z)
    if len(Znp.shape)==1:
        Znp=Znp.reshape(-1,1)
    assert len(Znp.shape) == 2
    assert Znp.shape[0] > Znp.shape[1]

    XZnp = np.concatenate((Xnp,Znp),axis=1)

    Hx= cal_entropy_raw_ND_discrete(Xnp)

    Hz = cal_entropy_raw_ND_discrete(Znp)

    Hxz = cal_entropy_raw_ND_discrete(XZnp)

    return Hx+Hz-Hxz

def cal_muinfo_raw_ND_discrete_torch(Xnp,Znp):
    """
    Calculate multual information of N dimensional discrete data X and Z
    I(X;Z) = H(X) + H(Z) - H(X,Z)
    :param X: [Ndata of D-dim value]
    :param Z: [Ndata of D-dim value]
    :return:
    """
    if len(Xnp.shape)==1:
        Xnp=Xnp.view(-1,1)
    assert len(Xnp.shape) == 2
    assert Xnp.shape[0] > Xnp.shape[1]

    if len(Znp.shape)==1:
        Znp=Znp.view(-1,1)
    assert len(Znp.shape) == 2
    assert Znp.shape[0] > Znp.shape[1]

    XZnp = torch.cat((Xnp,Znp),dim=1)

    Hx= cal_entropy_raw_ND_discrete_torch(Xnp)

    Hz = cal_entropy_raw_ND_discrete_torch(Znp)

    Hxz = cal_entropy_raw_ND_discrete_torch(XZnp)

    return Hx+Hz-Hxz