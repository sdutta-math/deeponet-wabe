#! /usr/bin/env python

import numpy as np
import scipy

class data_scaler:
    def __init__(self, u_min=None, u_max=None, v_min=0.0, v_max=1.0, axis=None):
        """
        Args:
            u_min: float, int, None, or list of floats or ints (default None).
                Minimum of the data before being scaled.
                If axis = None, must be an integer or float. If axis != None,
                must be a list of floats or ints of the same size as entries
                in the axis dimension. I.e. for an array of size (3,2,4), with
                axis = 1, u_min must have a size of 2. If u_min = None, its
                value will be inferred from the first call of data_scalarV2.
            u_max: float, int, None, or list of floats or ints (default None).
                Maximum of the data before being scaled
                If axis = None, must be an integer, float, or None. If axis != None,
                must be a list of floats or ints of the same size as entries
                in the axis dimension. I.e. for an array of size (3,2,4), with
                axis = 1, u_min must have a size of 2. If u_min = None, its
                value will be inferred from the first call of data_scalarV2.
            v_min: float or int, (default 0.0) minimum value of the data
                after it has been scaled
            v_max: float or int, (default 1.0) maximum value of the data 
                after it has been scaled
            axis: int or None (default None). Axis to scale arrays over. None
                results in a global scaling of the arrays.
        """
        self.u_min, self.u_max = u_min, u_max
        self.v_min, self.v_max = v_min, v_max
        self.axis=axis

    def __call__(self, u):
        """
        Takes a tuple of numpy arrays and scales them based
        linearly from [u_min, u_max] to [v_min, v_max]. Returns
        a tuple of scaled arrays in the same order they were passed.
        If axis != None, the arrays are scaled along that axis.
        If axis != None, all arrays must have the same number of
        dimensions, and they must have the same number entries over
        the specified axis.
        """
        assert isinstance(u,tuple), "Must put arrays in as tuple"
        
        if self.axis is not None:
            ndim = u[0].ndim
            nchannels = u[0].shape[self.axis]
            slc = [slice(None)]*ndim
            
            for x in u:
                assert x.ndim == ndim, "Number of dimensions must match for axis != None"
                assert x.shape[self.axis] == nchannels, f"Number of entries in axis {self.axis} do not match across data"
            
            if self.u_max == None and self.u_min == None:
                self.u_max, self.u_min = [], []
                for i in range(nchannels):
                    slc[self.axis] = i
                    self.u_max.append(np.array([x[tuple(slc)].max() for x in u]).max())
                    self.u_min.append(np.array([x[tuple(slc)].min() for x in u]).min())
            else:
                assert len(self.u_max) == nchannels and len(self.u_min) == nchannels, \
                f"u_max and u_min must be arrays of size {nchannels}"
            
            v = []
            for x in u:
                v_temp = np.zeros(x.shape)
                for i in range(nchannels):
                    slc[self.axis] = i                   
                    x_std = (x[tuple(slc)] - self.u_min[i])/(self.u_max[i] - self.u_min[i])
                    v_temp[tuple(slc)] = x_std*(self.v_max - self.v_min) + self.v_min
                v.append(v_temp)
            v = tuple(v)
                  
        else:
            if self.u_max == None and self.u_min == None:
                self.u_max = np.array([x.max() for x in u]).max()
                self.u_min = np.array([x.min() for x in u]).min()

            v = tuple((u - self.u_min)*(self.v_max - self.v_min)/(self.u_max - self.u_min) + self.v_min for u in u)
        return v
    
    def scale_inverse(self, v):
        """
        Takes a tuple of numpy arrays and scales them based
        linearly from [v_min, v_max] to [u_min, u_min] and returns
        a tuple of scaled arrays
        """
        assert isinstance(v,tuple), "Must put arrays in as tuple"
        
        if self.axis is not None:
            ndim = v[0].ndim
            nchannels = v[0].shape[self.axis]
            slc = [slice(None)]*ndim
            u = []
            for x in v:
                u_temp = np.zeros(x.shape)
                for i in range(nchannels):
                    slc[self.axis] = i                   
                    x_std = (x[tuple(slc)] - self.v_min)/(self.v_max - self.v_min)
                    u_temp[tuple(slc)] = x_std*(self.u_max[i] - self.u_min[i]) + self.u_min[i]
                u.append(u_temp)
            u = tuple(u)

        else:
            u = tuple((x - self.v_min)*(self.u_max - self.u_min)/(self.v_max - self.v_min) + self.u_min for x in v)
        return u


def compute_pod_multicomponent(S_pod,subtract_mean=True,subtract_initial=False,full_matrices=False):
    """
    Compute standard SVD [Phi,Sigma,W] for all variables stored in dictionary S_til
     where S_til[key] = Phi . Sigma . W is an M[key] by N[key] array
    Input:
    :param: S_pod -- dictionary of snapshots
    :param: subtract_mean -- remove mean or not
    :param: full_matrices -- return Phi and W as (M,M) and (N,N) [True] or (M,min(M,N)) and (min(M,N),N)
    Returns:
    S      : perturbed snapshots if requested, otherwise shallow copy of S_pod
    S_mean : mean of the snapshots
    Phi : left basis vector array
    sigma : singular values
    W   : right basis vectors
    """
    S, S_mean = {},{}
    Phi,sigma,W = {},{},{}

    for key in list(S_pod.keys()):
        if subtract_mean:
            S_mean[key] = np.mean(S_pod[key],1);
            S[key] = S_pod[key].copy();
            S[key]-= np.tile(S_mean[key],(S_pod[key].shape[1],1)).T
            Phi[key],sigma[key],W[key] = np.linalg.svd(S[key][:,1:],full_matrices=full_matrices)

        elif subtract_initial:
            S_mean[key] = S_pod[key][:,0]
            S[key] = S_pod[key].copy()
            S[key]-= np.tile(S_mean[key],(S_pod[key].shape[1],1)).T
            Phi[key],sigma[key],W[key] = np.linalg.svd(S[key][:,:],full_matrices=full_matrices)
        else:
            S_mean[key] = np.mean(S_pod[key],1)
            S[key] = S_pod[key]
            Phi[key],sigma[key],W[key] = np.linalg.svd(S[key][:,:],full_matrices=full_matrices)

    return S,S_mean,Phi,sigma,W


def compute_trunc_basis(D,U,eng_cap = 0.999999,user_rank={}):
    """
    Compute the number of modes and truncated basis to use based on getting 99.9999% of the 'energy'
    Input:
    D -- dictionary of singular values for each system component
    U -- dictionary of left singular basis vector arrays
    eng_cap -- fraction of energy to be captured by truncation
    user_rank -- user-specified rank to over-ride energy truncation (Empty dict means ignore)
    Output:
    nw -- list of number of truncated modes for each component
    U_r -- truncated left basis vector array as a list (indexed in order of dictionary keys in D)
    """

    nw = {}
    for key in list(D.keys()):
        if key in user_rank:
            nw[key] = user_rank[key]
            nw[key] = np.minimum(nw[key], D[key].shape[0]-2)
            print('User specified truncation level = {0} for {1}\n'.format(nw[key],key))
        else:
            nw[key] = 0
            total_energy = (D[key]**2).sum(); assert total_energy > 0.
            energy = 0.
            while energy/total_energy < eng_cap and nw[key] < D[key].shape[0]-2:
                nw[key] += 1
                energy = (D[key][:nw[key]]**2).sum()
            print('{3} truncation level for {4}% = {0}, \sigma_{1} = {2}'.format(nw[key],nw[key]+1,
                                                            D[key][nw[key]+1],key,eng_cap*100) )

    U_r = {}
    for key in list(D.keys()):
        U_r[key] = U[key][:,:nw[key]]

    return nw, U_r