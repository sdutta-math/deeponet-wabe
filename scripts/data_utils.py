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