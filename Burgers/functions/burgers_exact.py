#! /usr/bin/env python3
#

import numpy as np



def init_cond(x,Re):
    t0 = np.exp(Re/8)
    denominator = 1 + np.sqrt(1/t0) * np.exp(Re*(x**2)/4)
    return x/denominator

def true_solution(x,t,Re):
    t0 = np.exp(Re/8)
    numerator = x/(t+1)
    denominator = 1 + np.sqrt((t+1)/t0) * np.exp(Re*(x**2)/(4*t+4))
    return numerator/denominator


# ## Generate numerical solution
# uh = {}
# for r, Re in enumerate(Re_list):
    
#     uh[r] = {}
#     uh[r]['burgers'] = np.zeros((xx.shape[0],tt.shape[0]))
#     for it,tn in enumerate(tt):
#         uh[r]['burgers'][:,it] = true_solution(xx,tn,Re)

