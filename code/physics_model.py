import numpy as np


def FVD(arg, vi, delta_v, delta_d):
    alpha, lamda, v_0, b, beta = arg
    V_star = v_0 * (np.tanh(delta_d / b - beta) - np.tanh(-beta))
    ahat = alpha * (V_star - vi) + lamda * delta_v
    return ahat

def IDM(arg, vi, delta_v, delta_d):
    vf, A, b, s0, T = arg
    vi = np.asarray(vi)
    delta_v = np.asarray(delta_v)
    delta_d = np.asarray(delta_d)
    
    s_star = s0 + np.maximum(0, vi * T + (vi * delta_v) / (2 * (A * b) ** 0.5))
    
    #print('A*b',A*b)
    epsilon = 1e-5
    ahat = A*(1 - (vi/vf)**4 - (s_star/(delta_d+epsilon))**2)
    
    return ahat

