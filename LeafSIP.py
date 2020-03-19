# -*- coding: utf-8 -*-
"""
Leaf Radiative Transfer Model based on Spectral Invariant Properties

@author: Yelu Zeng, Shengbiao Wu, Min Chen, and Dalei Hao

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from numpy import exp, log, sin, pi
from scipy.special import expn

#%%

def LeafSIPmodel(Kparafile, LeafPara, alpha):
    # Kparafile: name of the file that contains key coefficients, see below
    # LeafPara: a 1D array of N, Cab, Ccar, Cant, Cbrw, Cw, Cm
    # alpha: incidence angle, the angle between the normal to 
    #        the leaf facet and illumination direction
    
    data = pd.read_csv(Kparafile)
    
    #%% Get coefficients (Ks) for leaf bioparameters
    
    # wavelengths
    lamda = data['lambda']

    # refractive index
    nr = data['nr']
    
    # K for chlorophyll (a+b) content
    Kab = data['Kab']
    
    # K for carotenoids
    Kcar = data['Kcar']
    
    # K for anthocyanins
    Kant = data['Kant']
    
    # K for brown pigments
    Kbrw = data['Kbrw']
    
    # K for water
    Kw = data['Kw']
    
    # K for dry matter
    Km = data['Km']
    
    #%% Get Leaf bioparameters
    
    # N:    Leaf 'layer' - similar definition as in PROSPECT
    # Cab:  Chlorophyll (a+b)(cm-2.microg)
    # Ccar: Carotenoids (cm-2.microg)
    # Cant: Anthocyanins (cm-2.microg)
    # Cbrw: Brown pigments (arbitrary units)
    # Cw:   Water  (cm)
    # Cm:   Dry matter (cm-2.g)
    N, Cab, Ccar, Cant, Cbrw, Cw, Cm = LeafPara
    
    #%% Calculate spectral invariants
    
    # K: absorption parameter
    K = Cab*Kab + Ccar*Kcar + Cant*Kant + Cbrw*Kbrw + Cw*Kw + Cm*Km
    K0 = K / N
    K1 = K0 * (N-1)
    
    # reference leaf albedo
    w0 = exp(-K1)
    
    # p: recollision probability
    fN = -0.1812*N**4 + 2.1960*N**3 - 10.0600*N**2 + 20.47*N - 12.30
    p = 1 - (1-exp(-fN)) / fN
    
    # q: scattering asymmetry parameter 
    tmp1 = 0.6035 - 1.2671*N**2 + 0.1902*N**4
    tmp2 = 1 - 0.8048*N**2 + 0.2329*N**4
    q = tmp1/tmp2
    
    # Single scattering albedo
    w = w0 * (1-p) / (1 - p*w0)
    
    #%% reflectance and transmittance

    # transmittance coefficient of a thin leaf plate
    theta = (1-K0) * exp(-K0) + K0**2 * expn(1,K0)

    # transmissivity of a leaf-air interface at alpha and 90 degree
    tav_alpha = calctav(alpha, nr)
    tav_90    = calctav(90, nr)
    
    # Ta: transmittance of the top 1 plate
    tmp1 = tav_90 * tav_alpha * theta * nr**2
    tmp2 = nr**4 - theta**2 * (nr**2 - tav_90)**2
    Ta = tmp1 / tmp2

    # Ra: reflectance of the top 1 plate    
    tmp1 = tav_90 * tav_alpha * theta**2 * (nr**2 - tav_90)
    tmp2 = nr**4 - theta**2 * (nr**2 - tav_90)**2
    Ra = 1 - tav_alpha + tmp1/tmp2
    
    # T90: transmittance of the top 1 plate when alpha = 90
    tmp1 = tav_90 * tav_90 * theta * nr**2
    tmp2 = nr**4 - theta**2 * (nr**2 - tav_90)**2
    T90 = tmp1 / tmp2

    # R90: reflectance of the top 1 plate when alpha = 90    
    tmp1 = tav_90 * tav_90 * theta**2 * (nr**2 - tav_90)
    tmp2 = nr**4 - theta**2 * (nr**2 - tav_90)**2
    R90 = 1 - tav_90 + tmp1/tmp2

    # Tb and Rb: transmittance and reflectance of 
    # the bottom N-1 plate
    Rb = w * (1/2 + q/2 * (1-p*w0)/(1-p*q*w0))
    Tb = w * (1/2 - q/2 * (1-p*w0)/(1-p*q*w0))
    
    # total reflectance (R) and transmittance (T)
    R = Ra + Ta * Rb * T90 / (1 - R90*Rb)
    T = Ta * Tb / (1 - R90*Rb)
    
    #%% output
    outdf = pd.DataFrame({'lambda': lamda, 
                          'R': R,
                          'T': T,
                          'p': p,
                          'q': q,
                          'w0': w0,
                          'w': w,
                          'alpha': alpha})
    return outdf

#%%
def calctav(alfa, nr):
    # ***********************************************************************
    # Stern F. (1964), Transmission of isotropic radiation across an
    # interface between two dielectrics, Appl. Opt., 3(1):111-113.
    # Allen W.A. (1973), Transmission of isotropic light across a
    # dielectric surface in two and three dimensions, J. Opt. Soc. Am.,
    # 63(6):664-666.
    # ***********************************************************************

    rd = pi / 180
    n2 = nr ** 2
    np = n2 + 1
    nm = n2 - 1
    a  = (nr + 1) * (nr + 1) / 2
    k  = -(n2 - 1) * (n2 - 1) / 4
    sa  =  sin(alfa * rd)

    b1  = (alfa != 90) * ((sa**2 - np/2) * (sa**2 - np/2) + k) ** 0.5
    b2  = sa**2 - np/2
    b   = b1 - b2
    b3  = b ** 3
    a3  = a ** 3
    ts  = (k**2/(6*b3) + k/b - b/2) - (k**2/(6*a3) + k/a - a/2)

    tp1 = -2 * n2 * (b-a) / (np**2)
    tp2 = -2 * n2 * np * log(b/a) / (nm**2)
    tp3 = n2 * (1/b - 1/a) / 2;
    tp4 = 16 * n2**2 * (n2**2 + 1) * log((2*np*b - nm**2) / 
                        (2*np*a - nm**2)) / (np**3 * nm**2)
    tp5 = 16 * n2**3 * (1/(2*np*b - nm**2) - 1/(2*np*a - nm**2)) / (np**3)
    tp  = tp1 + tp2 + tp3 + tp4 + tp5
    tav = (ts+tp) / (2 * sa**2)

    return tav