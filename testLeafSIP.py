#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 02:40:04 2020

@author: chen671
"""

import matplotlib.pyplot as plt
import seaborn as sns; 
sns.set(style="ticks", color_codes=True)

from LeafSIP import LeafSIPmodel

Ccar=10;       # Carotenoids (cm-2.microg)
Cant=0;        # Anthocyanins (cm-2.microg)
Cbrw=0.0;      # brown pigments (arbitrary units)
N=1.518;       # Leaf layer
Cab=58.0;      # Chlorophyll (a+b)(cm-2.microg)
Cw=0.013100;   # Water  (cm)
Cm=0.003662;   # dry matter (cm-2.g)
Kparafile = 'dataSpec_PDB.csv'
LeafPara = [N, Cab, Ccar, Cant, Cbrw, Cw, Cm]
alpha = 40

df = LeafSIPmodel(Kparafile, LeafPara, alpha)
df1 = pd.melt(df, id_vars = ['lambda','alpha'], 
        value_vars = ['R','T','w'])

g = sns.relplot(x = "lambda", y = "value", 
                col = "variable", col_wrap = 2,
                hue = "variable",
                kind="line", data = df1)

