# %%
%matplotlib inline
from graphviz import Graph
import numpy as np
import re
import warnings
import colorsys
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from palettable.wesanderson import Zissou_5

Zissou_5.show_as_blocks()

# %%
def hex_to_rgb(h):
    rgb = tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0,2,4))
    return rgb

def rgb_to_hex(rgb):
    h = '#' + ('{:02x}'*3).format(*rgb)
    return h

def hsl_to_rgb(hsl):
    # 0<=H<=360, 0<=S<=1, 0<=L<=1
    H, S, L = hsl
    C = (1-abs(2*L-1))*S
    X = C*(1-abs((H/60)%2-1))
    m = L-C/2
    if 0<=H<60:
        RGB = (C,X,0)
    elif 60<=H<120:
        RGB = (X,C,0)
    elif 120<=H<180:
        RGB = (0,C,X)
    elif 180<=H<240:
        RGB = (0,X,C)
    elif 240<=H<300:
        RGB = (X,0,C)
    elif 300<=H<360:
        RGB = (C,0,X)
    else:
        warnigns.warn('Hue value must lie between 0... and 360°')
        return (0,0,0)
    rgb = tuple(int((v+m)*255) for v in RGB)
    return rgb

def rgb_to_hsl(rgb):
    # 0<=R<=255, 0<=G<=255, 0<=B<=255
    R, G, B = tuple(v/255 for v in rgb)
    C_lim = min(R,G,B), max(R,G,B)
    C_del = C_lim[1]-C_lim[0]
    L = (C_lim[0] + C_lim[1]) / 2
    if C_del==0:
        H = 0
        S = 0
    else:
        S = C_del/(1-abs(2*L-1))
        if C_lim[1]==R:
            H = 60*((G-B)/C_del %6)
        elif C_lim[1]==G:
            H = 60*((B-R)/C_del + 2)
        elif C_lim[1]==B:
            H = 60*((R-G)/C_del + 4)
        else:
            warnings.warn('Invalide rgb values')
            return(0,0,0)
    hsl = H,S,L
    return hsl

class mycolor:
    def __init__(self, cols):
        if type(cols) is not list:
            cols = [cols]
        scols = cols
        for n, col in enumerate(cols):
            if type(col) is tuple:
                print('tuple')
                if len(col)==3 and all([np.isscalar(v) for v in col]):
                    print('scalar length')
                    if all([0 <= v <= 255 for v in col]):
                        print('range')
                        scols[n] = col
            elif col is str:
                if bool(re.match(r'^#(?:[0-9a-fA-F]{3}){1,2}', col)):
                    scols[n] = hex_to_rgb(col)
            else:
                scols[n] = (0,0,0)
                warnings.warn('Invalid color')
        self.cols = scols

    def rgb(self,idx=None):
        num = len(self.cols)
        if idx == None:
            return self.cols
        elif -num<=idx<num:
            return self.cols[idx]
    def hsl(self,idx=None):
        num = len(self.cols)
        if idx == None:
            return [rgb_to_hsl(col) for col in self.cols]
        elif -num<=idx<num:
            return rgb_to_hsl(self.cols[idx])
    def float(self,idx=None):
        num = len(self.cols)
        if idx == None:
            return [tuple(float(v)/255 for v in col) for col in self.cols]
        elif -num<=idx<num:
            return tuple(float(v)/255 for v in self.cols[idx])
    def hex(self,idx=None):
        num = len(self.cols)
        if idx == None:
            return [rgb_to_hex(col) for col in self.cols]
        elif -num<=idx<num:
            return rgb_to_hex(self.cols[idx])
    def set_hsl(self,cols,idx=None):
        scols = self.cols
        num = len(scols)
        if idx == None:
            self.cols = [hsl_to_rgb(col) for col in cols]
        elif -num<=idx<num:
            scols[idx] = hsl_to_rgb(cols)
            self.cols = scols
    def change_hsl(self,cols,rel=False,idx=None):
        # change all values that are not None and inside range
        curr = self.hsl()
        new = [list(col) for col in curr]
        num = len(curr)
        start = 0
        if idx is not None:
            if -num<=idx<num:
                if idx<0:
                    start = num-idx
                else:
                    start = idx
        if type(cols) is list:
            arr = cols
        else:
            if idx is None:
                arr = [cols] * num
            else:
                arr = [cols]
        for i, col in enumerate(arr):
            for n, v in enumerate(col):
                if v is not None:
                    if rel:
                        v = v * curr[i+start][n]
                    if n==0 and 0<=v<=360:
                        new[i+start][n] = v
                    elif 0<=v<=1:
                        new[i+start][n] = v
        rgb = [hsl_to_rgb(tuple(col)) for col in new]
        self.cols = rgb


colors = Zissou_5.colors
mycol = mycolor(colors)
print(mycol.rgb())
print(mycol.hsl())
mycol.change_hsl((None, None, 0.5),True)
print('\n')
print(mycol.rgb())
print(mycol.hsl())


# Attribute presets
class dict_cond(dict):
    def __init__(self, cond=lambda x: x is not None):
        self.cond = cond
    def __setitem__(self, key, value):
        if key in self or self.cond(value):
            dict.__setitem__(self, key, value)

def box_attr(name, color=None, label=None, frame=True):
    d = dict_cond()
    if label is None:
        label = name
    d['name'] = name
    d['label'] = label
    d['shape'] = 'box'
    if color is not None:
        d['style'] = 'filled'
        d['fillcolor'] = color
    return d

# %%
g = Graph('G')
g.node('A', 'King Arthur')
g.node('B', 'Sir Bedevere the Wise')
with g.subgraph(name='child', node_attr=box_attr('name', 'blue', 'label')) as c:
    c.edge('foo', 'bar')

g.render('out.dot', view=False)
# %%
