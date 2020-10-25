"""
colpal.py

A module of gen_colors_tex
"""

__author__ = "Roman Rüttimann"
#%%
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import warnings
    import matplotlib as mpl
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
    import copy
    import pandas as pd
except Exception as e:
    print("Not all Modules have been loaded. {}".format(e))

def format_axes(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(direction='in', length=0, width=0)
    ax.grid(color='black', linestyle='-', linewidth=2)
    for child in ax.get_children():
        if isinstance(child, mpl.spines.Spine):
            child.set_color('black')
            child.set_linewidth(2)


def preview_cmap_axes(cmap, num=None):
    if num == None or num > cmap.N:
        num = cmap.N

    bounds = [0,1,cmap.N]
    norm = BoundaryNorm(bounds, cmap.N)
    data = np.atleast_2d(np.arange(num))

    ax = plt.axes()
    ax.imshow(data, extent=[0,num,0,1], cmap=cmap)
    format_axes(ax)
    ax.set_xticks(data[0,:])
    ax.set_yticks([0,1])

def listOfTuples(l0, l1):
    return list(map(lambda x, y:(x,y), l0, l1))

class Colpal:
    #------------------------------------------------------------------
    # HELPER FUNCTIONS
    def hex_to_rgb(_,h):
        rgb = tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0,2,4))
        return rgb

    def rgb_to_hex(_,rgb):
        h = '#' + ('{:02x}'*3).format(*rgb)
        return h

    def hsl_to_rgb(_,hsl):
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

    def rgb_to_hsl(_,rgb):
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
    
    def rgb_to_rel(_,rgb):
        relative = tuple(v/255 for v in rgb)
        return relative
    
    #------------------------------------------------------------------------
    # CLASS METHODS
    def __init__(self, obj):
        if isinstance(obj, pd.core.frame.DataFrame):
            self.name = obj.ColorName.iloc[0]
            self.type = obj.Type.iloc[0]
            colors = np.zeros([len(obj),3], dtype=np.uint8)
            for i,v in enumerate(obj.RGB_HTML):
                colors[i,:] = np.array([int(v.lstrip('#')[i:i+2], 16) for i in (0,2,4)], dtype=np.uint8)
            self.colors = colors
        elif isinstance(obj, Colpal):
            self.name = obj.name
            self.type = obj.type
            self.colors = obj.colors
    
    def get_name(self):
        return copy.copy(self.name)

    def set_name(self, name):
        self.name = name
    
    def get_type(self):
        return copy.copy(self.type)

    def set_type(self, type_):
        self.type = type_

    def get_colors(self):
        return copy.copy(self.colors)

    def set_colors(self, colors):
        self.colors = colors

    def __len__(self):
        return self.colors.shape[0]
    
    def size(self):
        return self.colors.shape[0]
    
    def rgb(self, idx=None):
        if idx == None:
            idx = range(self.size())
        colarr = np.atleast_2d(self.colors[idx,:])
        if colarr.shape[0] <= 1:
            return tuple(colarr[0,:])
        else:
            return list(map(tuple, colarr))
    
    def rgb_rel(self, idx=None):
        if idx == None:
            idx = range(self.size())
        colarr = np.atleast_2d(self.colors[idx,:])
        relarr = np.apply_along_axis(self.rgb_to_rel,1,colarr)
        if colarr.shape[0] <= 1:
            return tuple(relarr[0,:])
        else:
            return list(map(tuple, relarr))
    
    def rgb_hex(self, idx=None):
        if idx == None:
            idx = range(self.size())
        colarr = np.atleast_2d(self.colors[idx,:])
        hexarr = np.apply_along_axis(self.rgb_to_hex,1,colarr)
        if colarr.shape[0] <= 1:
            return hexarr[0]
        else:
            return list(hexarr)

    def hsl(self, idx=None):
        if idx == None:
            idx = range(self.size())
        colarr = np.atleast_2d(self.colors[idx,:])
        hslarr = np.apply_along_axis(self.rgb_to_hsl,1,colarr)
        if colarr.shape[0] <= 1:
            return tuple(hslarr[0,:])
        else:
            return list(map(tuple, hslarr))

    def set_hsl_value(self, value, idx=None, rel=True, valtype=0, invert=False):
        maxima = (360, 1, 1)
        m = 1
        if type(value) is list:
            m = len(value)
        n = self.size()
        if idx == None:
            p = n
            idx = list(range(n))
        else:
            p = 1
            if isinstance(idx, list):
                p = len(idx)
        colarr = self.colors
        hslarr = np.apply_along_axis(self.rgb_to_hsl,1,np.atleast_2d(colarr))
        currvals = np.atleast_1d(hslarr[:,valtype])
        if m == 1 and p == n:
            newvals = np.repeat(value, p)
        elif m <= p:
            newvals = np.array(value)
        elif m > p:
            newvals = np.array(value[range(p)])
        if rel:
            if invert:
                newvals = maxima[valtype] - np.multiply(maxima[valtype]-currvals[:newvals.size], newvals)
            else:
                newvals = np.multiply(newvals, currvals[:newvals.size])
        else:
            if invert:
                newvals = maxima[valtype] - newvals
        hslarr[:newvals.size, valtype] = newvals
        newcolrows = np.apply_along_axis(self.hsl_to_rgb,1,hslarr)
        if isinstance(idx, list):
            for i,v in enumerate(idx):
                colarr[v,:] = newcolrows[i,:]
        else:
            colarr[idx,:] = newcolrows[0,:]
        self.set_colors(colarr)
    
    def set_hue(self, value, idx=None, rel=True):
        self.set_hsl_value(value, idx, rel, 0)
    
    def set_saturation(self, value, idx=None, rel=True):
        self.set_hsl_value(value, idx, rel, 1)

    def set_lightness(self, value, idx=None, rel=True):
        self.set_hsl_value(value, idx, rel, 2)
    
    def set_darkness(self, value, idx=None, rel=True):
        self.set_hsl_value(value, idx, rel, 2, True)
    
    def colmap_array(self):
        rel_rgb = np.apply_along_axis(self.rbg_to_rel,1,self.colors)
        cmap_arr = np.append(rel_rgb, np.ones((rel_rgb.size[0],1)), axis=1)
        return cmap_arr

    def ListedColormap(self, name=None):
        if name == None:
            name = self.name
        rel_rgb = np.apply_along_axis(self.rgb_to_rel,1,self.colors)
        cmap = ListedColormap(np.append(rel_rgb, np.ones([self.size(),1]), axis=1), name=name)
        cmap = ListedColormap(np.append(rel_rgb, np.ones([self.size(),1]), axis=1))
        return cmap
    
    def LinearSegmentedColormap(self, name=None):
        if name == None:
            name = self.name
        cmap = LinearSegmentedColormap.from_list(name, self.rgb_rel())
        return cmap

    def preview_cmap(self, num=None):
        return preview_cmap_axes(self.ListedColormap(), num)

    def previewColors(self, num=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax = self.preview_cmap(num)
        plt.show()
# %%
