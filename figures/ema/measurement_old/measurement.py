# %%
%matplotlib inline
from graphviz import Graph
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# # if using color spectrum (for map plots)
# from palettable.wesanderson import Zissou_5

colpal_std = ['#0099E6', '#12255A', '#F23814', '#DFB78B', '#B6C3C5']
colpal_2xlight = ['#75D2FF', '#597CDE', '#F99C8B', '#EEDAC3', '#DAE0E1']
colpal_4xlight = ['#B8E8FF', '#AABCEE', '#FCCEC5', '#F7EEE3', '#EEF1F2']

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
    d['style'] = 'filled'
    if color is None:
        color = 'white'
    d['color'] = color
    return d



# %%
g = Graph('G')
g.node('Structure', shape='none')
g.node(**box_attr('Exciter', colpal_4xlight[0]))
g.node(**box_attr('Sens', colpal_4xlight[1], 'Sensing Element'))
g.node(**box_attr('Converter', colpal_4xlight[2]))
g.node(**box_attr('Amplifier', colpal_4xlight[3]))
g.node(**box_attr('Transmission', colpal_))


for i in range(1,6):
    g.node('Structure'.format(i), shape='rect')

with g.subgraph() as s:
    s.attr(rank='same')
    s.node('PO')
    for j in range(1,3):
        s.node('dot{:d}'.format(j), shape='point', width='0')
    s.edge('PO', 'dot1')
    s.edge('dot1', 'dot2', arrowhead='none')

g.node('WO')
g.edge('dot1', 'WO')

with g.subgraph() as s:
    s.attr(rank='same')
    for j in range(1,3):
        s.node('dot2{:d}'.format(j), shape='point', width='0')
    s.edge('WO', 'dot21')
    s.edge('dot21', 'dot22')

g.node('mt', shape='none', label='', image='mt_scheme.svg')
g.edge('dot2', 'mt')

g.render('out', view=False)


# %%
g.attr['rankdir'] = 'LR'
# ranks
for i in range(4):
    g.node(str(i), style='invis')
    if i:
        g.edge(str(i-1), str(i), style='invis')
# head
g.attr('node', **box_attr(None, colpal_4xlight[0], None))
g.node(*list(head_str[0]))
# mount
g.attr('node', **box_attr(None, colpal_4xlight[1], None))
for i in range(mount_str.shape[0]):
    g.node(*list(mount_str[i]))
    g.edge(head_str[0][0], mount_str[i][0])
# materials
g.attr('node', **box_attr(None, colpal_4xlight[2], None))
for i in range(mat_str.shape[0]):
    g.node(*list(mat_str[i]))
    match = next((x for x in mount_str[:,0] if x in mat_str[i,0]), False)
    if match:
        g.edge(match, mat_str[i,0])
# pins
g.attr('node', **box_attr(None, colpal_4xlight[3], None))
for i in range(pin_str.shape[0]):
    g.node(*list(pin_str[i]))
    g.edge('surf_pla', pin_str[i,0])
# power
g.attr('node', **box_attr(None, colpal_4xlight[4], None))
for i in range(pow_str.shape[0]):
    g.node(*list(pow_str[i]))
    match = next((x for x in np.concatenate((mat_str[:,0],pin_str[:,0])) if x in pow_str[i,0]), False)
    if match:
        g.edge(match, pow_str[i,0])
# packages
g.attr('node', **box_attr(None, None, None, False))
for i in range(pack_map.shape[0]):
    for name in pack_map[i,1]:
        g.node(name, name)
        g.edge(pack_map[i,0], name)
# ranking
edgelist = list(head_str[0,0])
edgelist = list(mount_str[:,0])
edgelist = list(mat_str[:,0])
edgelist = list(pin_str[:,0])
edgelist = list(pow_str[:,0])
edgelist = list(np.concatenate(pack_map[:,1]))
out = '\t{rank=end; ' + ('{}' + (len(edgelist)-1) * ' -- {}').format(*edgelist) + '; }'


g.render('out.dot', view=False)