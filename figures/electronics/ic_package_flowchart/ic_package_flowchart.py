# %%
%matplotlib inline
from graphviz import Graph
import numpy as np
import re
import warnings
import colorsys
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
head_str = np.array([('ichead', 'IC Packages')])
mount_str = np.array([
    ('through',      'Through Hole Packages'),
    ('surf',         'Surface Mount Packages'),
    ('contl',        'Contactless Packages')])
mat_str = np.array([
    ('through_cer',  'Ceramic'),
    ('through_pla',  'Plastic'),
    ('surf_cer',     'Ceramic'),
    ('surf_pla',     'Plastic'),
    ('surf_met',     'Metal')])
pin_str = np.array([
    ('dual',         'Dual'),
    ('quad',         'Quad'),
    ('array',        'Array')])
pow_str = np.array([
    ('through_cer_p', 'Power'),
    ('through_pla_p', 'Power'),
    ('surf_cer_p',   'Power'),
    ('dual_p',       'Power'),
    ('quad_p',       'Power'),
    ('array_p',      'Power')])
pack_map = np.array([
    ('through_cer_p',   ['HCPGA']),
    ('through_pla_p',   ['CDIL', 'CPGA']), 
    ('surf_cer_p',      ['DBS', 'HDIP', 'RBS', 'RDBS', 'SIL', 'TBS']),
    ('dual_p',          ['HSOP', 'HTSSOP']),
    ('quad_p',          ['HBCC', 'HLQFP', 'HQFP', 'HSQFP', 'HTQFP']),
    ('array_p',         ['HBGA']),
    ('through_cer',     ['CDIL', 'CPGA']),
    ('through_pla',     ['DIP', 'SDIP', 'SIL']),
    ('surf_cer',        ['CLLCC', 'CWQCCN']),
    ('dual',            ['PMFP', 'SO', 'SSOP', 'TSSOP', 'PMFP']),
    ('quad',            ['BBC', 'LQFP', 'PLCC', 'QFP', 'SQFP', 'TQFP', 'SQFP']),
    ('array',           ['BGA', 'LFBGA', 'TFBGA', 'VFBGA']),
    ('surf_met',        ['MSQFP'])], dtype=object)

# %%
g = Graph('G')
g.graph_attr['rankdir'] = 'LR'
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
#g.body.append(out)
# g.body.append('\t\{{str}\}')
#     edgelist = ['1', *list(mount_str[:,0])]
# for i in range(1,len(edgelist)):
#     c.edge(edgelist[i-1], edgelist[i])


# with g.subgraph(node_attr=box_attr(None, colpal_4xlight[0], None)) as c:
#     c.attr('node', box_attr(None, colpal_std[0], None))
# with g.subgraph(node_attr=box_attr(None, colpal_4xlight[1], None)) as c:
#     for i in range(mount_str.shape[0]):
#         c.node(*list(mount_str[i]))
#         c.edge(head_str[0][0], mount_str[i][0])
# with g.subgraph(node_attr=box_attr(None, colpal_4xlight[2], None)) as c:
#     for i in range(mat_str.shape[0]):
#         c.node(*list(mat_str[i]))
#         match = next((x for x in mount_str[:,0] if x in mat_str[i,0]), False)
#         if match:
#             c.edge(match, mat_str[i,0])
# with g.subgraph(node_attr=box_attr(None, colpal_4xlight[3], None)) as c:
#     for i in range(pin_str.shape[0]):
#         c.node(*list(pin_str[i]))
#         c.edge('surf_pla', pin_str[i,0])
# with g.subgraph(node_attr=box_attr(None, colpal_4xlight[4], None)) as c:
#     for i in range(pow_str.shape[0]):
#         c.node(*list(pow_str[i]))
#         match = next((x for x in np.concatenate((mat_str[:,0],pin_str[:,0])) if x in pow_str[i,0]), False)
#         if match:
#             c.edge(match, pow_str[i,0])
# with g.subgraph(node_attr=box_attr(None,None,None,False)) as c:
#     for i in range(pack_map.shape[0]):
#         for name in pack_map[i,1]:
#             c.node(name, name)
#             c.edge(pack_map[i,0], name)

g.render('out.dot', view=False)
# %%

