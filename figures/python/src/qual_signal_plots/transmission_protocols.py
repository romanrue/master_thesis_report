#%%
try:
    from pathlib import Path
    import numpy as np
    from scipy import signal
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from cycler import cycler
    from svgpathtools import svg2paths
    from svgpath2mpl import parse_path
    from matplotlib.markers import MarkerStyle
    from collections import OrderedDict

    # colors
    from colors.gen_colors import colpalettes
except Exception as e:
    print("Some Modules are missing. {}".format(e))

# linestyles
linestyle_tuple = [
    ('solid',                 (0,())),

    # ('loosely dashed',        (0, (4, 8))),
    ('dashed',                (0, (4, 3))),
    # ('densely dashed',        (0, (4, 1))),

    # ('loosely dotted',        (0, (1, 10))),
    ('dotted',                (0, (1, 1))),
    # ('densely dotted',        (0, (1, 1))),

    # ('loosely dashdotted',    (0, (3, 10, 1, 10))),
    ('dashdotted',            (0, (4, 3, 1, 3))),
    ('densely dashdotted',    (0, (4, 1, 1, 1))),

    # ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10)))
    # ('dashdotdotted',         (0, (4, 3, 1, 3, 1, 3))),
    ('densely dashdotdotted', (0, (4, 1, 1, 1, 1, 1))),
]
resize_factor = 0.8
for i, el in enumerate(linestyle_tuple):
    linestyle_tuple[i] = (el[0], (el[1][0], tuple([resize_factor*x for x in el[1][1]])))
linestyle_list = [linestyle for (_,linestyle) in linestyle_tuple]

# colors
for el in colpalettes:
    if el.name == 'WesMixL':
        color_list = el.rgb_hex()
        # set color cycle as standard
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color',el.rgb_hex())
    elif el.name == 'WesMixUL':
        ulcolors = el.rgb_hex()

# custom cycle must be called in every plot
ccs = [ (cycler(color=color_list[:len(linestyle_list)]) + cycler(linestyle=linestyle_list)),
       (cycler(color= color_list[:6]) + cycler(linestyle=linestyle_list[:2]*3))]


testenv = True
save_output = True
filenames = {
    'volt_out': 'op_amp_transfer_volt',
    'curr_out': 'op_amp_transfer_curr',
}

mpl.rcParams['figure.figsize'] = [dim/2.54 for dim in [9,6.75]]
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['font.size'] = 11
mpl.rcParams['legend.fontsize'] = 'medium'
mpl.rcParams['axes.labelsize'] = 'medium'
mpl.rcParams['figure.titlesize'] = 'medium'

mpl.rcParams['lines.linewidth'] = 2

if not testenv:
    mpl.use('PDF')
mpl.rc('font',**{'family': 'serif', 'sans-serif': ['Helvetica']})
#mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rc('pdf', use14corefonts=True)

dir_path = Path.cwd()

### Import svg paths of markers
def importsvgpaths(directory):
    if isinstance(directory,str):
        directory=Path(directory)
    svgs = list(directory.glob('*.svg'))
    svgpath_dict = {}
    for f in svgs:
        key = f.stem
        paths, _ = svg2paths(str(f))
        svgpath_dict[key] = paths[0].d()
        if key.endswith('_r'):
            base_name = key[:-2]
            for i,ending in enumerate(('_u', '_l', '_d')):
                angle = (i+1)*90
                svgpath_dict[''.join([base_name,ending])] = paths[0].rotated(angle).d()
    return svgpath_dict

def dict_svgpath_to_mpl(svgpathdict):
    marker_dict={}
    for key in svgpath_dict:
        marker_dict[key] = parse_path(svgpathdict[key])
        # center horizontally
        marker_dict[key].vertices -= marker_dict[key].vertices.mean(axis=0)
    return marker_dict

svgpath_dict = importsvgpaths(dir_path.parent / 'markers')
marker_dict = dict_svgpath_to_mpl(svgpath_dict)

# %%
### Visual parameters
aspRatio=4/3            # figure aspect ratio
keepAspRatio=True       # if false height is taken into account
width=2.2               # [cm] figure width
height=2.2              # [cm] used iff not keepAspectratio

plotLineWidth=1.2       # [pt] line width of plotted lines
axisLineWidth=0.9       # [pt] line width of plotted axes
axisArrow_relSize=8    # [-] defines marker size w.r.t. axisLineWidth
xaxis_rel_length=1.15   # [-] relative x-axis bound to data
yaxis_rel_length=1.25   # [-] relative y-axis bound to data

labelSize=11            # [pt] label font size
xlabelshift=np.array([0.0, 0.34])  # [-] relative x-label shift to x-axis enpoint
ylabelshift=np.array([0.15, 0.05])  # [-] relative y-label shift to y-axis enpoint

### Plot presets
def axis_setup(fig=plt.gcf(), np.limits=np.zeros((2,2)), axlinewidth=0.9, labelshift=np.zeros((2,2)), labelsize=11, arrow_relsize=8):
    ax = fig.add_subplot()
    ax.set_prop_cycle(ccs[0])
    ax.grid(True)
    # Axis visibility and linewidth
    for side in ['left', 'bottom']:
        ax.spines[side].set_position('zero')
        ax.spines[side].set_linewidth(axisLineWidth)
    for side in ['right', 'top']:
        ax.spines[side].set_visible(False)
    # Ticks orientation
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # Toggle Ticks invisible
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(axislables[0], fontsize=labelsize)
    ax.set_ylabel(axislables[1], fontsize=labelsize, rotation=0)
    # Change axis limits
    if not np.count_nonzeros(limits):
        ax.set_xlim(limts[0,:])
        ax.set_ylim(limts[1,:])
    # Axis reposition labels
    xspine_pt = ax.spines['bottom']._path._vertices[1,:]
    yspine_pt = ax.spines['left']._path._vertices[1,:]
    trans = ax.transLimits.transform
    xspine_coord = trans(xspine_pt)
    if xspine_coord[0] < 0.5:    # hard coded bug fix
        xspine_coord[0] = 1
    xlabel_coord = xspine_coord + labelshift[0,:]
    ylabel_coord = trans(yspine_pt) + labelshift[1,:]
    ax.xaxis.set_label_coords(*xlabel_coord)
    ax.yaxis.set_label_coords(*ylabel_coord)
    # Draw axis arrows
    ax.plot((1), (0), ls="", marker=marker_dict['arrow_r'], ms=arrow_relsize*axisLineWidth, color="k",
        transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot((0), (1), ls="", marker=marker_dict['arrow_u'], ms=arrow_relsize*axisLineWidth, color="k",
        transform=ax.get_xaxis_transform(), clip_on=False)
    return ax
#%% SPI
### Plot

### Contentual parameters
axislables=['t','V']
x = np.arange(0,5,0.05)
y = (signal.square(np.sin((x)/1*np.pi))+1)/2*(1/3)+(2/3)
z = (signal.square(np.sin((x/2)/1*np.pi))+1)/2*(1/3)+0
#plt.ioff() # turn off interactive mode

def plot(x,y,z, ax=None):
    if ax is None:
        ax = plt.gca()
    # Plots
    ax.plot(x,y,zorder=10)
    ax.plot(x,z,zorder=11)
    # Axis limits (extend)
    xlim = ax.axes.get_xlim()
    ax.axes.set_xlim((0,xlim[1]*xaxis_rel_length))
    ylim = ax.axes.get_ylim()
    ax.axes.set_ylim((ylim[0], ylim[0]+(ylim[1]-ylim[0])*yaxis_rel_length))

fig = plt.figure()
ax = axis_setup(fig)
ax.plot(x,y)
ax.plot(x,z)

if save_output:
    plt.savefig('{}.svg'.format(filenames['curr_out']))
plt.show()
# %%
