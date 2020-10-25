#%%
try:
    from pathlib import Path
    import importlib.machinery
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from svgpathtools import svg2paths
    from svgpath2mpl import parse_path
    from matplotlib.markers import MarkerStyle
except Exception as e:
    print("Some Modules are missing {}".format(e))

mpl.use('PDF') 
mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rc('pdf', use14corefonts=True)

path = Path(__file__).resolve()

def import_file(full_name, path):
    """Import a python module from a path 3.5+ only.
    Does not call sys.modules[full_name] = path
    """
    from importlib import util
    loader = importlib.machinery.SourceFileLoader('temp', path)
    spec = util.spec_from_loader(loader.name, loader)
    mod = util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod

# Colors (acces by colpalette)
colfound = False
colpath = None
colpath_parts = path.parts
for i, part in reversed(list(enumerate(colpath_parts))):
    if part.lower() == 'figures':
        colpath = Path(*colpath_parts[:(i+1)]) / 'colors'
        colfound=True
        break
if not colfound:
    print('Color palettes were not found. Using standard colors')
### File locations
# input
markerdir = Path('markers')
# output
outdir = Path('out')
filename = '' # if empty this files name is used

### Import svg paths of markers
svgs = list(markerdir.glob('*.svg'))
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

marker_dict={}
for key in svgpath_dict:
    marker_dict[key] = parse_path(svgpath_dict[key])
    # center horizontally
    marker_dict[key].vertices -= marker_dict[key].vertices.mean(axis=0)

# %%
### Visual parameters
aspRatio=4/3            # figure aspect ratio
keepAspRatio=True       # if false height is taken into account
width=2.5               # [cm] figure width
height=2.5              # [cm] used iff not keepAspectratio

plotLineWidth=1.2       # [pt] line width of plotted lines
axisLineWidth=0.9       # [pt] line width of plotted axes
axisArrow_relSize=10    # [-] defines marker size w.r.t. axisLineWidth
xaxis_rel_length=1.15   # [-] relative x-axis bound to data
yaxis_rel_length=1.25   # [-] relative y-axis bound to data

labelSize=11            # [pt] label font size
xlabelshift=np.array([0.0, 0.34])  # [-] relative x-label shift to x-axis enpoint
ylabelshift=np.array([0.15, 0.05])  # [-] relative y-label shift to y-axis enpoint

### Contentual parameters
axislables=['t','x']
x = np.linspace(0,11.25*np.pi,num=1000)
y = np.sin(x)
z = np.sin(x/5)
xp = np.array([2.5,7.5])*np.pi
yp = np.array([1, -1])

#%%
### Plot

plt.ioff() # turn off interactive mode

def plot(x,y,z, ax=None):
    if ax is None:
        ax = plt.gca()
    # Plots
    ax.plot(x,y,zorder=10)
    ax.plot(x,z,zorder=11)
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
    ax.set_xlabel(axislables[0], fontsize=labelSize)
    ax.set_ylabel(axislables[1], fontsize=labelSize, rotation=0)
    # Axis limits (extend)
    xlim = ax.axes.get_xlim()
    ax.axes.set_xlim((0,xlim[1]*xaxis_rel_length))
    ylim = ax.axes.get_ylim()
    ax.axes.set_ylim((ylim[0], ylim[0]+(ylim[1]-ylim[0])*yaxis_rel_length))
    # Axis reposition labels
    xspine_pt = ax.spines['bottom']._path._vertices[1,:]
    yspine_pt = ax.spines['left']._path._vertices[1,:]
    trans = ax.transLimits.transform
    xspine_coord = trans(xspine_pt)
    if xspine_coord[0] < 0.5:    # hard coded bug fix
        xspine_coord[0] = 1
    xlabel_coord = xspine_coord + xlabelshift
    ylabel_coord = trans(yspine_pt) + ylabelshift
    ax.xaxis.set_label_coords(*xlabel_coord)
    ax.yaxis.set_label_coords(*ylabel_coord)
    # Draw axis arrows
    ax.plot((1), (0), ls="", marker=marker_dict['arrow_r'], ms=axisArrow_relSize*axisLineWidth, color="k",
        transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot((0), (1), ls="", marker=marker_dict['arrow_u'], ms=axisArrow_relSize*axisLineWidth, color="k",
        transform=ax.get_xaxis_transform(), clip_on=False)

cm = 1/2.54
if keepAspRatio:
    height = width/aspRatio
fig = plt.figure(figsize=(width*cm, height*cm))
ax = fig.add_subplot(111)
plot(x,y,z,ax)

if filename is None or not filename:
    filename = Path(__file__).stem
plt.savefig('{}.pdf'.format(filename))

plt.show()