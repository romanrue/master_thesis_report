#%%
try:
    from pathlib import Path
    import numpy as np
    from control import tf, bode, bode_plot
    from scipy.signal import windows
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from cycler import cycler
    from svgpathtools import svg2paths
    from svgpath2mpl import parse_path
    from matplotlib.markers import MarkerStyle

    # colors
    from colors.gen_colors import colpalettes
except Exception as e:
    print("Some Modules are missing. {}".format(e))

for el in colpalettes:
    if el.name == 'WesMixL':
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color',el.rgb_hex())

mpl.rcParams['figure.figsize'] = [dim/2.54 for dim in [9,4.5]]
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['font.size'] = 11
mpl.rcParams['legend.fontsize'] = 'medium'
mpl.rcParams['axes.labelsize'] = 'medium'
mpl.rcParams['figure.titlesize'] = 'medium'

# mpl.rcParams['lines.linewidth'] = 2

mpl.use('PDF')
mpl.rc('font',**{'family': 'serif', 'sans-serif': ['Helvetica']})
#mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rc('pdf', use14corefonts=True)
# mpl.rcParams['text.latex.preamble'] = [r'''
#     \usepackage[utf8]{inputenc}'
#     \usepackage{times}
#     \usepackage{helvet}
#     \usepackage[T1]{fontenc}
#     \usepackage{amsmath}
#     \usepackage{amssymb}
#     \usepackage{amsthm}
#     \usepackage{physics}
#     \usepackage{siunitx}
#         \sisetup{per-mode=symbol-or-fraction, range-phrase=\dots,range-units=single}
#     '''
# ]

dir_path = Path(__file__).parent

# Colors (acces by colpalette)
### File locations
outdir = dir_path / 'out'
filename = '' # if empty this files name is used


#%%
### Plot
a = 5
a_ratio = 0.5
x = np.linspace(0,2*np.pi,500)
y_normal = a*a_ratio*np.sin(x)
y_rr = a*np.sin(x)

fig = plt.figure()
ax = fig.add_subplot()

ax.plot(x,y_normal, label=r'Normal')
ax.plot(x,y_rr,'--', label=r'Rail to rail')

ax.legend(handlelength=1)
ax.set_xlabel(r'Time')
ax.set_ylabel(r'Signal [$V$]')

yticks = np.append(np.arange(0,a,2.5),a)
yticks = np.unique(np.append(np.flip(-yticks),yticks))
ax.set_yticks(yticks)
ax.get_xticklabels()[0].set_color('C3')
ax.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False,
)
yticklabels = ax.get_yticklabels()
ytickgridlines = ax.yaxis.get_gridlines()
for t_idx in [0,-1]:
    yticklabels[t_idx].set_color('C3')
    ytickgridlines[t_idx].set_color('C3')
ax.xaxis.grid(True)
ax.yaxis.grid(True)

if filename is None or not filename:
    filename = Path(__file__).stem
plt.savefig('{}.svg'.format(filename))

plt.show()
# %%
