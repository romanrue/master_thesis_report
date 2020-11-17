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

mpl.rcParams['figure.figsize'] = [dim/2.54 for dim in [12,6]]
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

    
axislables=['Time','LSB error']


#%%
### Plot
mu, sigma = 0, 0.01
x = np.linspace(0,500,500)
y = np.random.normal(mu, sigma, x.size)
count, hist, ignored = plt.hist(y)

fig = plt.figure()
gs = fig.add_gridspec(1,4)
gs.update(hspace=0.3)
axs = []
ax0 = fig.add_subplot(gs[0,:3])
ax1 = fig.add_subplot(gs[0,3], sharey=ax0)
axs = [ax0, ax1]
axs[0].plot(x,y, color='C1')

axs[1].hist(y, bins=20, density=True, orientation='horizontal', color='C1', alpha=0.2)
ysort = np.sort(y)
axs[1].plot(1/(sigma*np.sqrt(2*np.pi))*np.exp(-((ysort-mu)/sigma)**2/2), ysort, color='C1')

axs[0].set_title(r'Time-domain' '\n' r'noise')
axs[0].set_xlabel(r'Time $[ms]$')
axs[0].set_ylabel(r'Voltage $[V]$')
axs[1].set_title(r'Noise' '\n' r'distribution')
axs[1].set_xlabel(r'Probability' '\n' r'density $[-]$')

axs[1].tick_params(
    axis='y',
    which='both',
    left=False,
    right=False,
    labelleft=False,
)
for ax in axs:
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)

if filename is None or not filename:
    filename = Path(__file__).stem
plt.savefig('{}.svg'.format(filename))

plt.show()
# %%
