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

mpl.rcParams['figure.figsize'] = [dim/2.54 for dim in [12,9]]
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['font.size'] = 11
mpl.rcParams['legend.fontsize'] = 'medium'
mpl.rcParams['axes.labelsize'] = 'medium'
mpl.rcParams['figure.titlesize'] = 'medium'

mpl.rcParams['lines.linewidth'] = 2

mpl.use('PDF')
mpl.rc('font',**{'family': 'serif', 'sans-serif': ['Helvetica']})
#mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rc('pdf', use14corefonts=True)

dir_path = Path(__file__).parent

# Colors (acces by colpalette)
### File locations
outdir = dir_path / 'out'
filename = '' # if empty this files name is used

    
axislables=['Time','LSB error']


#%%
### Plot
dec_res = 8
lsb_size = 3/dec_res
x = np.arange(0,2*np.pi,0.01)
y0a = 1.5*np.sin(x) + 1.5
y1 = (y0a-lsb_size/2) % lsb_size - lsb_size/2
y0b = y0a-y1

fig = plt.figure()
gs = fig.add_gridspec(3,1)
gs.update(hspace=0.3)
axs = []
ax0 = fig.add_subplot(gs[:2,0])
ax1 = fig.add_subplot(gs[2,0])
axs = [ax0, ax1]
axs[0].plot(x,y0a, label='Analog')
axs[0].plot(x,y0b, label='Digital')
axs[1].plot(x,y1, color='C1')
axs[1].set_xlabel(r'Time')
axs[0].set_ylabel(r'Signal $[V]$')
axs[1].set_ylabel(r'LSB error $[V]$')
axs[0].legend()
axs[1].yaxis.set_label_coords(0.5,-0.5)

for i, ax in enumerate(axs):
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False,
    )
    ax.yaxis.set_label_coords(-0.15,0.5)


if filename is None or not filename:
    filename = Path(__file__).stem
plt.savefig('{}.svg'.format(filename))

plt.show()
# %%
