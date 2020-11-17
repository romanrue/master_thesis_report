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
x = np.linspace(0,11.25*np.pi,1000)
ya = np.sin(x)
yb = np.sin(x/a)

# x_stem_sol1 = 2*np.pi*a/(a-1)*np.arange(5)
# x_stem_sol2 = np.pi*a/(a+1)*(1+2*np.arange(7))
# x_stem = np.concatenate((x_stem_sol1, x_stem_sol2))
# x_stem.sort(kind='mergesort')
x_stem = np.arange(11.25*np.pi, step=1.5*np.pi)
ya_stem = np.sin(x_stem)
yb_stem = np.sin(x_stem/a)

fig = plt.figure()
ax = fig.add_subplot()

ax.plot(x,ya, label=r'$f>f_{\max}$')
ax.plot(x,yb,'--', label=r'$f<f_{\max}$')
ax.stem(x_stem,ya_stem, linefmt='C0', markerfmt='C0o', basefmt=' ')
ax.stem(x_stem,yb_stem, linefmt='C1', markerfmt='C1o', basefmt=' ')

ax.set_ylim(-1.1,2.1)
ax.legend()
ax.set_xlabel(r'Time')
ax.set_ylabel(r'Signal')

ax.tick_params(
    axis='both',
    which='both',
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False,
)
ax.xaxis.grid(True)
ax.yaxis.grid(True)

if filename is None or not filename:
    filename = Path(__file__).stem
plt.savefig('{}.svg'.format(filename))

plt.show()
# %%
