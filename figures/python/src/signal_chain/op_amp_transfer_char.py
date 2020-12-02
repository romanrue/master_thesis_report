#%%
try:
    from pathlib import Path
    import re
    import numpy as np
    from scipy import special
    from scipy import interpolate
    from scipy import fftpack
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

dir_path = Path(__file__).parent

# Colors (acces by colpalette)
### File locations
outdir = dir_path / 'out'

#%% Polynomials (expanded form)
def find_nearest(a, a0):
    idx = np.abs(a - a0).argmin()
    return (idx, a[idx])

def clean_str(s):
    # replace special characters
    s = re.sub('[^a-zA-Z0-9 \.\:\\/]','', s)
    # replace specific characters
    s = re.sub('[^a-zA-Z0-9]', '_', s)
    return s

def axis_setup(fig=plt.gcf()):
    ax = fig.add_subplot()
    ax.set_prop_cycle(ccs[0])
    ax.grid(True)
    return ax

def linear_step(x,x0=0,x1=1):
    y = np.piecewise(x, [
        x < x0,
        (x >= x0) & (x <= x1),
        x > x1],
            [0.,
            lambda x: x/(x1-x0) + x0/(x0-x1),
            1.])
    return y

def smooth_linear_step(x,x0=0,x1=1,n=6,r=0.075):
    s = 1./(x1-x0)
    a = np.array([
        [0, 0, x0-(n-1)*r/s],
        [1, 0, x1+(n-1)*r/s]])
    a[0,1] = s/(n*(n*r/s)**(n-1))
    a[1,1] = -s/(n*(n*r/s)**(n-1))
    y = np.piecewise(x, [
        x < a[0,2],
        (x >= a[0,2]) & (x < x0+r/s),
        (x >= x0+r/s) & (x < x1-r/s),
        (x >= x1-r/s) & (x < a[1,2]),
        x >= a[1,2]],
            [0.,
            lambda x: a[0,0] + a[0,1]*(x-a[0,2])**n,
            lambda x: s*(x-x0),
            lambda x: a[1,0] + a[1,1]*(-x+a[1,2])**n,
            1.])
    return y

def sigmoid(x):
    y = 1/(1+(np.exp(x)))
    return y

def norm_weighted_sum(xs, ws):
    wsn = ws/(np.sum(ws)*len(xs))
    ys_arr = np.vstack(xs)
    y = np.sum(np.multiply(ys_arr, wsn[:, np.newaxis]), axis=0)
    return y
#%
#%% volt out
N = int(1e6)
A = 12

x = 1e-4*np.linspace(-1.8,1.8,N)
y_lin = (linear_step(x,-1.1e-4, 1.1e-4)-0.5)
y_lins = 2*A*smooth_linear_step(x,-1.1e-4,1.1e-4)-A

R = [0, 0, x[find_nearest(y_lins,10)[0]], 10]

fig_step = plt.figure()
ax = axis_setup(fig_step)

ax.add_patch(mpl.patches.Rectangle((1e6*R[0], R[1]),
  1e6*R[2],R[3], ls=':', lw=2, fc='none', ec='k'))
ax.plot(1e6*x, y_lins)
ax.set_xlabel(r'$V_D$ [$\mu V$]')
ax.set_ylabel(r'$V_o$ [$V$]')
ax.annotate('VV-OPA\nCV-OPA',
    xy=(0.045,0.8), xycoords='axes fraction',
    textcoords='offset points',
    size=11,
    bbox=dict(boxstyle="round", fc=ulcolors[6], ec="none"))
ax.annotate(r'$V_{o,\min}$',
    xy=(-150, -10.2), xycoords="data",
    va="center", ha="center", size=11,
    bbox=dict(fc='w', ec='none', pad=0.2))
ax.annotate(r'$V_{o,\max}$',
    xy=(150, 10.2), xycoords="data",
    va="center", ha="center", size=11,
    bbox=dict(fc='w', ec='none', pad=0.2))
if save_output:
    plt.savefig('{}.svg'.format(filenames['volt_out']))
plt.show()
#%% curr out
N = int(1e6)
A = 12

x = 1e-4*np.linspace(-1.8,1.8,N)
y_lin = (linear_step(x,-1.1e-4, 1.1e-4)-0.5)
y_lins = 2*A*smooth_linear_step(x,-1.1e-4,1.1e-4)-A

R = [0, 0, x[find_nearest(y_lins,10)[0]], 10]

fig_step = plt.figure()
ax = axis_setup(fig_step)

ax.add_patch(mpl.patches.Rectangle((1e6*R[0], R[1]),
  1e6*R[2],R[3], ls=':', lw=2, fc='none', ec='k'))
ax.plot(1e6*x, y_lins)
ax.set_xlabel(r'$V_D$ [$\mu V$]')
ax.set_ylabel(r'$I_o$ [$mA$]')
ax.annotate('VC-OPA\nCC-OPA',
    xy=(0.045,0.8), xycoords='axes fraction',
    textcoords='offset points',
    size=11,
    bbox=dict(boxstyle="round", fc=ulcolors[5], ec="none"))
ax.annotate(r'$I_{o,\min}$',
    xy=(-150, -10.2), xycoords="data",
    va="center", ha="center", size=11,
    bbox=dict(fc='w', ec='none', pad=0.2))
ax.annotate(r'$I_{o,\max}$',
    xy=(150, 10.2), xycoords="data",
    va="center", ha="center", size=11,
    bbox=dict(fc='w', ec='none', pad=0.2))
if save_output:
    plt.savefig('{}.svg'.format(filenames['curr_out']))
plt.show()
# %%
