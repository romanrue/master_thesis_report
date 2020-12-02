#%%
try:
    from pathlib import Path
    import re
    import numpy as np
    from control import tf, step_response, bode, bode_plot
    from control import dcgain, freqresp, impulse_response
    from scipy import signal
    from scipy import fft
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

# custom cycle must be called in every plot
ccs = [ (cycler(color=color_list[:len(linestyle_list)]) + cycler(linestyle=linestyle_list)),
       (cycler(color= color_list[:6]) + cycler(linestyle=linestyle_list[:2]*3))]

mpl.rcParams['figure.figsize'] = [dim/2.54 for dim in [12,9]]
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['font.size'] = 11
mpl.rcParams['legend.fontsize'] = 'medium'
mpl.rcParams['axes.labelsize'] = 'medium'
mpl.rcParams['figure.titlesize'] = 'medium'

mpl.rcParams['lines.linewidth'] = 1.2

mpl.use('PDF')
mpl.rc('font',**{'family': 'serif', 'sans-serif': ['Helvetica']})
#mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rc('pdf', use14corefonts=True)

dir_path = Path(__file__).parent

# Colors (acces by colpalette)
### File locations
outdir = dir_path / 'out'
filenames = {
    'ordcmp': 'lp_gain_ord_comparison'
}

save_output = True
filenames = {
    'step': 'lp_filter_4ord_step',
    'imp': 'lp_filter_4ord_imp',
    'mag': 'lp_filter_mag',
    'pha': 'lp_filter_pha',
    'grd': 'lp_filter_grd'
}

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

def define_keys(in_list):
    keys = []
    for key in in_list:
        if isinstance(key, str):
            keys.append(clean_str(key))
        elif isinstance(key, tuple):
            for v in key[1]:
                keys.append(clean_str(key[0] + '{:g}'.format(v)))
    return keys

type_keys = define_keys(['crit', 'bessel', 'butter',
                         ('cheby', [0.5, 1, 2, 3]),
                         ('ellip', [0.5, 1, 2, 3])])
#%%
poly_values = {
    'crit':[#
        [1, 1, 1.0000, 0.0000, 1.000, None],
        [2, 1, 1.2872, 0.4142, 1.000, 0.50],
        [3, 1, 0.5098, 0.0000, 1.961, None],
        [3, 2, 1.0197, 0.2599, 1.262, 0.50],
        [4, 1, 0.8700, 0.1892, 1.480, 0.50],
        [4, 2, 0.8700, 0.1892, 1.480, 0.50],
        [5, 1, 0.3856, 0.0000, 2.593, None],
        [5, 2, 0.7712, 0.1487, 1.669, 0.50],
        [5, 3, 0.7712, 0.1487, 1.669, 0.50],
        [6, 1, 0.6999, 0.1225, 1.839, 0.50],
        [6, 2, 0.6999, 0.1225, 1.839, 0.50],
        [6, 3, 0.6999, 0.1225, 1.839, 0.50],
        [7, 1, 0.3226, 0.0000, 3.100, None],
        [7, 2, 0.6453, 0.1041, 1.995, 0.50],
        [7, 3, 0.6453, 0.1041, 1.995, 0.50],
        [7, 4, 0.6453, 0.1041, 1.995, 0.50],
        [8, 1, 0.6017, 0.0905, 2.139, 0.50],
        [8, 2, 0.6017, 0.0905, 2.139, 0.50],
        [8, 3, 0.6017, 0.0905, 2.139, 0.50],
        [8, 4, 0.6017, 0.0905, 2.139, 0.50],
        [9, 1, 0.2829, 0.0000, 3.534, None],
        [9, 2, 0.5659, 0.0801, 2.275, 0.50],
        [9, 3, 0.5659, 0.0801, 2.275, 0.50],
        [9, 4, 0.5659, 0.0801, 2.275, 0.50],
        [9, 5, 0.5659, 0.0801, 2.275, 0.50],
        [10, 1, 0.5358, 0.0718, 2.402, 0.50],
        [10, 2, 0.5358, 0.0718, 2.402, 0.50],
        [10, 3, 0.5358, 0.0718, 2.402, 0.50],
        [10, 4, 0.5358, 0.0718, 2.402, 0.50],
        [10, 5, 0.5358, 0.0718, 2.402, 0.50]
    ]
}
#%% plot magnitude

# def crit_damped(N,wn=1):
#   alpha = wn*np.sqrt(2**(1/N)-1)
#   H = 1 / np.prod(N*[tf([1,alpha],1)])
#   H = H/np.asscalar(dcgain(H))
#   a = H.den[0][0]
#   b = H.num[0][0]
#   b, a = signal.normalize(b,a)
#   return (b, a)

def crit_damped(N, wn=1, dataset=poly_values):
    poly_list = list(filter(lambda params:
                            params[0] == N, dataset['crit']))
    H = 1 / np.prod([tf([1,params[2],params[3]],1) for params in poly_list])
    a = H.den[0][0]
    b = H.num[0][0]
    return (b,a)

# get denominator and numerator of transferfunctions as dictionary of keys
def get_filter_sys(N, wn=1, type_keys=type_keys):
    H_list = []
    for key in type_keys:
        if key=='crit':
            H_list.append(crit_damped(N,wn))
        elif key=='bessel':
            H_list.append(signal.bessel(N, wn, 'low', analog=True, norm='mag'))
        elif key=='butter':
            H_list.append(signal.butter(N, wn, 'low', analog=True))
        elif 'cheby' in key:
            rp = float('.'.join(re.findall(r'\d+', key)))
            if rp==3:
                v = wn
            elif rp==0.5:
                corr_arr = np.array([0.349, 0.720, 0.857, 0.915, 0.944, 0.961, 0.971, 0.978, 0.982, 0.986])
                v = corr_arr[N-1]*wn
            H_list.append(signal.cheby1(N, rp, v, 'low', analog=True))
        elif 'ellip' in key:
            rp = float('.'.join(re.findall(r'\d+', key)))
            H_list.append(signal.ellip(N, rp, 10*wn, wn, 'low', analog=True))
    H = dict(zip(type_keys, H_list))
    return H

# #manually update corr_arr
# N_t = list(range(1,11))
# H_t = [get_filter_sys(N, 1, ['cheby0_5']) for N in N_t]
# w = np.geomspace(0.9,1.1,10000)
# plt.figure()
# plt.axhline(-3)
# plt.axvline(1)
# for i,N in enumerate(N_t):
#   _, m, _ = signal.bode(signal.TransferFunction(*H_t[i]['cheby0_5']),w)
#   idx, val = find_nearest(m,-3)
#   plt.semilogx(w,m)
#   print('N: {:d},\tw: {:f},\t h: {:f}'.format(N,w[idx],val))
# plt.xlim([0.9,1.1])
# plt.ylim([-5,-1])
# plt.show()

#%%
### sinlge plots
# inputs
single_keys = [type_keys[i] for i in [1,2,3,6]]
single_N = 4
single_H = get_filter_sys(single_N,1,single_keys)

single_limits = [0,5]
single_res = 1000

single_calc_limits = [0, 6.3*single_limits[1]]
t = np.linspace(*single_calc_limits, single_res)
w = np.linspace(0.99,1.01,1000) # for grd
# single axis setup function
def axis_setup(fig=plt.gcf()):
    ax = fig.add_subplot()
    ax.set_prop_cycle(ccs[0])
    ax.grid(True)
    return ax

def gen_label_str(key):
    if key=='crit':     return r'Critically damped'
    elif key=='bessel': return r'Bessel'
    elif key=='butter': return r'Butterworth'
    elif 'cheby' in key:
        rp = float('.'.join(re.findall(r'\d+', key)))
        return r'Chebyshev, {:g}dB ripple'.format(rp)
    elif 'ellip' in key:
        rp = float('.'.join(re.findall(r'\d+', key)))
        return r'Elliptic, {:g}dB ripple'.format(rp)

# calculation
single_list = [None]*len(single_keys)
for i,key in enumerate(single_keys):
    H = list(single_H[key])
    if 'cheby' in key:
        # fix chebishev gain
        rp = float('.'.join(re.findall(r'\d+', key)))
        H[0] = H[0] * 10**(rp/20)
    _, step = signal.step2(H, T=t)
    _, _, p = signal.bode(signal.TransferFunction(*H),w)
    _, imp = signal.impulse(H, T=t)
    grd = -np.diff(p/360)/np.diff(w)
    grd_c = grd[find_nearest(w,1)[0]]
    single_list[i] = {'tn': t/grd_c, 'step': step, 'imp': imp}
single_dict = dict(zip(single_keys, single_list))
# plots
#%% step function
fig_step = plt.figure()
ax = axis_setup(fig_step)

for i,key in enumerate(single_keys):
    ax.plot(t/(2*np.pi), single_dict[key]['step'], label=gen_label_str(key))
ax.set_xlim(single_limits)
ax.set_xlabel(r'$\frac{t}{T_c}$ [-]', fontsize='x-large')
ax.set_ylabel(r'$\frac{U_o}{U_i}$ [-]', fontsize='x-large')
ax.legend(handlelength=1)

if save_output:
    plt.savefig('{}.svg'.format(filenames['step']))
plt.show()
#%% impulse function
fig_imp = plt.figure()
ax = axis_setup(fig_imp)

for i,key in enumerate(single_keys):
    ax.plot(t/(2*np.pi), single_dict[key]['imp'], label=gen_label_str(key))
ax.set_xlim(single_limits)
ax.set_xlabel(r'$\frac{t}{T_c}$ [-]', fontsize='x-large')
ax.set_ylabel(r'$\frac{U_o}{U_i}$ [-]', fontsize='x-large')
ax.legend(handlelength=1)
if save_output:
    plt.savefig('{}.svg'.format(filenames['imp']))
plt.show()

#%%
#### Quatro plots

# inputs
quatro_keys = [type_keys[i] for i in [1,2,3,6]]
quatro_N = list(range(1,6+1))
quatro_H = [get_filter_sys(i,1,quatro_keys) for i in quatro_N]

out_limits = ([0.01,30],[-40,10])
calc_lim = out_limits[0]
w = np.geomspace(*calc_lim,10000)

# x axis ticks calculation
x_major_explim = [int(np.log10(i)) for i in calc_lim]
x_major_t = 10.0 ** np.arange(x_major_explim[0], x_major_explim[1]+1)
x_minor_t = np.multiply.outer(10.0 ** np.arange(-2,x_major_explim[1]+1), np.arange(1,10)).flatten()

# quatro axes setup function
def axes_setup(fig=plt.gcf(), x_major_t=x_major_t, x_minor_t=x_minor_t):
    gs = fig.add_gridspec(2,2)
    gs.update(wspace= 0.3, hspace=0.35)
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[0,1])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])
    axs = [ax0, ax1, ax2, ax3]

    for ax in axs:
        ax.set_prop_cycle(ccs[1])
    ax.set_xscale('log')
    ax.set_xticks(x_major_t)
    ax.set_xticks(x_minor_t, minor=True)
    ax.grid(True, which='major')
    ax.grid(True, which='minor', linestyle='dotted', linewidth=0.5)
    return axs

def gen_title_str(key):
    if key=='crit':     return r'Critical damping'
    elif key=='bessel': return r'Bessel'
    elif key=='butter': return r'Butterworth'
    elif 'cheby' in key:
        rp = float('.'.join(re.findall(r'\d+', key)))
        return r'Chebyshev, {:g}dB ripple'.format(rp)
    elif 'ellip' in key:
        rp = float('.'.join(re.findall(r'\d+', key)))
        return r'Elliptic, {:g}dB ripple'.format(rp)

# calculation
quatro_y_list = [None]*len(quatro_keys)
for i,key in enumerate(quatro_keys):
    quatro_y_list[i] = []
    for j,N in enumerate(quatro_N):
        _, m, p = signal.bode(signal.TransferFunction(*quatro_H[j][key]),w)
        grd = -np.diff(p/360)/np.diff(w)
        quatro_y_list[i].append({'m':m,'p':p,'grd':grd})
quatro_y_dict = dict(zip(quatro_keys, quatro_y_list))

# plots
#%% gains
fig_mag = plt.figure(figsize=[dim/2.54 for dim in [22,22]])
y = lambda data: np.insert(data, 0, data[0])
for i, ax in enumerate(axes_setup(fig_mag)):
    key = quatro_keys[i]
    for j, N in enumerate(quatro_N):
        ax.plot(np.insert(w,0,0), y(quatro_y_dict[key][j]['m']), label=r'N={}'.format(N))
        ax.set_xlim(out_limits[0])
        ax.set_ylim(out_limits[1])
        ax.title.set_text(gen_title_str(key))
        ax.set_xlabel(r'$\omega_n$ [-]')
        ax.set_ylabel(r'$Gain$ [dB]')
        ax.legend(handlelength=1)

if save_output:
    plt.savefig('{}.svg'.format(filenames['mag']))
plt.show()

#%% phases
fig_pha = plt.figure(figsize=[dim/2.54 for dim in [22,22]])
y = lambda data: np.insert(data, 0, data[0])
for i, ax in enumerate(axes_setup(fig_pha)):
    key = quatro_keys[i]
    for j, N in enumerate(quatro_N):
        ax.plot(np.insert(w,0,0), y(quatro_y_dict[key][j]['p']), label=r'N={}'.format(N))
        ax.set_xlim(out_limits[0])
        ax.set_yticks(np.arange(0,-np.floor(N/2+1)*180,-180))
        ax.title.set_text(gen_title_str(key))
        ax.set_xlabel(r'$\omega_n$ [-]')
        ax.set_ylabel(r'$Phase$ [$^{\circ}$]')
        ax.legend(handlelength=1)

if save_output:
    plt.savefig('{}.svg'.format(filenames['pha']))
plt.show()

#%% group delays
fig_grd = plt.figure(figsize=[dim/2.54 for dim in [22,22]])
y = lambda data: np.insert(data, 0, data[0])
for i, ax in enumerate(axes_setup(fig_grd)):
    key = quatro_keys[i]
    for j, N in enumerate(quatro_N):
        ax.plot(w, y(quatro_y_dict[key][j]['grd']), label=r'N={}'.format(N))
        ax.set_xlim(out_limits[0])
        ax.title.set_text(gen_title_str(key))
        ax.set_xlabel(r'$\omega_n$ [-]')
        ax.set_ylabel(r'$T_{gr}$ [-]')
        ax.legend(handlelength=1)

if save_output:
    plt.savefig('{}.svg'.format(filenames['grd']))
plt.show()
# %%
