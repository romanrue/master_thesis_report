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

mpl.rcParams['figure.figsize'] = [dim/2.54 for dim in [9,6.75]]
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
filenames = {
  'ordcmp': 'lp_gain_ord_comparison'
}

save_output = True
filenames = {
  'bp1': 'lp_filter_2ord_bessel_bp',
  'bp2': 'lp_filter_2ord_cheby_bp',
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

def crit_damped(N,wn=1):
  alpha = wn*np.sqrt(2**(1/N)-1)
  H = 1 / np.prod(N*[tf([1,alpha],1)])
  H = H/np.asscalar(dcgain(H))
  a = H.den[0][0]
  b = H.num[0][0]
  b, a = signal.normalize(b,a)
  return (b, a)

# def crit_damped(N, wn=1, dataset=poly_values):
#   poly_list = list(filter(lambda params:
#     params[0] == N, dataset['crit']))
#   H = 1 / np.prod([tf([1,params[2],params[3]],1) for params in poly_list])
#   a = H.den[0][0]
#   b = H.num[0][0]
#   return (b,a)

# get denominator and numerator of transferfunctions as dictionary of keys
def get_filter_sys(N, wn=1, type_keys=type_keys):
  H_list = []
  for i,key in enumerate(type_keys):
    if isinstance(N, list):
      Np = N[i]
    else:
      Np =N
    if key=='crit':
      H_list.append(crit_damped(Np,wn))
    elif key=='bessel':
      H_list.append(signal.bessel(Np, wn, 'low', analog=True, norm='mag'))
    elif key=='butter':
      H_list.append(signal.butter(Np, wn, 'low', analog=True))
    elif 'cheby' in key:
      rp = float('.'.join(re.findall(r'\d+', key)))
      H_list.append(signal.cheby1(Np, rp, wn, 'low', analog=True))
    elif 'ellip' in key:
      rp = float('.'.join(re.findall(r'\d+', key)))
      H_list.append(signal.ellip(Np, rp, 10*wn, wn, 'low', analog=True))
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
single_keys = type_keys
single_N = [2]*(len(single_keys))
single_H = get_filter_sys(single_N,1,single_keys)

single_limits = ([0.01,10],[-25,10])
single_res = 10000
w = np.geomspace(*single_limits[0],single_res)

# x axis ticks calculation
def log_ticks_loc(limits):
  major_explim = [int(np.log10(i)) for i in limits]
  major_t = 10.0 ** np.arange(major_explim[0], major_explim[1]+1)
  minor_t = np.multiply.outer(10.0 ** np.arange(-2,major_explim[1]+1), np.arange(1,10)).flatten()
  return (major_t, minor_t)

# single axis setup function
def axis_setup(fig=plt.gcf(), x_major_t=[], x_minor_t=[]):
  ax = fig.add_subplot()
  ax.set_prop_cycle(ccs[0])
  ax.set_xscale('log')
  ax.set_xticks(x_major_t)
  ax.set_xticks(x_minor_t, minor=True)
  ax.grid(True, which='major')
  ax.grid(True, which='minor', linestyle='dotted', linewidth=0.5)
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
  _, m, p = signal.bode(signal.TransferFunction(*H),w)
  single_list[i] = {'m': m,'p': p}
single_dict = dict(zip(single_keys, single_list))

# %% special legend
class YellowLineBox(object):
  def legend_artist(self, legend, orig_handle, fontsize, handlebox):
    x0, y0 = handlebox.xdescent, handlebox.ydescent
    width, height = handlebox.width, handlebox.height
    background = mpl.patches.Rectangle([x0, y0], width, height, alpha=0.2, color='C2', lw=2, transform=handlebox.get_transform())
    bottom = mpl.lines.Line2D([x0, x0+width], [y0]*2, color='C2', lw=2, transform=handlebox.get_transform())
    top = mpl.lines.Line2D([x0, x0+width], [y0+height]*2, color='C2', lw=2, transform=handlebox.get_transform())
    patch = mpl.collections.PatchCollection([background, bottom, top])
    handlebox.add_artist(patch)
    handlebox.add_artist(top)
    handlebox.add_artist(bottom)
    return patch

class OrangeLineBox(object):
  def legend_artist(self, legend, orig_handle, fontsize, handlebox):
    x0, y0 = handlebox.xdescent, handlebox.ydescent
    width, height = handlebox.width, handlebox.height
    background = mpl.patches.Rectangle([x0, y0], width, height, alpha=0.2, color='C1', lw=2, transform=handlebox.get_transform())
    bottom = mpl.lines.Line2D([x0, x0+width], [y0]*2, color='C1', lw=2, transform=handlebox.get_transform())
    top = mpl.lines.Line2D([x0, x0+width], [y0+height]*2, color='C1', lw=2, transform=handlebox.get_transform())
    patch = mpl.collections.PatchCollection([background, bottom, top])
    handlebox.add_artist(patch)
    handlebox.add_artist(top)
    handlebox.add_artist(bottom)
    return patch

# %update legend colors (not working!)
def update_legend(leg, color='C2', alpha=0.2):
  idx, curr = 0, leg

  def recurs_navigation(idx, curr):
    for child in curr.get_children():
      if isinstance(child, mpl.offsetbox.VPacker):
        if np.any(idx==np.array([0,2])):
          recurs_navigation(idx+1, child)
      elif isinstance(child, mpl.offsetbox.HPacker):
        if np.any(idx==np.array([1,3])):
          recurs_navigation(idx+1, child)
      elif isinstance(child, mpl.offsetbox.DrawingArea):
        if idx==4:
          recurs_navigation(idx+1, child)
      elif isinstance(child, mpl.collections.PatchCollection):
        if idx==5:
          child.set_color=color
          child.set_alpha=alpha

  recurs_navigation(idx, curr)
# plots
#%% band pass plot 1
ytlabel_loc = np.append(np.arange(*single_limits[1], 5),[single_limits[1][1], -3])
ytlabel_loc = np.delete(ytlabel_loc, np.s_[2:-1:2])
gamma = find_nearest(single_dict['bessel']['m'], ytlabel_loc[1])
ytlabel_loc[1] = gamma[1]
ytlabels= [r'{:g}'.format(num) for num in list(ytlabel_loc)]
ytlabels[1] = r'$\zeta$'

fig_bp1 = plt.figure()
ax = axis_setup(fig_bp1, *log_ticks_loc(single_limits[0]))
y = lambda data: np.insert(data, 0, data[0])
ax.add_patch(mpl.patches.Rectangle((0, -3), 1, 3, alpha=0.2, color='C2', facecolor='none'))
ax.hlines([-3,0], 0, 1, colors='C2')
ax.add_patch(mpl.patches.Rectangle((w[gamma[0]], single_limits[1][0]),
  single_limits[0][1]-w[gamma[0]],
  gamma[1]-single_limits[1][0], alpha=0.2, color='C1', facecolor='none'))
ax.hlines([single_limits[1][0], gamma[1]], w[gamma[0]], single_limits[0][1], colors='C1')
ax.plot(np.insert(w,0,0), y(single_dict['bessel']['m']))
ax.set_xlim(single_limits[0])
ax.set_ylim(single_limits[1])
ax.set_yticks(ytlabel_loc)
ax.set_yticklabels(ytlabels)
ax.set_xlabel(r'$\omega_n$ [-]')
ax.set_ylabel(r'Gain [dB]')
leg = ax.legend([1, 2],
  ['Passband', 'Stopband'],
  loc='lower center',
  handler_map={1: YellowLineBox(), 2: OrangeLineBox()})
update_legend(leg)

if save_output:
  plt.savefig('{}.svg'.format(filenames['bp1']))
plt.show()

#%% band pass plot 2
ytlabel_loc = np.append(np.arange(*single_limits[1], 5), [single_limits[1][1], -2])
ytlabel_loc = np.delete(ytlabel_loc, np.s_[2:-1:2])
gamma = find_nearest(single_dict['cheby2']['m'], ytlabel_loc[1])
ytlabel_loc[1] = gamma[1]
ytlabels= [r'{:g}'.format(num) for num in list(ytlabel_loc)]
ytlabels[1] = r'$\zeta$'
ytlabels[-1] = r'$-\delta$'

fig_bp2 = plt.figure()
ax = axis_setup(fig_bp2, *log_ticks_loc(single_limits[0]))
y = lambda data: np.insert(data, 0, data[0])
ax.add_patch(mpl.patches.Rectangle((0, -2), 1, 2, alpha=0.2, color='C2', facecolor='none'))
ax.hlines([-2,0], 0, 1, colors='C2')
ax.add_patch(mpl.patches.Rectangle((w[gamma[0]], single_limits[1][0]),
  single_limits[0][1]-w[gamma[0]],
  gamma[1]-single_limits[1][0], alpha=0.2, color='C1', facecolor='none'))
ax.hlines([single_limits[1][0], gamma[1]], w[gamma[0]], single_limits[0][1], colors='C1')
ax.plot(np.insert(w,0,0), y(single_dict['cheby2']['m']))
ax.set_xlim(single_limits[0])
ax.set_ylim(single_limits[1])
ax.set_yticks(ytlabel_loc)
ax.set_yticklabels(ytlabels)
ax.set_xlabel(r'$\omega_n$ [-]')
ax.set_ylabel(r'Gain [dB]')
leg = ax.legend([1, 2],
  ['Passband', 'Stopband'],
  loc='lower center',
  handler_map={1: YellowLineBox(), 2: OrangeLineBox()})
update_legend(leg)

if save_output:
  plt.savefig('{}.svg'.format(filenames['bp2']))
plt.show()



# %%
