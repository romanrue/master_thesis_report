#%%
try:
  from pathlib import Path
  import numpy as np
  from control import tf, step_response, bode, bode_plot
  from control import dcgain, freqresp, impulse_response
  from scipy.signal import windows, group_delay
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
linestyle_list = [linestyle for (_,linestyle) in linestyle_tuple]

# colors
for el in colpalettes:
  if el.name == 'WesMixL':
    color_list = el.rgb_hex()
    # set color cycle as standard
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color',el.rgb_hex())

# custom cycle must be called in every plot
cc = (cycler(color=color_list[:len(linestyle_list)]) + cycler(linestyle=linestyle_list))

mpl.rcParams['figure.figsize'] = [dim/2.54 for dim in [14,10.5]]
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['font.size'] = 11
mpl.rcParams['legend.fontsize'] = 'medium'
mpl.rcParams['axes.labelsize'] = 'medium'
mpl.rcParams['figure.titlesize'] = 'medium'

# mpl.rcParams['lines.linewidth'] = 2

#mpl.use('PDF')
mpl.rc('font',**{'family': 'serif', 'sans-serif': ['Helvetica']})
#mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rc('pdf', use14corefonts=True)

dir_path = Path(__file__).parent

# Colors (acces by colpalette)
### File locations
outdir = dir_path / 'out'
filename = '' # if empty this files name is used

    
axislables=['Time','LSB error']

#%% Polynomials (expanded form)
poly_keys = ('type','N', 'i', 'ai', 'bi', 'fcior', 'Qi')
type_keys = ('crit', 'bessel', 'butterworth', 'cheby0_5', 'cheby1', 'cheby2', 'cheby3')
type_names = dict(zip(type_keys,[
  r'Critically damped',
  r'Bessel',
  r'Butterworth',
  r'Chebychev, 0.5dB ripple',
  r'Chebychev, 1dB ripple',
  r'Chebychev, 2dB ripple',
  r'Chebychev, 3dB ripple']))

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
  ],
  'bessel': [#
    [1, 1, 1.0000, 0.0000, 1.000, None],
    [2, 1, 1.3617, 0.6180, 1.000, 0.58],
    [3, 1, 0.7560, 0.0000, 1.323, None],
    [3, 2, 0.9996, 0.4772, 1.414, 0.69],
    [4, 1, 1.3397, 0.4889, 0.978, 0.52],
    [4, 2, 0.7743, 0.3890, 1.797, 0.81],
    [5, 1, 0.6656, 0.0000, 1.502, None],
    [5, 2, 1.1402, 0.4128, 1.184, 0.56],
    [5, 3, 0.6216, 0.3245, 2.138, 0.92],
    [6, 1, 1.2217, 0.3887, 1.063, 0.51],
    [6, 2, 0.9686, 0.3505, 1.431, 0.61],
    [6, 3, 0.5131, 0.2756, 2.447, 1.02],
    [7, 1, 0.5937, 0.0000, 1.684, None],
    [7, 2, 1.0944, 0.3395, 1.207, 0.53],
    [7, 3, 0.8304, 0.3011, 1.695, 0.66],
    [7, 4, 0.4332, 0.2381, 2.731, 1.13],
    [8, 1, 1.1112, 0.3162, 1.164, 0.51],
    [8, 2, 0.9754, 0.2979, 1.381, 0.56],
    [8, 3, 0.7202, 0.2621, 1.963, 0.71],
    [8, 4, 0.3728, 0.2087, 2.992, 1.23],
    [9, 1, 0.5386, 0.0000, 1.857, None],
    [9, 2, 1.0244, 0.2834, 1.277, 0.52],
    [9, 3, 0.8710, 0.2636, 1.574, 0.59],
    [9, 4, 0.6320, 0.2311, 2.226, 0.76],
    [9, 5, 0.3257, 0.1854, 3.237, 1.32],
    [10, 1, 1.0215, 0.2650, 1.264, 0.50],
    [10, 2, 0.9393, 0.2549, 1.412, 0.54],
    [10, 3, 0.7815, 0.2351, 1.780, 0.62],
    [10, 4, 0.5604, 0.2059, 2.479, 0.81],
    [10, 5, 0.2883, 0.1665, 3.466, 1.42]
  ],
  'cheby0_5': [#
    [1, 1, 1.0000, 0.0000, 1.000, None],
    [2, 1, 1.3614, 1.3827, 1.000, 0.86],
    [3, 1, 1.8636, 0.0000, 0.537, None],
    [3, 2, 0.6402, 1.1931, 1.335, 1.71],
    [4, 1, 2.6282, 3.4341, 0.538, 0.71],
    [4, 2, 0.3648, 1.1509, 1.419, 2.94],
    [5, 1, 2.9235, 0.0000, 0.342, None],
    [5, 2, 1.3025, 2.3534, 0.881, 1.18],
    [5, 3, 0.2290, 1.0833, 1.480, 4.54],
    [6, 1, 3.8645, 6.9797, 0.366, 0.68],
    [6, 2, 0.7528, 1.8573, 1.078, 1.81],
    [6, 3, 0.1589, 1.0711, 1.495, 6.51],
    [7, 1, 4.0211, 0.0000, 0.249, None],
    [7, 2, 1.8729, 4.1795, 0.645, 1.09],
    [7, 3, 0.4861, 1.5676, 1.208, 2.58],
    [7, 4, 0.1156, 1.0443, 1.517, 8.84],
    [8, 1, 5.1117, 11.9607, 0.276, 0.68],
    [8, 2, 1.0639, 2.9365, 0.844, 1.61],
    [8, 3, 0.3439, 1.4206, 1.284, 3.47],
    [8, 4, 0.0885, 1.0407, 1.521, 11.53],
    [9, 1, 5.1318, 0.0000, 0.195, None],
    [9, 2, 2.4283, 6.6307, 0.506, 1.06],
    [9, 3, 0.6839, 2.2908, 0.989, 2.21],
    [9, 4, 0.2559, 1.3133, 1.344, 4.48],
    [9, 5, 0.0695, 1.0272, 1.532, 14.58],
    [10, 1, 6.3648, 18.3695, 0.222, 0.67],
    [10, 2, 1.3582, 4.3453, 0.689, 1.53],
    [10, 3, 0.4822, 1.9440, 1.091, 2.89],
    [10, 4, 0.1994, 1.2520, 1.381, 5.61],
    [10, 5, 0.0563, 1.0263, 1.533, 17.99]
  ],
  'cheby1': [#
    [1, 1, 1.0000, 0.0000, 1.000, None],
    [2, 1, 1.3022, 1.5515, 1.000, 0.96],
    [3, 1, 2.2156, 0.0000, 0.451, None],
    [3, 2, 0.5442, 1.2057, 1.353, 2.02],
    [4, 1, 2.5904, 4.1301, 0.540, 0.78],
    [4, 2, 0.3039, 1.1697, 1.417, 3.56],
    [5, 1, 3.5711, 0.0000, 0.280, None],
    [5, 2, 1.1280, 2.4896, 0.894, 1.40],
    [5, 3, 0.1872, 1.0814, 1.486, 5.56],
    [6, 1, 3.8437, 8.5529, 0.366, 0.76],
    [6, 2, 0.6292, 1.9124, 1.082, 2.20],
    [6, 3, 0.1296, 1.0766, 1.493, 8.00],
    [7, 1, 4.9520, 0.0000, 0.202, None],
    [7, 2, 1.6338, 4.4899, 0.655, 1.30],
    [7, 3, 0.3987, 1.5834, 1.213, 3.16],
    [7, 4, 0.0937, 1.0423, 1.520, 10.90],
    [8, 1, 5.1019, 14.7608, 0.276, 0.75],
    [8, 2, 0.8916, 3.0426, 0.849, 1.96],
    [8, 3, 0.2806, 1.4334, 1.285, 4.27],
    [8, 4, 0.0717, 1.0432, 1.520, 14.24],
    [9, 1, 6.3415, 0.0000, 0.158, None],
    [9, 2, 2.1252, 7.1711, 0.514, 1.26],
    [9, 3, 0.5624, 2.3278, 0.994, 2.71],
    [9, 4, 0.2076, 1.3166, 1.346, 5.53],
    [9, 5, 0.0562, 1.0258, 1.533, 18.03],
    [10, 1, 6.3634, 22.7468, 0.221, 0.75],
    [10, 2, 1.1399, 4.5167, 0.694, 1.86],
    [10, 3, 0.3939, 1.9665, 1.093, 3.56],
    [10, 4, 0.1616, 1.2569, 1.381, 6.94],
    [10, 5, 0.0455, 1.0277, 1.532, 22.26]
  ],
  'cheby2': [#
    [1, 1, 1.0000, 0.0000, 1.000, None],
    [2, 1, 1.1813, 1.7775, 1.000, 1.13],
    [3, 1, 2.7994, 0.0000, 0.357, None],
    [3, 2, 0.4300, 1.2036, 1.378, 2.55],
    [4, 1, 2.4025, 4.9862, 0.550, 0.93],
    [4, 2, 0.2374, 1.1896, 1.413, 4.59],
    [5, 1, 4.6345, 0.0000, 0.216, None],
    [5, 2, 0.9090, 2.6036, 0.908, 1.78],
    [5, 3, 0.1434, 1.0750, 1.493, 7.23],
    [6, 1, 3.5880, 10.4648, 0.373, 0.90],
    [6, 2, 0.4925, 1.9622, 1.085, 2.84],
    [6, 3, 0.0995, 1.0826, 1.491, 10.46],
    [7, 1, 6.4760, 0.0000, 0.154, None],
    [7, 2, 1.3258, 4.7649, 0.665, 1.65],
    [7, 3, 0.3067, 1.5927, 1.218, 4.12],
    [7, 4, 0.0714, 1.0384, 1.523, 14.28],
    [8, 1, 4.7743, 18.1510, 0.282, 0.89],
    [8, 2, 0.6991, 3.1353, 0.853, 2.53],
    [8, 3, 0.2153, 1.4449, 1.285, 5.58],
    [8, 4, 0.0547, 1.0461, 1.518, 18.69],
    [9, 1, 8.3198, 0.0000, 0.120, None],
    [9, 2, 1.7299, 7.6580, 0.522, 1.60],
    [9, 3, 0.4337, 2.3549, 0.998, 3.54],
    [9, 4, 0.1583, 1.3174, 1.349, 7.25],
    [9, 5, 0.0427, 1.0232, 1.536, 23.68],
    [10, 1, 5.9618, 28.0376, 0.226, 0.89],
    [10, 2, 0.8947, 4.6644, 0.697, 2.41],
    [10, 3, 0.3023, 1.9858, 1.094, 4.66],
    [10, 4, 0.1233, 1.2614, 1.380, 9.11],
    [10, 5, 0.0347, 1.0294, 1.531, 27.27]
  ],
  'cheby3': [#
    [1, 1, 1.0000, 0.0000, 1.000, None],
    [2, 1, 1.0650, 1.9305, 1.000, 1.30],
    [3, 1, 3.3496, 0.0000, 0.299, None],
    [3, 2, 0.3559, 1.1923, 1.396, 3.07],
    [4, 1, 2.1853, 5.5339, 0.557, 1.08],
    [4, 2, 0.1964, 1.2009, 1.410, 5.58],
    [5, 1, 5.6334, 0.0000, 0.178, None],
    [5, 2, 0.7620, 2.6530, 0.917, 2.14],
    [5, 3, 0.1172, 1.0686, 1.500, 8.82],
    [6, 1, 3.2721, 11.6773, 0.379, 1.04],
    [6, 2, 0.4077, 1.9873, 1.086, 3.46],
    [6, 3, 0.0815, 1.0861, 1.489, 12.78],
    [7, 1, 7.9064, 0.0000, 0.126, None],
    [7, 2, 1.1159, 4.8963, 0.670, 1.98],
    [7, 3, 0.2515, 1.5944, 1.222, 5.02],
    [7, 4, 0.0582, 1.0348, 1.527, 17.46],
    [8, 1, 4.3583, 20.2948, 0.286, 1.03],
    [8, 2, 0.5791, 3.1808, 0.855, 3.08],
    [8, 3, 0.1765, 1.4507, 1.285, 6.83],
    [8, 4, 0.0448, 1.0478, 1.517, 22.87],
    [9, 1, 10.1759, 0.0000, 0.098, None],
    [9, 2, 1.4585, 7.8971, 0.526, 1.93],
    [9, 3, 0.3561, 2.3651, 1.001, 4.32],
    [9, 4, 0.1294, 1.3165, 1.351, 8.87],
    [9, 5, 0.0348, 1.0210, 1.537, 29.00],
    [10, 1, 5.4449, 31.3788, 0.230, 1.03],
    [10, 2, 0.7414, 4.7363, 0.699, 2.94],
    [10, 3, 0.2479, 1.9952, 1.094, 5.70],
    [10, 4, 0.1008, 1.2638, 1.380, 11.15],
    [10, 5, 0.0283, 1.0304, 1.530, 35.85]
  ],
  'butterworth': [#
    [1,	1,	1.0000,	0.0000,	1.000,	None],
    [2,	1,	1.4142,	1.0000,	1.000,	0.71],
    [3,	1,	1.0000,	0.0000,	1.000,	None],
    [3, 2,	1.0000,	1.0000,	1.272,	1.00],
    [4,	1,	1.8478,	1.0000,	0.719,	0.54],
    [4, 2,	0.7654,	1.0000,	1.390,	1.31],
    [5,	1,	1.0000,	0.0000,	1.000,	None],
    [5, 2,	1.6180,	1.0000,	0.859,	0.62],
    [5, 3,	0.6180,	1.0000,	1.448,	1.62],
    [6,	1,	1.9319,	1.0000,	0.676,	0.52],
    [6, 2,	1.4142,	1.0000,	1.000,	0.71],
    [6, 3,	0.5176,	1.0000,	1.479,	1.93],
    [7,	1,	1.0000,	0.0000,	1.000,	None],
    [7, 2,	1.8019,	1.0000,	0.745,	0.55],
    [7, 3,	1.2470,	1.0000,	1.117,	0.80],
    [7, 4,	0.4450,	1.0000,	1.499,	2.25],
    [8,	1,	1.9616,	1.0000,	0.661,	0.51],
    [8, 2,	1.6629,	1.0000,	0.829,	0.60],
    [8, 3,	1.1111,	1.0000,	1.206,	0.90],
    [8, 4,	0.3902,	1.0000,	1.512,	2.56],
    [9,	1,	1.0000,	0.0000,	1.000,	None],
    [9, 2,	1.8794,	1.0000,	0.703,	0.53],
    [9, 3,	1.5321,	1.0000,	0.917,	0.65],
    [9, 4,	1.0000,	1.0000,	1.272,	1.00],
    [9, 5,	0.3473,	1.0000,	1.521,	2.88],
    [10,	1,	1.9754,	1.0000,	0.655,	0.51],
    [10, 2,	1.7820,	1.0000,	0.756,	0.56],
    [10, 3,	1.4142,	1.0000,	1.000,	0.71],
    [10, 4,	0.9080,	1.0000,	1.322,	1.10],
    [10, 5,	0.3129,	1.0000,	1.527,	3.20]
  ]
}

filter_poly_list = []
for key in poly_values:
  for param_set in poly_values[key]:
    filter_poly_list.append(
      dict(zip(poly_keys,[key]+param_set)))
# dictionary = dict(zip(poly_keys,['bessel']+bessel_values[0]))
#bessel data

def filter_tf(typename, N, A_0=1, dataset=filter_poly_list):
  poly_list = list(filter(lambda params:
    params['type'] == typename and
    params['N'] == N, 
    dataset))
  H = A_0 / np.prod([tf([1,params['ai'],params['bi']],1) for params in poly_list])
  return H

# def filter_tf(filterlist, order):
#   if isinstance(filterlist[0], dict):
#     s = TransferFunction.s
#     for d in filterlist:
#       if d[n] == order:
#         H = np.prod([(s**2+el) for el in d[w2si]]) / np.prod([(s**2+d[a][i]*s+d[b][i]) for i in range(len(d[b]))])
#         if order % 2:
#           H = H / (s+d[a][-1])
#         break
#   elif isinstance(filterlist[0], list):
#     H = tf([1],filterlist[order-1])
#   return H

#%%
### Plot
H = dict(zip(
  type_keys,
  [filter_tf(filtertype,2) for filtertype in type_keys]))
for key in H: # normalize over dc gain
  H[key] = H[key]/np.asscalar(dcgain(H[key]))

#%%
def find_nearest(a, a0):
  idx = np.abs(a - a0).argmin()
  return (idx, a[idx])

def find_fc(TF, init_range=[0,1000], res_step=1000, tol=1e-2, target=1/np.sqrt(2)):
  mag, _, omega = [x.flatten() for x in freqresp(TF,np.linspace(*init_range,res_step))]
  def fc_recurs_approx(mag, omega):
    res = find_nearest(mag, target)
    diff = target - res[1]
    print('{:d}: {:f}'.format(*res))
    if np.abs(diff) > tol:
      # assume negative slope
      if diff < 0:
        search_arr = target-mag[res[0]:]
        idx = np.asscalar(np.argwhere((search_arr > 0))[0])
        new_range = [res[0], res[0]+idx]
        # print('{:d}: {:f}'.format(res[0]+idx,mag[res[0]+idx]))
      else:
        search_arr = target-mag[:res[0]]
        idx = np.asscalar(np.argwhere((search_arr < 0))[-1])
        new_range = [idx, res[0]]
        # print('{:d}: {:f}'.format(idx,mag[idx]))
      mag, _, omega = [x.flatten() for x in freqresp(TF, np.linspace(*new_range, res_step))]
      res, omega = fc_recurs_approx(mag, omega)
    return (res, omega)

  res, omega = fc_recurs_approx(mag, omega)
  return (omega[res[0]], res[1])

#%%
fig = plt.figure()
ax = fig.add_subplot()
ax.set_prop_cycle(cc)

plot_keys = [type_keys[i] for i in [0,1,2,3,6]]
for key in plot_keys:
  x, y = step_response(H[key])
  #x = x/find_fc(H[key])[0]
  ax.plot(x,y, label=type_names[key]) 

ax.legend()
ax.set_xlabel(r'Time')
ax.set_ylabel(r'Signal')

# ax.tick_params(
#     axis='x',
#     which='both',
#     bottom=False,
#     top=False,
#     left=False,
#     right=False,
#     labelbottom=False,
#     labelleft=False,
# )
ax.grid()

if filename is None or not filename:
  filename = Path(__file__).stem
plt.savefig('{}.svg'.format(filename))

plt.show()
# %%
