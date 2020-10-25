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

dir_path = Path(__file__).parent

# Colors (acces by colpalette)
### File locations
outdir = dir_path / 'out'
filename = '' # if empty this files name is used

    
axislables=['t','x']

zeta = [0.05, 0.1, 0.2, 0.707, 1]
n = len(zeta)
res = 1000

colidx = None
cmap = None
for i, el in enumerate(colpalettes):
    if el.name == 'WesMixL':
        colidx = i
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color',el.rgb_hex())
    if el.name == 'ZissouSTD' and el.type == 'div':
        cmap = el.LinearSegmentedColormap()
colors = [None] * n
lines = [None] * n
lineth = [None] * n
zeta_norm = list(map(lambda x: ((x-min(zeta))/max(zeta))**0.4, zeta))
print(zeta_norm)
for i,el in enumerate(zeta):
    colors[i] = cmap(zeta_norm[i])
    print(colors[i])
    if i == n-1:
        lines[i] = '-'
        lineth[i] = 1.2
    else:
        lines[i] = '-'
        lineth[i] = 0.9
custom_cycler = (cycler(color=colors) + cycler(linestyle=lines) + cycler(linewidth=lineth))
plt.rc('axes', prop_cycle=custom_cycler)

#%%
### Plot
fig = plt.figure()
for el in zeta:
    plotsys = ([1], [1,el,1])
    sys = tf(*plotsys)
    # mag, phase, omg = bode(sys, omega_num=res, Plot=False)
    # omg_n = np.abs(phase+np.pi/2).argmin()
    # plt.plot()
    bode_plot(sys, omega_num=res, label='$\zeta = {}$'.format(el))
axes_list = fig.axes
for ax in axes_list:
    ax.legend()


if filename is None or not filename:
    filename = Path(__file__).stem
plt.savefig('{}.svg'.format(filename))

plt.show()
# %%
