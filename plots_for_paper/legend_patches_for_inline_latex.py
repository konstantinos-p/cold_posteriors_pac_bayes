from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D

'''
Generate legend patches of the different colors and patterns used in the experiment Figures. These patches will be used
in the main text of the paper.
'''

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif"
})

size_font_title = 20
size_font_legend = 30.5
size_font_axis = 20
tick_size = 17#17
border_linewidth = 1.5


colors = ['#66c2a5','#fc8d62','#8da0cb']
cmap = plt.colormaps["plasma"]

fig1, ax1 = plt.subplots(figsize=(10,10))
patches = []

bound, = ax1.plot([1, 2, 2], label="Line 1",  linewidth=5, linestyle='-',c=cmap(0))
bound_trials, = ax1.plot([1, 2, 2], label="Line 1",  linewidth=5, linestyle='-',c=cmap(0),alpha=0.5)
test, = ax1.plot([3, 2, 1], label="Line 2", linewidth=5, linestyle='--',c=cmap(0.5))
barrier, = ax1.plot([3, 2, 2], label="Line 2", linewidth=5, linestyle='--', alpha=0.5)

'''
patches.append(Line2D([0], [0], marker='--', color='w', label='$Z_{\mathrm{test}}$',
                      markerfacecolor=cmap(0), markersize=15))
patches.append(Line2D([0], [0], marker='s', color='w', label='$\mathcal{B}_{\mathrm{Alquier}}$',
                      markerfacecolor=cmap(0.5), markersize=15))
patches.append(Line2D([0], [0], marker='s', color='w', label='$\mathcal{B}_{\mathrm{Alquier}}$',
                      markerfacecolor='#1f77b4', markersize=15,alpha=0.5))

ax1.legend(loc=1, handles=patches, fontsize=size_font_legend)
'''

first_legend = ax1.legend(handles=[bound,test,barrier,bound_trials], loc='upper right', fontsize=size_font_legend)

end = 1