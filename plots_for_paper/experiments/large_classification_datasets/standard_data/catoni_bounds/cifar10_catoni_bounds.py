import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from utils.plot_utils import multiple_bounds_concatenation,keep_last_elements

"""
Plot for the cifar10 dataset original bound.
"""
def scale01(x):
    return x/np.max(x)


path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/' \
       'standard_data/classification_experiments/cifar10/results/bounds/grid_search/'

models = [0,3,4,6,7]
metric = 'zero_one'
bound_type = 'original'
bound_components = ['bound_0']

test,bounds,lambdas =  multiple_bounds_concatenation(path=path,bound_components=bound_components,models=models,
                                                     metric=metric,bound_type=bound_type)

#Preprocess
axis =0

test_means = np.mean(test,axis=0)
bound_means = np.mean(bounds,axis=0)

test_stds = np.std(test,axis=0)
bound_stds = np.std(bounds,axis=0)

threshold = 2e-2
test_means,bound_means,lambdastmp = keep_last_elements(test_means,bound_means,lambdas,threshold=threshold)
test,bounds,lambdas = keep_last_elements(test,bounds,lambdas,threshold=threshold)

# Figure 1
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

size_font_title = 20
size_font_legend = 13.5
size_font_axis = 20
tick_size = 17#17
border_linewidth = 1.5

fig1, ax1 = plt.subplots(figsize=(3.35, 3))
plt.xlabel('$\lambda$', fontsize=size_font_axis)
plt.ylabel('Catoni',fontsize=size_font_title)



linewidth_m = 7
smoothness = 0.1
area = (8 * 0.7) ** 2  # 0 to 15 point radii

colors = ['#66c2a5','#fc8d62','#8da0cb']
cmap = plt.colormaps["plasma"]

# Plots
std_scale = 2
offset1 = 0
offset3 = 0


patches = []
ax2 = ax1.twinx()
ax1.plot(lambdas,test_means,linewidth=5, c=cmap(0),linestyle='-')
for test_num in range(test.shape[0]):
    ax1.plot(lambdas, test[test_num, :], linewidth=5, c=cmap(0), linestyle='-', alpha=0.2)
ax2.plot(lambdas,bound_means,linewidth=5, c=cmap(0.5),linestyle='--')

patches.append(Line2D([0], [0], marker='s', color='w', label='$Z_{\mathrm{test}}$',
                  markerfacecolor=cmap(0), markersize=15))
patches.append(Line2D([0], [0], marker='s', color='w', label='$\mathcal{B}_{\mathrm{Catoni}}$',
                  markerfacecolor=cmap(0.5), markersize=15))

plt.legend(loc=0, handles=patches, fontsize=size_font_legend)
plt.title('CIFAR-10',fontsize=size_font_title)

# Figure formating
plt.grid(linestyle=':', color='grey',axis='y')
[i.set_linewidth(border_linewidth) for i in ax1.spines.values()]
plt.tight_layout()
ax1.set_xscale('log')

ax1.tick_params(axis='both', which='major', labelsize=tick_size)
ax2.tick_params(axis='both', which='major', labelsize=tick_size)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)

end=1
