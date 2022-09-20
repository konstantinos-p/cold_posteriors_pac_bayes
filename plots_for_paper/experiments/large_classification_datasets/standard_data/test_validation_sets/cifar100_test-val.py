import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from utils.plot_utils import multiple_runs_concatenation

"""
Plot the requires metrics for the test and validation sets of the cifar100 dataset.
"""
def scale01(x):
    return x/np.max(x)
def softmax(x):
    return np.maximum(x,0.01)


path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/' \
       'classification_experiments/cifar100/results/runs/isotropic/'

metric = 'zero_one'
nlls = {}
original_bounds= {}
models = [0,1,2,3,4,5,6,7,8]#[0,1,2,3] or  [0,1,4]

runs = ['run_0','run_1','run_2']

test,val,lambdas = multiple_runs_concatenation(path=path,runs=runs,models=models,metric=metric)

#Preprocess
axis= 0
nlls_means = {}
nlls_stds = {}
original_bounds_means = {}
original_bounds_stds = {}

test_means = np.mean(test,axis=0)
val_means = np.mean(val,axis=0)

test_std = np.std(test,axis=0)
val_std = np.std(val,axis=0)




# Figure 1
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


size_font_title = 20
size_font_legend = 15
size_font_axis = 20
tick_size = 10

fig1, ax1 = plt.subplots(figsize=(3,3))
plt.xlabel('$\lambda$', fontsize=size_font_axis)

linewidth_m = 7
smoothness = 0.1
area = (8 * 0.7) ** 2  # 0 to 15 point radii

colors = ['#66c2a5','#fc8d62','#8da0cb']

# Plots
std_scale = 2


plt.plot(lambdas, test_means, linewidth=5, c=colors[1])
ax1.fill_between(lambdas, softmax(test_means - std_scale * test_std),
                 test_means + std_scale * test_std, color=colors[1],
                 alpha=.5)
plt.plot(lambdas, val_means, linewidth=5, c=colors[2],linestyle='--')
ax1.fill_between(lambdas, softmax(val_means - std_scale * val_std),
                 val_means + std_scale * val_std, color=colors[2],
                 alpha=.5)

patches = []

patches.append(Line2D([0], [0], marker='s', color='w', label='$Z_{\mathrm{test}}$',
                      markerfacecolor=colors[1], markersize=15))
patches.append(Line2D([0], [0], marker='s', color='w', label='$Z_{\mathrm{validation}}$',
                      markerfacecolor=colors[2], markersize=15))

plt.legend(loc=1, handles=patches, fontsize=size_font_legend)


# Figure formating
plt.grid(linestyle=':', color='k')
plt.tight_layout()
ax1.set_xscale('log')
#ax1.set_yscale('log')


end=1
