import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from utils.plot_utils import multiple_runs_concatenation,keep_last_elements

"""
Plot the requires metrics for the test and validation sets of the cifar10 dataset.
"""
def scale01(x):
    return x/np.max(x)
def softmax(x):
    return np.maximum(x,0.01)


path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/' \
       'classification_experiments/cifar10/results/runs/kron/'

metric = 'zero_one'
models = [0,3,4,6,7]#[0,3,4,6,7]

runs = ['run_0','run_1','run_2']#['run_3']

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

threshold = 1e-7
test_means,val_means,lambdastmp = keep_last_elements(test_means,val_means,lambdas,threshold=threshold)
test,val,lambdas = keep_last_elements(test,val,lambdas,threshold=threshold)


# Figure 1
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


size_font_title = 20
size_font_legend = 13.5
size_font_axis = 20
tick_size = 17#17
border_linewidth = 1.5

fig1, ax1 = plt.subplots(figsize=(3.35,3))
plt.xlabel('$\lambda$', fontsize=size_font_axis)
plt.ylabel('KFAC',fontsize=size_font_title)

linewidth_m = 7
smoothness = 0.1
area = (8 * 0.7) ** 2  # 0 to 15 point radii

colors = ['#66c2a5','#fc8d62','#8da0cb']
cmap = plt.colormaps["plasma"]

# Plots
std_scale = 2


plt.plot(lambdas, test_means, linewidth=5,  c=cmap(0))
for test_num in range(test.shape[0]):
    ax1.plot(lambdas, test[test_num, :], linewidth=5,  c=cmap(0), linestyle='-', alpha=0.2)
plt.plot(lambdas, val_means, linewidth=5,  c=cmap(0.5),linestyle='--')


patches = []

patches.append(Line2D([0], [0], marker='s', color='w', label='$Z_{\mathrm{test}}$',
                      markerfacecolor=cmap(0), markersize=15))
patches.append(Line2D([0], [0], marker='s', color='w', label='$Z_{\mathrm{validation}}$',
                      markerfacecolor=cmap(0.5), markersize=15))

plt.legend(loc=1, handles=patches, fontsize=size_font_legend)


# Figure formating
plt.grid(linestyle=':', color='grey',axis='y')
[i.set_linewidth(border_linewidth) for i in ax1.spines.values()]
plt.tight_layout()
ax1.set_xscale('log')

ax1.tick_params(axis='both', which='major', labelsize=tick_size)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)



end=1
