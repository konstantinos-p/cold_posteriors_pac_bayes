import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from utils.plot_utils import multiple_bounds_concatenation

"""
Plot for the cifar10 dataset original bound.
"""
def scale01(x):
    return x/np.max(x)


path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/' \
       'standard_data/classification_experiments/cifar10/results/bounds/marginal_likelihood/'

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



# Figure 1
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

size_font_title = 20
size_font_legend = 15
size_font_axis = 20
tick_size = 10

fig1, ax1 = plt.subplots(figsize=(3, 3))
plt.xlabel('$\lambda$', fontsize=size_font_axis)

linewidth_m = 7
smoothness = 0.1
area = (8 * 0.7) ** 2  # 0 to 15 point radii

colors = ['#66c2a5','#fc8d62','#8da0cb']

# Plots
std_scale = 2
offset1 = 0
offset3 = 0


patches = []
ax2 = ax1.twinx()
ax1.plot(lambdas,test_means,linewidth=5, c=colors[1],linestyle='-')
ax2.plot(lambdas,bound_means,linewidth=5, c=colors[2],linestyle='--')
patches.append(Line2D([0], [0], marker='s', color='w', label='NLL',
                  markerfacecolor=colors[0], markersize=15))


# Figure formating
plt.grid(linestyle=':', color='k')
plt.tight_layout()
ax1.set_xscale('log')

end=1
