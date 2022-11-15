from utils.plot_utils import normalize,to_zero_one,keep_last_elements
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from utils.plot_utils import multiple_bounds_concatenation
from matplotlib.colors import Normalize
import matplotlib.colors as colors
"""
Plot the correlation of the bound with the risk for the case of regression.
"""

# cifar10
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/' \
       'standard_data/classification_experiments/cifar10/results/bounds/grid_search/'

models = [0,3,4,6,7]
metric = 'zero_one'
bound_type = 'original'
bound_components = ['bound_0']

test,bounds,lambdas_cifar10 =  multiple_bounds_concatenation(path=path,bound_components=bound_components,models=models,
                                                     metric=metric,bound_type=bound_type)

#Preprocess
test_means_cifar10 = np.mean(test,axis=0)
bound_means_cifar10 = np.mean(bounds,axis=0)

test_stds_cifar10 = np.std(test,axis=0)
bound_stds_cifar10 = np.std(bounds,axis=0)

threshold = 2e-2
test_means_cifar10,bound_means_cifar10,lambdas_cifar10 = keep_last_elements(test_means_cifar10,bound_means_cifar10,
                                                                            lambdas_cifar10,threshold=threshold)


# cifar100
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/' \
       'standard_data/classification_experiments/cifar100/results/bounds/grid_search/'

models = [0,1,2,3,4,5,6,7,8]
metric = 'zero_one'
bound_type = 'original'
bound_components = ['bound_0']

test,bounds,lambdas_cifar100 =  multiple_bounds_concatenation(path=path,bound_components=bound_components,models=models,
                                                     metric=metric,bound_type=bound_type)

#Preprocess
test_means_cifar100 = np.mean(test,axis=0)
bound_means_cifar100 = np.mean(bounds,axis=0)

test_stds_cifar100 = np.std(test,axis=0)
bound_stds_cifar100 = np.std(bounds,axis=0)

threshold = 1e-2
test_means_cifar100,bound_means_cifar100,lambdas_cifar100 = keep_last_elements(test_means_cifar100,bound_means_cifar100,
                                                                               lambdas_cifar100,threshold=threshold)

# fashionmnist
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/' \
       'standard_data/classification_experiments/fashionmnist/results/bounds/grid_search/'

models = [0,1,2,3,4,5,6,7,8,9]
metric = 'zero_one'
bound_type = 'original'
bound_components = ['bound_1']

test,bounds,lambdas_fashionmnist =  multiple_bounds_concatenation(path=path,bound_components=bound_components,models=models,
                                                     metric=metric,bound_type=bound_type)

#Preprocess
test_means_fashionmnist = np.mean(test,axis=0)
bound_means_fashionmnist = np.mean(bounds,axis=0)

test_stds_fashionmnist = np.std(test,axis=0)
bound_stds_fashionmnist = np.std(bounds,axis=0)
#threshold = 1e-5
#test_means_fashionmnist,bound_means_fashionmnist,lambdas_fashionmnist = keep_last_elements(test_means_fashionmnist,
#                                                                                           bound_means_fashionmnist,
#                                                                                           lambdas_fashionmnist,
#                                                                                           threshold=threshold)


# svhn
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/' \
       'standard_data/classification_experiments/svhn/results/bounds/grid_search/'

models = [0,1,2,3,4,5,6,7,8,9]
metric = 'zero_one'
bound_type = 'original'
bound_components = ['bound_0']

test,bounds,lambdas_svhn =  multiple_bounds_concatenation(path=path,bound_components=bound_components,models=models,
                                                     metric=metric,bound_type=bound_type)

#Preprocess
test_means_svhn = np.mean(test,axis=0)
bound_means_svhn = np.mean(bounds,axis=0)

test_stds_svhn = np.std(test,axis=0)
bound_stds_svhn = np.std(bounds,axis=0)

threshold = 1e-1
test_means_svhn,bound_means_svhn,lambdas_svhn = keep_last_elements(test_means_svhn,bound_means_svhn,lambdas_svhn,
                                                                                           threshold=threshold)


#Normalization

test_means_cifar10 = normalize(test_means_cifar10)
test_means_cifar100 = normalize(test_means_cifar100)
test_means_fashionmnist = normalize(test_means_fashionmnist)
test_means_svhn = normalize(test_means_svhn)

bound_means_cifar10 = normalize(bound_means_cifar10)
bound_means_cifar100 = normalize(bound_means_cifar100)
bound_means_fashionmnist = normalize(bound_means_fashionmnist)
bound_means_svhn = normalize(bound_means_svhn)


test_means_cifar10 = np.reshape(test_means_cifar10,(-1,1))
test_means_cifar100 = np.reshape(test_means_cifar100,(-1,1))
test_means_fashionmnist = np.reshape(test_means_fashionmnist,(-1,1))
test_means_svhn = np.reshape(test_means_svhn,(-1,1))

bound_means_cifar10 = np.reshape(bound_means_cifar10,(-1,1))
bound_means_cifar100 = np.reshape(bound_means_cifar100,(-1,1))
bound_means_fashionmnist = np.reshape(bound_means_fashionmnist,(-1,1))
bound_means_svhn = np.reshape(bound_means_svhn,(-1,1))

# Figure 1
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

size_font_title = 20
size_font_legend = 15/1.5
size_font_axis = 15
tick_size = 10
border_linewidth = 1.5

fig1, ax1 = plt.subplots(figsize=(3.8,3))
plt.xlabel('$\mathcal{B}_{\mathrm{Catoni}}$', fontsize=size_font_axis)
plt.ylabel('Test 0-1 Loss', fontsize=size_font_axis)


linewidth_m = 7
smoothness = 0.1
area = (12 * 1) ** 2  # 0 to 15 point radii

#colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33']
cmap = plt.colormaps["plasma"]

# Plots
#Normalization
ax1.scatter(bound_means_cifar10,test_means_cifar10, s=area, marker='o', c=cmap(to_zero_one(np.log(lambdas_cifar10))),alpha=0.7)
ax1.scatter(bound_means_cifar100,test_means_cifar100, s=area, marker='s', c=cmap(to_zero_one(np.log(lambdas_cifar100))),alpha=0.7)
ax1.scatter(bound_means_fashionmnist,test_means_fashionmnist, s=area, marker='^', c=cmap(to_zero_one(np.log(lambdas_fashionmnist))),alpha=0.7)
ax1.scatter(bound_means_svhn,test_means_svhn, s=area, marker='*', c=cmap(to_zero_one(np.log(lambdas_svhn))),alpha=0.7)

#Plot optimal correlation line
x_linspace = np.reshape(np.linspace(-1,4,20),(-1,1))
plt.plot(x_linspace,x_linspace,color='black',linestyle='--',linewidth=2)




patch1 = Line2D([0], [0], marker='o', color='w', label='CIFAR-10',
                      markerfacecolor=cmap(1), markersize=10)
patch2 = Line2D([0], [0], marker='s', color='w', label='CIFAR-100',
                      markerfacecolor=cmap(1), markersize=10)
patch3 = Line2D([0], [0], marker='^', color='w', label='FMNIST',
                      markerfacecolor=cmap(1), markersize=10)
patch4 = Line2D([0], [0], marker='*', color='w', label='SVHN',
                      markerfacecolor=cmap(1), markersize=10)


plt.legend(loc=4, handles=[patch1,patch2,patch3,patch4], fontsize=size_font_legend)
cb = plt.colorbar(plt.cm.ScalarMappable(norm=colors.LogNorm(1e-7, 1e+4), cmap=cmap),
             ax=ax1, label="$\lambda$")
cb.outline.set_linewidth(border_linewidth)
# Figure formating
plt.grid(linestyle=':', color='k')
plt.xlim(np.amin(x_linspace),np.amax(x_linspace))
plt.ylim(np.amin(x_linspace),np.amax(x_linspace))
ax1.tick_params(axis='both', which='major', labelsize=tick_size)
[i.set_linewidth(border_linewidth) for i in ax1.spines.values()]
plt.tight_layout()

end =1
