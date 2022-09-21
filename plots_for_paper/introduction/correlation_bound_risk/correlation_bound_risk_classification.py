from utils.plot_utils import normalize,to_zero_one
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from utils.plot_utils import multiple_bounds_concatenation
from matplotlib.colors import Normalize

"""
Plot the correlation of the bound with the risk for the case of regression.
"""

# cifar10
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/' \
       'standard_data/classification_experiments/cifar10/results/bounds/marginal_likelihood/'

models = [0,3,4,6,7]
metric = 'zero_one'
bound_type = 'original'
bound_components = ['bound_0']

test,bounds,lambdas_cifar10 =  multiple_bounds_concatenation(path=path,bound_components=bound_components,models=models,
                                                     metric=metric,bound_type=bound_type)

#Preprocess
test_means_cifar10 = np.mean(test[:,1:],axis=0)
bound_means_cifar10 = np.mean(bounds[:,1:],axis=0)

test_stds_cifar10 = np.std(test[:,1:],axis=0)
bound_stds_cifar10 = np.std(bounds[:,1:],axis=0)
lambdas_cifar10 = lambdas_cifar10[1:]
# cifar100
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/' \
       'standard_data/classification_experiments/cifar100/results/bounds/marginal_likelihood/'

models = [0,1,2,3,4,5,6,7,8]
metric = 'zero_one'
bound_type = 'original'
bound_components = ['bound_0','bound_1']

test,bounds,lambdas_cifar100 =  multiple_bounds_concatenation(path=path,bound_components=bound_components,models=models,
                                                     metric=metric,bound_type=bound_type)

#Preprocess
test_means_cifar100 = np.mean(test,axis=0)
bound_means_cifar100 = np.mean(bounds,axis=0)

test_stds_cifar100 = np.std(test,axis=0)
bound_stds_cifar100 = np.std(bounds,axis=0)

# fashionmnist
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/' \
       'standard_data/classification_experiments/fashionmnist/results/bounds/marginal_likelihood/'

models = [0,1,2,3,4,5,6,7,8,9]
metric = 'zero_one'
bound_type = 'original'
bound_components = ['bound_0','bound_1']

test,bounds,lambdas_fashionmnist =  multiple_bounds_concatenation(path=path,bound_components=bound_components,models=models,
                                                     metric=metric,bound_type=bound_type)

#Preprocess
test_means_fashionmnist = np.mean(test,axis=0)
bound_means_fashionmnist = np.mean(bounds,axis=0)

test_stds_fashionmnist = np.std(test,axis=0)
bound_stds_fashionmnist = np.std(bounds,axis=0)

# svhn
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/' \
       'standard_data/classification_experiments/svhn/results/bounds/marginal_likelihood/'

models = [0,1,2,3,4,5,6,7,8,9]
metric = 'zero_one'
bound_type = 'original'
bound_components = ['bound_0','bound_1']

test,bounds,lambdas_svhn =  multiple_bounds_concatenation(path=path,bound_components=bound_components,models=models,
                                                     metric=metric,bound_type=bound_type)

#Preprocess
test_means_svhn = np.mean(test,axis=0)
bound_means_svhn = np.mean(bounds,axis=0)

test_stds_svhn = np.std(test,axis=0)
bound_stds_svhn = np.std(bounds,axis=0)

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
size_font_legend = 15
size_font_axis = 20
tick_size = 10

fig1, ax1 = plt.subplots(figsize=(3.8,3))
plt.xlabel('$\mathcal{B}_{\mathrm{original}}$', fontsize=size_font_axis)
plt.ylabel('$\mathrm{Test \; nll}$', fontsize=size_font_axis)


linewidth_m = 7
smoothness = 0.1
area = (8 * 1) ** 2  # 0 to 15 point radii

colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33']
cmap = plt.colormaps["plasma"]

# Plots
#Normalization
plt.scatter(bound_means_cifar10,test_means_cifar10, s=area, marker='o', c=cmap(to_zero_one(lambdas_cifar10)),alpha=0.7)
plt.scatter(bound_means_cifar100,test_means_cifar100, s=area, marker='s', c=cmap(to_zero_one(lambdas_cifar100)),alpha=0.7)
plt.scatter(bound_means_fashionmnist,test_means_fashionmnist, s=area, marker='^', c=cmap(to_zero_one(lambdas_fashionmnist)),alpha=0.7)
plt.scatter(bound_means_svhn,test_means_svhn, s=area, marker='^', c=cmap(to_zero_one(lambdas_svhn)),alpha=0.7)

#Plot optimal correlation line
x_linspace = np.reshape(np.linspace(-1,1,20),(-1,1))
plt.plot(x_linspace,x_linspace,color='black',linestyle='--',linewidth=2)




patch1 = Line2D([0], [0], marker='o', color='w', label='CIFAR-10',
                      markerfacecolor=cmap(1), markersize=10)
patch2 = Line2D([0], [0], marker='s', color='w', label='CIFAR-100',
                      markerfacecolor=cmap(1), markersize=10)
patch3 = Line2D([0], [0], marker='^', color='w', label='FMNIST',
                      markerfacecolor=cmap(1), markersize=10)
patch4 = Line2D([0], [0], marker='^', color='w', label='SVHN',
                      markerfacecolor=cmap(1), markersize=10)


plt.legend(loc=4, handles=[patch1,patch2,patch3,patch4], fontsize=size_font_legend/1.3)
plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(0.1, 1e+4), cmap=cmap),
             ax=ax1, label="$\lambda$")

# Figure formating
plt.grid(linestyle=':', color='k')
plt.tight_layout()
plt.xlim(np.amin(x_linspace),np.amax(x_linspace))
plt.ylim(np.amin(x_linspace),np.amax(x_linspace))



end =1
