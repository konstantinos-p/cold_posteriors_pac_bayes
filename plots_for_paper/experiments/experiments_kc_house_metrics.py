import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import torch

"""
Plot for the KC_House dataset metrics for Figure 3 of the paper.
"""
def scale01(x):
    return x/np.max(x)


path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/' \
       'standard_data/regression_experiments/kc_house/results/kc_house_lambda_0_to_10/'

nlls = {}
original_bounds= {}
priors = [1,2,3]
for prior in priors:
    nlls[str(prior)]=[]
    original_bounds[str(prior)]=[]
for i in range(10):
    for prior in  priors:
        results_file1 = open(path+'model_'+str(i)+"/results_"+str(prior)+".pkl", "rb")
        output1 = pickle.load(results_file1)
        nlls[str(prior)].append(np.reshape(output1['GaussianNLL'][1],(1,-1)))
        original_bounds[str(prior)].append(np.reshape(output1['original'][1],(1,-1)))


lambdas = output1['GaussianNLL'][0]

#Preprocess
axis= 0
nlls_means = {}
nlls_stds = {}
original_bounds_means = {}
original_bounds_stds = {}
for prior in priors:
    nlls[str(prior)] = np.concatenate(nlls[str(prior)],axis=axis)
    original_bounds[str(prior)] = np.concatenate(original_bounds[str(prior)],axis=axis)

    nlls_means[str(prior)] = np.mean(nlls[str(prior)],axis=0)
    original_bounds_means[str(prior)] = np.mean(original_bounds[str(prior)],axis=0)

    nlls_stds[str(prior)] = np.std(nlls[str(prior)],axis=0)
    original_bounds_stds[str(prior)] = np.std(original_bounds[str(prior)],axis=0)


prior_variance = np.round(torch.linspace(0.00001,0.1,20).numpy(),3)


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
for prior,color_num in zip(priors,np.arange(len(priors))):
    plt.plot(lambdas,offset1+nlls_means[str(prior)],linewidth=5, c=colors[color_num])

    ax1.fill_between(lambdas, offset1+(nlls_means[str(prior)]-std_scale*nlls_stds[str(prior)]),
                     offset1+(nlls_means[str(prior)]+std_scale*nlls_stds[str(prior)]), color=colors[color_num], alpha=.5)


    patches.append(Line2D([0], [0], marker='s', color='w', label='$\sigma^2_{\pi}='+str(prior_variance[prior])+'$',
                      markerfacecolor=colors[color_num], markersize=15))



plt.legend(loc=1, handles=patches, fontsize=size_font_legend)

# Figure formating
plt.grid(linestyle=':', color='k')
plt.tight_layout()
ax1.set_xscale('log')

end=1
