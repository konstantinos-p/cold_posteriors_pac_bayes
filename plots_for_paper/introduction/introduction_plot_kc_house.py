from utils.plot_utils import normalize
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

"""
Plot for KC_House for the introduction of the paper (Figure 1).
"""
def scale01(x):
    return x/np.max(x)


path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/' \
       'standard_data/regression_experiments/kc_house/results/kc_house_lambda_0_to_10/'

nlls = []
eces = []
original_bounds= []

for i in range(9):
    results_file1 = open(path+'model_'+str(i)+"/results_1.pkl", "rb")
    output1 = pickle.load(results_file1)
    nlls.append(np.reshape(output1['GaussianNLL'][1],(1,-1)))
    original_bounds.append(np.reshape(output1['original'][1],(1,-1)))


lambdas = output1['GaussianNLL'][0]

axis= 0

nlls = np.concatenate(nlls,axis=axis)
original_bounds = np.concatenate(original_bounds,axis=axis)

nlls_mean = np.mean(nlls,axis=0)
original_bounds_mean = np.mean(original_bounds,axis=0)

nlls_std = np.std(nlls,axis=0)
original_bounds_std = np.std(original_bounds,axis=0)


# Figure 1
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

size_font_title = 20
size_font_legend = 15
size_font_axis = 20
tick_size = 10

fig1, ax1 = plt.subplots(figsize=(2.5, 4))
plt.xlabel('$\lambda$', fontsize=size_font_axis)

linewidth_m = 7
smoothness = 0.1
area = (8 * 0.7) ** 2  # 0 to 15 point radii

colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33']

# Plots
#Normalization
nlls_mean = normalize(nlls_mean)
original_bounds_mean = normalize(original_bounds_mean)

nlls_std = scale01(nlls_std)
original_bounds_std = scale01(original_bounds_std)


std_scale = 2
offset1 = 0
offset3 = 1


plt.plot(lambdas,offset1+nlls_mean,linewidth=5, c=colors[0])
plt.plot(lambdas,offset3+original_bounds_mean,linewidth=5, c='black',linestyle='--')

ax1.fill_between(lambdas, offset1+(nlls_mean-std_scale*nlls_std), offset1+(nlls_mean+std_scale*nlls_std), color=colors[0], alpha=.5)

patch1 = Line2D([0], [0], marker='s', color='w', label='NLL',
                      markerfacecolor=colors[0], markersize=15)
patch2 = Line2D([0], [0], marker='s', color='w', label='ECE',
                      markerfacecolor=colors[1], markersize=15)
patch3 = Line2D([0], [0], marker='s', color='w', label='$\mathcal{B}_{\mathrm{original}}$',
                      markerfacecolor='black', markersize=15)


plt.legend(loc=1, handles=[patch1,patch3], fontsize=size_font_legend)

# Figure formating
plt.grid(linestyle=':', color='k')
plt.tight_layout()


end=1
