import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

"""
Plot for the Abalone dataset original bounds for Figure 3 of the paper.
"""
def scale01(x):
    return x/np.max(x)


path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/' \
       'standard_data/regression_experiments/abalone/results/abalone_lambda_0_to_20/'

nlls = {}
original_bounds= {}
priors = [1]
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




# Figure 1
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif', weight='bold')
#plt.rc('font', weight='bold')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif"
})
#plt.rcParams.update({
#    "text.usetex": True,
#})
#plt.rc('text.latex', preamble=r'\usepackage{sfmath} \boldmath')

size_font_title = 20
size_font_legend = 13.5
size_font_axis = 20
tick_size = 17#17
border_linewidth = 1.5


fig1, ax1 = plt.subplots(figsize=(3.35,3))
plt.xlabel('$\lambda$', fontsize=size_font_axis)
plt.ylabel('Alquier',fontsize=size_font_axis)
ax2 = ax1.twinx()

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
for prior,color_num in zip(priors,np.arange(len(priors))):
    ax1.plot(lambdas, nlls_means[str(prior)], linewidth=5, c=cmap(0), linestyle='-')
    for nll_num in range(nlls[str(prior)].shape[0]):
        ax1.plot(lambdas, nlls[str(prior)][nll_num,:], linewidth=5, c=cmap(0), linestyle='-',alpha=0.2)


    ax2.plot(lambdas,offset3+original_bounds_means[str(prior)],linewidth=5, c=cmap(0.5),linestyle='--')
    patches.append(Line2D([0], [0], marker='s', color='w', label='NLL',
                      markerfacecolor=colors[color_num], markersize=15))
plt.title('ABALONE',fontsize=size_font_title)
ax1.vlines(1, 0.95,
           0.98, linewidth=5, linestyle='--',
           zorder=-1, alpha=0.5)
# Figure formating
plt.grid(linestyle=':', color='grey')
[i.set_linewidth(border_linewidth) for i in ax1.spines.values()]
plt.tight_layout()
ax1.set_xscale('log')
ax1.tick_params(axis='both', which='major', labelsize=tick_size)
ax2.tick_params(axis='both', which='major', labelsize=tick_size)

patches = []
patches.append(Line2D([0], [0], marker='s', color='w', label='$Z_{\mathrm{test}}$',
                      markerfacecolor=cmap(0), markersize=15))
patches.append(Line2D([0], [0], marker='s', color='w', label='$\mathcal{B}_{\mathrm{Alquier}}$',
                      markerfacecolor=cmap(0.5), markersize=15))
ax1.legend(loc=1, handles=patches, fontsize=size_font_legend)
ax2.legend(loc=1, handles=patches, fontsize=size_font_legend)


ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)

ax2.tick_params(axis='y', colors=cmap(0.5))
ax1.tick_params(axis='y', colors=cmap(0))

plt.savefig('abalone_alquier.png',dpi=300)




end=1
