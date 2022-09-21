import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

"""
Plot for the cifar10 dataset original bound.
"""
def scale01(x):
    return x/np.max(x)


path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/' \
       'standard_data/classification_experiments/cifar10/results/bounds/marginal_likelihood/bound_0'

bound_log_file = open(path + "/bound_log_0.pkl", "rb")
bound_log_file = pickle.load(bound_log_file)


nlls=[]
original_bounds=[]
models = [0,3,4,6,7]

for i in models:
    results_file1 = open(path+"/results_"+str(i)+".pkl", "rb")
    output1 = pickle.load(results_file1)
    nlls.append(np.reshape(output1['zero_one'][1],(1,-1)))
    original_bounds.append(np.reshape(output1['original'][1],(1,-1)))


lambdas = output1['nll'][0]

#Preprocess
axis =0

nlls = np.concatenate(nlls,axis=axis)
original_bounds = np.concatenate(original_bounds,axis=axis)

nlls_means = np.mean(nlls,axis=0)
original_bounds_means = np.mean(original_bounds,axis=0)

nlls_stds = np.std(nlls,axis=0)
original_bounds_stds = np.std(original_bounds,axis=0)


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
ax1.plot(lambdas,nlls_means,linewidth=5, c=colors[1],linestyle='-')
ax2.plot(lambdas,original_bounds_means,linewidth=5, c=colors[2],linestyle='--')
patches.append(Line2D([0], [0], marker='s', color='w', label='NLL',
                  markerfacecolor=colors[0], markersize=15))


# Figure formating
plt.grid(linestyle=':', color='k')
plt.tight_layout()
ax1.set_xscale('log')

end=1
