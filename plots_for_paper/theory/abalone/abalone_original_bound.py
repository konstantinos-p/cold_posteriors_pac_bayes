import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import pickle
from utils.plot_utils import CPU_Unpickler

'''
Abalone original bound, Figure 3 Appendix.
'''

#Load data
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/' \
       'regression_experiments/abalone/results/abalone_lambda_0_to_20/model_0/'

log_file_1 = open(path+"bound_log_7.pkl", "rb")

contents = CPU_Unpickler(log_file_1).load()

results_file3 = open(path+"results_5.pkl", "rb")
output1 = pickle.load(results_file3)

#Calculate terms
Empicical_Risk = contents['empirical_risk_MC']
Moment = contents['moment_MC']
KL = contents['kl']

lambdas = output1['original'][0]


'''
Plot Figure
'''
#Figure 1
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


size_font_title = 20
size_font_legend = 15
size_font_axis = 20
tick_size=10

fig1, ax1 = plt.subplots(figsize=(4,4))
plt.xlabel(r'$\lambda$',fontsize=size_font_axis)



linewidth_m = 7
smoothness= 0.1
area = (8 * 0.7)**2  # 0 to 15 point radii

colors=['#377eb8','#e41a1c','#4daf4a','#8da0cb','#ff7f00','#ffff33','#fe01b1']


#Plots
plt.plot(lambdas,Empicical_Risk,linewidth=5,c=colors[1])
plt.plot(lambdas,Moment,linewidth=5,c=colors[2])
plt.plot(lambdas,KL,linewidth=5,c=colors[0])
plt.plot(lambdas,Empicical_Risk+Moment+KL,linewidth=5,c='black',linestyle='--')


#Figure formating

patch0 = Line2D([0], [0], marker='s', color='w', label=r'E. Risk',
       markerfacecolor=colors[1], markersize=15)
patch1 = Line2D([0], [0], marker='s', color='w', label=r'Moment',
       markerfacecolor=colors[2], markersize=15)
patch2 = Line2D([0], [0], marker='s', color='w', label=r'KL',
       markerfacecolor=colors[0], markersize=15)
patch3 = Line2D([0], [0], marker='s', color='w', label=r'$\mathcal{B}_{\mathrm{original}}$',
       markerfacecolor='black', markersize=15)

plt.grid(linestyle=':',color='k')
plt.tight_layout()
ax1.set_xscale('log')
ax1.set_yscale('log')

ax1.set_ylim(ymin=0.01, ymax=1000)
patches= [patch0,patch1,patch2,patch3,]
plt.legend(loc=1, handles=patches, fontsize=size_font_legend)
plt.savefig('abalone_original.png',dpi=300)
end = 1