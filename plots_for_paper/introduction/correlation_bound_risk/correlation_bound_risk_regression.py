from utils.plot_utils import normalize
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression

"""
Plot the correlation of the bound with the risk for the case of regression.
"""

# Abalone
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/' \
       'standard_data/regression_experiments/abalone/results/abalone_lambda_0_to_20/'

nlls = []
original_bounds= []

for i in range(10):
    results_file1 = open(path+'model_'+str(i)+"/results_1.pkl", "rb")
    output1 = pickle.load(results_file1)
    nlls.append(np.reshape(output1['GaussianNLL'][1],(1,-1)))
    original_bounds.append(np.reshape(output1['original'][1],(1,-1)))


lambdas_abalone = output1['GaussianNLL'][0]

axis= 0

nlls = np.concatenate(nlls,axis=axis)
original_bounds = np.concatenate(original_bounds,axis=axis)

nlls_mean_abalone = np.mean(nlls,axis=0)
original_bounds_mean_abalone = np.mean(original_bounds,axis=0)

# Diamonds
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/' \
       'standard_data/regression_experiments/diamonds/results/diamonds_lambda_0_to_10/'

nlls = []
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

nlls_mean_diamonds = np.mean(nlls,axis=0)
original_bounds_mean_diamonds = np.mean(original_bounds,axis=0)

# KC_House
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


lambdas_kc_house = output1['GaussianNLL'][0]

axis= 0

nlls = np.concatenate(nlls,axis=axis)
original_bounds = np.concatenate(original_bounds,axis=axis)

nlls_mean_kc_house = np.mean(nlls,axis=0)
original_bounds_mean_kc_house = np.mean(original_bounds,axis=0)


#Normalization
nlls_mean_kc_house = normalize(nlls_mean_kc_house)
nlls_mean_diamonds = normalize(nlls_mean_diamonds)
nlls_mean_abalone = normalize(nlls_mean_abalone)

original_bounds_mean_kc_house = normalize(original_bounds_mean_kc_house)
original_bounds_mean_diamonds = normalize(original_bounds_mean_diamonds)
original_bounds_mean_abalone = normalize(original_bounds_mean_abalone)

nlls_mean_kc_house = np.reshape(nlls_mean_kc_house,(-1,1))
nlls_mean_diamonds = np.reshape(nlls_mean_diamonds,(-1,1))
nlls_mean_abalone = np.reshape(nlls_mean_abalone,(-1,1))

original_bounds_mean_kc_house = np.reshape(original_bounds_mean_kc_house,(-1,1))
original_bounds_mean_diamonds = np.reshape(original_bounds_mean_diamonds,(-1,1))
original_bounds_mean_abalone = np.reshape(original_bounds_mean_abalone,(-1,1))

# Figure 1
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

size_font_title = 20
size_font_legend = 15
size_font_axis = 20
tick_size = 10

fig1, ax1 = plt.subplots(figsize=(3,3))
plt.xlabel('$\mathcal{B}_{\mathrm{original}}$', fontsize=size_font_axis)
plt.ylabel('$\mathrm{Test \; nll}$', fontsize=size_font_axis)


linewidth_m = 7
smoothness = 0.1
area = (7 * 1) ** 2  # 0 to 15 point radii

colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33']

# Plots
#Normalization
plt.scatter(original_bounds_mean_abalone,nlls_mean_abalone, s=area, marker='o', c=colors[0])
plt.scatter(original_bounds_mean_diamonds,nlls_mean_diamonds, s=area, marker='s', c=colors[1])
plt.scatter(original_bounds_mean_kc_house,nlls_mean_kc_house, s=area, marker='^', c=colors[2])

#Plot optimal correlation line
x_linspace = np.reshape(np.linspace(-1,4.5,20),(-1,1))

reg1 = LinearRegression().fit(original_bounds_mean_abalone, nlls_mean_abalone)
plt.plot(x_linspace,reg1.predict(x_linspace),color=colors[0],linestyle='-',linewidth=2)

reg2 = LinearRegression().fit(original_bounds_mean_diamonds, nlls_mean_diamonds)
plt.plot(x_linspace,reg2.predict(x_linspace),color=colors[1],linestyle='-',linewidth=2)

reg3 = LinearRegression().fit(original_bounds_mean_kc_house, nlls_mean_kc_house)
plt.plot(x_linspace,reg3.predict(x_linspace),color=colors[2],linestyle='-',linewidth=2)

plt.plot(x_linspace,x_linspace,color='black',linestyle='--',linewidth=2)




patch1 = Line2D([0], [0], marker='s', color='w', label='Abalone',
                      markerfacecolor=colors[0], markersize=10)
patch2 = Line2D([0], [0], marker='s', color='w', label='Diamonds',
                      markerfacecolor=colors[1], markersize=10)
patch3 = Line2D([0], [0], marker='s', color='w', label='KC_House',
                      markerfacecolor=colors[2], markersize=10)


plt.legend(loc=4, handles=[patch1,patch2,patch3], fontsize=size_font_legend/1.3)

# Figure formating
plt.grid(linestyle=':', color='k')
plt.tight_layout()
plt.xlim(np.amin(x_linspace),np.amax(x_linspace))
plt.ylim(np.amin(x_linspace),np.amax(x_linspace))



end =1
