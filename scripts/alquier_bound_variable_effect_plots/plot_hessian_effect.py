import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from utils import posterior_variance, empirical_risk, moment, kl, bound

'''
In this script we plot the optimized Alquier bound value for different lambda
when we change the values of the hessian "h".

All other variables are set to have the value of 1. This is meant to
be illustrative of the loss landscape with respect to lambda. This loss landscape is
expected to be convex and have a single minimum.
'''

#seemingly more influence
prior_variance = 0.1
x_variance = 1
#seemingly less influence
h = 1
prior_posterior_distance = 1
#Unobservable/or not modified for the same architecture.
delta = 0.05
d = 1
oracle_l2 = 1
epsilon_variance = 1

lambda_bound = np.linspace(0,1/(2*prior_variance*x_variance),num = 100)





#Figure 1
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

size_font_title = 20
size_font_legend = 15
size_font_axis = 20
tick_size=10

fig1, ax1 = plt.subplots(figsize=(4,4))
plt.xlabel(r'$\lambda$',fontsize=size_font_axis)
plt.ylabel('Bound Value',fontsize=size_font_axis)



linewidth_m = 7
smoothness= 0.1
area = (8 * 0.7)**2  # 0 to 15 point radii

colors=['#377eb8','#e41a1c','#4daf4a','#8da0cb','#ff7f00','#ffff33','#fe01b1']


#Plot Non Vacuous
count = 0
for h in [1,500,10000]:
    bound_vals = bound(lambda_bound,h,d,prior_variance,oracle_l2,epsilon_variance,x_variance,prior_posterior_distance,delta)
    min_loc = np.argmin(bound_vals)

    bound_minimum_val = bound_vals[min_loc]
    lambda_minimum_val = lambda_bound[min_loc]

    plt.plot(lambda_bound,bound_vals,linewidth=5,c=colors[count])
    plt.scatter(lambda_bound[min_loc],bound_vals[min_loc],marker='x',facecolor='k',zorder=10,s=70)
    count +=1



#Figure formating

patch0 = Line2D([0], [0], marker='s', color='w', label=r'h=1',
       markerfacecolor=colors[0], markersize=15)
patch1 = Line2D([0], [0], marker='s', color='w', label=r'h=500',
       markerfacecolor=colors[1], markersize=15)
patch2 = Line2D([0], [0], marker='s', color='w', label=r'h=10000',
       markerfacecolor=colors[2], markersize=15)

plt.legend(loc = 1,handles=[patch0,patch1,patch2],fontsize=size_font_legend)


plt.grid(linestyle=':',color='k')
plt.tight_layout()

#ax1.set_xlim(xmin=2, xmax=100)
ax1.set_ylim(ymin=0, ymax=30)


end = 1