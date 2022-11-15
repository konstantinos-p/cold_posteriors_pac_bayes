from utils.plot_utils import normalize,to_zero_one,keep_last_elements
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from utils.plot_utils import multiple_bounds_concatenation
from matplotlib.colors import Normalize
import matplotlib.colors as colors
"""
Plot the outline of our approach.
"""

def approximatly_quadratic_function(x):
    '''
    Defines an approximately quadratic function.
    '''
    res = np.zeros(x.size)
    res[np.where(x<0)] = 0.1*x[np.where(x<0)]**2
    res[np.where(x>0)] = x[np.where(x>=0)]**2
    return res

def quadratic_function(x):
    '''
    Defines an approximately quadratic function.
    '''
    res = 1.1*x**2
    return res

def gaussian(x,mean,sigma):
    '''
    Gaussian function.
    '''
    return (1/(sigma*np.sqrt(2*3.14)))*np.exp(-(1/2)*((x-mean)/sigma)**2)


# Figure 1
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

size_font_title = 20
size_font_legend = 15/1.5
size_font_axis = 15
tick_size = 10
border_linewidth = 1.5

fig1, ax1 = plt.subplots(figsize=(3.8,3))
#plt.xlabel('$\mathcal{B}_{\mathrm{Catoni}}$', fontsize=size_font_axis)
#plt.ylabel('Test 0-1 Loss', fontsize=size_font_axis)


linewidth_m = 7
smoothness = 0.1
area = (12 * 1) ** 2  # 0 to 15 point radii

#colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33']
cmap = plt.colormaps["plasma"]

# Plots

sigma_prior =0.4
sigma_ELBO =0.2

#Plot optimal correlation line
x_linspace = np.linspace(-4,4,100)
plt.plot(x_linspace,approximatly_quadratic_function(x_linspace),color='black',linestyle='-',linewidth=2,alpha=0.5)
x_linspace2 = np.linspace(-1.5,1.5,100)
#posterior = prior
plt.plot(x_linspace2,quadratic_function(x_linspace2),color='black',linestyle='--',linewidth=2)
plt.fill_between(x_linspace2,gaussian(x_linspace2,0,sigma_prior),color=cmap(0),alpha=0.7)
plt.fill_between(x_linspace2+3,gaussian(x_linspace2+3,3,sigma_prior),color=cmap(0),alpha=0.1)

#ELBO posterior
plt.fill_between(x_linspace2,gaussian(x_linspace2,0,sigma_ELBO),color=cmap(0.5),alpha=0.7)

#frozen posterior
plt.arrow(0,0,0,4,width=0.1,color=cmap(0.99),edgecolor='black')

#MAP
plt.scatter(0,0,marker='*',s=area,color='red',zorder=5)
plt.text(-0.2,-0.6,'MAP',fontsize=12)

#Prior
plt.scatter(3,0,marker='*',s=area)
plt.text(2.8,-0.6,'Prior',fontsize=12)

patch1 = Line2D([0], [0], marker='', color='w', label=r'$\frac{1}{\sigma^2_{\hat{\rho}}(\lambda) }=\frac{\lambda h}{d}+\frac{1}{\sigma_{\pi}^2}$',
                      markerfacecolor=cmap(1), markersize=10)
patch2 = Line2D([0], [0], marker='s', color='w', label='$\lambda=1e-7$',
                      markerfacecolor=cmap(0), markersize=10)
patch3 = Line2D([0], [0], marker='s', color='w', label='$\lambda=1$',
                      markerfacecolor=cmap(0.5), markersize=10)
patch4 = Line2D([0], [0], marker='s', color='w', label='$\lambda=1e+4$',
                      markerfacecolor=cmap(0.99), markersize=10)


#plt.legend(loc=1, handles=[patch2,patch3,patch4], fontsize=size_font_legend)
#cb = plt.colorbar(plt.cm.ScalarMappable(norm=colors.LogNorm(1e-7, 1e+4), cmap=cmap),
#             ax=ax1, label="$\lambda$")
#cb.outline.set_linewidth(border_linewidth)
# Figure formating
#plt.grid(linestyle=':', color='k')
plt.xlim(-2,4)
plt.ylim(-1,5)
ax1.tick_params(axis='both', which='major', labelsize=tick_size)
[i.set_linewidth(border_linewidth) for i in ax1.spines.values()]
plt.axis('off')
#plt.title(r'$\frac{1}{\sigma^2_{\hat{\rho}}(\lambda) }=\frac{\lambda h}{d}+\frac{1}{\sigma_{\pi}^2}$',size=size_font_title)
plt.text(1,4,r'$\frac{1}{\sigma^2_{\hat{\rho}}(\lambda) }=\frac{\lambda h}{d}+\frac{1}{\sigma_{\pi}^2}$',
         size=size_font_title,bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),
                   ))
plt.tight_layout()
plt.show()

#asdasd

end =1