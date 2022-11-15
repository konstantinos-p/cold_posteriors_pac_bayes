from utils.plot_utils import common_matplotlib_plot
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import normalize
"""
Plot for the introduction of the paper.
"""
def scale01(x):
    return x/np.max(x)

def normalize_sklearn(xy):

    x = []
    y = []
    for res in xy:
        x.append(np.reshape(res[0],(1,-1)))
        y.append(np.reshape(res[1],(1,-1)))

    y = np.concatenate(y,axis=0)

    y = normalize(y)

    for i in range(len(xy)):
        xy[i][1] = y[i,:]

    return xy





def return_preprocessed(path,approximate=False):
    '''
    Finds nll and bounds and returns them
    '''

    nlls = {}
    original_bounds= {}
    mixed_bounds = {}
    approximate_bounds = {}
    priors = [9]
    for prior in priors:
        nlls[str(prior)]=[]
        original_bounds[str(prior)]=[]
        mixed_bounds[str(prior)] = []
        approximate_bounds[str(prior)] = []
    for i in range(8):
        for prior in  priors:
            results_file1 = open(path+'model_'+str(i)+"/results_"+str(prior)+".pkl", "rb")
            output1 = pickle.load(results_file1)
            if approximate == True:
                nlls[str(prior)].append(output1['GaussianNLL'])
            else:
                nlls[str(prior)].append(output1['nll'])
            original_bounds[str(prior)].append(output1['original'])
            mixed_bounds[str(prior)].append(output1['mixed'])
            if approximate == True:
                approximate_bounds[str(prior)].append(output1['approximate'])
    if approximate == True:
        return nlls,original_bounds,mixed_bounds,approximate_bounds
    else:
        return nlls,original_bounds,mixed_bounds

def plot(xy,path,linestyle='-'):
    # Figure 1
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    #plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

    size_font_title = 20
    size_font_legend = 15
    size_font_axis = 20
    tick_size = 10

    fig1, ax1 = plt.subplots(figsize=(3,3))
    plt.xlabel('$\lambda$', fontsize=size_font_axis)
    #plt.ylabel(y_axis, fontsize=size_font_axis)

    linewidth_m = 7
    smoothness = 0.1
    area = (8 * 0.7) ** 2  # 0 to 15 point radii

    #colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33']
    colors = ['#66c2a5','#fc8d62','#8da0cb']

    # Plots
    std_scale = 2
    offset1 = 0
    offset3 = 0




    plt.plot(xy[0],xy[1],linewidth=5, c=colors[2],linestyle=linestyle)


    #plt.legend(loc=1, handles=patches, fontsize=size_font_legend)

    # Figure formating
    plt.grid(linestyle=':', color='k')
    plt.tight_layout()
    ax1.set_xscale('log')

    #ax1.set_xlim(xmin=0, xmax=axis_x_limit)
    #ax1.set_ylim(ymin=0, ymax=axis_y_limit)

    plt.savefig(path,dpi=600)
    plt.close()

def plot_all_in_one(xy,path,linestyle='-',num_to_plot=3,caption=None,dataset=None):
    # Figure 1
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    #plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

    size_font_title = 20
    size_font_legend = 15
    size_font_axis = 20
    tick_size = 10

    fig1, ax1 = plt.subplots(figsize=(3,3))
    plt.xlabel('$\lambda$', fontsize=size_font_axis)
    #if dataset == 'abalone/':
    #    plt.ylabel(caption, fontsize=size_font_axis)

    linewidth_m = 7
    smoothness = 0.1
    area = (8 * 0.7) ** 2  # 0 to 15 point radii

    #colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33']
    colors = ['#66c2a5','#fc8d62','#8da0cb']
    #colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']


    # Plots
    std_scale = 2
    offset1 = 0
    offset3 = 0

    #xy = normalize_sklearn(xy)

    for res,color in zip(xy,np.arange(len(xy))):
        if color < num_to_plot:
            plt.plot(res[0],res[1],linewidth=5, c=colors[color],linestyle=linestyle)


    #plt.legend(loc=1, handles=patches, fontsize=size_font_legend)

    # Figure formating
    plt.grid(linestyle=':', color='k')
    plt.tight_layout()
    ax1.set_xscale('log')
    #ax1.set_yscale('log')

    #ax1.set_xlim(xmin=0, xmax=axis_x_limit)
    #ax1.set_ylim(ymin=0, ymax=axis_y_limit)

    plt.savefig(path,dpi=600)
    plt.close()



