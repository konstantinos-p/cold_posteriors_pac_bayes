from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pickle
import torch
import io

def common_matplotlib_plot(results,exclude=[],transform = None,x_axis='$x$',y_axis='$y$',map_test_risk=None,axis_y_limit=None,axis_x_limit=None):
    '''
    Plot Figure
    '''




    # Figure 1
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    #plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

    size_font_title = 20
    size_font_legend = 15
    size_font_axis = 20
    tick_size = 10

    fig1, ax1 = plt.subplots(figsize=(4, 4))
    plt.xlabel(x_axis, fontsize=size_font_axis)
    plt.ylabel(y_axis, fontsize=size_font_axis)

    linewidth_m = 7
    smoothness = 0.1
    area = (8 * 0.7) ** 2  # 0 to 15 point radii

    colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33']

    # Plots
    patches = []
    for key,plt_number in zip(list(results.keys()),np.arange(len(list(results.keys())))):
        if not key in exclude:
            x = results[key][0]
            y = results[key][1]
            if transform != None:
                y = transform(y)
            plt.plot(x,y, linewidth=5, c=colors[plt_number])
            patches.append(Line2D([0], [0], marker='s', color='w', label=key,
                        markerfacecolor=colors[plt_number], markersize=15))

    if map_test_risk !=None:
        plt.axhline(y=map_test_risk, color='black', linestyle='--',zorder=5)

    plt.legend(loc=1, handles=patches, fontsize=size_font_legend)

    # Figure formating
    plt.grid(linestyle=':', color='k')
    plt.tight_layout()

    if axis_x_limit != None:
        ax1.set_xlim(xmin=0, xmax=axis_x_limit)
    if axis_y_limit != None:
        ax1.set_ylim(ymin=0, ymax=axis_y_limit)

    return

def amin_bound(lambdas,bound):
    '''
    Finds the minimal value of the bound as well
    as the corresponding value of lambda.
    '''
    return lambdas[np.argmin(bound)],bound[np.argmin(bound)]

def normalize(x):
    '''
    Make x zero mean and unit variance.
    '''

    x = x-np.mean(x)
    x = x/np.std(x)
    return x

def log_scale(x):
    '''
    Make x log scaled.
    '''

    return np.log(x)

def zero_mean(x):
    '''
    Center x.
    '''
    return x-np.mean(x)

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)




