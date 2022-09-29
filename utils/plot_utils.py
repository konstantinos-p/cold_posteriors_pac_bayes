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

def multiple_runs_concatenation(path,runs,models,metric):
    '''
    Concatenates the results of multiple runs.

    Parameters
    ----------
    path: string
        The path of the multiple runs.
    runs: Python list
        The names of the runs to concatenate. The list should contain strings of the form
        run_0,run_1
    models: Python list
        A list of numbers of different models to use.
    metric: {'zero_one','nll','ECE'}

    Returns
    -------
    mul_run_results_test: torch.tensor
        The concatenated test metric results from multiple runs and models.
    mul_run_results_validation: torch.tensor
        The concatenated validation metric results from multiple runs and models.
    mul_run_lambdas: torch.tensor
        The concatenated lambdas from multiple runs and models.
    '''
    mul_run_results_test = []
    mul_run_results_validation = []
    mul_run_lambdas = []

    for run in runs:
        test=[]
        val=[]
        for model in models:
            results_file = open(path +run+ "/results_" + str(model) + ".pkl", "rb")
            output = pickle.load(results_file)
            test.append(np.reshape(np.array(output[0]['test'][metric][1]), (1, -1)))
            val.append(np.reshape(np.array(output[0]['validation'][metric][1]), (1, -1)))

        lambdas = np.reshape(np.array(output[0]['test'][metric][0]), (-1))

        test = np.concatenate(test, axis=0)
        val = np.concatenate(val, axis=0)

        mul_run_results_test.append(test)
        mul_run_results_validation.append(val)
        mul_run_lambdas.append(lambdas)

    mul_run_results_test = np.concatenate(mul_run_results_test, axis=1)
    mul_run_results_validation = np.concatenate(mul_run_results_validation, axis=1)
    mul_run_lambdas = np.concatenate(mul_run_lambdas)

    sort_seq = np.argsort(mul_run_lambdas)
    mul_run_lambdas = mul_run_lambdas[sort_seq]
    mul_run_results_test = mul_run_results_test[:,sort_seq]
    mul_run_results_validation = mul_run_results_validation[:,sort_seq]


    return mul_run_results_test,mul_run_results_validation,mul_run_lambdas

def multiple_bounds_concatenation(path,bound_components,models,metric,bound_type):
    '''
    Concatenates the results of multiple bounds.

    Parameters
    ----------
    path: string
        The path of the multiple bounds.
    runs: Python list
        The names of the bounds to concatenate. The list should contain strings of the form
        bound_0,bound_1
    models: Python list
        A list of numbers of different models to use.
    metric: {'zero_one','nll','ECE'}

    Returns
    -------
    mul_run_results_test: torch.tensor
        The concatenated test metric results from multiple bound_components and models.
    mul_run_results_bound: torch.tensor
        The concatenated bound results from multiple bound_components and models.
    mul_run_lambdas: torch.tensor
        The concatenated lambdas from multiple bound_components and models.
    '''
    mul_run_results_test = []
    mul_run_results_bound = []
    mul_run_lambdas = []

    for bound_component in bound_components:
        test=[]
        bound=[]
        for model in models:
            results_file1 = open(path + bound_component +"/results_" + str(model) + ".pkl", "rb")
            output1 = pickle.load(results_file1)
            test.append(np.reshape(output1[metric][1], (1, -1)))
            bound.append(np.reshape(output1[bound_type][1], (1, -1)))

        lambdas = np.reshape(np.array(output1[bound_type][0]), (-1))

        test = np.concatenate(test, axis=0)
        bound = np.concatenate(bound, axis=0)

        mul_run_results_test.append(test)
        mul_run_results_bound.append(bound)
        mul_run_lambdas.append(lambdas)

    mul_run_results_test = np.concatenate(mul_run_results_test, axis=1)
    mul_run_results_bound = np.concatenate(mul_run_results_bound, axis=1)
    mul_run_lambdas = np.concatenate(mul_run_lambdas)

    sort_seq = np.argsort(mul_run_lambdas)
    mul_run_lambdas = mul_run_lambdas[sort_seq]
    mul_run_results_test = mul_run_results_test[:,sort_seq]
    mul_run_results_bound = mul_run_results_bound[:,sort_seq]


    return mul_run_results_test,mul_run_results_bound,mul_run_lambdas

def to_zero_one(x):
    if np.amin(x)<0:
        x+=np.abs(np.amin(x))
    x = x/np.amax(x)
    return x

def keep_last_elements(test,bound,lambdas,threshold):
    '''
    Test the test, bound and lambdas arrays and outputs a subsection of them after thresholding lambda.
    '''

    if len(test.shape)==1:
        test = test[np.where(lambdas > threshold)]
        bound = bound[np.where(lambdas > threshold)]
        lambdas = lambdas[np.where(lambdas>threshold)]
    else:
        test = test[:,np.where(lambdas > threshold)]
        bound = bound[:,np.where(lambdas > threshold)]
        lambdas = lambdas[np.where(lambdas>threshold)]
        test = test.squeeze()
        bound = bound.squeeze()
    return test,bound,lambdas

def plot_catoni(ax,xlabel,lambdas,test_means,bound_means,test,title,legend,size_font_axis,
                size_font_title,size_font_legend,border_linewidth,tick_size,vertline=True,ylabel1=None,ylabel2=None):
    '''
    Basic Catoni plot.
    '''
    if xlabel != None:
        ax.set_xlabel(xlabel, fontsize=size_font_axis)
    if ylabel1 != None:
        ax.set_ylabel(ylabel1, fontsize=size_font_title)

    cmap = plt.colormaps["plasma"]

    # Plots
    patches = []
    ax2 = ax.twinx()
    if ylabel2 != None:
        ax2.set_ylabel(ylabel2, fontsize=size_font_title)
    ax.plot(lambdas, test_means, linewidth=5, c=cmap(0), linestyle='-')
    for test_num in range(test.shape[0]):
        ax.plot(lambdas, test[test_num, :], linewidth=5, c=cmap(0), linestyle='-', alpha=0.2)
    ax2.plot(lambdas, bound_means, linewidth=5, c=cmap(0.5), linestyle='--')
    if legend == True:
        patches.append(Line2D([0], [0], marker='s', color='w', label='$Z_{\mathrm{test}}$',
                              markerfacecolor=cmap(0), markersize=15))
        patches.append(Line2D([0], [0], marker='s', color='w', label='$\mathcal{B}_{\mathrm{Catoni}}$',
                              markerfacecolor=cmap(0.5), markersize=15))

        ax.legend(loc=0, handles=patches, fontsize=size_font_legend)
    if vertline ==True:
        ax.vlines(1,np.amin(test[np.isfinite(test)]),np.amax(test[np.isfinite(test)]), linewidth=5, linestyle='--',
                   zorder=-1, alpha=0.5)
    if title != None:
        ax.set_title(title, fontsize=size_font_title)

    # Figure formating
    ax.grid(linestyle=':', color='grey', axis='y')
    [i.set_linewidth(border_linewidth) for i in ax.spines.values()]
    ax.set_xscale('log')

    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax2.tick_params(axis='both', which='major', labelsize=tick_size)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(axis='y', colors=cmap(0.5))
    ax.tick_params(axis='y', colors=cmap(0))
    return


def plot_test_val(ax1,xlabel,ylabel,lambdas,test_means,val_means,test,title,legend,size_font_axis,
                size_font_title,size_font_legend,border_linewidth,tick_size,vertline=True):
    # Figure 1
    if xlabel != None:
        ax1.set_xlabel(xlabel, fontsize=size_font_axis)
    if ylabel != None:
        ax1.set_ylabel(ylabel, fontsize=size_font_title)


    cmap = plt.colormaps["plasma"]

    # Plots
    std_scale = 2

    ax1.plot(lambdas, test_means, linewidth=5, c=cmap(0))
    for test_num in range(test.shape[0]):
        ax1.plot(lambdas, test[test_num, :], linewidth=5, c=cmap(0), linestyle='-', alpha=0.2)
    ax1.plot(lambdas, val_means, linewidth=5, c=cmap(0.5), linestyle='--')
    if legend == True:
        patches = []
        patches.append(Line2D([0], [0], marker='s', color='w', label='$Z_{\mathrm{test}}$',
                              markerfacecolor=cmap(0), markersize=15))
        patches.append(Line2D([0], [0], marker='s', color='w', label='$Z_{\mathrm{validation}}$',
                              markerfacecolor=cmap(0.5), markersize=15))

        ax1.legend(loc=1, handles=patches, fontsize=size_font_legend)
    if vertline ==True:
        ax1.vlines(1,np.amin(test[np.isfinite(test)]),np.amax(test[np.isfinite(test)]), linewidth=5, linestyle='--',
                   zorder=-1, alpha=0.5)
    if title != None:
        ax1.set_title(title, fontsize=size_font_title)

    # Figure formating
    ax1.grid(linestyle=':', color='grey', axis='y')
    [i.set_linewidth(border_linewidth) for i in ax1.spines.values()]
    plt.tight_layout()
    ax1.set_xscale('log')

    ax1.tick_params(axis='both', which='major', labelsize=tick_size)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    #ax1.tick_params(axis='y', colors=cmap(0))
    return

def plot_ECE_zero_one(ax1,xlabel,ylabel,test_means1,test_means2,title,legend,size_font_axis,
                size_font_title,size_font_legend,border_linewidth,tick_size,vertline=True):
    # Figure 1
    if xlabel != None:
        ax1.set_xlabel(xlabel, fontsize=size_font_axis)
    if ylabel != None:
        ax1.set_ylabel(ylabel, fontsize=size_font_title)


    cmap = plt.colormaps["plasma"]

    a = is_pareto_efficient(np.hstack([np.reshape(test_means1,(-1,1)),np.reshape(test_means2,(-1,1))]))
    test_means1 = test_means1[a]
    test_means2 = test_means2[a]

    ax1.scatter(test_means1, test_means2, s=165, c='violet',marker='*')

    if legend == True:
        patches = []
        patches.append(Line2D([0], [0], marker='s', color='w', label='$Z_{\mathrm{test}}$',
                              markerfacecolor=cmap(0), markersize=15))
        patches.append(Line2D([0], [0], marker='s', color='w', label='$Z_{\mathrm{validation}}$',
                              markerfacecolor=cmap(0.5), markersize=15))

        ax1.legend(loc=1, handles=patches, fontsize=size_font_legend)

    if title != None:
        ax1.set_title(title, fontsize=size_font_title)

    # Figure formating
    ax1.grid(linestyle=':', color='grey', axis='y')
    [i.set_linewidth(border_linewidth) for i in ax1.spines.values()]
    plt.tight_layout()
    #ax1.set_xscale('log')

    ax1.tick_params(axis='both', which='major', labelsize=tick_size)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    #ax1.tick_params(axis='y', colors=cmap(0))
    return

def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient
