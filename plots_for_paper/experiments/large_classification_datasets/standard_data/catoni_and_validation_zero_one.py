import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from utils.plot_utils import multiple_bounds_concatenation,keep_last_elements,plot_catoni,multiple_runs_concatenation,\
    plot_test_val


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
f, axs = plt.subplots(3, 4,figsize = (13, 9))

size_font_title = 20
size_font_legend = 16
size_font_axis = 20
tick_size = 17  # 17
border_linewidth = 1.5

metric = 'zero_one'
bound_type = 'original'

#Subfigure 00
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/' \
       'standard_data/classification_experiments/cifar10/results/bounds/grid_search/'

models = [0,3,4,6,7]
bound_components = ['bound_0']

test,bounds,lambdas =  multiple_bounds_concatenation(path=path,bound_components=bound_components,models=models,
                                                     metric=metric,bound_type=bound_type)

axis =0

test_means = np.mean(test,axis=0)
bound_means = np.mean(bounds,axis=0)

test_stds = np.std(test,axis=0)
bound_stds = np.std(bounds,axis=0)

threshold = 2e-2
test_means,bound_means,lambdastmp = keep_last_elements(test_means,bound_means,lambdas,threshold=threshold)
test,bounds,lambdas = keep_last_elements(test,bounds,lambdas,threshold=threshold)

plot_catoni(ax = axs[0,0],xlabel = '$\lambda$',lambdas = lambdas,
            test_means=test_means,bound_means=bound_means,test=test,title='CIFAR-10',
            legend = True,size_font_axis=size_font_axis,
                size_font_title=size_font_title,size_font_legend = size_font_legend,
            border_linewidth=border_linewidth,tick_size=tick_size,ylabel1 = 'Catoni')


#Subfigure 10
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/' \
       'classification_experiments/cifar10/results/runs/isotropic/'

models = [0,3,4,6,7]#[0,3,4,6,7]

runs = ['run_3']

test,val,lambdas = multiple_runs_concatenation(path=path,runs=runs,models=models,metric=metric)

axis= 0
nlls_means = {}
nlls_stds = {}
original_bounds_means = {}
original_bounds_stds = {}


test_means = np.mean(test,axis=0)
val_means = np.mean(val,axis=0)

test_std = np.std(test,axis=0)
val_std = np.std(val,axis=0)

threshold = 1e-1
test_means,val_means,lambdastmp = keep_last_elements(test_means,val_means,lambdas,threshold=threshold)
test,val,lambdas = keep_last_elements(test,val,lambdas,threshold=threshold)


plot_test_val(ax1 = axs[1,0],xlabel = '$\lambda$',lambdas = lambdas,
            test_means=test_means,val_means=val_means,test=test,title=None,
            legend = True,size_font_axis=size_font_axis,
                size_font_title=size_font_title,size_font_legend = size_font_legend,
            border_linewidth=border_linewidth,tick_size=tick_size,ylabel = 'Isotropic')

#Subfigure 20
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/' \
       'classification_experiments/cifar10/results/runs/kron/'

models = [0,3,4,6,7]#[0,3,4,6,7]

runs = ['run_0','run_1','run_2']#['run_3']

test,val,lambdas = multiple_runs_concatenation(path=path,runs=runs,models=models,metric=metric)

axis= 0
nlls_means = {}
nlls_stds = {}
original_bounds_means = {}
original_bounds_stds = {}


test_means = np.mean(test,axis=0)
val_means = np.mean(val,axis=0)

test_std = np.std(test,axis=0)
val_std = np.std(val,axis=0)

threshold = 1e-7
test_means,val_means,lambdastmp = keep_last_elements(test_means,val_means,lambdas,threshold=threshold)
test,val,lambdas = keep_last_elements(test,val,lambdas,threshold=threshold)


plot_test_val(ax1 = axs[2,0],xlabel = '$\lambda$',lambdas = lambdas,
            test_means=test_means,val_means=val_means,test=test,title=None,
            legend = True,size_font_axis=size_font_axis,
                size_font_title=size_font_title,size_font_legend = size_font_legend,
            border_linewidth=border_linewidth,tick_size=tick_size,ylabel = 'KFAC')


#Subfigure 01
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/' \
       'standard_data/classification_experiments/cifar100/results/bounds/grid_search/'

models = [0,1,2,3,4,5,6,7,8]
bound_components = ['bound_0']

test,bounds,lambdas =  multiple_bounds_concatenation(path=path,bound_components=bound_components,models=models,
                                                     metric=metric,bound_type=bound_type)
axis =0

test_means = np.mean(test,axis=0)
bound_means = np.mean(bounds,axis=0)

test_stds = np.std(test,axis=0)
bound_stds = np.std(bounds,axis=0)

threshold = 1e-2
test_means,bound_means,lambdastmp = keep_last_elements(test_means,bound_means,lambdas,threshold=threshold)
test,bounds,lambdas = keep_last_elements(test,bounds,lambdas,threshold=threshold)

plot_catoni(ax = axs[0,1],xlabel = '$\lambda$',lambdas = lambdas,
            test_means=test_means,bound_means=bound_means,test=test,title='CIFAR-100',
            legend = False,size_font_axis=size_font_axis,
                size_font_title=size_font_title,size_font_legend = size_font_legend,
            border_linewidth=border_linewidth,tick_size=tick_size)

#Subfigure 11
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/' \
       'classification_experiments/cifar100/results/runs/isotropic/'


models = [0,1,2,3,4,5,6,7,8]#[0,1,2,3] or  [0,1,4]

runs = ['run_3']

test,val,lambdas = multiple_runs_concatenation(path=path,runs=runs,models=models,metric=metric)

#Preprocess
axis= 0
nlls_means = {}
nlls_stds = {}
original_bounds_means = {}
original_bounds_stds = {}

test_means = np.mean(test,axis=0)
val_means = np.mean(val,axis=0)

test_std = np.std(test,axis=0)
val_std = np.std(val,axis=0)

threshold = 1e-2
test_means,val_means,lambdastmp = keep_last_elements(test_means,val_means,lambdas,threshold=threshold)
test,val,lambdas = keep_last_elements(test,val,lambdas,threshold=threshold)


plot_test_val(ax1 = axs[1,1],xlabel = '$\lambda$',ylabel = None,lambdas = lambdas,
            test_means=test_means,val_means=val_means,test=test,title=None,
            legend = False,size_font_axis=size_font_axis,
                size_font_title=size_font_title,size_font_legend = size_font_legend,
            border_linewidth=border_linewidth,tick_size=tick_size)


#Subfigure 21
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/' \
       'classification_experiments/cifar100/results/runs/kron/'


models = [0,1,2,3,4,5,6,7,8]#[0,1,2,3] or  [0,1,4]

runs = ['run_0','run_1','run_2']#['run_3']

test,val,lambdas = multiple_runs_concatenation(path=path,runs=runs,models=models,metric=metric)

#Preprocess
axis= 0
nlls_means = {}
nlls_stds = {}
original_bounds_means = {}
original_bounds_stds = {}

test_means = np.mean(test,axis=0)
val_means = np.mean(val,axis=0)

test_std = np.std(test,axis=0)
val_std = np.std(val,axis=0)

threshold = 1e-2
test_means,val_means,lambdastmp = keep_last_elements(test_means,val_means,lambdas,threshold=threshold)
test,val,lambdas = keep_last_elements(test,val,lambdas,threshold=threshold)


plot_test_val(ax1 = axs[2,1],xlabel = '$\lambda$',ylabel = None,lambdas = lambdas,
            test_means=test_means,val_means=val_means,test=test,title=None,
            legend = False,size_font_axis=size_font_axis,
                size_font_title=size_font_title,size_font_legend = size_font_legend,
            border_linewidth=border_linewidth,tick_size=tick_size)

#Figure 02
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/' \
       'standard_data/classification_experiments/svhn/results/bounds/grid_search/'

models = [0,1,2,3,4,5,6,7,8,9]
bound_components = ['bound_0']

test,bounds,lambdas =  multiple_bounds_concatenation(path=path,bound_components=bound_components,models=models,
                                                     metric=metric,bound_type=bound_type)

#Preprocess
axis =0

test_means = np.mean(test,axis=0)
bound_means = np.mean(bounds,axis=0)

test_stds = np.std(test,axis=0)
bound_stds = np.std(bounds,axis=0)

threshold = 2e-2
test_means,bound_means,lambdastmp = keep_last_elements(test_means,bound_means,lambdas,threshold=threshold)
test,bounds,lambdas = keep_last_elements(test,bounds,lambdas,threshold=threshold)

plot_catoni(ax = axs[0,2],xlabel = '$\lambda$',lambdas = lambdas,
            test_means=test_means,bound_means=bound_means,test=test,title='SVHN',
            legend = False,size_font_axis=size_font_axis,
                size_font_title=size_font_title,size_font_legend = size_font_legend,
            border_linewidth=border_linewidth,tick_size=tick_size)

#Figure 12
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/' \
       'classification_experiments/svhn/results/runs/isotropic/'


models = [0,1,2,3,4,5]

runs = ['run_3']

test,val,lambdas = multiple_runs_concatenation(path=path,runs=runs,models=models,metric=metric)

#Preprocess
axis= 0
nlls_means = {}
nlls_stds = {}
original_bounds_means = {}
original_bounds_stds = {}

test_means = np.mean(test,axis=0)
val_means = np.mean(val,axis=0)

test_std = np.std(test,axis=0)
val_std = np.std(val,axis=0)

threshold = 2e-2
test_means,val_means,lambdastmp = keep_last_elements(test_means,val_means,lambdas,threshold=threshold)
test,val,lambdas = keep_last_elements(test,val,lambdas,threshold=threshold)

plot_test_val(ax1 = axs[1,2],xlabel = '$\lambda$',ylabel = None,lambdas = lambdas,
            test_means=test_means,val_means=val_means,test=test,title=None,
            legend = False,size_font_axis=size_font_axis,
                size_font_title=size_font_title,size_font_legend = size_font_legend,
            border_linewidth=border_linewidth,tick_size=tick_size)


#Figure 22
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/' \
       'classification_experiments/svhn/results/runs/kron/'


models = [0,1,2,3,4,5]

runs = ['run_0','run_1','run_2']

test,val,lambdas = multiple_runs_concatenation(path=path,runs=runs,models=models,metric=metric)

#Preprocess
axis= 0
nlls_means = {}
nlls_stds = {}
original_bounds_means = {}
original_bounds_stds = {}

test_means = np.mean(test,axis=0)
val_means = np.mean(val,axis=0)

test_std = np.std(test,axis=0)
val_std = np.std(val,axis=0)

threshold = 2e-2
test_means,val_means,lambdastmp = keep_last_elements(test_means,val_means,lambdas,threshold=threshold)
test,val,lambdas = keep_last_elements(test,val,lambdas,threshold=threshold)

plot_test_val(ax1 = axs[2,2],xlabel = '$\lambda$',ylabel = None,lambdas = lambdas,
            test_means=test_means,val_means=val_means,test=test,title=None,
            legend = False,size_font_axis=size_font_axis,
                size_font_title=size_font_title,size_font_legend = size_font_legend,
            border_linewidth=border_linewidth,tick_size=tick_size)


#Figure 03
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/' \
       'standard_data/classification_experiments/fashionmnist/results/bounds/grid_search/'

models = [0,1,2,3,4,5,6,7,8,9]
bound_components = ['bound_1']

test,bounds,lambdas =  multiple_bounds_concatenation(path=path,bound_components=bound_components,models=models,
                                                     metric=metric,bound_type=bound_type)

#Preprocess
axis =0

test_means = np.mean(test,axis=0)
bound_means = np.mean(bounds,axis=0)

test_stds = np.std(test,axis=0)
bound_stds = np.std(bounds,axis=0)

plot_catoni(ax = axs[0,3],xlabel = '$\lambda$',lambdas = lambdas,
            test_means=test_means,bound_means=bound_means,test=test,title='FMNIST',
            legend = False,size_font_axis=size_font_axis,
                size_font_title=size_font_title,size_font_legend = size_font_legend,
            border_linewidth=border_linewidth,tick_size=tick_size)

#Figure 13
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/' \
       'classification_experiments/fashionmnist/results/runs/isotropic/'


models = [0,1,2,3,4,5,6,7,8,9]

runs = ['run_3']

test,val,lambdas = multiple_runs_concatenation(path=path,runs=runs,models=models,metric=metric)

#Preprocess
axis= 0
nlls_means = {}
nlls_stds = {}
original_bounds_means = {}
original_bounds_stds = {}

test_means = np.mean(test,axis=0)
val_means = np.mean(val,axis=0)

test_std = np.std(test,axis=0)
val_std = np.std(val,axis=0)

plot_test_val(ax1 = axs[1,3],xlabel = '$\lambda$',ylabel = None,lambdas = lambdas,
            test_means=test_means,val_means=val_means,test=test,title=None,
            legend = False,size_font_axis=size_font_axis,
                size_font_title=size_font_title,size_font_legend = size_font_legend,
            border_linewidth=border_linewidth,tick_size=tick_size)

#Figure 23
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/' \
       'classification_experiments/fashionmnist/results/runs/kron/'

models = [0,1,2,3,4,5,6,7,8,9]

runs = ['run_0','run_1','run_2']#['run_3']

test,val,lambdas = multiple_runs_concatenation(path=path,runs=runs,models=models,metric=metric)

#Preprocess
axis= 0
nlls_means = {}
nlls_stds = {}
original_bounds_means = {}
original_bounds_stds = {}

test_means = np.mean(test,axis=0)
val_means = np.mean(val,axis=0)

test_std = np.std(test,axis=0)
val_std = np.std(val,axis=0)

plot_test_val(ax1 = axs[2,3],xlabel = '$\lambda$',ylabel = None,lambdas = lambdas,
            test_means=test_means,val_means=val_means,test=test,title=None,
            legend = False,size_font_axis=size_font_axis,
                size_font_title=size_font_title,size_font_legend = size_font_legend,
            border_linewidth=border_linewidth,tick_size=tick_size)










#Final adjustments
f.tight_layout()
#f.savefig('all_figs_exp_class_main_text.png',dpi=300,bbox_inches='tight')



end=1
