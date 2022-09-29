import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from utils.plot_utils import multiple_bounds_concatenation,keep_last_elements,plot_catoni,multiple_runs_concatenation, \
    plot_ECE_zero_one


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
f, axs = plt.subplots(2, 4,figsize = (13, 6.5))

size_font_title = 20
size_font_legend = 16
size_font_axis = 20
tick_size = 17  # 17
border_linewidth = 1.5

metric1 = 'ECE'
metric2 = 'zero_one'
bound_type = 'original'




#Subfigure 00
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/' \
       'classification_experiments/cifar10/results/runs/isotropic/'

models = [0,3,4,6,7]#[0,3,4,6,7]

runs = ['run_3']

test,val,lambdas = multiple_runs_concatenation(path=path,runs=runs,models=models,metric=metric1)
test_means1 = np.mean(test,axis=0)
test,val,lambdas = multiple_runs_concatenation(path=path,runs=runs,models=models,metric=metric2)
test_means2 = np.mean(test,axis=0)


threshold = 1e-1
test_means1,test_means2,lambdastmp = keep_last_elements(test_means1,test_means2,lambdas,threshold=threshold)



plot_ECE_zero_one(ax1 = axs[0,0],xlabel = metric1,ylabel = '0-1 Loss',test_means1=test_means1,
                  test_means2=test_means2,title='CIFAR-10',
            legend = False,size_font_axis=size_font_axis,
                size_font_title=size_font_title,size_font_legend = size_font_legend,
            border_linewidth=border_linewidth,tick_size=tick_size,vertline=True)

#Subfigure 10
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/' \
       'classification_experiments/cifar10/results/runs/kron/'

models = [0,3,4,6,7]#[0,3,4,6,7]

runs = ['run_0','run_1','run_2']#['run_3']

test,val,lambdas = multiple_runs_concatenation(path=path,runs=runs,models=models,metric=metric1)
test_means1 = np.mean(test,axis=0)
test,val,lambdas = multiple_runs_concatenation(path=path,runs=runs,models=models,metric=metric2)
test_means2 = np.mean(test,axis=0)

threshold = 1e-7
test_means1,test_means2,lambdastmp = keep_last_elements(test_means1,test_means2,lambdas,threshold=threshold)


plot_ECE_zero_one(ax1 = axs[1,0],xlabel = metric1,ylabel = '0-1 Loss',test_means1=test_means1,
                  test_means2=test_means2,title=None,
            legend = False,size_font_axis=size_font_axis,
                size_font_title=size_font_title,size_font_legend = size_font_legend,
            border_linewidth=border_linewidth,tick_size=tick_size,vertline=True)



#Subfigure 01
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/' \
       'classification_experiments/cifar100/results/runs/isotropic/'


models = [0,1,2,3,4,5,6,7,8]#[0,1,2,3] or  [0,1,4]

runs = ['run_3']

test,val,lambdas = multiple_runs_concatenation(path=path,runs=runs,models=models,metric=metric1)
test_means1 = np.mean(test,axis=0)
test,val,lambdas = multiple_runs_concatenation(path=path,runs=runs,models=models,metric=metric2)
test_means2 = np.mean(test,axis=0)

threshold = 1e-2
test_means1,test_means2,lambdastmp = keep_last_elements(test_means1,test_means2,lambdas,threshold=threshold)


plot_ECE_zero_one(ax1 = axs[0,1],xlabel = metric1,ylabel = '0-1 Loss',test_means1=test_means1,
                  test_means2=test_means2,title='CIFAR-100',
            legend = False,size_font_axis=size_font_axis,
                size_font_title=size_font_title,size_font_legend = size_font_legend,
            border_linewidth=border_linewidth,tick_size=tick_size,vertline=True)


#Subfigure 11
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/' \
       'classification_experiments/cifar100/results/runs/kron/'


models = [0,1,3,4,5,6]#[0,1,2,3] or  [0,1,4]

runs = ['run_0','run_1','run_2']#['run_3']

test,val,lambdas = multiple_runs_concatenation(path=path,runs=runs,models=models,metric=metric1)
test_means1 = np.mean(test,axis=0)
test,val,lambdas = multiple_runs_concatenation(path=path,runs=runs,models=models,metric=metric2)
test_means2 = np.mean(test,axis=0)

threshold = 1e-2
test_means1,test_means2,lambdastmp = keep_last_elements(test_means1,test_means2,lambdas,threshold=threshold)


plot_ECE_zero_one(ax1 = axs[1,1],xlabel = metric1,ylabel = '0-1 Loss',test_means1=test_means1,
                  test_means2=test_means2,title=None,
            legend = False,size_font_axis=size_font_axis,
                size_font_title=size_font_title,size_font_legend = size_font_legend,
            border_linewidth=border_linewidth,tick_size=tick_size,vertline=True)



#Figure 02
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/' \
       'classification_experiments/svhn/results/runs/isotropic/'


models = [0,1,2,3,4,5]

runs = ['run_3']

test,val,lambdas = multiple_runs_concatenation(path=path,runs=runs,models=models,metric=metric1)
test_means1 = np.mean(test,axis=0)
test,val,lambdas = multiple_runs_concatenation(path=path,runs=runs,models=models,metric=metric2)
test_means2 = np.mean(test,axis=0)

threshold = 2e-2
test_means1,test_means2,lambdastmp = keep_last_elements(test_means1,test_means2,lambdas,threshold=threshold)

plot_ECE_zero_one(ax1 = axs[0,2],xlabel = metric1,ylabel = '0-1 Loss',test_means1=test_means1,
                  test_means2=test_means2,title='SVHN',
            legend = False,size_font_axis=size_font_axis,
                size_font_title=size_font_title,size_font_legend = size_font_legend,
            border_linewidth=border_linewidth,tick_size=tick_size,vertline=True)


#Figure 12
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/' \
       'classification_experiments/svhn/results/runs/kron/'


models = [0,1,2,3,4,5]

runs = ['run_0','run_1','run_2']

test,val,lambdas = multiple_runs_concatenation(path=path,runs=runs,models=models,metric=metric1)
test_means1 = np.mean(test,axis=0)
test,val,lambdas = multiple_runs_concatenation(path=path,runs=runs,models=models,metric=metric2)
test_means2 = np.mean(test,axis=0)

threshold = 1e-8#2e-2
test_means1,test_means2,lambdastmp = keep_last_elements(test_means1,test_means2,lambdas,threshold=threshold)

plot_ECE_zero_one(ax1 = axs[1,2],xlabel = metric1,ylabel = '0-1 Loss',test_means1=test_means1,
                  test_means2=test_means2,title=None,
            legend = False,size_font_axis=size_font_axis,
                size_font_title=size_font_title,size_font_legend = size_font_legend,
            border_linewidth=border_linewidth,tick_size=tick_size,vertline=True)


#Figure 03
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/' \
       'classification_experiments/fashionmnist/results/runs/isotropic/'


models = [0,1,2,3,4,5,6,7,8,9]

runs = ['run_3']

test,val,lambdas = multiple_runs_concatenation(path=path,runs=runs,models=models,metric=metric1)
test_means1 = np.mean(test,axis=0)
test,val,lambdas = multiple_runs_concatenation(path=path,runs=runs,models=models,metric=metric2)
test_means2 = np.mean(test,axis=0)

plot_ECE_zero_one(ax1 = axs[0,3],xlabel = metric1,ylabel = '0-1 Loss',test_means1=test_means1,
                  test_means2=test_means2,title='FMNIST',
            legend = False,size_font_axis=size_font_axis,
                size_font_title=size_font_title,size_font_legend = size_font_legend,
            border_linewidth=border_linewidth,tick_size=tick_size,vertline=True)

#Figure 13
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/' \
       'classification_experiments/fashionmnist/results/runs/kron/'

models = [0,1,2,3,4,5,6,7,8,9]

runs = ['run_0','run_1','run_2']#['run_3']

test,val,lambdas = multiple_runs_concatenation(path=path,runs=runs,models=models,metric=metric1)
test_means1 = np.mean(test,axis=0)
test,val,lambdas = multiple_runs_concatenation(path=path,runs=runs,models=models,metric=metric2)
test_means2 = np.mean(test,axis=0)

plot_ECE_zero_one(ax1 = axs[1,3],xlabel = metric1,ylabel = '0-1 Loss',test_means1=test_means1,
                  test_means2=test_means2,title=None,
            legend = False,size_font_axis=size_font_axis,
                size_font_title=size_font_title,size_font_legend = size_font_legend,
            border_linewidth=border_linewidth,tick_size=tick_size,vertline=True)










#Final adjustments
f.tight_layout()
#f.savefig('all_figs_exp_class_main_text.png',dpi=300,bbox_inches='tight')



end=1
