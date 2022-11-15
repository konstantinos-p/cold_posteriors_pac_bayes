from utils.plot_utils import common_matplotlib_plot,normalize
import pickle
"""
Plot the results for the regression test.
"""

model = 'MNIST/results/MNIST_lambda_0_to_20/model_9/'
#model = 'MNIST/model_0/'

results_file1 = open(model+"results_7.pkl", "rb")
results_file2 = open(model+"results_8.pkl", "rb")
results_file3 = open(model+"results_9.pkl", "rb")
output1 = pickle.load(results_file1)
output2 = pickle.load(results_file2)
output3 = pickle.load(results_file3)

#metric = 'zero_one'
#metric = 'ECE'
metric = 'nll'

mse = {'NLL1':output1[metric],
       'NLL2':output2[metric],
       'NLL3':output3[metric]
           }

original = {'Original1':output1['original'],
            'Original2':output2['original'],
            'Original3':output3['original'],
           }

log_file_1 = open(model+"model_log.pkl", "rb")
log1 = pickle.load(log_file_1)

common_matplotlib_plot(mse,exclude=['approximate'],x_axis='$\lambda$',y_axis='GaussianNLL',map_test_risk=log1['test_loss_prior'][metric])#,axis_x_limit=20
#common_matplotlib_plot(original,exclude=['approximate'],x_axis='$\lambda$',y_axis='$\mathcal{B}_{\mathrm{original}}$')



end=1
