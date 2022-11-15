from utils.plot_utils import common_matplotlib_plot
import pickle
"""
Plot the results for the MNIST-10 bound calculation.
"""

#Load data
results_file1 = open("MNIST/results/MNIST_lambda_0_to_1000/model_0/results_0.pkl", "rb")
results_file2 = open("MNIST/results/MNIST_lambda_0_to_1000/model_0/results_5.pkl", "rb")
results_file3 = open("MNIST/results/MNIST_lambda_0_to_1000/model_0/results_9.pkl", "rb")
output1 = pickle.load(results_file1)
output2 = pickle.load(results_file2)
output3 = pickle.load(results_file3)

#Set quantities to plot
mse = {'NLL1':output1['zero_one'],
       'NLL2':output2['zero_one'],
       'NLL3':output3['zero_one']
           }

original = {'Original1':output1['original'],
            'Original2':output2['original'],
            'Original3':output3['original'],
           }

#Load log file
#log_file_1 = open("MNIST/results/MNIST_lambda_0_to_1000/model_0/model_log.pkl", "rb")
log_file_1 = open("/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/"
                  "standard_data/regression_experiments/kc_house/results/kc_house_lambda_0_to_10/model_0/model_log.pkl", "rb")
log1 = pickle.load(log_file_1)

#Plot
common_matplotlib_plot(mse,exclude=['approximate'],x_axis='$\lambda$',y_axis='GaussianNLL',map_test_risk=log1['test_loss_prior']['zero_one'])
common_matplotlib_plot(original,exclude=['approximate'],x_axis='$\lambda$',y_axis='$\mathcal{B}_{\mathrm{original}}$')



end=1
