from utils.plot_utils import common_matplotlib_plot,normalize
import pickle
"""
Plot the results for the regression test.
"""

folder = 'kc_house/results/kc_house_lambda_0_to_10/'
#folder = 'diamonds/results/diamonds_lambda_0_to_10/'
#folder = 'diamonds/'
#folder = 'kc_house/'

model = 'model_8/'

results_file1 = open(folder+model+"results_0.pkl", "rb")
results_file2 = open(folder+model+"results_1.pkl", "rb")
results_file3 = open(folder+model+"results_9.pkl", "rb")
output1 = pickle.load(results_file1)
output2 = pickle.load(results_file2)
output3 = pickle.load(results_file3)

mse = {'NLL1':output1['GaussianNLL'],
       'NLL2':output2['GaussianNLL'],
        'NLL3':output3['GaussianNLL']
           }

original = {'Original1':output1['original'],
           'Original2':output2['original'],
            'Original3':output3['original'],
           }

log_file_1 = open(folder+model+"model_log.pkl", "rb")
log1 = pickle.load(log_file_1)

common_matplotlib_plot(mse,exclude=['approximate'],x_axis='$\lambda$',y_axis='GaussianNLL',map_test_risk=log1['test_loss_prior']['GaussianNLL'])
common_matplotlib_plot(original,exclude=['approximate'],x_axis='$\lambda$',y_axis='$\mathcal{B}_{\mathrm{original}}$')#,axis_x_limit=20



end=1
