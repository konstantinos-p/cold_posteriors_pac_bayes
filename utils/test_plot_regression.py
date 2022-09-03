from utils.plot_utils import common_matplotlib_plot,normalize
import pickle
"""
Plot the results for the regression test.
"""


results_file = open("results/regression/results.pkl", "rb")
output = pickle.load(results_file)

common_matplotlib_plot(output,exclude=['approximate','mixed','original'])

common_matplotlib_plot(output,exclude=['mixed','original','GaussianNLL'],transform=normalize)


end=1