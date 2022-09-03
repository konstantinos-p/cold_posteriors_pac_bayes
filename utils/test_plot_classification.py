from utils.plot_utils import common_matplotlib_plot,normalize,log_scale,zero_mean
import pickle
"""
Plot the results for the classification test.
"""


results_file = open("results/classification/results.pkl", "rb")
output = pickle.load(results_file)

def composite_function(f, g):
    return lambda x : f(g(x))

common_matplotlib_plot(output,exclude=['mixed','original'],transform=zero_mean)

common_matplotlib_plot(output,exclude=['nll','ECE','zero_one'])

end=1