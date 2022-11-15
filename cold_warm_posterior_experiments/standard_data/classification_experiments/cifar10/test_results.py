import pickle
'''
Check that the results I am saving have the correct format and values.
'''

map_results = open("results/model_3/model_log.pkl", "rb")
map = pickle.load(map_results)

#laplace_results = open("results/model_0/la_metrics.pkl", "rb")
#laplace = pickle.load(laplace_results)



end = 1