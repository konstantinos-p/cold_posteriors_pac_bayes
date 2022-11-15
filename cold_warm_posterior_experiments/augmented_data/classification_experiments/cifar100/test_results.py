import pickle
'''
Check that the results I am saving have the correct format and values.
'''

results_file = open("results/model_0/model_log.pkl", "rb")
output1 = pickle.load(results_file)

end = 1