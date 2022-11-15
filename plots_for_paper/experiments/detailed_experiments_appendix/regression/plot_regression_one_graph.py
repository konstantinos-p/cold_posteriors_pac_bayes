from plots_for_paper.experiments.detailed_experiments_appendix.utils_plot_detailed import return_preprocessed,plot,\
    plot_all_in_one
import torch
import numpy as np

'''
Using this script we plot the Figure 1 of the Appendix. Specifically we plot samples from the original, approximate
and mixed bounds as well as samples from the NLL. By samples here we mean the results from different model parameters
that we have found using 10 different initializations of the same architecture.
'''



path1 = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/' \
       'standard_data/regression_experiments/abalone/results/abalone_lambda_0_to_20/'

path2 = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/' \
       'standard_data/regression_experiments/diamonds/results/diamonds_lambda_0_to_10/'

path3 = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/' \
       'standard_data/regression_experiments/kc_house/results/kc_house_lambda_0_to_10/'

paths = [path1,path2,path3]

prior_variances = torch.linspace(0.00001,0.1,20).numpy()

for path,dataset_name in zip(paths,['abalone/','diamonds/','kc_house/']):

    results = return_preprocessed(path, approximate=True)

    for result, nameofresult,caption in  zip(results,['nll', 'original_bounds', 'mixed_bounds', 'approximate_bounds'],
                                     ['NLL', '$\mathcal{B}_{\mathrm{original}}$', '$\mathcal{B}_{\mathrm{mixed}}$',
                                      '$\mathcal{B}_{\mathrm{approximate}}$']):

        dict_keys = list(result.keys())

        for temps,pos in zip(dict_keys,np.arange(len(dict_keys))):

            networks = result[temps]

            for network,network_number in zip(networks,np.arange(len(networks))):

                if network_number in [0]:
                    path_to_save = 'img/'+dataset_name+nameofresult+'_network_'+str(network_number)+'_temperature_'+str(pos)

#                    if nameofresult != 'nll':
#                        plot(xy=network, path=path_to_save, linestyle='--')

            path_to_save = 'img/' + dataset_name + nameofresult +'_temperature_' + str(pos)
            plot_all_in_one(xy=networks,path=path_to_save,linestyle='-',caption=caption,dataset=dataset_name)




