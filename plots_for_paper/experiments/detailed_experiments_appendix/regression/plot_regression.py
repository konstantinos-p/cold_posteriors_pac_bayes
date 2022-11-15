from plots_for_paper.experiments.detailed_experiments_appendix.utils_plot_detailed import return_preprocessed,plot
import torch
import numpy as np

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

    for result, nameofresult in  zip(results,['nll', 'original_bounds', 'mixed_bounds', 'approximate_bounds']):

        dict_keys = list(result.keys())

        for temps,pos in zip(dict_keys,np.arange(len(dict_keys))):

            networks = result[temps]

            for network,network_number in zip(networks,np.arange(len(networks))):

                if network_number in [0,1,2,3,4]:
                    path_to_save = 'img/'+dataset_name+nameofresult+'_network_'+str(network_number)+'_temperature_'+str(pos)

                    if nameofresult == 'nll':
                        plot(xy=network, path=path_to_save, linestyle='-')
                    else:
                        plot(xy=network, path=path_to_save, linestyle='--')


