from torch.utils.data import Dataset, DataLoader
from scripts.regression_datasets.regression_torch_datasets import AbaloneDataset,data_split
'''
Some testing scripts for the Abalone torch dataset object
'''

def test_without_dataloader(csv_file):
    '''
    This test function iterates over a split of the Abalone dataset without using a
    dataloader.
    '''


    abalone_dataset = AbaloneDataset(csv_file)

    for i in range(len(abalone_dataset)):
        X,y = abalone_dataset[i]

        print('Features')
        print(X)

        print('Targetss')
        print(y)

        pause =1

def test_with_dataloader(csv_file):
    '''
    This test function iterates over a split of the Abalone dataset by using a
    dataloader.
    '''

    abalone_dataset = AbaloneDataset(csv_file)

    dataloader = DataLoader(abalone_dataset, batch_size=40,
                            shuffle=True)

    for i_batch, (X,y) in enumerate(dataloader):

        print('Features')
        print(X)

        print('Targets')
        print(y)

        pause =1





#test_without_dataloader('dataset/data.csv')

#test_with_dataloader('dataset/data.csv')

test,train,train_suffix,validation,true,true_suffix = \
    data_split(AbaloneDataset('dataset/data.csv'),test_size=835,train_size=835+418,suffix_size = 0,validation_size=418,seed_split=42)

end =1