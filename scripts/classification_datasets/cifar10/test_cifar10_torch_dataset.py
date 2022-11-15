from scripts.classification_datasets.cifar10.cifar10_torch_dataset import CIFAR10_split,get_dataloaders

'''
In this module I test the CIFAR10 dataset.
'''

test,train,validation = CIFAR10_split(dir='dataset',image_transforms=True)

print('The sizes of the sets are: \nTest: {:n}, \nTrain: {:n}, , \nValidation: {:n}'.format(len(test),len(train),
                                                                                            len(validation)) )
test_dataloader,train_dataloader,validation_dataloader = get_dataloaders(dir='dataset')


end =1