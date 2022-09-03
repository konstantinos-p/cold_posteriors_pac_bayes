from scripts.classification_datasets.cifar100.cifar100_torch_dataset import CIFAR100_split,get_dataloaders

'''
In this module I test the CIFAR100 dataset.
'''

test,train,validation = CIFAR100_split(dir='dataset',image_transforms=True)

print('The sizes of the sets are: \nTest: {:n} \nTrain: {:n} \nValidation: {:n}'.format(len(test),len(train),
                                                                                            len(validation)) )
test_dataloader,train_dataloader,validation_dataloader = get_dataloaders(dir='dataset')


end =1