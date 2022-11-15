from scripts.classification_datasets.fashionmnist.fashionmnist_torch_dataset import  FashionMNIST_split,get_dataloaders

'''
In this module I test the fashion_mnist dataset.
'''

test,train,validation = FashionMNIST_split(dir='dataset',image_transforms=False)

print('The sizes of the sets are: \nTest: {:n} \nTrain: {:n} \nValidation: {:n}'.format(len(test),len(train),
                                                                                            len(validation)) )
test_dataloader,train_dataloader,validation_dataloader = get_dataloaders(dir='dataset')


end =1