from scripts.classification_datasets.mnist.mnist_torch_dataset import MNIST_split
from torch.utils.data import DataLoader

'''
In this module I test the MNIST_split dataset function.
'''

test,train,train_suffix,validation,true,true_suffix = MNIST_split(dir='dataset',mode='classification',suffix_size = 5000,validation_size=5000,seed_split=42)

print('The sizes of the sets are: \nTest: {:n}, \nTrain: {:n}, \nTrain Suffix: {:n}, \nValidation: {:n}, \nTrue: {:n}, \nTrue Suffix: {:n}'.format(len(test),len(train),len(train_suffix),len(validation),len(true),len(true_suffix)) )

#MNIST
for dataset in [test,train,train_suffix,validation,true,true_suffix]:
    dataloader = DataLoader(dataset, batch_size=40,
                                shuffle=True)
    x_mnist,y_mnist = next(iter(dataloader))
    print(x_mnist)
    print(y_mnist)


end =1