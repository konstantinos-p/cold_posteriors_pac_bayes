This folder includes some initial tests of the Laplace redux package.

During initial testing of version 0.1a2 I noted the following bugs/restrictions:

#1 When using the last layer version you can’t sample from the prior directly.
self.n_samples doesn’t have a value before applying la.fit.

#2 Backpack rejects models that are not nn.sequential.

#3 Last layer works even in more complex models because it determines layers in a different way.

#4 The mean of the Laplace approximation is not set before applying la.fit.
