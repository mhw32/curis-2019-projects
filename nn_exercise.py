"""
@author Milan Mosse (milan.mosse@gmail.com)
@version May 2019

(i) Generate a dataset as follows: sample 10000 points from N(0, 1) as inputs (x).
Define outputs (y) as y = 5*x**3 + 10.
Reserve 80% of the dataset for training and save 20% for evaluation.

(ii) train a linear regression model (built in PyTorch!) using gradient descent on the training
dataset. Evaluate on the reserved 20%. How does your model perform on training and evaluation data?
(compute the mean squared error between predicted y and true y). 

(iii) Add more layers and some nonlinearities to your linear regression model (nn.TanH lets say).
Compute training and evaluation performance again. What happens?

Ask Mike later re: cuda (is_available, etc.)
"""
import torch
from torch.autograd import Variable
import numpy as np

# (i) generate data set
N = 10000
T = int(.8 * N)	# end pnt for train data

mu, sigma = 0, 1
x = np.random.normal(mu,sigma,N).astype(np.float32)
y = (5*(x**3)) + 10

# (ii) train linear regression model
# ref: https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817

#	(a) set up linear regression model, loss function
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

inDim, outDim = 1, 1
lrnRt = 0.01

model = linearRegression(inDim, outDim)
criterion = torch.nn.MSELoss()
# what does this \/ do?
optimizer = torch.optim.SGD(model.parameters(), lr=lrnRt)

# 	(b) train the model

x_train = x[:T]
y_train = y[:T]

epochs = 100
for epoch in range(epochs):
	# convert to Variable
	inputs = Variable(torch.from_numpy(x_train))
	labels = Variable(torch.from_numpy(y_train))

	# we don't want to accumulate gradient
	optimizer.zero_grad()
	print(inputs.size())

	# get ouput
	output = model(inputs)

	# get loss
	loss = criterion(outputs,labels)
	print(loss)
	# get gradients w.r.t. params
	loss.backward()

	# update params
	optimizer.step()

#	(c) evaluate
"""
x_eval = x[T:]
y_eval = y[T:]

with torch.no_grad():
	predicted = model(Variable(torch.from_numpy(y_eval))).data.numpy()

	mse = (np.square(y_eval - predicted)).mean()
	print("Mean squared error after ", epochs, " epochs: ", mse)

	plt.clf()
	plt.plot(x_eval, y_eval, 'go', label='True data', alpha=.5)
	plt.plot(x_eval, predicted, '--', label='Predictions', alpha=.5)
	plt.legend(loc = 'best')
	plt.show()
"""

# (iii) Add more layers and some nonlinearities
