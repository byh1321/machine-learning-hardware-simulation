#Original Source : https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
					help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
					help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
					help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
					help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
					help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
					help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
					help='how many batches to wait before logging training status')
parser.add_argument('--targetlayer', type=int, default=0.2, metavar='N',
					help='choose which noise layer to use')
parser.add_argument('--load', type=int, default=1, metavar='N',
					help='load trained data from test.dat file')
parser.add_argument('--filename', default="output.txt", metavar='N',
					help='output filename')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
	datasets.MNIST('../data', train=True, download=True,
				   transform=transforms.Compose([
					   transforms.ToTensor(),
					   transforms.Normalize((0.1307,), (0.3081,))
				   ])),
	batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
	datasets.MNIST('../data', train=False, transform=transforms.Compose([
					   transforms.ToTensor(),
					   transforms.Normalize((0.1307,), (0.3081,))
				   ])),
	batch_size=args.test_batch_size, shuffle=True, **kwargs)

def meanstdextract(input):
	f = open(args.filename, 'a+')
#	print(input.size())
#	print(torch.mean(input))
	print("Mean : " ,torch.mean(input), "sigma : ", torch.std(input), file = f)
	f.close()

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)

	def forward(self, x):
		if args.targetlayer == 0:
			meanstdextract(x)

		x = self.conv1(x)
		if args.targetlayer == 1:
			meanstdextract(x)

		x = F.max_pool2d(x,2)
		if args.targetlayer == 2:
			meanstdextract(x)

		x = F.relu(x)
		if args.targetlayer == 3:
			meanstdextract(x)
			
		x = self.conv2(x)
		if args.targetlayer == 4:
			meanstdextract(x)

		x = self.conv2_drop(x)
		if args.targetlayer == 5:
			meanstdextract(x)
			
		x = F.max_pool2d(x,2)
		if args.targetlayer == 6:
			meanstdextract(x)

		x = F.relu(x)
		if args.targetlayer == 7:
			meanstdextract(x)

		x = x.view(-1, 320)
		if args.targetlayer == 8:
			meanstdextract(x)
			
		x = self.fc1(x)
		if args.targetlayer == 9:
			meanstdextract(x)

		x = F.relu(x)
		if args.targetlayer == 10:
			meanstdextract(x)

		x = F.dropout(x, training=self.training)
		if args.targetlayer == 11:
			meanstdextract(x)

		x = self.fc2(x)
		if args.targetlayer == 12:
			meanstdextract(x)

		return F.log_softmax(x)

model = Net()
#noise = DynamicGNoise(10,10)
#print(len(noise))

if args.cuda:
	model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def test():
	model.eval()
	test_loss = 0
	correct = 0
	for data, target in test_loader:
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data, volatile=True), Variable(target)
		output = model(data)
		test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
		pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()

	test_loss /= len(test_loader.dataset)
#	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#		test_loss, correct, len(test_loader.dataset),
#		100. * correct / len(test_loader.dataset)))
	f = open("record1.txt", 'a+')
	print('{}'.format(correct), end='\t',file = f)
	#f.write(str(correct),end='\t')
	f.close()


#for epoch in range(1, args.epochs + 1):
global is_testing
f = open('record1.txt', 'a+')
is_testing = 0
if args.load == 1:
	model = torch.load('test1.dat')
elif args.load == 2:
	model = torch.load('test2.dat')
elif args.load == 3:
	model = torch.load('test3.dat')
is_testing = 1
test()
f.close()
