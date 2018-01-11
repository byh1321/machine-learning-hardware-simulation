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
parser.add_argument('--noiselayer', type=int, default=0.2, metavar='N',
					help='choose which noise layer to use')
parser.add_argument('--std', type=int, default=1, metavar='N',
					help='change standard diviation of noise layer')
parser.add_argument('--load', type=int, default=0, metavar='N',
					help='load trained data from test.dat file')
parser.add_argument('--pprec', type=int, default=0, metavar='N',
					help='parameter precision for layer weight')
parser.add_argument('--aprec', type=int, default=0, metavar='N',
					help='Arithmetic precision for internal arithmetic')
parser.add_argument('--iwidth', type=int, default=10, metavar='N',help='integer bitwidth for internal part')
torch.set_printoptions(precision=10)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
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

def gaussian(ins, stddev=args.std):
	global is_testing
	if is_testing:
		return ins + Variable(torch.randn(ins.size()).cuda() * stddev)
	return ins

def round_max(input):
	maximum = 2**args.iwidth-1
	minimum = -maximum-1
	input = F.relu(torch.add(input, -minimum))
	input = F.relu(torch.add(torch.neg(input), maximum-minimum))
	input = torch.add(torch.neg(input), maximum)
	return input	

def FixedPointConv(input, ins, outs, kernel_size, layer, stride = 1, padding = 0):
	outputsize = int((input.size()[2]-kernel_size+2*padding)/stride + 1)
	input.cuda()
	weight, bias = layer.parameters()
	weight = torch.round(weight / (2 ** (-args.pprec))) * (2 ** (-args.pprec))
	bias = torch.round(bias / (2 ** (-args.pprec))) * (2 ** (-args.pprec))
	#print(weight[0,0,0:5,0:5])
	#print(bias)
	#print(input[0,0,0:5,0:5])
	weight.cuda()
	bias.cuda()
	output = torch.FloatTensor(input.size()[0],weight.size()[0],outputsize,outputsize).zero_()
	output = output.cuda()
	#for i in range(0,input.size()[0]):
	for i in range(0,1): #Which image do you wanna use
		for j in range(0,weight.size()[0]): #Which filter do you wanna use
			for k in range(0,outputsize): #feature row index
				for l in range(0, outputsize): #feature column index
					a_1 = k*stride
					a_2 = a_1 + kernel_size
					b_1 = l*stride 
					b_2 = b_1 + kernel_size
					sum = 0
					for m in range(0,weight.size()[1]): # Have to repeat filter convoultion for thickness of filter times.
						tmp = torch.mul(weight[j,m,:,:], input[i,m,a_1:a_2,b_1:b_2]).cuda()
						tmp = tmp.view(-1,1)
						tmp = torch.round(tmp / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
						tmp = round_max(tmp)
						sum = torch.add(torch.sum(tmp),sum).cuda() # Accumulating 2D filter convoultion values
					output[i,j,k,l] = sum.data[0]
					flag = 0
				output[i,j,k,:] = torch.add(output[i,j,k,:].cuda(),bias.data[j]).cuda()
				tmp = round_max(tmp)
				tmp = torch.round(tmp / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
	'''print("writing output_fixed.txt")
	f = open('output_fixed.txt','w+')
	for i in output[0:10]:
		for j in i:
			print(j,file = f)'''
	return output.cuda()

def FixedPointFC(input, layer):
	weight, bias = layer.parameters()
	weight = torch.round(weight / (2 ** (-args.pprec))) * (2 ** (-args.pprec))
	bias = torch.round(bias / (2 ** (-args.pprec))) * (2 ** (-args.pprec))
	weight = round_max(weight)
	bias = round_max(bias)
	weight = torch.transpose(weight, 0, 1)
	input = torch.round(input / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
	input = round_max(input)
	output = torch.addmm(bias, input.cuda(), weight)
	output = round_max(output)
	output = torch.round(output / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
	return output.cuda()

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)

	def forward(self, x):
		x = torch.round(x / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
		conv1_input = x
		print("conv1_input size : ",x.size())
		x = FixedPointConv(x, 1, 10, 5, self.conv1)
		x = torch.round(x / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
		print("conv1_output size : ",x.size())
		conv1_output = x
		#x = self.conv1(x)
		
		x = F.max_pool2d(x,2)
		x = F.relu(x)

		x = torch.round(x / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
		print("conv2_input size : ",x.size())
		conv2_input = x

		x = FixedPointConv(x, 10, 20, 5, self.conv2)

	
		x = torch.round(x / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
		print("conv2_output size : ",x.size())
		conv2_output = x
		#x = self.conv2(x)
		
		x = self.conv2_drop(x)
		x = F.max_pool2d(x, 2)
		x = F.relu(x)

		x = x.view(-1, 320)
		
		x = torch.round(x / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
		print("fc1_input size : ",x.size())
		fc1_input = x
		x = FixedPointFC(x, self.fc1)
		
		x = torch.round(x / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
		print("fc1_output size : ",x.size())
		fc1_output = x
		#x = self.fc1(x)
		x = F.relu(x)

		x = F.dropout(x, training=self.training)
		
		x = torch.round(x / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
		print("fc2_input size : ",x.size())
		fc2_input = x
		x = FixedPointFC(x, self.fc2)
		
		x = torch.round(x / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
		print("fc2_output size : ",x.size())
		fc2_output = x
		#x = self.fc2(x)

		return (F.log_softmax(x),conv1_input,conv1_output,conv2_input,conv2_output, fc1_input, fc1_output, fc2_input, fc2_output)

	def extract_weight(self):
		f = open('conv1_param_fixed.txt','w')
		weight, bias = self.conv1.parameters()
		weight = torch.round(weight / (2 ** (-args.pprec))) * (2 ** (-args.pprec))
		bias = torch.round(bias / (2 ** (-args.pprec))) * (2 ** (-args.pprec))
		for i in range(0,weight.size()[0]):
			for j in range(0,weight.size()[1]):
				for k in range(0,weight.size()[2]):
					for l in range(0,weight.size()[3]):
						print(weight[i,j,k,l].data[0],file = f)
		print(bias, file=f)
		f.close()
		f = open('conv2_param_fixed.txt','w')
		weight, bias = self.conv2.parameters()
		weight = torch.round(weight / (2 ** (-args.pprec))) * (2 ** (-args.pprec))
		bias = torch.round(bias / (2 ** (-args.pprec))) * (2 ** (-args.pprec))
		for i in range(0,weight.size()[0]):
			for j in range(0,weight.size()[1]):
				for k in range(0,weight.size()[2]):
					for l in range(0,weight.size()[3]):
						print(weight[i,j,k,l].data[0],file = f)
		print(bias, file=f)
		f.close()
		f = open('fc1_param_fixed.txt','w')
		weight, bias = self.fc1.parameters()
		weight = torch.round(weight / (2 ** (-args.pprec))) * (2 ** (-args.pprec))
		bias = torch.round(bias / (2 ** (-args.pprec))) * (2 ** (-args.pprec))
		for i in range(0,weight.size()[0]):
			for j in range(0,weight.size()[1]):
				print(weight[i,j].data[0],file = f)
		print(bias, file=f)
		f.close()
		f = open('fc2_param_fixed.txt','w')
		weight, bias = self.fc2.parameters()
		weight = torch.round(weight / (2 ** (-args.pprec))) * (2 ** (-args.pprec))
		bias = torch.round(bias / (2 ** (-args.pprec))) * (2 ** (-args.pprec))
		for i in range(0,weight.size()[0]):
			for j in range(0,weight.size()[1]):
				print(weight[i,j].data[0],file = f)
		print(bias, file=f)
		f.close()

model = Net()
#noise = DynamicGNoise(10,10)
#print(len(noise))

if args.cuda:
	model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.data[0]))

def test():
	model.eval()
	test_loss = 0
	correct = 0
	for data, target in test_loader:
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data, volatile=True), Variable(target)
		(output,conv1_input,conv1_output,conv2_input,conv2_output,
			fc1_input, fc1_output, fc2_input, fc2_output) = model(data)
		'''f = open('conv1_input_fixed.txt','w+')
		out = conv1_input[0].view(-1,1)
		print(out.data[:],file = f)
		f.close()
		f = open('conv1_output_fixed.txt','w+')
		out = conv1_output[0].view(-1,1)
		print(out,file = f)
		f.close()
		f = open('conv2_input_fixed.txt','w+')
		out = conv2_input[0].view(-1,1)
		print(out.data[:],file = f)
		f.close()
		f = open('conv2_output_fixed.txt','w+')
		out = conv2_output[0].view(-1,1)
		print(out,file = f)
		f.close()
		f = open('fc1_input_fixed.txt','w+')
		out = fc1_input[0].view(-1,1)
		print(out.data[:],file = f)
		f.close()
		f = open('fc1_output_fixed.txt','w+')
		out = fc1_output[0].view(-1,1)
		print(out,file = f)
		f.close()
		f = open('fc2_input_fixed.txt','w+')
		out = fc2_input[0].view(-1,1)
		print(out.data[:],file = f)
		f.close()
		f = open('fc2_output_fixed.txt','w+')
		out = fc2_output[0].view(-1,1)
		print(out,file = f)		
		f.close()'''
		
		f = open('conv1_input_fixed.txt','w+')
		for j in range(0,conv1_input.size()[1]):
			for k in range(0,conv1_input.size()[2]):
				for l in range(0,conv1_input.size()[3]):
					print(conv1_input[0,j,k,l].data[0],file=f)
		f.close()
		f = open('conv1_output_fixed.txt','w+')
		for j in range(0,conv1_output.size()[1]):
			for k in range(0,conv1_output.size()[2]):
				for l in range(0,conv1_output.size()[3]):
					print(conv1_output[0,j,k,l],file=f)
		f.close()
		f = open('conv2_input_fixed.txt','w+')
		for j in range(0,conv2_input.size()[1]):
			for k in range(0,conv2_input.size()[2]):
				for l in range(0,conv2_input.size()[3]):
					print(conv2_input[0,j,k,l].data[0],file=f)
		f.close()
		f = open('conv2_output_fixed.txt','w+')
		for j in range(0,conv2_output.size()[1]):
			for k in range(0,conv2_output.size()[2]):
				for l in range(0,conv2_output.size()[3]):
					print(conv2_output[0,j,k,l],file=f)
		f.close()
		f = open('fc1_input_fixed.txt','w+')
		for i in range(0,fc1_input.size()[0]):
			for j in range(0,fc1_input.size()[1]):
				print(fc1_input[i,j].data[0],file=f)
		f.close()
		f = open('fc1_output_fixed.txt','w+')
		for i in range(0,fc1_output.size()[0]):
			for j in range(0,fc1_output.size()[1]):
				print(fc1_output[i,j].data[0],file=f)
		f.close()
		f = open('fc2_input_fixed.txt','w+')
		for i in range(0,fc2_input.size()[0]):
			for j in range(0,fc2_input.size()[1]):
				print(fc2_input[i,j].data[0],file=f)
		f.close()
		f = open('fc2_output_fixed.txt','w+')
		for i in range(0,fc2_output.size()[0]):
			for j in range(0,fc2_output.size()[1]):
				print(fc2_output[i,j].data[0],file=f)
		f.close()

		'''f = open('conv1_input_fixed.txt','w+')
		for i in conv1_input[0]:
			print(i,file = f)
		f.close()
		f = open('conv1_output_fixed.txt','w+')
		for i in conv1_output[0]:
			print(i,file = f)
		f.close()
		f = open('conv2_input_fixed.txt','w+')
		for i in conv2_input[0]:
			print(i,file = f)
		f.close()
		f = open('conv2_output_fixed.txt','w+')
		for i in conv2_output[0]:
			print(i,file = f)
		f.close()
		f = open('fc1_input_fixed.txt','w+')
		#for i in fc1_input:
		print(fc1_input[0].data,file = f)
		f.close()
		f = open('fc1_output_fixed.txt','w+')
		#for i in fc1_input:
		#	print(i,file = f)
		print(fc1_output[0].data,file = f)
		f.close()
		f = open('fc2_input_fixed.txt','w+')
		#for i in fc2_input:
		#	print(i,file = f)
		print(fc2_input[0].data,file = f)
		f.close()
		f = open('fc2_output_fixed.txt','w+')
		#for i in fc2_output:
		#	print(i,file = f)
		print(fc2_output[0].data,file = f)
		f.close()'''
		test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
		pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()

	test_loss /= len(test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))

pprec = args.pprec
for param in model.conv1.parameters():
	param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
for param in model.conv2.parameters():
	param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))

for param in model.fc1.parameters():
	param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
for param in model.fc2.parameters():
	param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))


global is_testing
is_testing = 0
if args.load == 0:
	train(epoch)
elif args.load == 1:
	model = torch.load('test1.dat')
elif args.load == 2:
	model = torch.load('test2.dat')
elif args.load == 3:
	model = torch.load('test3.dat')
elif args.load == 4:
	model = torch.load('test4.dat')
elif args.load == 5:
	model = torch.load('test5.dat')
model.extract_weight()
is_testing = 1
test()
