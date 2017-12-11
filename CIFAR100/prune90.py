'''
some parts of code are extracted from "https://github.com/kuangliu/pytorch-cifar"
I modified some parts for our experiment
'''

from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import progress_bar

import os
import argparse
# import VGG16

import struct
import random

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--se', default=0, type=int, help='start epoch')
parser.add_argument('--ne', default=0, type=int, help='number of epoch')
parser.add_argument('--pr', default=0, type=int, help='pruning') # mode=1 is pruning, mode=0 is no pruning
parser.add_argument('--ldpr', default=0, type=int, help='pruning') # mode=1 load pruned trained data. mode=0 is trained, but not pruned data
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--mode', default=1, type=int, help='train or inference') #mode=1 is train, mode=0 is inference
parser.add_argument('--pprec', type=int, default=20, metavar='N',help='parameter precision for layer weight')
parser.add_argument('--aprec', type=int, default=20, metavar='N',help='Arithmetic precision for internal arithmetic')
parser.add_argument('--fixed', type=int, default=0, metavar='N',help='fixed=0 - floating point arithmetic')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy

use_cuda = torch.cuda.is_available()

transform_train = transforms.Compose([transforms.RandomCrop(32,padding=4),
									  transforms.RandomHorizontalFlip(),
									  transforms.ToTensor(),
									  transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
transform_test = transforms.Compose([transforms.ToTensor(),
									 transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

cifar_train = dset.CIFAR100("./", train=True, transform=transform_train, target_transform=None, download=True)
cifar_test = dset.CIFAR100("./", train=False, transform=transform_test, target_transform=None, download=True)

train_loader = torch.utils.data.DataLoader(cifar_train,batch_size=args.bs, shuffle=True,num_workers=8,drop_last=False)
test_loader = torch.utils.data.DataLoader(cifar_test,batch_size=10000, shuffle=False,num_workers=8,drop_last=False)

mode = args.mode

mask_conv0 = torch.cuda.FloatTensor(64,3,3,3)
mask_conv3 = torch.cuda.FloatTensor(64,64,3,3)
mask_conv7 = torch.cuda.FloatTensor(128,64,3,3)
mask_conv10 = torch.cuda.FloatTensor(128,128,3,3)
mask_conv14 = torch.cuda.FloatTensor(256,128,3,3)
mask_conv17 = torch.cuda.FloatTensor(256,256,3,3)
mask_conv20 = torch.cuda.FloatTensor(256,256,3,3)
mask_conv24 = torch.cuda.FloatTensor(512,256,3,3)
mask_conv27 = torch.cuda.FloatTensor(512,512,3,3)
mask_conv30 = torch.cuda.FloatTensor(512,512,3,3)
mask_conv34 = torch.cuda.FloatTensor(512,512,3,3)
mask_conv37 = torch.cuda.FloatTensor(512,512,3,3)
mask_conv40 = torch.cuda.FloatTensor(512,512,3,3)

mask_fc1 = torch.cuda.FloatTensor(512,512)
mask_fc4 = torch.cuda.FloatTensor(512,512)
mask_fc6 = torch.cuda.FloatTensor(100,512)

def roundmax(input):
	maximum = 2**args.pprec-1
	minimum = -maximum-1
	input = F.relu(torch.add(input, -minimum))
	input = F.relu(torch.add(torch.neg(input), maximum-minimum))
	input = torch.add(torch.neg(input), maximum)
	return input	

class CNN(nn.Module):
	def __init__(self):
		super(CNN,self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(3,64,3,padding=1,bias=False), #layer0
			nn.BatchNorm2d(64), # batch norm is added because dataset is changed
			nn.ReLU(inplace=True),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(64,64,3,padding=1, bias=False), #layer3
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
		)
		self.maxpool1 = nn.Sequential(
			nn.MaxPool2d(2,2), # 16*16* 64
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(64,128,3,padding=1, bias=False), #layer7
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
		)
		self.conv4 = nn.Sequential(
			nn.Conv2d(128,128,3,padding=1, bias=False),#layer10
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
		)
		self.maxpool2 = nn.Sequential(
			nn.MaxPool2d(2,2), # 8*8*128
		)
		self.conv5 = nn.Sequential(
			nn.Conv2d(128,256,3,padding=1, bias=False), #layer14
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.conv6 = nn.Sequential(
			nn.Conv2d(256,256,3,padding=1, bias=False), #layer17
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.conv7 = nn.Sequential(
			nn.Conv2d(256,256,3,padding=1, bias=False), #layer20
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.maxpool3 = nn.Sequential(
			nn.MaxPool2d(2,2), # 4*4*256
		)
		self.conv8 = nn.Sequential(
			nn.Conv2d(256,512,3,padding=1, bias=False), #layer24
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.conv9 = nn.Sequential(
			nn.Conv2d(512,512,3,padding=1, bias=False), #layer27
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.conv10 = nn.Sequential(
			nn.Conv2d(512,512,3,padding=1, bias=False), #layer30
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.maxpool4 = nn.Sequential(
			nn.MaxPool2d(2,2), # 2*2*512
		)
		self.conv11 = nn.Sequential(
			nn.Conv2d(512,512,3,padding=1, bias=False), #layer34
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.conv12 = nn.Sequential(
			nn.Conv2d(512,512,3,padding=1, bias=False), #layer37
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.conv13 = nn.Sequential(
			nn.Conv2d(512,512,3,padding=1, bias=False), #layer40
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.maxpool5 = nn.Sequential(
			nn.MaxPool2d(2,2) # 1*1*512
		)
		self.fc1 = nn.Sequential(
			nn.Dropout(p=0.5),
			nn.Linear(512,512, bias=False), #fc_layer1
			nn.ReLU(inplace=True),
		)
		self.fc2 = nn.Sequential(
			nn.Dropout(p=0.5),
			nn.Linear(512,512, bias=False), #fc_layer4
			nn.ReLU(inplace=True),
		)
		self.fc3 = nn.Sequential(
			nn.Linear(512,100, bias=False) #fc_layer6
		)

	def forward(self,x):
		x = roundmax(x)
		out1 = self.conv1(x) # 1250*64*32*32
		if args.fixed:
			out1 = torch.round(out1 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out1 = roundmax(out1)

		out2 = self.conv2(out1) # 1250*64*32*32
		if args.fixed:
			out2 = torch.round(out2 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out2 = roundmax(out2)

		out3 = self.maxpool1(out2)

		out4 = self.conv3(out3) # 1250*128*16*16
		if args.fixed:
			out4 = torch.round(out4 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out4 = roundmax(out4)

		out5 = self.conv4(out4) # 1250*128*16*16
		if args.fixed:
			out5 = torch.round(out5 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out5 = roundmax(out5)

		out6 = self.maxpool2(out5)

		out7 = self.conv5(out6) # 1250*256*8*8
		if args.fixed:
			out7 = torch.round(out7 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out7 = roundmax(out7)

		out8 = self.conv6(out7) # 1250*256*8*8
		if args.fixed:
			out8 = torch.round(out8 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out8 = roundmax(out8)

		out9 = self.conv7(out8) # 1250*256*8*8
		if args.fixed:
			out9 = torch.round(out9 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out9 = roundmax(out9)

		out10 = self.maxpool3(out9)

		out11 = self.conv8(out10) # 1250*512*4*4
		if args.fixed:
			out11 = torch.round(out11 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out11 = roundmax(out11)

		out12 = self.conv9(out11) # 1250*512*4*4
		if args.fixed:
			out12 = torch.round(out12 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out12 = roundmax(out12)

		out13 = self.conv10(out12) # 1250*512*4*4
		if args.fixed:
			out13 = torch.round(out13 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out13 = roundmax(out13)

		out14 = self.maxpool4(out13)

		out15 = self.conv11(out14) # 1250*512*2*2
		if args.fixed:
			out15 = torch.round(out15 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out15 = roundmax(out15)

		out16 = self.conv12(out15) # 1250*512*2*2
		if args.fixed:
			out16 = torch.round(out16 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out16 = roundmax(out16)

		out17 = self.conv13(out16) # 1250*512*2*2
		if args.fixed:
			out17 = torch.round(out17 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out17 = roundmax(out17)

		out18 = self.maxpool5(out17)

		out19 = out18.view(out18.size(0),-1)

		out20 = self.fc1(out19) # 1250*512
		if args.fixed:
			out20 = torch.round(out20 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out20 = roundmax(out20)

		out21 = self.fc2(out20) # 1250*512
		if args.fixed:
			out21 = torch.round(out21 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out21 = roundmax(out21)

		out22 = self.fc3(out21) # 1250*10
		if args.fixed:
			out22 = torch.round(out22 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out22 = roundmax(out22)

		return out22

# Model
if args.resume:
	# Load checkpoint.
	print('==> Resuming from checkpoint..')
	assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
	checkpoint = torch.load('./checkpoint/ckpt.t1')
	best_acc = 0 
	if args.ldpr:
		checkpoint = torch.load('./checkpoint/ckpt_prune90.t1')	
		net = checkpoint['net']
	else:
		net = checkpoint['net']


else:
	print('==> Building model..')
	net = CNN()

if use_cuda:
	net.cuda()
	net = torch.nn.DataParallel(net, device_ids=range(0,8))
	cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

start_epoch = args.se
num_epoch = args.ne

# Training
def train(epoch):
	print('\nEpoch: %d' % epoch)
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(train_loader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		optimizer.zero_grad()
		inputs, targets = Variable(inputs), Variable(targets)
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()

		train_loss += loss.data[0]
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum()

		progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test():
	global best_acc
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(test_loader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		inputs, targets = Variable(inputs, volatile=True), Variable(targets)
		outputs = net(inputs)
		loss = criterion(outputs, targets)

		test_loss += loss.data[0]
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum()

		progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
		f = open('result_prune90.txt','a+')
		print('{:.5f}'.format(100. * correct / len(test_loader.dataset)), end='\t', file=f)
		f.close()


	# Save checkpoint.
	acc = 100.*correct/total
	if args.mode:
		if acc > best_acc:
			print('Saving..')
			state = {
				'net': net.module if use_cuda else net,
				'acc': acc,
			}
			if not os.path.isdir('checkpoint'):
				os.mkdir('checkpoint')
			torch.save(state, './checkpoint/ckpt_prune90.t1')
			best_acc = acc
	
	return acc

# Retraining
def retrain(epoch,mask_conv0,mask_conv3,mask_conv7,mask_conv10,mask_conv14,mask_conv17,mask_conv20,mask_conv24,mask_conv27,mask_conv30,mask_conv34,mask_conv37,mask_conv40,mask_fc1,mask_fc4,mask_fc6):
	print('\nEpoch: %d' % epoch)
	global best_acc
	net.train()
	train_loss = 0
	total = 0
	correct = 0
	mask = torch.load('mask_90.dat')
	mask_conv0 = mask['mask_conv0']
	mask_conv3 = mask['mask_conv3']
	mask_conv7 = mask['mask_conv7']
	mask_conv10 = mask['mask_conv10']
	mask_conv14 = mask['mask_conv14']
	mask_conv17 = mask['mask_conv17']
	mask_conv20 = mask['mask_conv20']
	mask_conv24 = mask['mask_conv24']
	mask_conv27 = mask['mask_conv27']
	mask_conv30 = mask['mask_conv30']
	mask_conv34 = mask['mask_conv34']
	mask_conv37 = mask['mask_conv37']
	mask_conv40 = mask['mask_conv40']
	mask_fc1 = mask['mask_fc1']
	mask_fc4 = mask['mask_fc4']
	mask_fc6 = mask['mask_fc6']
	for batch_idx, (inputs, targets) in enumerate(train_loader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		optimizer.zero_grad()
		inputs, targets = Variable(inputs), Variable(targets)
		outputs = net(inputs)
		loss = criterion(outputs, targets)

		loss.backward()
		
		for child in net.children():
			for param in child.conv1[0].parameters():
				param.grad.data = torch.mul(param.grad.data, mask_conv0)
				param.data = torch.mul(param.data,mask_conv0)
		for child in net.children():
			for param in child.conv2[0].parameters():
				param.grad.data = torch.mul(param.grad.data, mask_conv3)
				param.data = torch.mul(param.data,mask_conv3)
		for child in net.children():
			for param in child.conv3[0].parameters():
				param.grad.data = torch.mul(param.grad.data, mask_conv7)
				param.data = torch.mul(param.data,mask_conv7)
		for child in net.children():
			for param in child.conv4[0].parameters():
				param.grad.data = torch.mul(param.grad.data, mask_conv10)
				param.data = torch.mul(param.data,mask_conv10)
		for child in net.children():
			for param in child.conv5[0].parameters():
				param.grad.data = torch.mul(param.grad.data, mask_conv14)
				param.data = torch.mul(param.data,mask_conv14)
		for child in net.children():
			for param in child.conv6[0].parameters():
				param.grad.data = torch.mul(param.grad.data, mask_conv17)
				param.data = torch.mul(param.data,mask_conv17)
		for child in net.children():
			for param in child.conv7[0].parameters():
				param.grad.data = torch.mul(param.grad.data, mask_conv20)
				param.data = torch.mul(param.data,mask_conv20)
		for child in net.children():
			for param in child.conv8[0].parameters():
				param.grad.data = torch.mul(param.grad.data, mask_conv24)
				param.data = torch.mul(param.data,mask_conv24)
		for child in net.children():
			for param in child.conv9[0].parameters():
				param.grad.data = torch.mul(param.grad.data, mask_conv27)
				param.data = torch.mul(param.data,mask_conv27)
		for child in net.children():
			for param in child.conv10[0].parameters():
				param.grad.data = torch.mul(param.grad.data, mask_conv30)
				param.data = torch.mul(param.data,mask_conv30)
		for child in net.children():
			for param in child.conv11[0].parameters():
				param.grad.data = torch.mul(param.grad.data, mask_conv34)
				param.data = torch.mul(param.data,mask_conv34)
		for child in net.children():
			for param in child.conv12[0].parameters():
				param.grad.data = torch.mul(param.grad.data, mask_conv37)
				param.data = torch.mul(param.data,mask_conv37)
		for child in net.children():
			for param in child.conv13[0].parameters():
				param.grad.data = torch.mul(param.grad.data, mask_conv40)
				param.data = torch.mul(param.data,mask_conv40)

		for child in net.children():
			for param in child.fc1[0].parameters():
				param.grad.data = torch.mul(param.grad.data, mask_fc1)
				param.data = torch.mul(param.data,mask_fc1)
		for child in net.children():
			for param in child.fc2[0].parameters():
				param.grad.data = torch.mul(param.grad.data, mask_fc4)
				param.data = torch.mul(param.data,mask_fc4)
		for child in net.children():
			for param in child.fc3[0].parameters():
				param.grad.data = torch.mul(param.grad.data, mask_fc6)
				param.data = torch.mul(param.data,mask_fc6)

		

		optimizer.step()

		train_loss += loss.data[0]
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum()

		progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
		acc = 100.*correct/total

prune = args.pr


if prune:
	'''print("pruning CONV1 weights")
	for child in net.children():
		for param in child.conv1[0].parameters():
			for i in range(0,64):
				for j in range(0,3):
					for k in range(0,3):
						for l in range(0,3):
							if param.data[i][j][k][l] <= 0.01662:
								if param.data[i][j][k][l] >= -0.01662:
									mask_conv0[i][j][k][l] = 0
								else:
									mask_conv0[i][j][k][l] = 1
							else:
								mask_conv0[i][j][k][l] = 1

	print("pruning CONV2 weights")
	for child in net.children():
		for param in child.conv2[0].parameters():
			for i in range(0,64):
				for j in range(0,64):
					for k in range(0,3):
						for l in range(0,3):
							if param.data[i][j][k][l] <= 0.01662:
								if param.data[i][j][k][l] >= -0.01662:
									mask_conv3[i][j][k][l] = 0
								else:
									mask_conv3[i][j][k][l] = 1
							else:
								mask_conv3[i][j][k][l] = 1

	print("pruning CONV3 weights")
	for child in net.children():
		for param in child.conv3[0].parameters():
			for i in range(0,128):
				for j in range(0,64):
					for k in range(0,3):
						for l in range(0,3):
							if param.data[i][j][k][l] <= 0.01662:
								if param.data[i][j][k][l] >= -0.01662:
									mask_conv7[i][j][k][l] = 0
								else:
									mask_conv7[i][j][k][l] = 1
							else:
								mask_conv7[i][j][k][l] = 1

	print("pruning CONV4 weights")
	for child in net.children():
		for param in child.conv4[0].parameters():
			for i in range(0,128):
				for j in range(0,128):
					for k in range(0,3):
						for l in range(0,3):
							if param.data[i][j][k][l] <= 0.01662:
								if param.data[i][j][k][l] >= -0.01662:
									mask_conv10[i][j][k][l] = 0
								else:
									mask_conv10[i][j][k][l] = 1
							else:
								mask_conv10[i][j][k][l] = 1

	print("pruning CONV5 weights")
	for child in net.children():
		for param in child.conv5[0].parameters():
			for i in range(0,256):
				for j in range(0,128):
					for k in range(0,3):
						for l in range(0,3):
							if param.data[i][j][k][l] <= 0.01662:
								if param.data[i][j][k][l] >= -0.01662:
									mask_conv14[i][j][k][l] = 0
								else:
									mask_conv14[i][j][k][l] = 1
							else:
								mask_conv14[i][j][k][l] = 1

	print("pruning CONV6 weights")
	for child in net.children():
		for param in child.conv6[0].parameters():
			for i in range(0,256):
				for j in range(0,256):
					for k in range(0,3):
						for l in range(0,3):
							if param.data[i][j][k][l] <= 0.01662:
								if param.data[i][j][k][l] >= -0.01662:
									mask_conv17[i][j][k][l] = 0
								else:
									mask_conv17[i][j][k][l] = 1
							else:
								mask_conv17[i][j][k][l] = 1

	print("pruning CONV7 weights")
	for child in net.children():
		for param in child.conv7[0].parameters():
			for i in range(0,256):
				for j in range(0,256):
					for k in range(0,3):
						for l in range(0,3):
							if param.data[i][j][k][l] <= 0.01662:
								if param.data[i][j][k][l] >= -0.01662:
									mask_conv20[i][j][k][l] = 0
								else:
									mask_conv20[i][j][k][l] = 1
							else:
								mask_conv20[i][j][k][l] = 1

	print("pruning CONV8 weights")
	for child in net.children():
		for param in child.conv8[0].parameters():
			for i in range(0,512):
				for j in range(0,256):
					for k in range(0,3):
						for l in range(0,3):
							if param.data[i][j][k][l] <= 0.01662:
								if param.data[i][j][k][l] >= -0.01662:
									mask_conv24[i][j][k][l] = 0
								else:
									mask_conv24[i][j][k][l] = 1
							else:
								mask_conv24[i][j][k][l] = 1

	print("pruning CONV9 weights")
	for child in net.children():
		for param in child.conv9[0].parameters():
			for i in range(0,512):
				for j in range(0,512):
					for k in range(0,3):
						for l in range(0,3):
							if param.data[i][j][k][l] <= 0.01662:
								if param.data[i][j][k][l] >= -0.01662:
									mask_conv27[i][j][k][l] = 0
								else:
									mask_conv27[i][j][k][l] = 1
							else:
								mask_conv27[i][j][k][l] = 1

	print("pruning CONV10 weights")
	for child in net.children():
		for param in child.conv10[0].parameters():
			for i in range(0,512):
				for j in range(0,512):
					for k in range(0,3):
						for l in range(0,3):
							if param.data[i][j][k][l] <= 0.01662:
								if param.data[i][j][k][l] >= -0.01662:
									mask_conv30[i][j][k][l] = 0
								else:
									mask_conv30[i][j][k][l] = 1
							else:
								mask_conv30[i][j][k][l] = 1

	print("pruning CONV11 weights")
	for child in net.children():
		for param in child.conv11[0].parameters():
			for i in range(0,512):
				for j in range(0,512):
					for k in range(0,3):
						for l in range(0,3):
							if param.data[i][j][k][l] <= 0.01662:
								if param.data[i][j][k][l] >= -0.01662:
									mask_conv34[i][j][k][l] = 0
								else:
									mask_conv34[i][j][k][l] = 1
							else:
								mask_conv34[i][j][k][l] = 1

	print("pruning CONV12 weights")
	for child in net.children():
		for param in child.conv12[0].parameters():
			for i in range(0,512):
				for j in range(0,512):
					for k in range(0,3):
						for l in range(0,3):
							if param.data[i][j][k][l] <= 0.01662:
								if param.data[i][j][k][l] >= -0.01662:
									mask_conv37[i][j][k][l] = 0
								else:
									mask_conv37[i][j][k][l] = 1
							else:
								mask_conv37[i][j][k][l] = 1

	print("pruning CONV13 weights")
	for child in net.children():
		for param in child.conv13[0].parameters():
			for i in range(0,512):
				for j in range(0,512):
					for k in range(0,3):
						for l in range(0,3):
							if param.data[i][j][k][l] <= 0.01662:
								if param.data[i][j][k][l] >= -0.01662:
									mask_conv40[i][j][k][l] = 0
								else:
									mask_conv40[i][j][k][l] = 1
							else:
								mask_conv40[i][j][k][l] = 1

	print("pruning FC1 weights")
	for child in net.children():
		for param in child.fc1[0].parameters():
			for i in range(0,512):
				for j in range(0,512):
					if param.data[i][j] <= 0.01662:
						if param.data[i][j] >= -0.01662:
							mask_fc1[i][j] = 0
						else:
							mask_fc1[i][j] = 1
					else:
						mask_fc1[i][j] = 1

	print("pruning FC2 weights")
	for child in net.children():
		for param in child.fc2[0].parameters():
			for i in range(0,512):
				for j in range(0,512):
					if param.data[i][j] <= 0.01662:
						if param.data[i][j] >= -0.01662:
							mask_fc4[i][j] = 0
						else:
							mask_fc4[i][j] = 1
					else:
						mask_fc4[i][j] = 1

	print("pruning FC3 weights")
	for child in net.children():
		for param in child.fc3[0].parameters():
			for i in range(0,100):
				for j in range(0,512):
					if param.data[i][j] <= 0.01662:
						if param.data[i][j] >= -0.01662:
							mask_fc6[i][j] = 0
						else:
							mask_fc6[i][j] = 1
					else:
						mask_fc6[i][j] = 1
	mask = {
		'mask_conv0': mask_conv0,
		'mask_conv3': mask_conv3,
		'mask_conv7': mask_conv7,
		'mask_conv10': mask_conv10,
		'mask_conv14': mask_conv14,
		'mask_conv17': mask_conv17,
		'mask_conv20': mask_conv20,
		'mask_conv24': mask_conv24,
		'mask_conv27': mask_conv27,
		'mask_conv30': mask_conv30,
		'mask_conv34': mask_conv34,
		'mask_conv37': mask_conv37,
		'mask_conv40': mask_conv40,
		'mask_fc1': mask_fc1,
		'mask_fc4': mask_fc4,
		'mask_fc6': mask_fc6
	}
	torch.save(mask, 'mask_90.dat')'''


	for epoch in range(0, 30):
		retrain(epoch,mask_conv0,mask_conv3,mask_conv7,mask_conv10,mask_conv14,mask_conv17,mask_conv20,mask_conv24,mask_conv27,mask_conv30,mask_conv34,mask_conv37,mask_conv40,mask_fc1,mask_fc4,mask_fc6)
		test()

# Train+inference vs. Inference
if mode == 1: # mode=1 is training & inference @ each epoch
	for epoch in range(start_epoch, start_epoch+num_epoch):
		train(epoch)
		test()
elif mode == 0: # only inference
	test()
else:
	pass

number_wv = 1

