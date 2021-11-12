## Code from https://github.com/dnddnjs/pytorch-cifar10/blob/master/resnet

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import argparse


# code from https://github.com/KellerJordan/ResNet-PyTorch-CIFAR10/blob/master/model.py
class IdentityPadding(nn.Module):
	def __init__(self, in_channels, out_channels, stride):
		super(IdentityPadding, self).__init__()
		
		self.pooling = nn.MaxPool2d(1, stride=stride)
		self.add_channels = out_channels - in_channels
    
	def forward(self, x):
		out = F.pad(x, (0, 0, 0, 0, 0, self.add_channels))
		out = self.pooling(out)
		return out
	
	
class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1, down_sample=False):
		super(ResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
			                   stride=stride, padding=1, bias=False) 
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)

		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
			                   stride=1, padding=1, bias=False) 
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.stride = stride
		
		if down_sample:
			self.down_sample = IdentityPadding(in_channels, out_channels, stride)
		else:
			self.down_sample = None


	def forward(self, x):
		shortcut = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.down_sample is not None:
			shortcut = self.down_sample(x)

		out += shortcut
		out = self.relu(out)
		return out


class ResNet(nn.Module):
	def __init__(self, num_layers, block, num_classes=10):
		super(ResNet, self).__init__()
        
		self.num_layers = num_layers
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, 
							   stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.relu = nn.ReLU(inplace=True)
		
		
		self.layers_2n = self.get_layers(block, 16, 16, stride=1) # feature map size = 32x32x16
		self.layers_4n = self.get_layers(block, 16, 32, stride=2) # feature map size = 16x16x32
		self.layers_6n = self.get_layers(block, 32, 64, stride=2) # feature map size = 8x8x64

		# output layers
		self.avg_pool = nn.AvgPool2d(8, stride=1)
		self.fc_out = nn.Linear(64, num_classes)
		
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', 
					                    nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
	
	def get_layers(self, block, in_channels, out_channels, stride):
		if stride == 2:
			down_sample = True
		else:
			down_sample = False
		
		layers_list = nn.ModuleList(
			[block(in_channels, out_channels, stride, down_sample)])
			
		for _ in range(self.num_layers - 1):
			layers_list.append(block(out_channels, out_channels))

		return nn.Sequential(*layers_list)
		
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.layers_2n(x)
		x = self.layers_4n(x)
		x = self.layers_6n(x)

		x = self.avg_pool(x)
		x = x.view(x.size(0), -1)
		x = self.fc_out(x)
		return x


def resnet():
	block = ResidualBlock
	# total number of layers if 6n + 2. if n is 5 then the depth of network is 32.
	model = ResNet(5, block) 
	return model













parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--resume', default=None, help='')
parser.add_argument('--batch_size', default=128, help='')
parser.add_argument('--batch_size_test', default=100, help='')
parser.add_argument('--num_worker', default=4, help='')
parser.add_argument('--logdir', type=str, default='logs', help='')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#print('==> Preparing data..')
transforms_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transforms_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataset_train = CIFAR10(root='../data', train=True, 
						download=False, transform=transforms_train)
dataset_test = CIFAR10(root='../data', train=False, 
					   download=False, transform=transforms_test)
train_loader = DataLoader(dataset_train, batch_size=args.batch_size, 
	                      shuffle=True, num_workers=args.num_worker)
test_loader = DataLoader(dataset_test, batch_size=args.batch_size_test, 
	                     shuffle=False, num_workers=args.num_worker)

# there are 10 classes so the dataset name is cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
	       'dog', 'frog', 'horse', 'ship', 'truck')

#print('==> Making model..')

net = resnet()
net = net.to(device)
num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
#print('The number of parameters of model is', num_params)
# print(net)

if args.resume is not None:
	checkpoint = torch.load('./save_model/' + args.resume)
	net.load_state_dict(checkpoint['net'])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

decay_epoch = [32000, 48000]
step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, 
								 milestones=decay_epoch, gamma=0.1)
#writer = SummaryWriter(args.logdir)


def train(epoch, global_steps):
	net.train()

	train_loss = 0
	correct = 0
	total = 0

	for batch_idx, (inputs, targets) in enumerate(train_loader):
		global_steps += 1
		step_lr_scheduler.step()
		inputs = inputs.to(device)
		targets = targets.to(device)
		outputs = net(inputs)
		loss = criterion(outputs, targets)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()
		
	acc = 100 * correct / total
	print('train epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(
		   epoch, batch_idx, len(train_loader), train_loss/(batch_idx+1), acc))

	#writer.add_scalar('log/train error', 100 - acc, global_steps)
	return global_steps


def test(epoch, best_acc, global_steps):
	net.eval()

	test_loss = 0
	correct = 0
	total = 0

	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(test_loader):
			inputs = inputs.to(device)
			targets = targets.to(device)
			outputs = net(inputs)
			loss = criterion(outputs, targets)

			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
	
	acc = 100 * correct / total
	print('test epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(
		   epoch, batch_idx, len(test_loader), test_loss/(batch_idx+1), acc))

	#writer.add_scalar('log/test error', 100 - acc, global_steps)
    
	if acc > best_acc:
		#print('==> Saving model..')
		state = {
		    'net': net.state_dict(),
		    'acc': acc,
		    'epoch': epoch,
		}
		if not os.path.isdir('save_model'):
		    os.mkdir('save_model')
		torch.save(state, './save_model/ckpt.pth')
		best_acc = acc

	return best_acc


if __name__=='__main__':
	best_acc = 0
	epoch = 0
	global_steps = 0
	
	if args.resume is not None:
		test(epoch=0, best_acc=0)
	else:
		while True:
			epoch += 1
			global_steps = train(epoch, global_steps)
			best_acc = test(epoch, best_acc, global_steps)
			print('best test accuracy is ', best_acc)
			
			if global_steps >= 64000:
				break