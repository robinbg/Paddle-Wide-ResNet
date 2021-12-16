from __future__ import print_function


import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.nn.functional as F
import config as cf
import paddle.vision as paddlevision
import paddle.vision.transforms as transforms
import os
import sys
import time
import argparse
import datetime
import numpy as np

from network import *

logfile = open("train_log.txt", "a")
def out(logfile, x="\n", end="\n"):
    logfile.write(str(x) + end)
    logfile.flush()

parser = argparse.ArgumentParser(description='Paddle CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=20, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
args = parser.parse_args()


best_acc = 0
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
]) # meanstd transformation

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

if(args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = paddlevision.datasets.Cifar10(mode='train', download=True, transform=transform_train)
    testset = paddlevision.datasets.Cifar10(mode='test', download=True, transform=transform_test)
    num_classes = 10

trainloader = paddle.io.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, use_shared_memory = True)
testloader = paddle.io.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, use_shared_memory = True)

# Return network & file name
def getNetwork(args):
    if (args.net_type == 'lenet'):
        net = LeNet(num_classes)
        file_name = 'lenet'
    elif (args.net_type == 'vggnet'):
        net = VGG(args.depth, num_classes)
        file_name = 'vgg-'+str(args.depth)
    elif (args.net_type == 'resnet'):
        net = ResNet(args.depth, num_classes)
        file_name = 'resnet-'+str(args.depth)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)

    return net, file_name

# Test only option
if (args.testOnly):
    print('\n[Test Phase] : Model setup')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    net, file_name = getNetwork(args)
    checkpoint = paddle.load('./checkpoint/'+args.dataset+os.sep+file_name+'.pdparam')
    net.set_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']



    net.eval()
    net.training = False
    acc = 0
    total = 0

    for batch_idx, data in enumerate(testloader):
        x_data = data[0]
        y_data = data[1]
        predicts = net(x_data)
        y_data = paddle.reshape(y_data, shape = (-1,1))
        acc += paddle.metric.accuracy(predicts, y_data)
        total += 1
    print("| Test Result\tAcc@1: %.2f%%" %(100.*acc/total))

    sys.exit(0)

# Model
print('\n[Phase 2] : Model setup')
if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    net, file_name = getNetwork(args)
    checkpoint = paddle.load('./checkpoint/'+args.dataset+os.sep+file_name+'.pdparam')
    net.set_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']
else:
    print('| Building net type [' + args.net_type + ']...')
    net, file_name = getNetwork(args)

# Training
def train(epoch):
    net.train()
    net.training = True
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.Momentum(parameters = net.parameters(), learning_rate=cf.learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx,data in enumerate(trainloader):
        optimizer.clear_grad()
        x_data = data[0]
        y_data = data[1]
        y_data = paddle.to_tensor(y_data)
        y_data = paddle.unsqueeze(y_data, 1)
        predicts = net(x_data)
        loss = F.cross_entropy(predicts, y_data)
        loss.backward()
        acc = paddle.metric.accuracy(predicts, y_data)
        optimizer.step()
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_idx+1,
                    (len(trainset)//batch_size)+1, loss.numpy(), 100.*acc.numpy()))
        sys.stdout.flush()
    out(logfile, 'Epoch [%3d/%3d] \t\tLoss: %.4f Training Set Accuracy: %.3f%%'
                %(epoch, num_epochs, loss.numpy(), 100.*acc.numpy()), end="  ")

def test(epoch):
    global best_acc
    net.eval()
    net.training = False
    acc = 0
    total = 0
    for batch_id, data in enumerate(testloader):
        x_data = data[0]
        y_data = data[1]
        y_data = paddle.reshape(y_data, shape = (-1,1))
        predicts = net(x_data)
        acc += paddle.metric.accuracy(predicts, y_data)
        total += 1
    acc = acc*100.0/total
    out(logfile,'Testing Set: %.3f%%'%(acc))
    if acc > best_acc:
        

        print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
        state = {
                    'net':net.state_dict(),
                    'epoch':epoch
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/'+args.dataset+os.sep
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        paddle.save(state, save_point+file_name+'.pdparam')
        best_acc = acc

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))

elapsed_time = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()

    train(epoch)
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))

print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = %.2f%%' %(best_acc))
