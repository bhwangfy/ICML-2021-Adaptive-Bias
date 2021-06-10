# This is a sample Python script.
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
#import matplotlib.pyplot as plt


class Net_mnist(nn.Module):
    def __init__(self):
        super(Net_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5,bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.fc1 = nn.Linear(64 * 5 * 5, 1024, bias=False)
        self.fc2 = nn.Linear(1024, 10, bias=False)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



def Experiment_MNIST(optimizer_type, with_momentum, batch_size_input,epoch_num):
    transform = transforms.Compose(
        [transforms.ToTensor()])
    batch_size = batch_size_input
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    dataiter = iter(trainloader)
    dataiter2 = iter(testloader)
    net = Net_mnist()
    net = net.cuda()
    criterion = nn.CrossEntropyLoss()
    if  optimizer_type=='AdaGrad':
        optimizer = optim.Adagrad(net.parameters())
    elif optimizer_type=='RMSProp':
        optimizer = optim.RMSprop(net.parameters(), lr=0.001, eps=10**(-5),alpha=0.9)
    elif optimizer_type=='Adam':
        if not with_momentum:
            optimizer = optim.Adam(net.parameters(), betas=(0, 0.9), eps=10 ** (-5))
        else:
            optimizer = optim.Adam(net.parameters(), betas=(0.9, 0.9), eps=10 ** (-5))
    elif optimizer_type=='SGD':
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    # show images
    # imshow(torchvision.utils.make_grid(images))
    # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    epoch = 0
    flag=0
    running_loss = 0
    epoch_list = []
    accuracy_list = []
    accuracy_list_training = []
    training_loss = []

    margin_processing = []
    margin_storage = []
    # loop over the dataset multiple times
    while epoch < epoch_num:
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].cuda(), data[1].cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        training_loss.append(running_loss * batch_size / 60000)
        epoch += 1
        epoch_list.append(epoch)
        # adding measure
        correct = 0
        total = 0
        correct_training = 0
        total_training = 0
        margin_processing = []
        corresponding_output = 0
        with torch.no_grad():
            for data2 in testloader:
                images2, labels2 = data2[0].cuda(), data2[1].cuda()
                outputs = net(images2)
                _, predicted = torch.max(outputs.data, 1)
                total += labels2.size(0)
                correct += (predicted == labels2).sum().item()
            for data_training in trainloader:
                images_training, labels_training = data_training[0].cuda(), data_training[1].cuda()
                outputs_training = net(images_training)
                _, predicted_training = torch.max(outputs_training.data, 1)
                total_training += labels_training.size(0)
                correct_training += (predicted_training == labels_training).sum().item()
                # calculate margin
                for i in range(len(labels_training)):
                    corresponding_output = outputs_training[i][labels_training[i]].item()
                    outputs_training[i][labels_training[i]] = -9999
                    margin_processing.append(corresponding_output - torch.max(outputs_training[i]).item())
            minimum = margin_processing[0]
            for j in range(len(margin_processing)):
                if margin_processing[j] < minimum:
                    minimum = margin_processing[j]
            l2norm_layer1 = torch.norm(net.conv1.weight).item()
            l2norm_layer2 = torch.norm(net.conv2.weight).item()
            l2norm_layer3 = torch.norm(net.fc1.weight).item()
            l2norm_layer4 = torch.norm(net.fc2.weight).item()
            l2norm = l2norm_layer1 ** 2 + l2norm_layer2 ** 2 + l2norm_layer3 ** 2 + l2norm_layer4 ** 2
            margin_storage.append(minimum / l2norm**2)
        print('Current Margin:%.10f ' % (minimum / l2norm**2))
        print('Accuracy of the network on test images: %.5f %%' % (
                100 * correct / total))
        print('Accuracy of the network on training images: %.5f %%' % (
                100 * correct_training / total_training))
        accuracy_list.append(100 * correct / total)
        accuracy_list_training.append(100 * correct_training / total_training)
    print('Finished Training')
    print('Test Accuracy')
    print(accuracy_list)
    print('Training Accuracy')
    print(accuracy_list_training)
    print('Training Loss')
    print(training_loss)
    print('Normalized Margin')
    print(margin_storage)
    return accuracy_list, accuracy_list_training, training_loss, margin_storage



if __name__ == '__main__':
    batch_size=1024
    epoch=2000
    RMSPROP_test_1024, RMSPROP_training_1024, RMSPROP_loss_1024, RMSPROP_margin_1024=Experiment_MNIST('RMSProp', False, batch_size,epoch)
    AdaGrad_test_1024, AdaGrad_training_1024, AdaGrad_loss_1024, AdaGrad_margin_1024 = Experiment_MNIST('RMSProp',
                                                                                                        False,
                                                                                                        batch_size,
                                                                                                        epoch)
    Adam_test_1024, Adam_training_1024, Adam_loss_1024, Adam_margin_1024 = Experiment_MNIST('Adam',
                                                                                                        False,
                                                                                                        batch_size,
                                                                                                        epoch)
    SGDM_test_1024, SGDM_training_1024, SGDM_loss_1024, SGDM_margin_1024 = Experiment_MNIST('SGD',
                                                                                            True,
                                                                                            batch_size,
                                                                                            epoch)
    AdaMomentum_test_1024, AdaMomentum_training_1024, AdaMomentum_loss_1024, AdaMomentum_margin_1024 = Experiment_MNIST('Adam',
                                                                                            True,
                                                                                            batch_size,
                                                                                            epoch)
    A=[]
    B=[]
    for i in range(0,len(SGDM_test_1024)):
        A.append(i)
    for i in range(len(SGDM_test_1024)):
        B.append(math.log2(i+1))

    plt.figure()
    l_adam,=plt.plot(A,Adam_test_1024,linestyle='-',linewidth=1.5,marker='s',markevery=50)
    l_rmsprop,=plt.plot(A,RMSPROP_test_1024,linestyle='-',linewidth=1.5,marker='p',markevery=50)
    l_adagrad,=plt.plot(A,AdaGrad_test_1024,linestyle='-',linewidth=1.5,marker='*',markevery=50)
    l_sgdm,=plt.plot(A,SGDM_test_1024,linestyle='-',linewidth=1.5,marker='o',markevery=50)
    l_adamomentum,=plt.plot(A,AdaMomentum_test_1024,linestyle='-',linewidth=1.5,marker='+',markevery=50)

    plt.ylim((99,99.5))
    plt.xlim((0,2000))
    plt.xlabel("Epoch")#x轴上的名字
    plt.ylabel("Test Accuracy")
    plt.legend(handles=[l_adam,l_rmsprop,l_adagrad,l_sgdm,l_adamomentum],labels=['Adam (w/m)','RMSPROP','AdaGrad','SGDM','Adam'])

    plt.figure()
    l_sgdm_training,=plt.plot(B,SGDM_training_1024,linestyle='-',linewidth=1.5,marker='o',markevery=50)
    l_adamomentum_training,=plt.plot(B,AdaMomentum_training_1024,linestyle='-',linewidth=1.5,marker='+',markevery=50)
    l_adam_training,=plt.plot(B,Adam_training_1024,linestyle='-',linewidth=1.5,marker='s',markevery=50)
    l_rmsprop_training,=plt.plot(B,RMSPROP_training_1024,linestyle='-',linewidth=1.5,marker='p',markevery=50)
    l_adagrad_training,=plt.plot(B,AdaGrad_training_1024,linestyle='-',linewidth=1.5,marker='*',markevery=50)

    plt.ylim((99.8,100.05))
    plt.xlim((3,math.log2(2000)))
    plt.xlabel("Epoch (log scale)")#x轴上的名字
    plt.ylabel("Training Accuracy")
    plt.legend(handles=[l_adam_training,l_rmsprop_training,l_adagrad_training,l_sgdm_training,l_adamomentum_training],labels=['Adam (w/m)','RMSPROP','AdaGrad','SGDM','Adam'])
    # plt.legend(handles=[l_sgdm_training,l_adamomentum_training],labels=['SGDM','Adam'])
    plt.figure()
    l_adam_loss, = plt.plot(B,Adam_loss_1024,linestyle='-',linewidth=1.5,marker='s',markevery=50)
    l_rmsprop_loss, = plt.plot(B,RMSPROP_loss_1024,linestyle='-',linewidth=1.5,marker='p',markevery=50)
    l_adagrad_loss, = plt.plot(B,AdaGrad_loss_1024,linestyle='-',linewidth=1.5,marker='*',markevery=50)
    l_sgdm_loss,=plt.plot(B,SGDM_loss_1024,linestyle='-',linewidth=1.5,marker='o',markevery=50)
    l_adamomentum_loss,=plt.plot(B,AdaMomentum_loss_1024,linestyle='-',linewidth=1.5,marker='+',markevery=50)
#l_sgd_loss, = plt.plot(A,SGD_loss_1024,linestyle='-',linewidth=1.5,marker='h',markevery=50)
    plt.ylim((0,0.005))
    plt.xlim((0,math.log2(2000)))
    plt.xlabel("Epoch (log scale)")#x轴上的名字
    plt.ylabel("Training Error")
#plt.legend(handles=[l_adam_loss,l_rmsprop_loss,l_adagrad_loss,l_sgd_loss],labels=['Adam (w/m)','RMSPROP','AdaGrad','SGD'])
# plt.legend(handles=[l_adam_loss,l_rmsprop_loss,l_adagrad_loss],labels=['Adam (w/m)','RMSPROP','AdaGrad'])
    plt.legend(handles=[l_adam_loss,l_rmsprop_loss,l_adagrad_loss,l_sgdm_loss,l_adamomentum_loss],
               labels=['Adam (w/m)','RMSPROP','AdaGrad','SGDM','Adam'])
    plt.figure()

    l_adam_margin,=plt.plot(A,Adam_margin_1024,linestyle='-',marker='s',linewidth=1.5,markevery=50)
    l_rmsprop_margin,=plt.plot(A, RMSPROP_margin_1024,linestyle='-',marker='p',linewidth=1.5,markevery=50)
    l_adagrad_margin,=plt.plot(A, AdaGrad_margin_1024,linestyle='-',marker='*',linewidth=1.5,markevery=50)
    l_sgdm_margin,=plt.plot(A,SGDM_margin_1024,linestyle='-',linewidth=1.5,marker='o',markevery=50)
    l_adamomentum_margin,=plt.plot(A,AdaMomentum_margin_1024,linestyle='-',linewidth=1.5,marker='+',markevery=50)
#l_sgd_margin,=plt.plot(B, SGD_margin_1024 ,linestyle='-',linewidth=1.5,marker='h',markevery=50)
    plt.ylim((0.2*10**(-6),4*10**(-6)))
    plt.xlim((0,2000))
    plt.xlabel("Epoch")#x轴上的名字
    plt.ylabel("Margin")
#plt.legend(handles=[l_adam_margin,l_rmsprop_margin,l_adagrad_margin,l_sgd_margin],labels=['Adam (w/m)','RMSPROP','AdaGrad','SGD'])
# plt.legend(handles=[l_adam_margin,l_rmsprop_margin,l_adagrad_margin],labels=['Adam (w/m)','RMSPROP','AdaGrad'])
    plt.legend(handles=[l_adam_margin,l_rmsprop_margin,l_adagrad_margin,l_sgdm_margin,l_adamomentum_margin],
               labels=['Adam (w/m)','RMSPROP','AdaGrad','SGDM','Adam'])
    plt.show()