import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

import torchvision
import torchvision.datasets
from torch.utils.data import DataLoader, sampler, random_split

import numpy as np
import matplotlib.pyplot as plt

img_size = 224
batch_size = 8
lr = 0.005
epochs = 50

# define training set transform
transforms_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(img_size),
    torchvision.transforms.CenterCrop(img_size),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.RandomRotation([-20, 20]),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# define validation set transform
transforms_validation = torchvision.transforms.Compose([
    torchvision.transforms.Resize(img_size),
    torchvision.transforms.CenterCrop(img_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# split dataset to trainset and validationset
dataset = torchvision.datasets.ImageFolder('./dataset/training_images', transform=transforms_train)
trainset, validset = torch.utils.data.random_split(dataset, [2700,300])
validset.dataset.transform = transforms_validation

# load dataset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, drop_last=True, shuffle=True)
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, drop_last=True)

# define model
model = torchvision.models.resnet152(pretrained=True)
for param in model.parameters():
    param = param.requires_grad_(False)
model.fc = nn.Linear(model.fc.in_features ,len(trainset.dataset.classes))
model.cuda()

# define optimizer and scheduler
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters() , lr = lr , momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0.0001, verbose=True, T_max=20)

min_valid_loss = np.inf
trainloss = []
validloss = []
best_valid_acc = 0
best_valid_epoch = 0

# =======================================================
# start to train model
for i in range(epochs):
    train_loss = 0.0
    valid_loss = 0.0
    
    # switch to train mode 
    model.train()
    for batch_i, (images, target) in enumerate(trainloader):
        images = images.cuda()
        target = target.cuda()
        
        optimizer.zero_grad()
        output = model(images) 
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss = train_loss + ((1/(batch_i+1)) * (loss.data - train_loss))
        trainloss.append(train_loss)

    # switch to validation mode
    correct = 0.
    total = 0.
    model.eval()
    for batch_i, (images, target) in enumerate(validloader):
        images = images.cuda()
        target = target.cuda()
        
        optimizer.zero_grad()
        output = model(images) 
        loss = criterion(output, target)
        # loss.backward()
        # optimizer.step()
        
        valid_loss = valid_loss + ((1/(batch_i+1)) * (loss.data - valid_loss))
        validloss.append(valid_loss)

        pred = output.data.max(1, keepdim=True)[1]
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += images.size(0)
        
    scheduler.step()
   
    print("Epoch: {}, Batch: {}, Training Loss: {}, Vaildation Loss: {}".format(i+1, batch_i+1, train_loss, valid_loss))

    # record where the best validation accuracy located
    valid_acc = 100. * correct / total
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        best_valid_epoch = i + 1
    
    print('\nValidate Accuracy: %2d%% (%2d/%2d)\n' % (100. * correct / total, correct, total))
    
    # save model as "model.pt"
    if valid_loss < min_valid_loss:
        torch.save(model, "model.pt")
        # print("Validation Loss Change from {} ---> {}".format(min_valid_loss, valid_loss))
        min_valid_loss = valid_loss

print('='*40)
print("\nBest valid acc is {}%, best epoch in {}".format(best_valid_acc, best_valid_epoch))