import os
import random
import torch
import torchvision
import glob
import cv2
import PIL
import time
import copy

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt

from torchvision import utils
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from collections import Counter


#determine the seed value
seed_value= 42 
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)


# np.random.seed(seed_value)
BATCH_SIZE=32

transformations = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45, resample=PIL.Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.32261464, 0.24428101, 0.20433183], std=[0.21060237, 0.1522986, 0.13312635])
])

dataset = torchvision.datasets.ImageFolder(root='add root folder', transform=transformations)
targets = dataset.targets

indices = np.arange(len(dataset.targets))
np.random.shuffle(indices)
total = len(indices)
a = int(np.round(0.7 * total))
b = int(np.round(a + 0.1 * total))

train_idx = indices[0:a]
val_idx = indices[a:b]
test_idx = indices[b:total]
print(train_idx)
print(val_idx)
print(test_idx)

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)
test_sampler = SubsetRandomSampler(test_idx)

trainloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
    num_workers=2)
valloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, sampler=val_sampler,
    num_workers=2)
testloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, sampler=test_sampler,
    num_workers=2)

print('Total:', dict(Counter(dataset.targets)))

train_classes = [dataset.targets[i] for i in train_idx]
print('Train Set:', dict(Counter(train_classes)))

val_classes = [dataset.targets[i] for i in val_idx]
print('Validation Set:', dict(Counter(val_classes)))

test_classes = [dataset.targets[i] for i in test_idx]
print('Test Set:', dict(Counter(test_classes)))

TRAIN_COUNT = len(train_idx)
VAL_COUNT = len(val_idx)
TEST_COUNT = len(test_idx)

total_dict = dict(Counter(dataset.targets))
NORMAL_COUNT=total_dict.get(0)
OT_COUNT=total_dict.get(1)


## function to show an image
def imshow(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.32261464, 0.24428101, 0.20433183])
    std = np.array([0.21060237, 0.1522986, 0.13312635])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    
## get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
        
        torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)

def train_model(model, criterion, optimizer, scheduler=None, num_epochs=25):
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=10, verbose=True)

    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        if scheduler:
          scheduler.step()

        epoch_loss = running_loss / TRAIN_COUNT
        epoch_acc = running_corrects.double() / TRAIN_COUNT

        print('TRAIN - Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        if epoch_acc > best_acc:
          best_acc = epoch_acc

        print()

        # VALIDATE MODEL
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
          for i, (inputs, labels) in enumerate(valloader, 0):
              inputs = inputs.to(device)
              labels = labels.to(device)
              outputs = model(inputs)
              _, preds = torch.max(outputs, 1)
              val_loss = criterion(outputs, labels)

              # statistics
              val_running_loss += val_loss.item() * inputs.size(0)
              val_running_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_running_loss / VAL_COUNT
        val_epoch_acc = val_running_corrects.double() / VAL_COUNT
        print('VALIDATION - Loss: {:.4f} Acc: {:.4f}'.format(val_epoch_loss, val_epoch_acc))

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_epoch_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Accuracy: {:4f}'.format(best_acc))

    # TEST MODEL
    model.eval()
    test_acc = 0.0
    correct_count = 0
    
    predlabels=torch.zeros(0,dtype=torch.long, device='cpu')
    truelabels=torch.zeros(0,dtype=torch.long, device='cpu')

    with torch.no_grad():
      for i, (inputs, labels) in enumerate(testloader, 0):
          inputs = inputs.to(device)
          labels = labels.to(device)
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)
          corrects = (torch.max(outputs, 1)[1].view(labels.size()).data == labels.data).sum()
          correct_count += corrects.item()

          predlabels=torch.cat([predlabels,preds.view(-1).cpu()])
          truelabels=torch.cat([truelabels,labels.view(-1).cpu()])

      test_acc = correct_count / TEST_COUNT 

    print()
    print('Test Accuracy: {:4f}'.format(test_acc))

    # Confusion matrix
    conf_mat=metrics.confusion_matrix(truelabels.numpy(), predlabels.numpy())
    # print('Confusion Matrix')
    # print(conf_mat)

    # Per-class accuracy
    class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
    # print('Class Accuacy')
    # print(class_accuracy)

    # Sensitivity & specificity
    tn, fp, fn, tp = conf_mat.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    fpr, tpr, thresholds = metrics.roc_curve(truelabels.numpy(), predlabels.numpy(), pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print('Sensitivity: {:4f}'.format(sensitivity))
    print('Specificity: {:4f}'.format(specificity))
    print('AUC: {:4f}'.format(auc))

    # Plot ROC
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return model
  
  class_names = { 0: "healty", 1: "non_healthy" }
criterion = nn.CrossEntropyLoss()

NUM_EPOCHS=50
LEARNING_RATE=1e-2
MOMENTUM=0.9

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names));

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

# We can try different optimizers
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=LEARNING_RATE)
# optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
# optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=LEARNING_RATE)
# optimizer_ft = torch.optim.RMSprop(model_ft.parameters(), lr=LEARNING_RATE)

#100 epochs with patience 10
model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler=None, num_epochs=100)


def visualize_model(model, num_images=2):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 9))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 5, images_so_far)
                ax.axis('off')
                item = preds[j].item()
                ax.set_title(class_names[item])
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

visualize_model(model_ft, num_images=25)


  
  
  
