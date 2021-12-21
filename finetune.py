from torch import optim
import torch.nn as nn
import torchvision.models as models
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt

from fineset import *

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.
    for batch_idx, (input, label) in enumerate(train_loader):

        input = input.to(DEVICE)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 1 == 0:
            print(f'epoch{epoch}, batch{batch_idx+1} loss: {running_loss / 15}')
            train_loss = running_loss / 1
            running_loss = 0.

    return train_loss

if __name__ == '__main__':

    train_dic = make_datapath_dic(phase='train')
    transform = ImageTransform(224)
    train_dataset = MyDataset(train_dic, transform=transform, phase='test')
    batch_size = 30

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 30

    #モデルのロード
    #model = EfficientNet.from_pretrained('efficientnet-b7') #Pretrained_modelのインポート
    model = models.resnet152(pretrained=True)
    num_ftrs = model._fc.in_features #全結合層の名前は"_fc"となっています
    model._fc = nn.Linear(num_ftrs, 10)
    model = model.to(DEVICE)

    #freeze layers except last layer
    #for param in model.parameters():
    #    param.requires_grad = False
    
    #last_layer = list(model.children())[-1]
    #print(f'except last layer: {last_layer}')
    #for param in last_layer.parameters():
    #    param.requires_grad = True
    #print(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum=0.9)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    torch.autograd.set_detect_anomaly(True)

    x_epoch_data = []
    y_train_loss_data = []

    for epoch in range(EPOCHS):

        loss_epoch = train(model, train_loader, criterion, optimizer, epoch)
        x_epoch_data.append(epoch)
        y_train_loss_data.append(loss_epoch)
        scheduler.step()

        #if epoch > (EPOCHS / 2) - 1:
            # unfreeze all layers
        #    for param in model.parameters():
        #        param.requires_grad = True


    plt.plot(x_epoch_data, y_train_loss_data, color='blue', label='train_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.title('loss')
    plt.show()

    figure = "loss" + str(y_train_loss_data[-1]) + ".png"
    plt.savefig(figure)

    model_name = 'fine' + str(y_train_loss_data[-1]) + '.pth'
    torch.save(model.state_dict(), model_name)
    print(f'Saved model as {model_name}')
