import os
import glob

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset,DataLoader
from torchvision import models, transforms
from efficientnet_pytorch import EfficientNet

def make_datapath_dic(categori):
    root_path = './imagenet_c/query/' + categori
    #root_path = './imagenet_c/search/' + categori
    class_list = os.listdir(root_path)
    datapath_dic = {}
    #print(class_list)
    for i, class_name in enumerate(class_list):
        data_list = []
        target_path = os.path.join(root_path, class_name)
        for path in glob.glob(target_path):
            data_list.append(path)
        datapath_dic[i] = data_list

    return datapath_dic

class ImageTransform():
    def __init__(self, size):
            self.data_transform  = {'train':    transforms.Compose([
                                                transforms.RandomResizedCrop(size=[size,size]),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
                                    ]),
                                    'test':     transforms.Compose([
                                                transforms.RandomResizedCrop(size=[size,size]),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
                                    ])
                                    }
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)

class MyDataset(Dataset):
    def __init__(self, datapath_dic, transform, phase):
        self.datapath_dic = datapath_dic
        self.transform = transform
        self.phase = phase

        all_datapath = []
        for data_list in self.datapath_dic.values():
            all_datapath += data_list
            
        self.all_datapath = all_datapath

    def __len__(self):
        return len(self.all_datapath)

    def __getitem__(self, idx):
        image_path = self.all_datapath[idx]
        image = self.transform(Image.open(image_path).convert('RGB'), self.phase)

        if 'airplane' in image_path:
            image_label = 0
        elif 'automobile' in image_path:
            image_label = 1
        elif 'bird' in image_path:
            image_label = 2
        elif 'cat' in image_path:
            image_label = 3
        elif 'deer' in image_path:
            image_label = 4
        elif 'dog' in image_path:
            image_label = 5
        elif 'frog' in image_path:
            image_label = 6
        elif 'horse' in image_path:
            image_label = 7
        elif 'ship' in image_path:
            image_label = 8
        elif 'truck' in image_path:
            image_label = 9
        
        return image, image_label

def image_show(train_loader,n):

  #Augmentationした画像データを読み込む
  tmp = iter(train_loader)
  images,labels = tmp.next()

  print(labels)

  #画像をtensorからnumpyに変換
  images = images.numpy()

  #n枚の画像を1枚ずつ取り出し、表示する
  for i in range(n):
    image = np.transpose(images[i],[1,2,0])
    plt.imshow(image)
    plt.show()

for i in range(10):
    classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    categori = classes[i]
    train_dic = make_datapath_dic(categori)
    #print(train_dic)
    transform = ImageTransform(224)
    test_dataset = MyDataset(datapath_dic=train_dic, transform=transform, phase='test')
    batch_size = 1

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #保存先
    save_dir = "./memory_bank(resnet)/target(3)/query"
    #save_dir = "./memory_bank(resnet)/target(3)/search"

    #読み込む画像確認
    #image_show(test_loader,1)

    #モデルの定義
    #finetuningのload
    model = models.resnet152(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,10)
    model_path = "/home/fine/ex1/FT/resnet3.pth"
    
    #model = EfficientNet.from_pretrained('efficientnet-b7', include_top = False)
    #num_ftrs = model._fc.in_features
    #model._fc = nn.Linear(num_ftrs, 10)
    #model_path = "/home/fine/ex1/FT/b73.pth"
    
    #densenet161 = models.densenet161(pretrained=True)
    #num_ftrs = densenet161.classifier.in_features
    #densenet161.classifier = nn.Linear(num_ftrs, 10)
    #print(densenet161)
    #model_path = "/home/fine/ex1/FT/densenet3.pth"
    #model = densenet161

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = nn.Sequential(*list(model.children())[:-1])
    #model.classifier = nn.Identity()
    print(model)

    model.eval()
    with torch.no_grad():
        for i, (anchor, label) in enumerate(test_loader):
            metric = model(anchor).detach().cpu().numpy()
            #imagenet,finetuningの場合
            metric = metric.reshape(metric.shape[0], metric.shape[1])
            #metric = metric.reshape(metric.shape[0], metric.shape[1], metric.shape[2])
            #特徴量を保存
            metric = metric[0]
            metric / np.linalg.norm(metric)  # Normalize
            path =  os.path.splitext(os.path.basename(train_dic[i][0]))[0]
            feature_path = Path( save_dir ) / ( path + ".npy")  # e.g., ./static/feature/xxx.npy
            print(metric.shape)
            np.save(feature_path, metric)
