import torchvision.models as models
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

resnet152 = models.resnet152(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg19 = models.vgg19(pretrained=True)
densenet161 = models.densenet161(pretrained=True)
efficientnet_b7  = EfficientNet.from_pretrained('efficientnet-b7') #Pretrained_modelのインポート

#finetunig-cifar10

#num_ftrs = resnet152.fc.in_features
#resnet152.fc = nn.Linear(num_ftrs, 10)
#print(resnet152)

#num_ftrs = alexnet.classifier[6].in_features
#alexnet.classifier[6] = nn.Linear(num_ftrs, 10)
#print(alexnet)

#num_ftrs = vgg19.classifier[6].in_features
#vgg19.fc = nn.Linear(num_ftrs, 10)
#print(vgg19)

#num_ftrs = densenet161.classifier.in_features
#densenet161.classifier = nn.Linear(num_ftrs, 10)
#print(densenet161)

#num_ftrs = efficientnet_b7._fc.in_features #全結合層の名前は"_fc"となっています
#efficientnet_b7._fc = nn.Linear(num_ftrs, 10)
#print(efficientnet_b7)