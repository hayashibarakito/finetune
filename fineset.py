import os
import glob
from PIL import Image

from torchvision.transforms import transforms
from torch.utils.data import Dataset,DataLoader

def make_datapath_dic(phase='train'):
    #root_path = './memory_bank(finetuning)/flickr-500_LOF/'
    root_path = './target_domain(3)/'
    class_list = os.listdir(root_path)
    class_list = [class_name for class_name in class_list if not class_name.startswith('.')]
    datapath_dic = {}
    for i, class_name in enumerate(class_list):
        data_list = []
        target_path = os.path.join(root_path, class_name, '*.jpg')
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
        else :
            image_label = 9
        
        return image, image_label

#print(make_datapath_dic("train"))