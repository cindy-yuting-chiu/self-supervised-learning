## setup
import torch
import torchvision
import torchvision.transforms as transforms
from aug_pair_generator import AugmentationPair

def get_color_distortion(s:float=1.0):
    """
    Function from the paper that create color distortion 
    s: float, the strength of color distortion 
    """
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


class ContrastLearningData:
    def __init__(self, download_folder:str) -> None:
        self.download_folder = download_folder
    def get_traindata(self):
        train_transform = transforms.Compose([
            transforms.GaussianBlur(23, sigma=(0.1, 2.0)),
            transforms.RandomResizedCrop(size=224,scale=(0.08,0.1),ratio=(0.75,1.33)),
            transforms.RandomHorizontalFlip(p=0.5),
            get_color_distortion(),
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.CIFAR10(root=self.download_folder, train=True,
                     download=True,transform=AugmentationPair(train_transform, n_views = 2))
        return trainset
        

