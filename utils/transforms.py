# encoding: utf-8
from torchvision import transforms as T
from utils.random_erasing import RandomErasing,Cutout

class TrainTransform(object):
    # def __init__(self):
        
    def __call__(self, x):
        x = T.Resize((256,128), interpolation=3)(x)
        x = T.RandomHorizontalFlip(0.5)(x)
        x = T.Pad(10)(x)
        x = T.RandomCrop((256,128))(x)
        
        x = T.ToTensor()(x)
        x = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
        # x = Cutout(probability = 0.5, size=64, mean=[0.0, 0.0, 0.0])(x)
        x = RandomErasing(mean=[0.485, 0.456, 0.406])(x)
        return x


class TestTransform(object):
    def __init__(self, flip=False):
        self.flip = flip

    def __call__(self, x=None):
        x = T.Resize((256, 128))(x)
        if self.flip:
            x = T.functional.hflip(x)
        x = T.ToTensor()(x)
        x = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])(x)
        return x
