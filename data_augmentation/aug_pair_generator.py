## this class is for creating two data augmentation per image 
class AugmentationPair(object):

    def __init__(self, base_transforms, n_views:int=2):
        self.base_transforms = base_transforms
        self.n_views = n_views #each images needs 2 data augmented pics 

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]