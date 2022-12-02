
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


###################### Setup #######################


import torchvision
import torchvision.transforms as transforms





class Block(nn.Module):
  def __init__(self, in_c: int, out_c: int, s: int):
    """
    A Block Model that takes in a few arguments. This will represent 1 block in the ResNet layer
    It should have 2 convoluational layer in each block
    1. in_c will indicate the number of input features 
    2. out_c will indicate the number of desired output features
    3. s will indicate the number of stride of the first conv layer in the block, it's also an
    indicator of whether the block is performing downsampling. 
    """
    super(Block, self).__init__()
    self.conv1 = nn.Conv2d(in_channels = in_c, out_channels = out_c, kernel_size = 3, stride= s, padding = 1)
    self.conv1_bn = nn.BatchNorm2d(out_c)
    self.conv2 = nn.Conv2d(in_channels = out_c, out_channels = out_c, kernel_size = 3, stride=1, padding = 1)
    self.conv2_bn = nn.BatchNorm2d(out_c)
    # if this is the first block in the layer (2nd & 3rd), we want to resize the identity
    # when stride !=1, that means we're downsampling in the first block of the layer
    self.identity = nn.Sequential() # if not downsample layer, then do nothing
    if s!=1: 
      self.identity = nn.Sequential(
          # we use option b 1x1 conv here
          nn.Conv2d(in_channels = in_c, out_channels = out_c, kernel_size = 1 , stride = 2, padding = 0),
          nn.BatchNorm2d(out_c)
      )

  def forward(self, x):  
    out = self.conv1(x)
    out = self.conv1_bn(out)
    out = F.relu(out)
    out = self.conv2(out)
    out = self.conv2_bn(out)
    # add in the identity here (make sure the size is correct)
    out += self.identity(x)
    out = F.relu(out)
    return out 


class ResNet18(nn.Module):
    def __init__(self, in_c, resblock):
      """
      This is the model that is representing the ResNet 20 archieture
      It should have 20 layers in total.
      1. in_c will indicate the image input feature channel number
      2. resblock will indicatet the Block that we created previously
      """
      super(ResNet18, self).__init__()

      # layer 0 
      # skip maxpooling in the first conv
      self.conv1 = nn.Conv2d(in_channels = in_c, out_channels = 64, kernel_size = 3, stride=1, padding= 1)
      self.conv1_bn = nn.BatchNorm2d(64)
      # define layers 
      self.layer1= nn.Sequential(
            resblock(in_c = 64, out_c = 64, s = 1),
            resblock(in_c = 64, out_c = 64, s = 1),
        )
      #downsampling layer
      self.layer2 = nn.Sequential(
            resblock(in_c = 64, out_c = 128, s = 2),
            resblock(in_c = 128, out_c = 128, s = 1),
        )
      #downsampling layer
      self.layer3 = nn.Sequential(
            resblock(in_c = 128, out_c = 256, s = 2),
            resblock(in_c = 256, out_c = 256, s = 1),
        )
      #downsampling layer
      self.layer4 = nn.Sequential(
            resblock(in_c = 256, out_c = 512, s = 2),
            resblock(in_c = 512, out_c = 512, s = 1),
        )
      
      # The MLP for g(.) consists of Linear->ReLU->Linear
      # self.mlp = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 128))
 

    def forward(self, x):
      # first conv
      out = self.conv1(x)
      out = self.conv1_bn(out)
      out = F.relu(out)
      # first layer
      out = self.layer1(out)
      # second layer
      out = self.layer2(out)
      # third layer
      out = self.layer3(out)
      # fourth layer 
      out = self.layer4(out)

      # apply avg pooling
      out = F.avg_pool2d(out, out.size()[3])
      # out = self.mlp(out)
      #out = out.view(out.size(0), -1)
      #out = self.linear(out)
      #out = self.linear(out)
      return out

class MLP(nn.Module):
    def __init__(self, in_c):
      """
      This is the model that is representing the MLP archieture
      """
      super(MLP, self).__init__()
      
      # The MLP for g(.) consists of Linear->ReLU->Linear
      self.mlp = nn.Sequential(nn.Linear(in_c, in_c), nn.ReLU(), nn.Linear(in_c, 128))
 

    def forward(self, x):
      out = x.view(x.size(0), -1)
      out = self.mlp(out)
      return out
