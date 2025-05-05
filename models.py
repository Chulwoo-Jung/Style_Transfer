# VGG19 Pre-trained model
# VGG19 convolutional layers split
# deep image representation

import torch
import torch.nn as nn
from torchvision.models import vgg19

conv = {
    'conv1_1' : 0, # style
    'conv2_1' : 5 , # style
    'conv3_1' : 10, # style
    'conv4_1' : 19, # style
    'conv5_1' : 28, # style
    'conv4_2' : 21 # content
}

class StyleTransfer(nn.Module):
    def __init__(self):
        super(StyleTransfer, self).__init__()
        #TODO: VGG19 Load
        self.vgg19_model = vgg19(pretrained=True)
        self.vgg19_features =  self.vgg19_model.features

        #TODO: VGG19 convolutional layers split
        self.style_layers = [conv['conv1_1'], conv['conv2_1'], conv['conv3_1'], conv['conv4_1'], conv['conv5_1']]
        self.content_layer = conv['conv4_2']

    
    def forward(self, x, mode:str):
        #TODO: slicing convolutional layer for each style and content layer
        features = []
        if mode == 'style':
            for i in range(len(self.vgg19_features)):
                x = self.vgg19_features[i](x)
                if i in self.style_layers:
                    features.append(x)

        elif mode == 'content':
            for i in range(len(self.vgg19_features)):
                x = self.vgg19_features[i](x)
                if i == self.content_layer:
                    features.append(x)

        return features

















































