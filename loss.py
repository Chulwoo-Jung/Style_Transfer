# content loss
# vgg19 feature map -> deep image representation
# MSE

# style loss
# gram matrix
# MSE


import torch
import torch.nn as nn
from torch.nn import MSELoss


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.mse = MSELoss()
    
    def forward(self, content_features:torch.Tensor, gen_content_features:torch.Tensor):
        loss = self.mse(content_features, gen_content_features)
        return loss


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.mse = MSELoss()
    
    def gram_matrix(self, x:torch.Tensor):
        """
        x : torch.Tensor shape : (batch_size, channel, height, width)
        reshape x to (batch_size, channel, height * width)
        notation in paper -> (batch_size, N , M )
        transpose x to (batch_size, M, N)
        matrix multiplication of x and x_transposed 
        then result matrix shape is (batch_size, N, N)
        """

        b,c,h,w = x.size()
        # reshape
        features = x.view(b, c, h*w)
        features_t = features.transpose(1, 2)
        gram = torch.bmm(features, features_t)
        return gram.div(c*h*w)
    
    def forward(self, style_features:torch.Tensor, gen_style_features:torch.Tensor):
        loss = self.mse(self.gram_matrix(style_features), self.gram_matrix(gen_style_features))
        return loss






