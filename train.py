# import
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import numpy as np
from PIL import Image
from models import StyleTransfer
from loss import ContentLoss, StyleLoss
import os
from tqdm import tqdm



def pre_process(image:Image.Image) -> torch.Tensor:
    # resize image to 512x512
    # convert image to tensor
    # normalize image
    transform = T.Compose([
        T.Resize(size=(512,512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # lambda function x -> (x - mean) / std
    ])
    image = transform(image).unsqueeze(dim=0)
    return image

def post_process(tensor:torch.Tensor) -> Image.Image:
    # shape : (1, channel, height, width)
    image:np.ndarray = tensor.cpu().detach().numpy()

    # shape : (channel, height, width)
    image = image.squeeze(0)  # squeeze accepts both dim=0 and 0 as valid arguments

    # shape : (height, width, channel)
    image = image.transpose(1, 2, 0)

    # denormalize image : image -> image * std + mean
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])

    # clip
    image = image.clip(0, 1) * 255
    #dtype unit8
    image = image.astype(np.uint8)

    # convert image to PIL image
    image = Image.fromarray(image)

    return image
    
    

def train():
    # setting device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    ## pre - process
    content_image = Image.open('./images/content.jpg')
    content_image = pre_process(content_image).to(device)

    style_image = Image.open('./images/style.jpg')
    style_image = pre_process(style_image).to(device)

    style_image2 = Image.open('./images/style2.jpg')
    style_image2 = pre_process(style_image2).to(device)


    # load model
    model = StyleTransfer().to(device)
    # load loss
    content_loss = ContentLoss().to(device)
    style_loss = StyleLoss().to(device) 

    # setting hyperparameters
    alpha = 1
    beta = 1e6
    learning_rate = 1

    # setting output directory
    output_dir = f'./{alpha}_{beta}_{learning_rate}'
    os.makedirs(output_dir, exist_ok=True)


    # setting optimizer
    # x = torch.randn((1,3,512,512), device=device, requires_grad=True)
    x = content_image.clone().requires_grad_(True)
    optimizer = optim.LBFGS([x], lr=learning_rate)


    # closure function
    def closure():
        optimizer.zero_grad()
        
        # content representation (content_image, x)
        content_features = model(content_image ,'content')        
        gen_content_features = model(x,'content')

        # style representation (style_image, x)
        style_features = model(style_image, 'style')
        gen_style_features = model(x, 'style')

        loss_content = 0
        loss_style = 0

        # content loss
        for content_feature, gen_content_feature in zip(content_features, gen_content_features):
            loss_content += content_loss(content_feature, gen_content_feature)

        # style loss
        for style_feature, gen_style_feature in zip(style_features, gen_style_features):
            loss_style += style_loss(style_feature, gen_style_feature)

        # total loss
        total_loss = alpha * loss_content + beta * loss_style
        
        # calculate gradients
        total_loss.backward()
        
        return total_loss
   
    # train loop
    epochs = 1000
    with tqdm(total=epochs, desc='[Training]', leave=True) as pbar:
        for epoch in range(epochs):
            total_loss = optimizer.step(closure)

            pbar.update(1)
            pbar.set_postfix({'total_loss':total_loss.item(), 'content_loss':loss_content.item(), 'style_loss':loss_style.item()})

            ## image generate and save
            if epoch % 100 == 0 or epoch == epochs - 1:
                with torch.no_grad():
                    content_features = model(content_image ,'content')        
                    gen_content_features = model(x,'content')

                    # style representation (style_image, x)
                    style_features = model(style_image, 'style')
                    gen_style_features = model(x, 'style')

                    loss_content = 0
                    loss_style = 0

                    # content loss
                    for content_feature, gen_content_feature in zip(content_features, gen_content_features):
                        loss_content += content_loss(content_feature, gen_content_feature)

                    for style_feature, gen_style_feature in zip(style_features, gen_style_features):
                        loss_style += style_loss(style_feature, gen_style_feature)

                    # total loss
                    total_loss = alpha * loss_content + beta * loss_style

                    # print loss
                    print(f"epoch {epoch} : total_loss {total_loss.item()} , content_loss {loss_content.item()} , style_loss {loss_style.item()}")

                    ## post - process and save 
                    generated_image = post_process(x.detach())
                    generated_image.save(os.path.join(output_dir, f'generated_{epoch+1}.jpg'))
                    print("Image saved at:", os.path.join(output_dir, f'generated_{epoch+1}.jpg'))



if __name__ == '__main__':
    train()

