#importing all the necessary models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image


#importing the baseline EfficientNet-B0 model
class EfficientNetB0Baseline(nn.Module):
    def __init__(self, weights=EfficientNet_B0_Weights.DEFAULT):
        super().__init__() #instantiating baseline model
        self.backbone = efficientnet_b0(weights=weights) #setting model with weights
        in_features = self.backbone.classifier[1].in_features  #1280 features
        self.backbone.classifier[1] = nn.Linear(in_features, 1) #manually setting model so it will give out real/fake (default is identifying features/objects)

    def forward(self, x): #when passing in data
        return self.backbone(x)  #x = (B (# of images), 3 (3 channels to set RGB), 224, 224 (sets images to 224x224 pixels))

#start of new algorithm
class EfficientNetBranch(nn.Module): #new model class
    def __init__(self, weights=EfficientNet_B0_Weights.DEFAULT, freeze=False):
        super().__init__() #instantiating new model
        self.backbone = efficientnet_b0(weights=weights) #setting weights
        self.backbone.classifier = nn.Identity()  #remove classifier --> features only
        self.out_dim = 1280 #output after classifying is 1280 numbers (default of EfficientNet-B0) --> setting that here

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False #not allowing weights to change if freeze is on

    def forward(self, x):
        return self.backbone(x)  #(B, 1280) because output is 1280 numbers


#new algorithm ctd
#start of residual cnn
class GaussianBlur(nn.Module): #getting the blurred image of the inputted image
    def __init__(self, channels=3, kernel_size=5, sigma=1.5): #self, 3 channels for RGB, kernel is a 5x5 square for each pixel, sigma is controlling bluriness
        super().__init__() 
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2 #setting distance from pixel: [-2 (far left), -1 (mid left), 0 (current pixel), 1 (mid right), 2, (far right)]
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum() #creating weights for each spot through bell-curve formula
        kernel_2d = torch.outer(g, g)  #(k, k) which is 5x5 square (multiplies by itself)
        kernel = kernel_2d.expand(channels, 1, kernel_size, kernel_size).contiguous() #3 copies of 5x5 square, 1 for each color channel (RGB)
        self.register_buffer("kernel", kernel)  #saves blur kernel to be used later 
        self.channels = channels
        self.pad = kernel_size // 2 #sets a 2 pixel border buffer around kernel

    def forward(self, x):
        x = F.pad(x, (self.pad,) * 4, mode="reflect")  #reflect avoids dark borders and keeps 224x224 shape
        return F.conv2d(x, self.kernel, groups=self.channels) #blurs for each color channel (RGB) to avoid blending/mixing


class ResBlock(nn.Module): #learning process where CNN stacks 4 times
    def __init__(self, in_ch, out_ch, stride=1): #self, input channels, output channels, kernel jumps 1 pixel each time
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False) #3x3 kernel that looks for patterns
        self.bn1 = nn.BatchNorm2d(out_ch) #helps stabilize learning
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False) #3x3 kernel that looks for patterns
        self.bn2 = nn.BatchNorm2d(out_ch) #helps stabilize learning

        self.skip = nn.Identity() #creates a copy of original image
        if stride != 1 or in_ch != out_ch: #makes output the same shape as the input, so they can compare
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x): #looks for more patterns between original input and output
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x)) #returns changes


class ResidualCNN(nn.Module): #actual residual cnn
    def __init__(self, out_dim=128, blur_kernel=5, blur_sigma=1.5):
        super().__init__()
        self.blur = GaussianBlur(3, blur_kernel, blur_sigma) #gets the blurred image

        #blur with 5x5 kernel and 1.5 sigma
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        #first layer that makes the image stay 224x224 and finds more patterns over each ResBlock call
        self.layers = nn.Sequential(
            ResBlock(32, 32),
            ResBlock(32, 64, stride=2),
            ResBlock(64, 128, stride=2),
            ResBlock(128, out_dim, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1) #averages channels into single number
        self.out_dim = out_dim #sets it as 128 number output

    def get_residual(self, x):
        return x - self.blur(x) #returns residual image (difference)

    def forward(self, x):
        residual = self.get_residual(x) #gets residual image
        feats = self.layers(self.stem(residual)) #finds features through 4 ResBlocks
        return self.pool(feats).flatten(1)  #(B, out_dim) shape output