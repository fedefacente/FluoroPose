#resnet 18
import torch.nn as nn
import torchvision.models as models
import torch
import numpy as np

class reg_gradcam(nn.Module):
    def __init__(self):
        super(reg_gradcam, self).__init__()

        # Load the pre-trained ResNet-18 model
        resnet = models.resnet18(weights = None, norm_layer=nn.InstanceNorm2d)

        # Modify the first layer to accept grayscale images
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(resnet.conv1.weight[:, 0:2, :, :])

        # Replace the remaining layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # self.dropout = nn.Dropout(0.2)

        # Replace the last fully connected layer
        num_features = resnet.fc.in_features
        self.fc1 = nn.Linear(num_features, 3)
        self.fc2 = nn.Linear(num_features, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = self.dropout(x)

        output1 = self.fc1(x)
        output2 = self.fc2(x)
        output2 = output2+ torch.tensor([np.float32(0), np.float32((0 + -675.0)), np.float32(0)],
                             dtype=x.dtype, device=x.device)
        return torch.cat([output1, output2], dim=1)


class reg_final(nn.Module):
    def __init__(self):
        super(reg_final, self).__init__()

        # Load the pre-trained ResNet-18 model
        resnet = models.resnet18(weights = None, norm_layer=nn.InstanceNorm2d)

        # Modify the first layer to accept grayscale images
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(resnet.conv1.weight[:, 0:2, :, :])

        # Replace the remaining layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # self.dropout = nn.Dropout(0.2)

        # Replace the last fully connected layer
        num_features = resnet.fc.in_features
        self.fc1 = nn.Linear(num_features, 3)
        self.fc2 = nn.Linear(num_features, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = self.dropout(x)
        #x = torch.sign(x) * torch.abs(x).pow(1.1)

        output1 = self.fc1(x)
        output2 = self.fc2(x)

        return x

class LatentFeaturePerturb(nn.Module):
    """
    Domain randomization applied to latent features (1D vector per sample).
    """
    def __init__(self, lam=0.5, eps=1e-5):
        super().__init__()
        self.lam = lam
        self.eps = eps

    def forward(self, f):
        """
        f: latent feature tensor of shape (B, C)
        """
        # Compute per-feature mean and std across the batch
        mu_f = f.mean(dim=0, keepdim=True)       # shape: (1, C)
        sigma_f = f.std(dim=0, unbiased=False, keepdim=True)  # shape: (1, C)

        # Sample random statistics for augmentation
        mu_a = torch.rand_like(mu_f)
        sigma_a = torch.rand_like(sigma_f)

        # Mix original and augmented statistics
        gamma_mix = self.lam * sigma_a + (1 - self.lam) * sigma_f
        beta_mix = self.lam * mu_a + (1 - self.lam) * mu_f

        # Apply normalization + perturbation
        f_norm = (f - mu_f) / (sigma_f + self.eps)
        f_mix = gamma_mix * f_norm + beta_mix

        return f_mix
class regressor(nn.Module):
    def __init__(self):
        super(regressor, self).__init__()

        # Load the pre-trained ResNet-18 model
        resnet = models.resnet18(weights = None, norm_layer=nn.InstanceNorm2d)
        #resnet = models.resnet18(weights = None, norm_layer=lambda num_features: nn.GroupNorm(8, num_features))
        #norm_layer=lambda num_features: nn.GroupNorm(8, num_features)
        # Modify the first layer to accept grayscale images
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(resnet.conv1.weight[:, 0:1, :, :])
        self.latent_perturb = LatentFeaturePerturb()

        # Replace the remaining layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        #self.dropout = nn.Dropout(0.2)

        # Replace the last fully connected layer
        num_features = resnet.fc.in_features

        self.num_features = resnet.fc.in_features
        self.fc1 = nn.Linear(num_features, 3)
        self.fc2 = nn.Linear(num_features, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = self.latent_perturb(x)

        #scale = 0.9 + 0.2 * torch.rand(x.size(0), 1, device=x.device)  # scale in [0.9, 1.1]
        #x = x * scale
        #gamma = 0.5 + 0.5 * torch.rand(1, device=x.device)
        #x = torch.sign(x) * torch.abs(x).pow(gamma)
        #x = self.dropout(x)
        #x = torch.tanh(1.5 * x) / 1.5

        output1 = self.fc1(x)
        output2 = self.fc2(x)
        return output1, output2
    def extract_features(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        return x

    
class registration(nn.Module):
    def __init__(self):
        super(registration, self).__init__()

        # Load the pre-trained ResNet-18 model
        resnet = models.resnet18(weights = None, norm_layer=nn.InstanceNorm2d)

        # Modify the first layer to accept grayscale images
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(resnet.conv1.weight[:, 0:2, :, :])
        self.latent_perturb = LatentFeaturePerturb()

        # Replace the remaining layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # self.dropout = nn.Dropout(0.2)

        # Replace the last fully connected layer
        num_features = resnet.fc.in_features
        self.fc1 = nn.Linear(num_features, 3)
        self.fc2 = nn.Linear(num_features, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = self.latent_perturb(x)
        #x = self.dropout(x)
        #scale = 0.9 + 0.2 * torch.rand(x.size(0), 1, device=x.device)  # scale in [0.9, 1.1]
        #x = x * scale
        #gamma = 0.5 + 0.5 * torch.rand(1, device=x.device)
        #x = torch.sign(x) * torch.abs(x).pow(0.8)
        #x = torch.tanh(1.5 * x) / 1.5

        output1 = self.fc1(x)
        output2 = self.fc2(x)
        return output1, output2

class registration2(nn.Module):
    def __init__(self):
        super(registration2, self).__init__()

        # Load the pre-trained ResNet-18 model
        resnet = models.resnet18(weights = None, norm_layer=nn.InstanceNorm2d)

        # Modify the first layer to accept grayscale images
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(resnet.conv1.weight[:, 0:3, :, :])

        # Replace the remaining layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # self.dropout = nn.Dropout(0.2)

        # Replace the last fully connected layer
        num_features = resnet.fc.in_features
        self.fc1 = nn.Linear(num_features, 3)
        self.fc2 = nn.Linear(num_features, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = self.dropout(x)

        output1 = self.fc1(x)
        output2 = self.fc2(x)
        return output1, output2

class regressor2fc(nn.Module):
    def __init__(self):
        super(regressor2fc, self).__init__()

        # Load the pre-trained ResNet-18 model
        resnet = models.resnet18(weights = None, norm_layer=nn.InstanceNorm2d)

        # Modify the first layer to accept grayscale images
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(resnet.conv1.weight[:, 0:1, :, :])

        # Replace the remaining layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        #self.dropout = nn.Dropout(0.2)

        # Replace the last fully connected layer
        num_features = resnet.fc.in_features
        self.fc1 = nn.Linear(num_features, 256)
        self.relu = nn.ReLU()
        self.do = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(256, 3)
        self.fc3 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = self.dropout(x)
        x = self.do(self.relu(self.fc1(x)))

        output1 = self.fc2(x)
        output2 = self.fc3(x)
        return output1, output2
    

class regressor_nopool(nn.Module):
    def __init__(self):
        super(regressor_nopool, self).__init__()

        # Load the pre-trained ResNet-18 model
        resnet = models.resnet18(weights = None, norm_layer=nn.InstanceNorm2d)

        # Modify the first layer to accept grayscale images
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(resnet.conv1.weight[:, 0:1, :, :])

        # Replace the remaining layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        #self.dropout = nn.Dropout(0.2)

        # Replace the last fully connected layer
        num_features = resnet.fc.in_features

        self.fc1 = nn.Linear(num_features, 3)
        self.fc2 = nn.Linear(num_features, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        print(x.size())
        output1 = self.fc1(x)
        output2 = self.fc2(x)
        return output1, output2
    
class ConvBlock1_enc(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.ConvBlock1_enc = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.ConvBlock1_enc(x)
        return x

class ConvBlock2_enc(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.ConvBlock2_enc = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.ConvBlock2_enc(x)
        return x

class ConvBlock3_enc(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.ConvBlock3_enc = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=1, padding=0),
            #nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.ConvBlock3_enc(x)
        return x

class FullyConnected(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.FullyConnected = nn.Sequential(nn.Linear(in_channels, out_channels),
                                            nn.ReLU(),
                                            nn.Dropout(p=0.2)
                                            )

    def forward(self, x):
        x = self.FullyConnected(x)
        return x





class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.conv_block1 = ConvBlock1_enc(in_channels=1, out_channels=16)
        self.conv_block2 = ConvBlock1_enc(in_channels=16, out_channels=32)
        self.conv_block3 = ConvBlock2_enc(in_channels=32, out_channels=64)
        self.conv_block4 = ConvBlock2_enc(in_channels=64, out_channels=128)
        self.conv_block5 = ConvBlock3_enc(in_channels=128, out_channels=128)

        self.fc1 = FullyConnected(in_channels=128, out_channels=2048)
        self.fc2 = FullyConnected(in_channels=2048, out_channels=1024)
        self.fc3 = FullyConnected(in_channels=1024, out_channels=128)

        self.fc4 = nn.Linear(128, 3)
        self.fc5 = nn.Linear(128, 3)
      

    def forward(self, x):
        x1 = self.conv_block1(x)
        x2 = self.conv_block2(x1)
        x3 = self.conv_block3(x2)
        x4 = self.conv_block4(x3)
        x5 = self.conv_block5(x4)
        '''x = self.gap(x5)  # Shape: (batch_size, 128, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # Shape: (batch_size, 128)'''
        x_fc = x5.view(x5.size(0), -1)
        x_fc1 = self.fc1(x_fc)
        x_fc1 = self.fc2(x_fc1)
        x_fc1 = self.fc3(x_fc1)

        x_1 = self.fc4(x_fc1)
        x_2 = self.fc5(x_fc1)
        return x_1, x_2


class regressor_new(nn.Module):
    def __init__(self, sid):
        super(regressor_new, self).__init__()

        # Load the pre-trained ResNet-18 model
        resnet = models.resnet18(weights=None, norm_layer=nn.InstanceNorm2d)
        # resnet = models.resnet18(weights = None, norm_layer=lambda num_features: nn.GroupNorm(8, num_features))
        # norm_layer=lambda num_features: nn.GroupNorm(8, num_features)
        # Modify the first layer to accept grayscale images
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(resnet.conv1.weight[:, 0:1, :, :])
        self.latent_perturb = LatentFeaturePerturb()

        # Replace the remaining layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # self.dropout = nn.Dropout(0.2)

        # Replace the last fully connected layer
        num_features = resnet.fc.in_features

        self.num_features = resnet.fc.in_features
        self.fc1 = nn.Linear(num_features, 3)
        self.fc2 = nn.Linear(num_features, 2)
        self.fc3 = nn.Linear(num_features, 1)
        nn.init.constant_(self.fc3.bias, sid)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.latent_perturb(x)

        # scale = 0.9 + 0.2 * torch.rand(x.size(0), 1, device=x.device)  # scale in [0.9, 1.1]
        # x = x * scale
        # gamma = 0.5 + 0.5 * torch.rand(1, device=x.device)
        # x = torch.sign(x) * torch.abs(x).pow(gamma)
        # x = self.dropout(x)
        # x = torch.tanh(1.5 * x) / 1.5

        output1 = self.fc1(x)
        output2 = self.fc2(x)
        output3 = self.fc3(x)

        return output1, output2, output3


class registration_new(nn.Module):
    def __init__(self):
        super(registration_new, self).__init__()

        # Load the pre-trained ResNet-18 model
        resnet = models.resnet18(weights=None, norm_layer=nn.InstanceNorm2d)

        # Modify the first layer to accept grayscale images
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(resnet.conv1.weight[:, 0:2, :, :])
        self.latent_perturb = LatentFeaturePerturb()

        # Replace the remaining layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # self.dropout = nn.Dropout(0.2)

        # Replace the last fully connected layer
        num_features = resnet.fc.in_features
        self.fc1 = nn.Linear(num_features, 3)
        self.fc2 = nn.Linear(num_features, 2)
        self.fc3 = nn.Linear(num_features, 1)
        #nn.init.constant_(self.fc3.bias, sid)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.latent_perturb(x)
        # x = self.dropout(x)
        # scale = 0.9 + 0.2 * torch.rand(x.size(0), 1, device=x.device)  # scale in [0.9, 1.1]
        # x = x * scale
        # gamma = 0.5 + 0.5 * torch.rand(1, device=x.device)
        # x = torch.sign(x) * torch.abs(x).pow(0.8)
        # x = torch.tanh(1.5 * x) / 1.5

        output1 = self.fc1(x)
        output2 = self.fc2(x)
        output3 = self.fc3(x)

        return output1, output2, output3


class regressor_mask(nn.Module):
    def __init__(self):
        super(regressor_mask, self).__init__()

        # Load the pre-trained ResNet-18 model
        resnet = models.resnet18(weights=None, norm_layer=nn.InstanceNorm2d)
        # resnet = models.resnet18(weights = None, norm_layer=lambda num_features: nn.GroupNorm(8, num_features))
        # norm_layer=lambda num_features: nn.GroupNorm(8, num_features)
        # Modify the first layer to accept grayscale images
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(resnet.conv1.weight[:, 0:1, :, :])
        self.latent_perturb = LatentFeaturePerturb()

        # Replace the remaining layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # self.dropout = nn.Dropout(0.2)

        # Replace the last fully connected layer
        num_features = resnet.fc.in_features

        self.num_features = resnet.fc.in_features
        self.fc1 = nn.Linear(num_features, 3)
        self.fc2 = nn.Linear(num_features, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.latent_perturb(x)

        # scale = 0.9 + 0.2 * torch.rand(x.size(0), 1, device=x.device)  # scale in [0.9, 1.1]
        # x = x * scale
        # gamma = 0.5 + 0.5 * torch.rand(1, device=x.device)
        # x = torch.sign(x) * torch.abs(x).pow(gamma)
        # x = self.dropout(x)
        # x = torch.tanh(1.5 * x) / 1.5

        output1 = self.fc1(x)
        output2 = self.fc2(x)
        return output1, output2


class registration_mask(nn.Module):
    def __init__(self):
        super(registration_mask, self).__init__()

        # Load the pre-trained ResNet-18 model
        resnet = models.resnet18(weights=None, norm_layer=nn.InstanceNorm2d)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(resnet.conv1.weight[:, 0:3, :, :])
        #self.latent_perturb = LatentFeaturePerturb()

        # Replace the remaining layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # self.dropout = nn.Dropout(0.2)

        # Replace the last fully connected layer
        num_features = resnet.fc.in_features
        self.fc1 = nn.Linear(num_features, 3)
        self.fc2 = nn.Linear(num_features, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        output1 = self.fc1(x)
        output2 = self.fc2(x)
        return output1, output2


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x_cat))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x
class regressor_attention(nn.Module):
    def __init__(self):
        super(regressor_attention, self).__init__()

        # Load the pre-trained ResNet-18 model
        resnet = models.resnet18(weights=None, norm_layer=nn.InstanceNorm2d)
        # resnet = models.resnet18(weights = None, norm_layer=lambda num_features: nn.GroupNorm(8, num_features))
        # norm_layer=lambda num_features: nn.GroupNorm(8, num_features)
        # Modify the first layer to accept grayscale images
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(resnet.conv1.weight[:, 0:1, :, :])
        self.latent_perturb = LatentFeaturePerturb()

        # Replace the remaining layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # self.dropout = nn.Dropout(0.2)

        # Replace the last fully connected layer
        num_features = resnet.fc.in_features
        self.cbam = CBAM(512)
        self.num_features = resnet.fc.in_features
        self.fc1 = nn.Linear(num_features, 3)
        self.fc2 = nn.Linear(num_features, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)


        output1 = self.fc1(x)
        output2 = self.fc2(x)
        return output1, output2

    def extract_features(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x



class registration_attention(nn.Module):
    def __init__(self):
        super(registration_attention, self).__init__()

        # Load the pre-trained ResNet-18 model
        resnet = models.resnet18(weights=None, norm_layer=nn.InstanceNorm2d)

        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(resnet.conv1.weight[:, 0:2, :, :])

        # Replace the remaining layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.cbam = CBAM(512)

        # self.dropout = nn.Dropout(0.2)

        # Replace the last fully connected layer
        num_features = resnet.fc.in_features
        self.fc1 = nn.Linear(num_features, 3)
        self.fc2 = nn.Linear(num_features, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        output1 = self.fc1(x)
        output2 = self.fc2(x)
        return output1, output2


import timm


class regressor_hrnet(nn.Module):
    def __init__(self):
        super(regressor_hrnet, self).__init__()

        # timm automatically handles the 1-channel modification if you specify in_chans
        # num_classes=0 removes the final classifier and returns the pooled feature vector
        self.hrnet = timm.create_model('hrnet_w18', pretrained=False, in_chans=1, num_classes=0)

        # HRNet-w18 pooled features output is typically 2048
        num_features = self.hrnet.num_features

        self.fc1 = nn.Linear(num_features, 3)
        self.fc2 = nn.Linear(num_features, 3)

    def forward(self, x):
        x = self.hrnet(x)

        output1 = self.fc1(x)
        output2 = self.fc2(x)
        return output1, output2



class RegistrationHRNet(nn.Module):
    def __init__(self):
        super(RegistrationHRNet, self).__init__()

        # timm automatically handles the 1-channel modification if you specify in_chans
        # num_classes=0 removes the final classifier and returns the pooled feature vector
        self.hrnet = timm.create_model('hrnet_w18', pretrained=False, in_chans=2, num_classes=0)

        # HRNet-w18 pooled features output is typically 2048
        num_features = self.hrnet.num_features

        self.fc1 = nn.Linear(num_features, 3)
        self.fc2 = nn.Linear(num_features, 3)

    def forward(self, x):
        x = self.hrnet(x)

        output1 = self.fc1(x)
        output2 = self.fc2(x)
        return output1, output2


class RegressorDecoderHRNet(nn.Module):
    def __init__(self, in_channels=1):
        super(RegressorDecoderHRNet, self).__init__()

        # 1. Load HRNet as a feature extractor
        self.encoder = timm.create_model(
            'hrnet_w18',
            pretrained=False,
            in_chans=in_channels,
            features_only=True
        )

        timm_channels = self.encoder.feature_info.channels()  # Length is 5
        num_blocks = len(timm_channels)

        # Prepend the input channel dimension for SMP compatibility
        self.encoder_channels = [in_channels] + list(timm_channels)
        self.decoder_channels = [256, 128, 64, 32, 16, 8][:num_blocks]

        # 2. Regression Branch
        highest_stage_channels = timm_channels[-1]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(highest_stage_channels, 3)
        self.fc2 = nn.Linear(highest_stage_channels, 3)

        # 3. Decoder Branch
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=self.encoder_channels,
            decoder_channels=self.decoder_channels,
            n_blocks=num_blocks,
            use_batchnorm=False,
            center=False,
            attention_type=None,
        )

        # Since Decoder Block 5 outputs full resolution (H, W),
        # we just map the final channel dimension (16) down to 1 mask channel.
        self.segmentation_head = nn.Conv2d(
            self.decoder_channels[-1],
            1,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        # 1. Extract Multi-Scale Features from HRNet
        timm_features = self.encoder(x)

        # 2. Regression Head
        deepest_features = timm_features[-1]
        pooled = self.avgpool(deepest_features)
        pooled = torch.flatten(pooled, 1)
        output1 = self.fc1(pooled)
        output2 = self.fc2(pooled)

        # 3. Decoder Head (SMP treats 'x' as the final restoration target)
        smp_compatible_features = [x] + timm_features
        decoder_output = self.decoder(*smp_compatible_features)

        # This outputs perfectly at (Batch, 1, H, W) without any blurring!
        segmentation = self.segmentation_head(decoder_output)

        return output1, output2, segmentation

import segmentation_models_pytorch as smp
import torch.nn.functional as F

class encoder_decoder(nn.Module):

    def __init__(self):
        super(encoder_decoder, self).__init__()

        # -------------------------
        # Encoder
        # -------------------------
        resnet = models.resnet18(
            weights=None,
            norm_layer=nn.InstanceNorm2d
        )

        self.conv1 = nn.Conv2d(
            1, 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.conv1.weight = nn.Parameter(
            resnet.conv1.weight[:, 0:1, :, :]
        )

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.cbam = CBAM(512)

        self.avgpool = resnet.avgpool

        # -------------------------
        # Regression heads
        # -------------------------
        num_features = resnet.fc.in_features

        self.fc1 = nn.Linear(num_features, 3)
        self.fc2 = nn.Linear(num_features, 3)

        # -------------------------
        # Decoder
        # -------------------------
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=[64, 64, 128, 256, 512],
            decoder_channels=[256, 128, 64, 32, 32],
            n_blocks=5,
            use_batchnorm=False,
            center=False,
            attention_type=None,
        )

        self.segmentation_head = nn.Conv2d(
            32,
            1,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        # -------------------------
        # Encoder
        # -------------------------
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)

        x1 = self.maxpool(x0)

        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        # x5 = self.cbam(x5)

        # -------------------------
        # Regression branch
        # -------------------------
        pooled = self.avgpool(x5)
        pooled = pooled.view(pooled.size(0), -1)

        output1 = self.fc1(pooled)
        output2 = self.fc2(pooled)

        # -------------------------
        # Decoder branch
        # -------------------------
        decoder_output = self.decoder(
            x0, x2, x3, x4, x5
        )

        segmentation = self.segmentation_head(
            decoder_output
        )

        input_size = x.shape[-2:]

        segmentation = F.interpolate(
            segmentation,
            size=input_size,
            mode='bilinear',
            align_corners=False
        )

        return output1, output2, segmentation


import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
def convert_bn_to_in(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            new = nn.InstanceNorm2d(
                child.num_features,
                affine=True,          # keep learnable scale/shift, like your head
                track_running_stats=False,  # per-sample stats at train AND eval
            )
            setattr(module, name, new)
        else:
            convert_bn_to_in(child)
    return module

class RegressorDecoderHRNetNew(nn.Module):
    def __init__(self, in_channels=1):
        super(RegressorDecoderHRNetNew, self).__init__()

        # ---------------------------------------------------------
        # 1. HRNet Backbone (Feature Extractor)
        # ---------------------------------------------------------
        self.encoder = timm.create_model(
            'hrnet_w18',
            pretrained=False,  # IMPORTANT: Set to True on login node to cache weights!
            in_chans=in_channels,
            features_only=True  # Keeps spatial dimensions intact
        )
        convert_bn_to_in(self.encoder)
        # Returns a list of channels for the 4 parallel stages (e.g., [18, 36, 72, 144])
        timm_channels = self.encoder.feature_info.channels()

        # ---------------------------------------------------------
        # 2. Multi-Scale Fusion Logic
        # ---------------------------------------------------------
        self.dec_channels = 64

        # Project all parallel stages to a unified channel size (64)
        self.projections = nn.ModuleList([
            nn.Conv2d(ch, self.dec_channels, kernel_size=1) for ch in timm_channels
        ])

        # Total channels after concatenating the 4 projected branches
        fused_channels = self.dec_channels * len(timm_channels)  # 64 * 4 = 256

        # ---------------------------------------------------------
        # 3. Segmentation Head
        # ---------------------------------------------------------
        # ---------------------------------------------------------
        self.segmentation_head = nn.Sequential(
            # 1. Process the Stride 2 fused features
            nn.Conv2d(fused_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),

            # 2. Refine at Stride 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),

            # 3. SINGLE Learnable Upsample: Stride 2 -> Stride 1 (Original Size)
            nn.Conv2d(64, 64 * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),  # → Outputs (B, 64, H, W)
            nn.ReLU(inplace=True),

            # 4. Final 1x1 Conv to get the single channel bone mask
            nn.Conv2d(64, 1, kernel_size=1),
        )
        # ---------------------------------------------------------
        # 4. Regression Head
        # ---------------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(fused_channels, 3)
        self.fc2 = nn.Linear(fused_channels, 3)

    def forward(self, x):
        # Step 1: Extract Multi-Scale Features from Backbone
        timm_features = self.encoder(x)

        # Step 2: Multi-Scale Fusion (Gather all resolutions to Stride 4)
        target_size = timm_features[0].shape[2:]  # Highest resolution (Stride 4)

        projected_features = []
        for i, feat in enumerate(timm_features):
            proj = self.projections[i](feat)
            # Upsample lower-res branches to match the Stride 4 resolution
            if proj.shape[2:] != target_size:
                proj = F.interpolate(proj, size=target_size, mode='bilinear', align_corners=False)
            projected_features.append(proj)

        # Concatenate features along the channel axis. Shape: (Batch, 256, H/4, W/4)
        fused_features = torch.cat(projected_features, dim=1)

        # Step 3: Segmentation Execution
        segmentation = self.segmentation_head(fused_features)

        # Final upsample from Stride 4 back to original input image resolution (Stride 1)

        # Step 4: Regression Execution
        # We pool directly from fused_features so the regression head has spatial awareness
        pooled = torch.flatten(self.avgpool(fused_features), 1)
        output1 = self.fc1(pooled)
        output2 = self.fc2(pooled)

        return output1, output2, segmentation


class RegistrationHRNetNew(nn.Module):
    def __init__(self, in_channels=2):
        super(RegistrationHRNetNew, self).__init__()

        # ---------------------------------------------------------
        # HRNet Backbone (Feature Extractor)
        # ---------------------------------------------------------
        self.encoder = timm.create_model(
            'hrnet_w18',
            pretrained=False,  # IMPORTANT: Set to True on login node to cache weights!
            in_chans=in_channels,
            features_only=True  # Keeps spatial dimensions intact
        )

        # Returns a list of channels for the 4 parallel stages (e.g., [18, 36, 72, 144])
        timm_channels = self.encoder.feature_info.channels()

        # ---------------------------------------------------------
        # Multi-Scale Fusion Logic
        # ---------------------------------------------------------
        self.dec_channels = 64

        # Project all parallel stages to a unified channel size (64)
        self.projections = nn.ModuleList([
            nn.Conv2d(ch, self.dec_channels, kernel_size=1) for ch in timm_channels
        ])

        # Total channels after concatenating the 4 projected branches
        fused_channels = self.dec_channels * len(timm_channels)  # 64 * 4 = 256

        # ---------------------------------------------------------
        #  Regression Head
        # ---------------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # MLPs map the 256 fused channels to your 6 parameters
        self.fc1 = nn.Linear(fused_channels, 3)
        self.fc2 = nn.Linear(fused_channels, 3)

    def forward(self, x):
        # Step 1: Extract Multi-Scale Features from Backbone
        timm_features = self.encoder(x)

        target_size = timm_features[0].shape[2:]  # Highest resolution

        projected_features = []
        for i, feat in enumerate(timm_features):
            proj = self.projections[i](feat)
            # Upsample lower-res branches to match the highest resolution
            if proj.shape[2:] != target_size:
                proj = F.interpolate(proj, size=target_size, mode='bilinear', align_corners=False)
            projected_features.append(proj)

        # Concatenate features along the channel axis. Shape: (Batch, 256, H/4, W/4)
        fused_features = torch.cat(projected_features, dim=1)


        # We pool directly from fused_features so the regression head has spatial awareness
        pooled = torch.flatten(self.avgpool(fused_features), 1)
        output1 = self.fc1(pooled)
        output2 = self.fc2(pooled)

        return output1, output2