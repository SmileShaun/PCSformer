import torch
import torch.nn as nn
import cv2
import robust_loss_pytorch
import torch.nn.functional as F
from torchvision import models


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()

        vgg_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        vgg_model = vgg_model.features[:16]
        for param in vgg_model.parameters():
            param.requires_grad = False

        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.l1_loss(dehaze_feature, gt_feature))
        return sum(loss)/len(loss)


class TotalLoss(nn.Module):
    def __init__(self, model_resolution):
        super(TotalLoss, self).__init__()
        self.VggLoss = VGGLoss()
        self.RobustLoss = robust_loss_pytorch.adaptive.AdaptiveImageLossFunction(
            image_size=(model_resolution, model_resolution, 3),
            float_dtype=torch.float32,
            device='cuda:0',
            color_space='RGB',
            representation='PIXEL')
        self.L1Loss = nn.L1Loss()

    def forward(self, result, gt):
        return self.L1Loss(result, gt) + 0.4*self.RobustLoss.lossfun(result-gt).mean() + 0.5*self.VggLoss(result, gt)

