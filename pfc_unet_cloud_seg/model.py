import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
from torchmetrics.classification import MulticlassJaccardIndex
import torchvision

class UNet(pl.LightningModule):
    def __init__(self, n_channels=4, n_classes=4, learning_rate=1e-3, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.lr = learning_rate
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.jaccard_index = MulticlassJaccardIndex(num_classes=n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        loss, y_pred, y = self._common_step(batch, batch_idx)
        jaccard_index = self.jaccard_index(y_pred, y)
        self.log_dict(
            {
                'train_loss': loss, 
                'train_jaccard_index': jaccard_index
            },
            on_step=False, 
            on_epoch=True, 
            prog_bar=True,
        )

        tensorboard_logs = {'jaccard_index': {'train': jaccard_index }, 'loss':{'train': loss }}

        return {'loss': loss, 'y_pred': y_pred, 'y': y, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        loss, y_pred, y = self._common_step(batch, batch_idx)
        jaccard_index = self.jaccard_index(y_pred, y)
        self.log_dict(
            {
                'validation_loss': loss, 
                'validation_jaccard_index': jaccard_index
            },
            on_step=False, 
            on_epoch=True, 
            prog_bar=True,
        )

        tensorboard_logs = {'jaccard_index': {'train': jaccard_index }, 'loss':{'train': loss }}

        return {'loss': loss, 'log': tensorboard_logs} 
    
    def test_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)
        jaccard_index = self.jaccard_index(y_pred, y)
        self.log_dict(
            {
                'test_loss': loss, 
                'test_jaccard_index': jaccard_index
            },
            on_step=False, 
            on_epoch=True, 
            prog_bar=True,
        )

        return loss
       
    def _common_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        y = y.long()
        x = x.to(torch.float32)
        y_pred = self.forward(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_pred, y)

        return loss, y_pred, y
    
    def predict_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        y = y.long()
        y_pred = self.forward(x)
        preds = torch.argmax(y_pred, dim=1)
        return preds
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "validation_loss"}

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss
