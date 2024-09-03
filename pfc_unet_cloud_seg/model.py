import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
import torchmetrics
import pandas as pd
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy, MulticlassPrecision, MulticlassF1Score, MulticlassRecall
import torchvision
from torchmetrics.functional import dice
import matplotlib.pyplot as plt
import seaborn as sn
import io
from PIL import Image
import config

#class IntHandler:
#    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
#        x0, y0 = handlebox.xdescent, handlebox.ydescent
#        text = plt.matplotlib.text.Text(x0, y0, str(orig_handle))
#        handlebox.add_artist(text)
#        return text

class UNet(pl.LightningModule):
    def __init__(self, n_channels=config.NUM_CHANNELS, n_classes=config.NUM_CLASSES, learning_rate=config.LEARNING_RATE, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.lr = learning_rate
        self.bilinear = bilinear

        self._label_ind_by_names = {'background': 0, 'nuvem_densa': 1, 'nuvem_fina': 2, 'sombra': 3}

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

        self.background_jaccard_index = MulticlassJaccardIndex(num_classes=n_classes, average=None)[0]
        self.background_accuracy = MulticlassAccuracy(num_classes=n_classes, average=None)[0]
        self.background_precision = MulticlassPrecision(num_classes=n_classes, average=None)[0]
        self.background_f1score = MulticlassF1Score(num_classes=n_classes, average=None)[0]
        self.background_recall = MulticlassRecall(num_classes=n_classes, average=None)[0]

        self.nuvem_densa_jaccard_index = MulticlassJaccardIndex(num_classes=n_classes, average=None)[1]
        self.nuvem_densa_accuracy = MulticlassAccuracy(num_classes=n_classes, average=None)[1]
        self.nuvem_densa_precision = MulticlassPrecision(num_classes=n_classes, average=None)[1]
        self.nuvem_densa_f1score = MulticlassF1Score(num_classes=n_classes, average=None)[1]
        self.nuvem_densa_recall = MulticlassRecall(num_classes=n_classes, average=None)[1]

        self.nuvem_fina_jaccard_index = MulticlassJaccardIndex(num_classes=n_classes, average=None)[2]
        self.nuvem_fina_accuracy = MulticlassAccuracy(num_classes=n_classes, average=None)[2]
        self.nuvem_fina_precision = MulticlassPrecision(num_classes=n_classes, average=None)[2]
        self.nuvem_fina_f1score = MulticlassF1Score(num_classes=n_classes, average=None)[2]
        self.nuvem_fina_recall = MulticlassRecall(num_classes=n_classes, average=None)[2]

        self.sombra_jaccard_index = MulticlassJaccardIndex(num_classes=n_classes, average=None)[3]
        self.sombra_accuracy = MulticlassAccuracy(num_classes=n_classes, average=None)[3]
        self.sombra_precision = MulticlassPrecision(num_classes=n_classes, average=None)[3]
        self.sombra_f1score = MulticlassF1Score(num_classes=n_classes, average=None)[3]
        self.sombra_recall = MulticlassRecall(num_classes=n_classes, average=None)[3]
        
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

        background_jaccard_index = self.background_jaccard_index(y_pred, y)
        background_accuracy = self.background_accuracy(y_pred, y)
        background_precision = self.background_precision(y_pred, y)
        background_f1score = self.background_f1score(y_pred, y)
        background_recall = self.background_recall(y_pred, y)

        nuvem_densa_jaccard_index = self.nuvem_densa_jaccard_index(y_pred, y)
        nuvem_densa_accuracy = self.nuvem_densa_accuracy(y_pred, y)
        nuvem_densa_precision = self.nuvem_densa_precision(y_pred, y)
        nuvem_densa_f1score = self.nuvem_densa_f1score(y_pred, y)
        nuvem_densa_recall = self.nuvem_densa_recall(y_pred, y)

        nuvem_fina_jaccard_index = self.nuvem_fina_jaccard_index(y_pred, y)
        nuvem_fina_accuracy = self.nuvem_fina_accuracy(y_pred, y)
        nuvem_fina_precision = self.nuvem_fina_precision(y_pred, y)
        nuvem_fina_f1score = self.nuvem_fina_f1score(y_pred, y)
        nuvem_fina_recall = self.nuvem_fina_recall(y_pred, y)

        sombra_jaccard_index = self.sombra_jaccard_index(y_pred, y)
        sombra_accuracy = self.sombra_accuracy(y_pred, y)
        sombra_precision = self.sombra_precision(y_pred, y)
        sombra_f1score = self.sombra_f1score(y_pred, y)
        sombra_recall = self.sombra_recall(y_pred, y)
        
        self.log_dict(
            {
                'train': loss,
                'train_background_jaccard_index': background_jaccard_index,
                'train_background_accuracy': background_accuracy,
                'train_background_precision': background_precision,
                'train_background_f1score': background_f1score,
                'train_background_recall': background_recall,
                'train_nuvem_densa_jaccard_index': nuvem_densa_jaccard_index,
                'train_nuvem_densa_accuracy': nuvem_densa_accuracy,
                'train_nuvem_densa_precision': nuvem_densa_precision,
                'train_nuvem_densa_f1score': nuvem_densa_f1score,
                'train_nuvem_densa_recall': nuvem_densa_recall,
                'train_nuvem_fina_jaccard_index': nuvem_fina_jaccard_index,
                'train_nuvem_fina_accuracy': nuvem_fina_accuracy,
                'train_nuvem_fina_precision': nuvem_fina_precision,
                'train_nuvem_fina_f1score': nuvem_fina_f1score,
                'train_nuvem_fina_recall': nuvem_fina_recall,
                'train_sombra_jaccard_index': sombra_jaccard_index,
                'train_sombra_accuracy': sombra_accuracy,
                'train_sombra_precision': sombra_precision,
                'train_sombra_f1score': sombra_f1score,
                'train_sombra_recall': sombra_recall,
            },
            on_step=True, 
            on_epoch=True, 
            prog_bar=True,
            sync_dist=True,
        )

        tensorboard_logs = {'background_jaccard_index': {'train': background_jaccard_index },'background_accuracy': {'train': background_accuracy },
                            'background_precision': {'train': background_precision },'background_f1score': {'train': background_f1score },
                            'background_recall': {'train': background_recall }, 'loss':{'train': loss }}

        
        return {'loss': loss, 'y_pred': y_pred, 'y': y, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        loss, y_pred, y = self._common_step(batch, batch_idx)

        background_jaccard_index = self.background_jaccard_index(y_pred, y)
        background_accuracy = self.background_accuracy(y_pred, y)
        background_precision = self.background_precision(y_pred, y)
        background_f1score = self.background_f1score(y_pred, y)
        background_recall = self.background_recall(y_pred, y)

        nuvem_densa_jaccard_index = self.nuvem_densa_jaccard_index(y_pred, y)
        nuvem_densa_accuracy = self.nuvem_densa_accuracy(y_pred, y)
        nuvem_densa_precision = self.nuvem_densa_precision(y_pred, y)
        nuvem_densa_f1score = self.nuvem_densa_f1score(y_pred, y)
        nuvem_densa_recall = self.nuvem_densa_recall(y_pred, y)

        nuvem_fina_jaccard_index = self.nuvem_fina_jaccard_index(y_pred, y)
        nuvem_fina_accuracy = self.nuvem_fina_accuracy(y_pred, y)
        nuvem_fina_precision = self.nuvem_fina_precision(y_pred, y)
        nuvem_fina_f1score = self.nuvem_fina_f1score(y_pred, y)
        nuvem_fina_recall = self.nuvem_fina_recall(y_pred, y)

        sombra_jaccard_index = self.sombra_jaccard_index(y_pred, y)
        sombra_accuracy = self.sombra_accuracy(y_pred, y)
        sombra_precision = self.sombra_precision(y_pred, y)
        sombra_f1score = self.sombra_f1score(y_pred, y)
        sombra_recall = self.sombra_recall(y_pred, y)
        
        self.log_dict(
            {
                'val': loss,
                'val_background_jaccard_index': background_jaccard_index,
                'val_background_accuracy': background_accuracy,
                'val_background_precision': background_precision,
                'val_background_f1score': background_f1score,
                'val_background_recall': background_recall,
                'val_nuvem_densa_jaccard_index': nuvem_densa_jaccard_index,
                'val_nuvem_densa_accuracy': nuvem_densa_accuracy,
                'val_nuvem_densa_precision': nuvem_densa_precision,
                'val_nuvem_densa_f1score': nuvem_densa_f1score,
                'val_nuvem_densa_recall': nuvem_densa_recall,
                'val_nuvem_fina_jaccard_index': nuvem_fina_jaccard_index,
                'val_nuvem_fina_accuracy': nuvem_fina_accuracy,
                'val_nuvem_fina_precision': nuvem_fina_precision,
                'val_nuvem_fina_f1score': nuvem_fina_f1score,
                'val_nuvem_fina_recall': nuvem_fina_recall,
                'val_sombra_jaccard_index': sombra_jaccard_index,
                'val_sombra_accuracy': sombra_accuracy,
                'val_sombra_precision': sombra_precision,
                'val_sombra_f1score': sombra_f1score,
                'val_sombra_recall': sombra_recall,
            },
            on_step=True, 
            on_epoch=True, 
            prog_bar=True,
            sync_dist=True,
        )

        tensorboard_logs = {'background_jaccard_index': {'train': background_jaccard_index },'background_accuracy': {'train': background_accuracy },
                            'background_precision': {'train': background_precision },'background_f1score': {'train': background_f1score },
                            'background_recall': {'train': background_recall }, 'loss':{'train': loss }}

        return {'loss': loss, , 'log': tensorboard_logs} 
    
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
        criterion1 = nn.CrossEntropyLoss()
        loss = 0.5 * criterion1(y_pred, y) + 0.5 * dice(y_pred, y, num_classes=self.n_classes)
        #criterion = L.JointLoss(L.FocalLoss(), L.LovaszLoss(), 1.0, 0.5)
        #loss = criterion(y_pred, y)

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
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val"}

    # def validation_epoch_end(self, outs):
    #     # see https://github.com/Lightning-AI/metrics/blob/ff61c482e5157b43e647565fa0020a4ead6e9d61/docs/source/pages/lightning.rst
    #     # each forward pass, thus leading to wrong accumulation. In practice do the following:
    #     tb = self.logger.experiment  # noqa

    #     outputs = torch.cat([tmp['y_pred'] for tmp in outs])
    #     labels = torch.cat([tmp['y'] for tmp in outs])

    #     confusion = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=self.n_classes).to(outputs.get_device())
    #     confusion(outputs, labels)
    #     computed_confusion = confusion.compute().detach().cpu().numpy().astype(int)

    #     # confusion matrix
    #     df_cm = pd.DataFrame(
    #         computed_confusion,
    #         index=self._label_ind_by_names.values(),
    #         columns=self._label_ind_by_names.values(),
    #     )

    #     fig, ax = plt.subplots(figsize=(10, 5))
    #     fig.subplots_adjust(left=0.05, right=.65)
    #     sn.set(font_scale=1.2)
    #     sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', ax=ax)
    #     ax.legend(
    #         self._label_ind_by_names.values(),
    #         self._label_ind_by_names.keys(),
    #         handler_map={int: IntHandler()},
    #         loc='upper left',
    #         bbox_to_anchor=(1.2, 1)
    #     )
    #     buf = io.BytesIO()

    #     plt.savefig(buf, format='jpeg', bbox_inches='tight')
    #     buf.seek(0)
    #     im = Image.open(buf)
    #     im = torchvision.transforms.ToTensor()(im)
    #     tb.add_image("val_confusion_matrix", im, global_step=self.current_epoch)

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
