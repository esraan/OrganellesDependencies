
import torch
from torch import nn



class ConvBlock(nn.Module):
    def __init__(self, ch_input_tensors, ch_output_tensor, kernel_size=3, padding=1):
        super().__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(ch_input_tensors, ch_output_tensor, kernel_size, padding),
                                        nn.BatchNorm2d(ch_output_tensor),
                                        nn.ReLU(inplace=True),  # added due to Naor code
                                        nn.Conv2d(ch_output_tensor, ch_output_tensor, kernel_size, padding),
                                        nn.BatchNorm2d(ch_output_tensor),
                                        nn.ReLU(inplace=True))

    def forward(self, inputs):
        x = self.conv_block(inputs)
        return x


class Encoder(nn.Module):
    def __init__(self, ch_input_tensors, ch_output_tensor, max_pool=2):
        super().__init__()
        # self.seq_encoder = nn.Sequential()
        self.conv = ConvBlock(ch_input_tensors, ch_output_tensor)
        self.pool = nn.MaxPool2d((max_pool, max_pool))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class Decoder(nn.Module):
    def __init__(self, ch_input_tensors, ch_output_tensor, kernel_size=2, padding=0, stride=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(ch_input_tensors, ch_output_tensor, kernel_size, stride, padding)
        ch_output_tensor_joined = ch_output_tensor + ch_output_tensor
        self.conv = ConvBlock(ch_output_tensor_joined, ch_output_tensor) #check adding both tensors if done correctly

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat((x, skip), dim=1)
        x = self.conv(x)
        return x


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        # """ Encoder """
        self.e1 = Encoder(3, 64)
        self.e2 = Encoder(64, 128)
        self.e3 = Encoder(128, 256)
        self.e4 = Encoder(256, 512)
        # """ Bottleneck """
        self.b = ConvBlock(512, 1024)
        # """ Decoder """
        self.d1 = Decoder(1024, 512)
        self.d2 = Decoder(512, 256)
        self.d3 = Decoder(256, 128)
        self.d4 = Decoder(128, 64)
        # """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        # """ Bottleneck """
        b = self.b(p4)
        # """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        # """ Classifier """
        outputs = self.outputs(d4)
        return outputs

# 
# class Unet(nn.Module):
# 
#     # fixed bug for pl loading model_checkpoint according to https://github.com/PyTorchLightning/pytorch-lightning/issues/2909
#     def __init__(self, *args, **kwargs):
#         super(Unet, self).__init__()
# 
#         if isinstance(kwargs, dict):
#             hparams = Namespace(**kwargs)
# 
#         self.save_hyperparameters(kwargs)
#         self.n_channels = hparams.n_input_channels
#         self.n_classes = hparams.n_classes
#         self.input_size = hparams.input_size
#         self.h = hparams.input_size[0]
#         self.w = hparams.input_size[1]
#         self.minimize_net_factor = hparams.minimize_net_factor
#         self.bilinear = True
# 
#         def double_conv(in_channels, out_channels):
#             return nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU(inplace=True),
#             )
# 
#         def down(in_channels, out_channels):
#             return nn.Sequential(
#                 nn.MaxPool2d(2),
#                 double_conv(in_channels, out_channels)
#             )
# 
#         class up(nn.Module):
#             def __init__(self, in_channels, out_channels, bilinear=True):
#                 super().__init__()
# 
#                 if bilinear:
#                     self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#                 else:
#                     self.up = torch.nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
#                                                        kernel_size=2, stride=2)
# 
#                 self.conv = double_conv(in_channels, out_channels)
# 
#             def forward(self, x1, x2):
#                 x1 = self.up(x1)
#                 # [?, C, H, W]
#                 diffY = x2.size()[2] - x1.size()[2]
#                 diffX = x2.size()[3] - x1.size()[3]
# 
#                 x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                                 diffY // 2, diffY - diffY // 2])
#                 x = torch.cat([x2, x1], dim=1)
#                 return self.conv(x)
# 
#         first_layer_depth = int(64 / self.minimize_net_factor)
#         self.inc = double_conv(self.n_channels, first_layer_depth)
#         self.down1 = down(first_layer_depth, first_layer_depth * 2)
#         self.down2 = down(first_layer_depth * 2, first_layer_depth * 4)
#         self.down3 = down(first_layer_depth * 4, first_layer_depth * 8)
#         self.down4 = down(first_layer_depth * 8, first_layer_depth * 8)
#         self.up1 = up(first_layer_depth * 16, first_layer_depth * 4)
#         self.up2 = up(first_layer_depth * 8, first_layer_depth * 2)
#         self.up3 = up(first_layer_depth * 4, first_layer_depth)
#         self.up4 = up(first_layer_depth * 2, first_layer_depth)
#         self.out = nn.Conv2d(first_layer_depth, self.n_classes, kernel_size=1)
# 
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         return self.out(x)
# 
#     def training_step(self, batch, batch_nb):
#         x, y = batch
#         x, y = x.to(self.device), y.to(self.device)
#         y_hat = self.forward(x)
#         loss = F.mse_loss(y_hat, y)
#         tensorboard_logs = {'train_loss': loss.detach()}
#         self.log('train_loss', loss.detach(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         return {'loss': loss, 'log': tensorboard_logs}
# 
#     def validation_step(self, batch, batch_nb):
#         _, loss, pcc = unify_test_function(self, batch)
#         self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         self.log('val_pcc', pcc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         return {'val_loss': loss}
# 
#     def validation_end(self, outputs):
#         avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
#         tensorboard_logs = {'val_loss': avg_loss.detach()}
#         self.log('avg_val_loss', avg_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
# 
#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-8)
# 
#     def configure_callbacks(self):
#         early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=6, verbose=True)
#         checkpoint = ModelCheckpoint(monitor="val_loss")
#         return [early_stop, checkpoint]
# 
# 
# def unify_test_function(model, batch):
#     if len(batch) == 3:
#         x, y, _ = batch
#     else:
#         x, y = batch
# 
#     x, y = x.to(model.device), y.to(model.device)
#     pred = process_image(model, x, model.input_size, model.n_channels)
#     loss = F.mse_loss(pred.detach(), y)
#     pcc = pearson_corrcoef(pred.reshape(-1), y.reshape(-1))
#     return pred.detach(), loss.detach(), pcc.detach()
