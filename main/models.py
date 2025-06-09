import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch import LightningModule
from torchvision.models import resnet18
from pytorch_optimizer import Ranger21
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

from main import matrix, utils, loss

# NOTE: -----------------------------------------------------------------
# NOTE: ----------------------- Main Framwork ---------------------------
# NOTE: -----------------------------------------------------------------

class LitModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        # Formal Build reported in Paper
        if cfg.ae.model == "ConvAE":
            self.model = ConvAE(cfg)
            cfg.exp.onset_mask = False # ConvAE 2021 literature use frame as input
        elif cfg.ae.model == "UNet":
            self.model = UNet(cfg)  # Our developed UNet
        
        self.recon_loss = loss.Recon_Loss(cfg)
        self.evaluator = loss.EvaluationMetrics()
        self.optim = self.configure_optimizers()
        self.lr_schedule = self.configure_lr_scheduler()
        
    def batch_to_input(self, batch):
        """Convert batch data into matrices for notes, frames, and velocities."""
        _input, _label = matrix.batch_to_matrix(batch, self.cfg, self.device)
        _onset = rearrange(_input[:, :, 0, :, :], 'b s h w -> (b s) h w')
        _frame = rearrange(_input[:, :, 1, :, :], 'b s h w -> (b s) h w')
        _velo =  rearrange(_input[:, :, 2, :, :], 'b s h w -> (b s) h w') / 128.0
        _velo = _velo * _onset if self.cfg.exp.onset_mask else _velo
        return _onset, _frame, _velo, _label

    def share_step(self, batch):
        """Shared step logic for training, validation, and testing."""
        gt_onset, gt_frame, gt_velo, _ = self.batch_to_input(batch)
        input_data = gt_onset if self.cfg.exp.onset_mask else gt_frame
        est_velo = self.model(input_data)
        loss = self.recon_loss.compute(est_velo, gt_velo, gt_onset)
        metrics = self.evaluator.train(est_velo, gt_velo, gt_onset)
        return (loss,) + metrics

    def training_step(self, batch, batch_idx):
        loss,\
            _, _, _, recall, _, _, _ = self.share_step(batch)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_recall', recall, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss,\
            _, _, _, recall, _, _, _  = self.share_step(batch)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_recall', recall, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        _,\
            mae, mse, sd_ae, recall, recall_5, sd, _ = self.share_step(batch)
        self.log(f'test_mae', mae, prog_bar=False)
        self.log(f'test_mse', mse, prog_bar=False)
        self.log(f'test_sd_ae', sd_ae, prog_bar=False)
        self.log(f'test_recall', recall, prog_bar=False)
        self.log(f'test_recall_5%', recall_5, prog_bar=False)
        self.log(f'test_sd', sd, prog_bar=False)
        return
    
    def configure_optimizers(self):
        """Configures the optimizer based on the configuration."""
        if self.cfg.opt.optimizer == "Adam":
            optimizer = Adam(self.parameters(), lr=self.cfg.opt.lr, betas=(0.9, 0.96), weight_decay=4.5e-2)
        elif self.cfg.opt.optimizer  == "Ranger":
            optimizer = Ranger21(self.parameters(), lr=self.cfg.opt.lr, betas=(0.9, 0.96), weight_decay=4.5e-2, num_iterations=10000)
        return optimizer

    def configure_lr_scheduler(self):
        """Configures the learning rate scheduler based on the configuration."""
        if self.cfg.opt.scheduler == "Warmup":
            lr_scheduler = utils.WarmupLinearLRSchedule(self.optim, init_lr=self.cfg.opt.lr_init, peak_lr=self.cfg.opt.lr, end_lr=self.cfg.opt.lr_end, warmup_epochs=self.cfg.opt.warmup_epochs, epochs=self.cfg.exp.epochs, current_step=0)
        elif self.cfg.opt.scheduler == "Exponential":
            lr_scheduler = ExponentialLR(self.optim, gamma=self.cfg.opt.lr_gamma)
        elif self.cfg.opt.scheduler == "ReduceLROnPlateau":
            lr_scheduler = ReduceLROnPlateau(self.optim, factor=self.cfg.opt.lr_factor, patience=self.cfg.opt.lr_patience)
        return lr_scheduler

    def reconstruct(self, input):
        """Reconstruct input data."""
        y = self.model(input)
        return y
    
    def show_attention(self, input):
        """
        Pass the input through the model with return_attention enabled,
        and return both the reconstructed output and the attention weights.
        """
        y, attention = self.model(input, return_attention=True)
        return y, attention
    
# NOTE: --------------------------------------------------------------------------
# NOTE: ------------ Architectures Reported in Paper (Formal Build) --------------
# NOTE: --------------------------------------------------------------------------
    
class UNet(nn.Module):
    """
    UNet AutoEncoder based on [Image Colorization 2022 by N. Wang]
    Paper (Model Architecture): https://www.mdpi.com/2073-8994/14/11/2295
    - Original paper channels: [64, 128, 256, 512] (Sigmoid)
    - Tested channels for our task:
      * [32, 64, 128, 256] - (overfitting)
      * [16, 32, 64, 128]
      * [16, 32, 64, 96] - (Best & Sigmoid better than ReLU)
      * [16, 32, 48, 64] 
      * [8, 16, 32, 64]
    """
    def __init__(self, cfg):
        super(UNet, self).__init__()
        self.cfg = cfg
        channels = cfg.ae.channels
        attn_window = cfg.ae.attn_window

        self.enc1 = self.final_block(1, channels[0])             # [batch, channels[0], H, W]
        self.enc2 = self.encoder_block(channels[0], channels[1]) # [batch, channels[1], H/2, W/2]
        self.enc3 = self.encoder_block(channels[1], channels[2]) # [batch, channels[2], H/4, W/4]
        self.enc4 = self.encoder_block(channels[2], channels[3]) # [batch, channels[3], H/8, W/8]

        self.mid = self.bottleneck(channels[3])        

        self.dec4 = self.decoder_block(channels[3] * (2 if self.cfg.ae.ablation != "no_skip" else 1), channels[2]) # [batch, channels[2], H/4, W/4]
        self.dec3 = self.decoder_block(channels[2] * (2 if self.cfg.ae.ablation != "no_skip" else 1), channels[1]) # [batch, channels[1], H/2, W/2]
        self.dec2 = self.decoder_block(channels[1] * (2 if self.cfg.ae.ablation != "no_skip" else 1), channels[0]) # [batch, channels[0], H, W]
        self.dec1 = self.final_block(channels[0], 1)                                                               # [batch, 1, H, W]

        # Attention blocks for skip connections (conditionally enabled)
        if self.cfg.ae.ablation=="attn":
            self.att4 = EfficientAttentionBlock(channels[3], channels[3], attn_window)
            self.att3 = EfficientAttentionBlock(channels[2], channels[2], attn_window)
            self.att2 = EfficientAttentionBlock(channels[1], channels[1], attn_window)
        else:
            self.att4 = self.att3 = self.att2 = None

        utils.initialize_weights(self, init_type=cfg.exp.init_type, component="Conv")

    def forward(self, x, return_attention=False):
        x = x.unsqueeze(1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        z = self.mid(e4)

        if self.cfg.ae.ablation == "no_skip": # No skip connections
            d4 = self.dec4(z)
            d3 = self.dec3(d4)
            d2 = self.dec2(d3)
        
        elif self.cfg.ae.ablation == "no_attn": # Skip connections without attention
            d4 = self.dec4(torch.cat((z, e4), dim=1))
            d3 = self.dec3(torch.cat((d4, e3), dim=1))
            d2 = self.dec2(torch.cat((d3, e2), dim=1))

        else: # Skip connections with attention
            if return_attention:
            # if self.cfg.interface.return_attention:
                # attn4_ref, attn4_in = e4, z
                attn4_out, attn4_weights = self.att4(e4, z, return_attention=True)
                d4 = self.dec4(torch.cat((z, attn4_out), dim=1))

                # attn3_ref, attn3_in = e3, d4
                attn3_out, attn3_weights = self.att3(e3, d4, return_attention=True)
                d3 = self.dec3(torch.cat((d4, attn3_out), dim=1))

                # attn2_ref, attn2_in = e2, d3
                attn2_out, attn2_weights = self.att2(e2, d3, return_attention=True)
                d2 = self.dec2(torch.cat((d3, attn2_out), dim=1))

                y = self.dec1(d2)
                return y.squeeze(1), (attn4_weights, attn3_weights, attn2_weights, # attention weights
                                      attn4_out, attn3_out, attn2_out, # attention output
                                      e4, e3, e2, # attention reference
                                      z, d4, d3, d2)  # attention input & decoder output
            else:
                d4 = self.dec4(torch.cat((z, self.att4(e4, z)), dim=1))
                d3 = self.dec3(torch.cat((d4, self.att3(e3, d4)), dim=1))
                d2 = self.dec2(torch.cat((d3, self.att2(e2, d3)), dim=1))
        
        y = self.dec1(d2)
        return y.squeeze(1)
    
    def encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True) if self.cfg.ae.activation == "ReLU" else nn.Sigmoid(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True) if self.cfg.ae.activation == "ReLU" else nn.Sigmoid(),
            nn.MaxPool2d(2))
    
    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True) if self.cfg.ae.activation == "ReLU" else nn.Sigmoid(),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True) if self.cfg.ae.activation == "ReLU" else nn.Sigmoid())

    def final_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid())
    
    def bottleneck(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True) if self.cfg.ae.activation == "ReLU" else nn.Sigmoid(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True) if self.cfg.ae.activation == "ReLU" else nn.Sigmoid())


class ConvAE(nn.Module):
    """
    Literature 2021 detailed implementation
    """
    def __init__(self, cfg):
        super(ConvAE, self).__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(4, 3), stride=1, padding="same"), nn.ReLU(True),  # Conv1
            nn.MaxPool2d(kernel_size=2, stride=2),                                          # MaxPool1
            nn.Conv2d(32, 32, kernel_size=(4, 3), stride=1, padding="same"), nn.ReLU(True), # Conv2
            nn.MaxPool2d(kernel_size=2, stride=2),                                          # MaxPool2
            nn.Conv2d(32, 32, kernel_size=(4, 3), stride=1, padding="same"), nn.ReLU(True), # Conv3
            nn.Conv2d(32, 32, kernel_size=(4, 3), stride=1, padding="same"), nn.ReLU(True), # Conv4
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(4, 3), stride=1, padding="same"), nn.ReLU(True), # Conv5
            nn.Conv2d(32, 32, kernel_size=(4, 3), stride=1, padding="same"), nn.ReLU(True), # Conv6
            nn.Upsample(scale_factor=2),                                                    # Upsample1
            nn.Conv2d(32, 32, kernel_size=(4, 3), stride=1, padding="same"), nn.ReLU(True), # Conv7
            nn.Upsample(scale_factor=2),                                                    # Upsample2
            nn.Conv2d(32, 1, kernel_size=(4, 3), stride=1, padding="same"),                 # Conv8
            nn.Sigmoid()  # Scale output to [0, 1]
        )
        utils.initialize_weights(self, init_type=cfg.exp.init_type, component="Conv")

    def forward(self, x):
        x = x.unsqueeze(1)
        z = self.encoder(x)
        y = self.decoder(z)
        return y.squeeze(1)
    
# NOTE: -------------------------------------------------------------------------
# NOTE: ------------------------- Attention Blocks ------------------------------
# NOTE: -------------------------------------------------------------------------

class EfficientAttentionBlock(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, window_size):
        """
        Memory-efficient attention block using local windows.
        
        Parameters:
        - encoder_channels: Number of channels in encoder output
        - decoder_channels: Number of channels in decoder input
        - window_size: Size of local attention window (e.g., 8x8)
        """
        super(EfficientAttentionBlock, self).__init__()
        self.window_size = window_size
        self.query = nn.Conv2d(decoder_channels, decoder_channels // 2, kernel_size=1)
        self.key =   nn.Conv2d(encoder_channels, decoder_channels // 2, kernel_size=1)
        self.value = nn.Conv2d(encoder_channels, decoder_channels, kernel_size=1)
        self.scale = (decoder_channels // 2) ** 0.5

    def forward(self, encoder_output, decoder_output, return_attention=False):
        """
        Forward pass using local window attention.
        """
        batch, _, height, width = decoder_output.shape # batch, channels, height, width
        
        # Compute query, key, and value
        query = self.query(decoder_output)  # [B, C/2, H, W]
        key = self.key(encoder_output)      # [B, C/2, H, W]
        value = self.value(encoder_output)  # [B, C, H, W]

        # Pad inputs to make them divisible by window_size
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        
        if pad_h > 0 or pad_w > 0:
            query = F.pad(query, (0, pad_w, 0, pad_h))
            key = F.pad(key, (0, pad_w, 0, pad_h))
            value = F.pad(value, (0, pad_w, 0, pad_h))
        
        # Reshape into local windows
        new_height = height + pad_h
        new_width = width + pad_w

        # Reshape tensors [B, C, H, W] to [B, C, num_windows_h, window_size, num_windows_w, window_size]
        query = query.view(batch, -1, new_height // self.window_size, self.window_size, new_width // self.window_size, self.window_size)
        key =     key.view(batch, -1, new_height // self.window_size, self.window_size, new_width // self.window_size, self.window_size)
        value = value.view(batch, -1, new_height // self.window_size, self.window_size, new_width // self.window_size, self.window_size)

        # Compute attention for each window
        query = query.permute(0, 2, 4, 1, 3, 5).contiguous()
        key =     key.permute(0, 2, 4, 1, 3, 5).contiguous()
        value = value.permute(0, 2, 4, 1, 3, 5).contiguous()
        
        # Reshape for matrix multiplication
        query = query.view(batch, -1, self.window_size * self.window_size, query.size(3))
        key =     key.view(batch, -1, key.size(3), self.window_size * self.window_size)
        value = value.view(batch, -1, self.window_size * self.window_size, value.size(3))

        # Compute scaled dot-product attention
        attention = torch.matmul(query, key) / self.scale   # [B, HW, HW]
        attention = F.softmax(attention, dim=-1)            # Normalize along the spatial dimension
        refined = torch.matmul(attention, value)            # [B, HW, C]

        # Reshape back to original format [B, C, H, W]
        refined = refined.view(batch, new_height // self.window_size, new_width // self.window_size, -1, self.window_size, self.window_size)
        refined = refined.permute(0, 3, 1, 4, 2, 5).contiguous()
        refined = refined.view(batch, -1, new_height, new_width)
        
        if pad_h > 0 or pad_w > 0:
            refined = refined[:, :, :height, :width]

        if return_attention: # <- Now returning attention weights
            return refined + decoder_output, attention

        return refined + decoder_output