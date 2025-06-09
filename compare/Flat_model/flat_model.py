import os, warnings, sys
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' #':16:8'  # or ":4096:8"
warnings.filterwarnings('ignore')
# warnings.filterwarnings('ignore',category=UserWarning)

import hydra
from omegaconf import OmegaConf
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import LearningRateMonitor
import torch
import torch.nn.functional as F
from einops import rearrange

from main import matrix, utils, dataloaders

class FlatModel(LightningModule):
    def __init__(self, cfg):
        super(FlatModel, self).__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        # Define the constant output value, normalized as required
        self.constant_value = cfg.flat.velo_value / 128.0

    def batch_to_input(self, batch):
        """Convert batch data into matrices for onsets, frames, and velocities."""
        # Convert the batch data to matrix format according to the configuration
        _input, _label = matrix.batch_to_matrix(batch, self.cfg, self.device)

        # Extract and reshape the onset, frame, and velocity matrices
        _onset = rearrange(_input[:, :, 0, :, :], 'b s h w -> (b s) h w')
        _frame = rearrange(_input[:, :, 1, :, :], 'b s h w -> (b s) h w')
        _velo = rearrange(_input[:, :, 2, :, :], 'b s h w -> (b s) h w') / 128.0  # Normalize velocity to [0, 1]

        # Apply onset mask to velocity if specified in the configuration
        _velo = _velo * _onset if self.cfg.exp.onset_mask else _velo

        return _onset, _frame, _velo, _label

    def forward(self, x):
        """Return a constant output tensor for any input x."""
        batch_size, height, width = x.size()
        return torch.full((batch_size, height, width), self.constant_value, device=x.device)

    def training_step(self, batch, batch_idx):
        """Prepare training data without logging any results."""
        _, _, _, _ = self.batch_to_input(batch)
        return None  # No further computation, just process data

    def validation_step(self, batch, batch_idx):
        """Prepare validation data without logging any results."""
        _, _, _, _ = self.batch_to_input(batch)
        return None  # No further computation, just process data

    def test_step(self, batch, batch_idx):
        """Test step for the Flat model, returning evaluation metrics for aggregation."""
        gt_onset, _, gt_velo, _ = self.batch_to_input(batch)
        est_velo = self.forward(gt_onset)  # Using onset as input
        # Calculate evaluation metrics
        mse, mae, sd_ae, sd, sd_gt, f1, recall, recall_5 = self.evaluation(est_velo, gt_velo, gt_onset)
        self.log('test_mse', mse, prog_bar=False)
        self.log('test_mae', mae, prog_bar=False)
        self.log('test_sd_ae', sd_ae, prog_bar=False)
        self.log('test_sd_pred', sd, prog_bar=False)
        # self.log('test_sd_groundtruth', sd_gt, prog_bar=False)
        # self.log('test_f1', f1, prog_bar=False)
        self.log('test_recall', recall, prog_bar=False)
        self.log('test_recall_5%', recall_5, prog_bar=False)
        return None

    def evaluation(self, est_velo, gt_velo, gt_onset):
        """Denomalised MSE, MAE, SD, and F1 score based on the provided ground truth and estimated velocities."""
        mask = gt_onset != 0
        # print("Debugging: est_velo has 1:", (est_velo == 1).any().item(), "| gt_velo has 1:", (gt_velo == 1).any().item())
        est_velo, gt_velo = (est_velo[mask] * 128).round(), (gt_velo[mask] * 128).round() 
        # Denormalized metrices
        mse = F.mse_loss(est_velo, gt_velo, reduction='mean')
        mae = F.l1_loss(est_velo, gt_velo, reduction='mean')
        sd_ae = torch.std(torch.abs(est_velo - gt_velo))  # Standard deviation of absolute error
        sd = torch.std(est_velo)   # Standard deviation of estimate
        sd_gt = torch.std(gt_velo) # Standard deviation of groundtruth
        f1 = 2 * ((sd / sd_gt) * (1 - 3 * mae)) / ((sd / sd_gt) + (1 - 3 * mae))
        recall = utils.recall(est_velo, gt_velo, threshold=12.7) # 10% error threshold of Velocity [0-127]
        recall_5 = utils.recall(est_velo, gt_velo, threshold=6.4) # 10% error threshold of Velocity [0-127]
        return mse, mae, sd_ae, sd, sd_gt, f1, recall, recall_5

    def configure_optimizers(self):
        return None  # No optimizer needed for the Flat model

@hydra.main(config_path="./conf", config_name="config")
def main(cfg: OmegaConf) -> None:
    # Preparation for the run (e.g., random seed, cleaning cache)
    utils.prep_run(cfg)
    lit_dataset = dataloaders.LitDataset(cfg)
    lit_model = FlatModel(cfg)

    # Set up Trainer with required callbacks
    callbacks = [LearningRateMonitor(logging_interval='epoch')]

    trainer = Trainer(
        accelerator="gpu",
        devices=cfg.exp.devices,
        max_epochs=1,  # Single epoch since this is preprocessing/evaluation only
        callbacks=callbacks,
        # logger=False,
        log_every_n_steps=1,  # This will ensure frequent logging to the terminal
        enable_progress_bar=False,
        enable_model_summary=False,      # Suppresses model summary display
    )
    trainer.fit(lit_model, datamodule=lit_dataset)
    trainer.test(lit_model, dataloaders=lit_dataset)

if __name__ == "__main__":
    main()