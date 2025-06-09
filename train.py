import os
import warnings
import hydra
import wandb
from omegaconf import OmegaConf
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from main import models, dataloaders, utils

# Setup environment
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
warnings.filterwarnings('ignore')


def get_trainer_config(cfg, callbacks):
    """Return trainer configuration."""
    return {
        "accelerator": "gpu",
        "devices": cfg.exp.devices,
        "max_epochs": cfg.exp.epochs,
        "callbacks": callbacks
    }

def setup_wandb_logger(cfg, model, exp_name=None):
    """Initialize WandB logger with given configuration."""
    if cfg.exp.testing_only:
        os.environ["WANDB_SILENT"] = "true"
    
    wandb_logger = WandbLogger(
        project=cfg.exp.group_name,
        name=exp_name or cfg.exp.exp_name,
        log_model=False
    )

    if not cfg.exp.testing_only:
        wandb_logger.watch(model, log='all')
        wandb_logger.log_hyperparams({**OmegaConf.to_container(cfg, resolve=True)})
    
    return wandb_logger

def test_one_checkpoint(trainer_config, lit_model, lit_dataset, cfg):
    """Test and plot the attention of specific one checkpoint."""
    # logger = setup_wandb_logger(cfg, lit_model, f"{cfg.exp.exp_name}_epoch{epoch}")
    trainer = Trainer(**trainer_config, logger=None)


def test_all_checkpoints(trainer_config, lit_model, lit_dataset, cfg):
    """Test all checkpoints in directory with proper logging."""
    epochs = utils.find_checkpoints(cfg.exp.test_ckpt_dir)
    if not epochs:
        raise ValueError(f"No checkpoints found in {cfg.exp.test_ckpt_dir}")
    
    print(f"Found {len(epochs)} checkpoints to evaluate")
    for epoch in epochs:
        wandb.finish()  # Cleanup previous run
        
        # Setup logger and trainer for this epoch
        logger = setup_wandb_logger(cfg, lit_model, f"{cfg.exp.exp_name}_epoch{epoch}")
        trainer = Trainer(**trainer_config, logger=logger)
        
        # Log hyperparameters if needed
        if cfg.exp.devices:  # Log for any GPU setup
            logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        
        # Run test
        ckpt_path = os.path.join(cfg.exp.test_ckpt_dir, f"epoch={epoch}.ckpt")
        print(f"\nTesting checkpoint: {ckpt_path}")
        trainer.test(lit_model, dataloaders=lit_dataset, ckpt_path=ckpt_path)


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: OmegaConf) -> None:
    # Initialize components
    utils.prep_run(cfg)
    lit_dataset = dataloaders.LitDataset(cfg)
    lit_model = models.LitModel(cfg)

    # Setup callbacks
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
            every_n_epochs=cfg.exp.save_n_eps,
            save_top_k=cfg.exp.save_k_ckpt,
            monitor="val_loss",
            dirpath=cfg.exp.save_ckpt_dir,
            filename='{epoch:02d}',
            mode='min',
            save_last=True
        ),
        EarlyStopping(
            monitor="val_loss",
            min_delta=cfg.exp.es_threshold,
            patience=cfg.exp.es_patience,
            verbose=False,
            mode="min"
        )
    ]
    callbacks.append(utils.TestEveryNEpochs(lit_dataset.test_dataloader(), every_n_epochs=cfg.exp.test_n_eps))

    trainer_config = get_trainer_config(cfg, callbacks)
    
    if not cfg.exp.testing_only:
        # Training mode
        logger = setup_wandb_logger(cfg, lit_model)
        trainer = Trainer(**trainer_config, logger=logger)
        trainer.fit(
            lit_model,
            datamodule=lit_dataset,
            ckpt_path=cfg.exp.load_ckpt_path if cfg.exp.training_restore else None
        )
        trainer.test(lit_model, dataloaders=lit_dataset, ckpt_path=None)
    else:
        # Testing mode - evaluate all checkpoints
        test_all_checkpoints(trainer_config, lit_model, lit_dataset, cfg)

if __name__ == "__main__":
    main()