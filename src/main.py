import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.data import MNISTDataModule
from src.model import MLPMixerLightning


def main():
    data_module = MNISTDataModule(data_dir="./data/", batch_size=128)

    # Define callbacks
    logger = TensorBoardLogger("lightning_logs", name="mlp-mixer-mnist")
    checkpoint_callback = ModelCheckpoint(
        dirpath="./",
        filename="mlp_mixer_mnist" + "-{epoch}-{validation_loss:.2f}",
        monitor="validation_loss",
        every_n_epochs=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    params = {
        "max_epochs": 10,
        "logger": logger,
        "log_every_n_steps": 10,
        "accumulate_grad_batches": 8,
        "callbacks": [lr_monitor, checkpoint_callback],
    }

    model = MLPMixerLightning(
        image_size=28,
        patch_size=7,
        input_channels=1,
        num_features=128,
        num_mixer_blocks=6,
        num_classes=10,
    )

    trainer = pl.Trainer(*params)

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
