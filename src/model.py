import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


def _number_of_patches(image_size: int, patch_size: int) -> int:
    """
    Calculate the number of patches to divide an image up into using a specific patch_size
    """
    assert (
        image_size % patch_size == 0
    ), "Patch size must be a multiple of the image_size"
    num_patches = (image_size // patch_size) ** 2
    return num_patches


class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self._mlp_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self._mlp_block(x)


class TokenMixer(nn.Module):
    def __init__(self, num_patches: int, num_features: int):
        super().__init__()

        self._layer_norm = nn.LayerNorm(num_features)
        self._mlp = MLPBlock(num_patches, num_features)

    def forward(self, x):
        # Keep track of residual
        # (batch_size, num_patches, num_features)
        residual = x

        # Perform layer normalisation
        # Transpose to mix tokens
        # (batch_size, num_features, num_patches)
        x = self._layer_norm(x)
        x_t = x.permute(0, 2, 1)
        o_t = self._mlp(x_t)

        # Transpose back to match residuals
        # (batch_size, num_patches, num_features)
        o = o_t.permute(0, 2, 1)

        return o + residual


class ChannelMixer(nn.Module):
    def __init__(self, num_patches, num_features):
        super().__init__()
        self._layer_norm = nn.LayerNorm(num_patches)
        self._mlp = MLPBlock(num_patches, num_features)

    def forward(self, x):
        # Keep track of residual
        residual = x

        # Perform layer normalisation
        x = self._layer_norm(x)

        # (batch_size, num_patches, num_features)
        o = self._mlp(x)

        return o + residual


class MixerBlock(nn.Module):
    def __init__(self, num_patches: int, num_features: int):
        super().__init__()
        self._token_mixer = TokenMixer(num_patches, num_features)
        self._channel_mixer = ChannelMixer(num_features, num_features)

    def forward(self, x):
        x = self._token_mixer(x)
        x = self._channel_mixer(x)

        return x


class MLPMixer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        input_channels: int,
        num_features: int,
        num_mixer_blocks: int,
        num_classes: int,
    ):
        super().__init__()
        self._num_patches = _number_of_patches(image_size, patch_size)
        self._num_features = num_features

        # A fully-connected per patch is equivalent to a Conv2d on a patch.
        self._linear_embedding = nn.Conv2d(
            input_channels,
            num_features,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self._mixer_blocks = nn.Sequential(
            *[
                MixerBlock(self._num_patches, num_features)
                for _ in range(num_mixer_blocks)
            ]
        )

        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # (batch_size, num_features, sqrt(num_patches), sqrt(num_patches))
        x = self._linear_embedding(x)

        # (batch_size, num_patches, num_features)
        x = x.view(-1, self._num_patches, self._num_features)

        # (batch_size, num_patches, num_features)
        x = self._mixer_blocks(x)

        # Global average pooling over the patches
        x = x.mean(1)

        # Linear classifier to map to class logits
        logits = self.classifier(x)

        return logits


class MLPMixerLightning(pl.LightningModule):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        input_channels: int,
        num_features: int,
        num_mixer_blocks: int,
        num_classes: int,
    ):
        super().__init__()
        self._model = MLPMixer(
            image_size,
            patch_size,
            input_channels,
            num_features,
            num_mixer_blocks,
            num_classes,
        )

        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self._model(x)
        loss = F.cross_entropy(logits, y)

        accuracy = float(sum(torch.argmax(logits, dim=1) == y) / len(y))

        self.log(
            "training_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "training_accuracy",
            accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self._model(x)
        loss = F.cross_entropy(logits, y)

        accuracy = float(sum(torch.argmax(logits, dim=1) == y) / len(y))

        self.log(
            "validation_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "validation_accuracy",
            accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss
