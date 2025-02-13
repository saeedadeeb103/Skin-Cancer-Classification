import torch.optim as optim
import pytorch_lightning as pl
import timm
from torchmetrics import Accuracy, Precision, Recall, F1Score
import torch



class timm_backbones(pl.LightningModule):
    """
    PyTorch Lightning model for image classification using a ResNet-18 architecture.

    This model uses a pre-trained ResNet-18 model and fine-tunes it for a specific number of classes.

    Args:
        num_classes (int, optional): The number of classes in the dataset. Defaults to 2.
        optimizer_cfg (DictConfig, optional): A Hydra configuration object for the optimizer.

    Methods:
        forward(x): Computes the forward pass of the model.
        configure_optimizers(): Configures the optimizer for the model.
        training_step(batch, batch_idx): Performs a training step on the model.
        validation_step(batch, batch_idx): Performs a validation step on the model.
        on_validation_epoch_end(): Called at the end of each validation epoch.
        test_step(batch, batch_idx): Performs a test step on the model.

    Example:
        model = ResNet18(num_classes=2, optimizer_cfg=cfg.model.optimizer)
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        trainer.test(model, dataloaders=test_dataloader)
    """
    def __init__(self, encoder='resnet18', num_classes=2, optimizer_cfg=None, l1_lambda=0.0):
        super().__init__()

        self.encoder = encoder
        self.model = timm.create_model(encoder, pretrained=True)
        if self.model.default_cfg["input_size"][1] == 3:  # If model expects 3 channels
            self.model.conv1 = torch.nn.Conv2d(
                in_channels=1,  # Change to single channel
                out_channels=self.model.conv1.out_channels,
                kernel_size=self.model.conv1.kernel_size,
                stride=self.model.conv1.stride,
                padding=self.model.conv1.padding,
                bias=False
            )

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.precision = Precision(task="multiclass", num_classes=num_classes)
        self.recall = Recall(task="multiclass", num_classes=num_classes)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes)

        self.l1_lambda = l1_lambda
        if hasattr(self.model, 'fc'):  # For models with 'fc' as the classification layer
            in_features = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(in_features, num_classes)
        elif hasattr(self.model, 'classifier'):  # For models with 'classifier'
            in_features = self.model.classifier.in_features
            self.model.classifier = torch.nn.Linear(in_features, num_classes)
        elif hasattr(self.model, 'head'):  # For models with 'head'
            in_features = self.model.head.in_features
            self.model.head = torch.nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model architecture for encoder: {encoder}")

        if optimizer_cfg is not None:
            optimizer_name = optimizer_cfg.name
            optimizer_lr = optimizer_cfg.lr
            optimizer_weight_decay = optimizer_cfg.weight_decay

            if optimizer_name == 'Adam':
                self.optimizer = optim.Adam(self.parameters(), lr=optimizer_lr, weight_decay=optimizer_weight_decay)
            elif optimizer_name == 'SGD':
                self.optimizer = optim.SGD(self.parameters(), lr=optimizer_lr, weight_decay=optimizer_weight_decay)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        else:
            self.optimizer = None

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()

        # Compute predictions and loss
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)

        # Add L1 regularization
        l1_norm = sum(param.abs().sum() for param in self.parameters())
        loss += self.l1_lambda * l1_norm

        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()

        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        accuracy = self.accuracy(y, preds)
        precision = self.precision(y, preds)
        recall = self.recall(y, preds)
        f1 = self.f1(y, preds)

        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log('val_acc', accuracy, prog_bar=True, on_epoch=True, on_step=True)
        self.log('val_precision', precision, prog_bar=True, on_epoch=True, on_step=True)
        self.log('val_recall', recall, prog_bar=True, on_epoch=True, on_step=True)
        self.log('val_f1', f1, prog_bar=True, on_epoch=True, on_step=True)

        return loss

    def on_validation_epoch_end(self):
        avg_loss = self.trainer.logged_metrics['val_loss_epoch']
        accuracy = self.trainer.logged_metrics['val_acc_epoch']

        self.log('val_loss', avg_loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', accuracy, prog_bar=True, on_epoch=True)

        return {'Average Loss:': avg_loss, 'Accuracy:': accuracy}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        accuracy = self.accuracy(y, preds)
        precision = self.precision(y, preds)
        recall = self.recall(y, preds)
        f1 = self.f1(y, preds)

        # Log test metrics
        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_acc', accuracy, prog_bar=True, logger=True)
        self.log('test_precision', precision, prog_bar=True, logger=True)
        self.log('test_recall', recall, prog_bar=True, logger=True)
        self.log('test_f1', f1, prog_bar=True, logger=True)

        return {'test_loss': loss, 'test_accuracy': accuracy, 'test_precision': precision, 'test_recall': recall, 'test_f1': f1}

