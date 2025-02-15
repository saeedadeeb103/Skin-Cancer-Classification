from torchvision import transforms
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
import hydra
from datasets import list_datasets, get_dataset_by_name
from encoders.encoders import timm_backbones
from torchaudio import transforms as T
from hydra.core.hydra_config import HydraConfig
from utils.helper_functions import collate_fn
import numpy as np
import random
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import os


def generate_report(metrics, trainer, model, test_loader, output_dir):
    """
    Generates a detailed PDF report with training metrics, validation metrics, and sample predictions.

    Args:
        metrics (dict): Dictionary containing training and validation metrics.
        trainer (pl.Trainer): PyTorch Lightning Trainer object.
        model (pl.LightningModule): Trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        output_dir (str): Directory to save the report and plots.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    if 'train_loss' in metrics:
        plt.plot(metrics['train_loss'], label='Train Loss', color='blue', linewidth=2)
    if 'val_loss' in metrics:
        plt.plot(metrics['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    loss_plot_path = os.path.join(output_dir, 'loss_plot.png')
    plt.savefig(loss_plot_path, bbox_inches='tight', dpi=300)
    plt.close()

    # Plot F1, Recall, Precision
    plt.figure(figsize=(10, 5))
    if 'val_f1' in metrics:
        plt.plot(metrics['val_f1'], label='F1 Score', color='green', linewidth=2)
    if 'val_recall' in metrics:
        plt.plot(metrics['val_recall'], label='Recall', color='red', linewidth=2)
    if 'val_precision' in metrics:
        plt.plot(metrics['val_precision'], label='Precision', color='purple', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Validation Metrics', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    metrics_plot_path = os.path.join(output_dir, 'metrics_plot.png')
    plt.savefig(metrics_plot_path, bbox_inches='tight', dpi=300)
    plt.close()

    # Save some sample images with predictions
    model.eval()
    sample_images = []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i >= 5:  # Save only 5 samples
                break
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            for j in range(x.size(0)):
                img = x[j].cpu().numpy().transpose(1, 2, 0)
                img = (img * 255).astype('uint8')
                plt.figure(figsize=(5, 5))
                plt.imshow(img)
                plt.title(f'GT: {y[j].item()}, Pred: {preds[j].item()}', fontsize=10)
                plt.axis('off')
                img_path = os.path.join(output_dir, f'sample_{i}_{j}.png')
                plt.savefig(img_path, bbox_inches='tight', dpi=150)
                plt.close()
                sample_images.append(img_path)

    # Generate PDF
    pdf_path = os.path.join(output_dir, 'report.pdf')
    c = canvas.Canvas(pdf_path, pagesize=letter)

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Training Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, 730, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Add loss plot
    if os.path.exists(loss_plot_path):
        c.drawImage(loss_plot_path, 100, 500, width=400, height=200)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(100, 480, "Training and Validation Loss")

    # Add metrics plot
    if os.path.exists(metrics_plot_path):
        c.drawImage(metrics_plot_path, 100, 250, width=400, height=200)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(100, 230, "Validation Metrics")

    # Add sample images
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, 200, "Sample Predictions")
    y_offset = 180
    for img_path in sample_images:
        if os.path.exists(img_path):
            c.drawImage(img_path, 100, y_offset, width=100, height=100)
            y_offset -= 120

    # Add summary statistics
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, y_offset - 20, "Summary Statistics")
    c.setFont("Helvetica", 10)
    if 'train_loss' in metrics:
        c.drawString(100, y_offset - 40, f"Final Training Loss: {metrics['train_loss'][-1]:.4f}")
    if 'val_loss' in metrics:
        c.drawString(100, y_offset - 60, f"Final Validation Loss: {metrics['val_loss'][-1]:.4f}")
    if 'val_f1' in metrics:
        c.drawString(100, y_offset - 80, f"Final Validation F1 Score: {metrics['val_f1'][-1]:.4f}")
    if 'val_recall' in metrics:
        c.drawString(100, y_offset - 100, f"Final Validation Recall: {metrics['val_recall'][-1]:.4f}")
    if 'val_precision' in metrics:
        c.drawString(100, y_offset - 120, f"Final Validation Precision: {metrics['val_precision'][-1]:.4f}")

    c.save()

    print(f"Report generated at {pdf_path}")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    hydra_cfg = HydraConfig.get()

    # Determine dataset name based on input type
    dataset_name = cfg.dataset_name

    # Print available datasets for debugging
    print(f"Available datasets: {list_datasets()}")
    print(f"Using dataset: {dataset_name}")

    # Define appropriate transformations
    try:
        target_size = tuple(cfg.dataset.target_size)
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = get_dataset_by_name(dataset_name, root_path=cfg.dataset.root_path, subset="train", transform=transform)
        val_dataset = get_dataset_by_name(dataset_name, root_path=cfg.dataset.root_path, subset="val", transform=transform)
        test_dataset = get_dataset_by_name(dataset_name, root_path=cfg.dataset.root_path, subset="test", transform=transform)
    except Exception:
        raise ValueError(f"Unsupported input_type: {cfg.input_type}")

    from encoders.encoders import timm_backbones
    import os
    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=cfg.batch_size, num_workers=min(os.cpu_count(), 7))
    val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=cfg.batch_size, num_workers=min(os.cpu_count(), 7))
    test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=cfg.batch_size, num_workers=min(os.cpu_count(), 7))
    model = timm_backbones(
        encoder=cfg.model.encoder,
        num_classes=cfg.num_classes,
        optimizer_cfg=cfg.model.optimizer,
    )
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath=f"{hydra_cfg.runtime.output_dir}/checkpoints/",
        filename="best_model"
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )

    # Define logger
    logger = TensorBoardLogger(save_dir="logs", name="outputloggs")

    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        min_epochs=cfg.min_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        precision=16 if torch.cuda.is_available() else 32,  # Mixed precision training
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accumulate_grad_batches=2  # Gradient accumulation for memory efficiency
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Evaluate the model
    train_val_metrics = trainer.logged_metrics.copy()  # Save a copy of the metrics
    trainer.test(model, test_loader)
    
    
    # generate pdf report
    generate_report(train_val_metrics, trainer, model, test_loader, hydra_cfg.runtime.output_dir)
    # Save the trained model
    model_path = f"{hydra_cfg.runtime.output_dir}/lora_only_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    import os
    print(f"Model size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()