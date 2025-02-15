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
import datetime
from collections import defaultdict

from tensorboard.backend.event_processing.event_processing import EventAccumulator
from tensorboard.backend.event_processing.event_accumulator import ScalarEvent
import os
import datetime
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import matplotlib.pyplot as plt
import torch

def generate_report(log_dir, trainer, model, test_loader, output_dir):
    """Generate comprehensive PDF report using TensorBoard metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load TensorBoard event data
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Extract metrics from TensorBoard logs
    metrics = defaultdict(list)
    for tag in event_acc.Tags()['scalars']:
        events = event_acc.Scalars(tag)
        metrics[tag] = [e.value for e in events]

    # Create visualizations
    plt.style.use('seaborn-v0_8')
    loss_fig = create_loss_plot(metrics)
    metrics_fig = create_metrics_plot(metrics)
    sample_images = generate_sample_predictions(model, test_loader, output_dir)

    # Generate PDF report
    pdf_path = create_pdf_document(
        output_dir=output_dir,
        metrics=metrics,
        loss_fig=loss_fig,
        metrics_fig=metrics_fig,
        sample_images=sample_images,
        model=model
    )
    
    cleanup_temp_files(output_dir, sample_images)
    return pdf_path

def create_loss_plot(metrics):
    """Create combined training/validation loss plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    has_data = False
    
    # Plot training loss
    if 'train_loss' in metrics:
        ax.plot(metrics['train_loss'], 
                label='Training Loss', 
                color='navy', 
                linewidth=2,
                alpha=0.8)
        has_data = True
    
    # Plot validation loss
    if 'val_loss' in metrics:
        ax.plot(metrics['val_loss'], 
                label='Validation Loss', 
                color='darkorange', 
                linewidth=2,
                linestyle='--')
        has_data = True
    
    if has_data:
        ax.set_title('Training and Validation Loss', fontsize=14, pad=20)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        plt.tight_layout()
        return fig
    return None

def create_metrics_plot(metrics):
    """Create validation metrics plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    has_data = False
    
    metric_colors = {
        'val_f1': 'green',
        'val_precision': 'blue',
        'val_recall': 'red'
    }
    
    for metric, color in metric_colors.items():
        if metric in metrics:
            ax.plot(metrics[metric], 
                    label=metric.replace('val_', '').title(), 
                    color=color,
                    linewidth=2,
                    alpha=0.8)
            has_data = True
    
    if has_data:
        ax.set_title('Validation Metrics', fontsize=14, pad=20)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        plt.tight_layout()
        return fig
    return None

def generate_sample_predictions(model, test_loader, output_dir, num_samples=6):
    """Generate sample prediction visualizations"""
    model.eval()
    samples = []
    denormalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            if len(samples) >= num_samples:
                break
            
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            
            for idx in range(x.size(0)):
                if len(samples) >= num_samples:
                    break
                
                img = denormalize(x[idx]).cpu().numpy().transpose(1, 2, 0)
                img = np.clip(img, 0, 1)
                
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(img)
                ax.set_title(f'True: {y[idx].item()}\nPred: {preds[idx].item()}',
                           fontsize=10, 
                           color='green' if y[idx] == preds[idx] else 'red')
                ax.axis('off')
                
                img_path = os.path.join(output_dir, f'sample_{len(samples)}.png')
                plt.savefig(img_path, bbox_inches='tight', dpi=120)
                plt.close()
                samples.append(img_path)
    return samples

def create_pdf_document(output_dir, metrics, loss_fig, metrics_fig, sample_images, model):
    """Create PDF report document"""
    pdf_path = os.path.join(output_dir, 'training_report.pdf')
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    
    # Header Section
    c.setFont('Helvetica-Bold', 16)
    c.drawString(72, height - 72, "Model Training Report")
    c.setFont('Helvetica', 10)
    c.drawString(72, height - 92, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Model Metadata
    metadata_y = height - 130
    c.setFont('Helvetica-Bold', 12)
    c.drawString(72, metadata_y, "Training Summary:")
    c.setFont('Helvetica', 10)
    metadata_items = [
        f"Model Architecture: {model.__class__.__name__}",
        f"Total Epochs Trained: {len(metrics.get('train_loss', []))}",
        f"Best Validation Loss: {min(metrics.get('val_loss', [0])):.4f}",
        f"Final Training Loss: {metrics.get('train_loss', [0])[-1]:.4f}"
    ]
    for item in metadata_items:
        metadata_y -= 20
        c.drawString(72, metadata_y, item)
    
    # Visualizations
    current_y = metadata_y - 40
    
    # Loss Plot
    if loss_fig:
        loss_path = os.path.join(output_dir, 'loss_plot.png')
        loss_fig.savefig(loss_path, bbox_inches='tight')
        c.drawImage(loss_path, 50, current_y - 250, width=500, height=250)
        current_y -= 300
    
    # Metrics Plot
    if metrics_fig:
        metrics_path = os.path.join(output_dir, 'metrics_plot.png')
        metrics_fig.savefig(metrics_path, bbox_inches='tight')
        c.drawImage(metrics_path, 50, current_y - 250, width=500, height=250)
        current_y -= 300
    
    # Sample Predictions
    if sample_images:
        c.setFont('Helvetica-Bold', 12)
        c.drawString(72, current_y - 20, "Sample Predictions:")
        img_y = current_y - 120
        x_offset = 72
        
        for img_path in sample_images:
            c.drawImage(img_path, x_offset, img_y, width=150, height=150)
            x_offset += 160
            if x_offset > 400:
                x_offset = 72
                img_y -= 160
                if img_y < 100:
                    c.showPage()
                    img_y = height - 100
    
    # Footer
    c.setFont('Helvetica-Oblique', 8)
    c.drawCentredString(width/2, 40, "Confidential - Generated by Model Training Framework")
    
    c.save()
    return pdf_path

def cleanup_temp_files(output_dir, sample_images):
    """Clean up temporary visualization files"""
    for f in os.listdir(output_dir):
        if f.startswith('sample_') or f.endswith('_plot.png'):
            os.remove(os.path.join(output_dir, f))

            
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
    # metrics_tracker = MetricsTracker()

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
    log_dir = logger.log_dir

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Evaluate the model
    train_val_metrics = trainer.logged_metrics.copy()  # Save a copy of the metrics
    trainer.test(model, test_loader)
    
    
    # generate pdf report
    generate_report(
        log_dir=log_dir,
        trainer=trainer,
        model=model,
        test_loader=test_loader,
        output_dir=hydra_cfg.runtime.output_dir
    )
    # Save the trained model
    model_path = f"{hydra_cfg.runtime.output_dir}/lora_only_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    import os
    print(f"Model size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()