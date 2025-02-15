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

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.units import inch
from hydra.core.hydra_config import HydraConfig

def generate_report(log_dir, model, test_loader, output_dir, cfg):
    """Generate comprehensive training report with all essential elements"""
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metrics from TensorBoard
    metrics = load_tensorboard_metrics(log_dir)
    
    # Create document structure
    doc = SimpleDocTemplate(
        os.path.join(output_dir, "training_report.pdf"),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Create story elements
    story = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontSize=18,
        spaceAfter=14,
        textColor=colors.HexColor('#2B3A67')
    )
    
    # Add header
    story.append(Paragraph("Training Report Summary", title_style))
    story.append(Spacer(1, 12))
    
    # Add metadata section
    story += create_metadata_section(model, cfg, styles)
    story.append(Spacer(1, 24))
    
    # Add hyperparameters table
    story += create_hyperparameters_table(cfg, styles)
    story.append(PageBreak())
    
    # Add training curves
    story += create_training_curves_section(metrics, output_dir, styles)
    story.append(PageBreak())
    
    # Add sample predictions
    story += create_predictions_section(model, test_loader, output_dir, styles)
    
    # Build final PDF
    doc.build(story)
    cleanup_temp_files(output_dir)
    return os.path.join(output_dir, "training_report.pdf")

def load_tensorboard_metrics(log_dir):
    """Load and parse TensorBoard event files"""
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    metrics = {}
    for tag in event_acc.Tags()['scalars']:
        events = event_acc.Scalars(tag)
        metrics[tag] = [e.value for e in events]
    return metrics

def create_metadata_section(model, cfg, styles, metrics):  # Add metrics parameter
    """Create model metadata section"""
    elements = []
    meta_style = ParagraphStyle(
        'Meta',
        parent=styles['BodyText'],
        fontSize=10,
        textColor=colors.HexColor('#4A4A4A')
    )
    
    metadata = [
        f"<b>Report Date:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"<b>Model Architecture:</b> {model.__class__.__name__}",
        f"<b>Training Duration:</b> {len(metrics.get('train_loss', []))} epochs",
        f"<b>Best Validation Loss:</b> {min(metrics.get('val_loss', [0])):.4f}",
        f"<b>Final Training Loss:</b> {metrics.get('train_loss', [0])[-1]:.4f}"
    ]
    
    for item in metadata:
        elements.append(Paragraph(item, meta_style))
        elements.append(Spacer(1, 4))
    
    return elements

def create_hyperparameters_table(cfg, styles):
    """Create formatted hyperparameters table"""
    elements = []
    header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['BodyText'],
        fontSize=12,
        textColor=colors.HexColor('#FFFFFF'),
        alignment=1
    )
    
    # Flatten Hydra config
    params = []
    for section in ['model', 'dataset', 'training']:
        if hasattr(cfg, section):
            params.extend([
                (f"<b>{key}</b>", str(value))
                for key, value in getattr(cfg, section).items()
            ])
    
    # Create table data
    table_data = [
        [Paragraph('Hyperparameter', header_style), 
         Paragraph('Value', header_style)]
    ] + params
    
    # Create table
    table = Table(table_data, colWidths=[2.5*inch, 3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#3A5A40')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8F9FA')),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#DEE2E6')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    
    elements.append(Paragraph("<b>Training Configuration</b>", styles['Heading2']))
    elements.append(Spacer(1, 12))
    elements.append(table)
    return elements

def create_training_curves_section(metrics, output_dir, styles):
    """Create section with training curves"""
    elements = []
    
    # Create and save plots
    loss_path = os.path.join(output_dir, 'loss_curves.png')
    create_combined_loss_plot(metrics).savefig(loss_path, bbox_inches='tight')
    
    metrics_path = os.path.join(output_dir, 'validation_metrics.png')
    create_metrics_plot(metrics).savefig(metrics_path, bbox_inches='tight')
    
    # Add to report
    elements.append(Paragraph("<b>Training Curves</b>", styles['Heading2']))
    elements.append(Spacer(1, 12))
    
    # Create two-column layout
    img_table = Table([
        [Image(loss_path, width=3.5*inch, height=2.5*inch),
         Image(metrics_path, width=3.5*inch, height=2.5*inch)]
    ], colWidths=[4*inch, 4*inch])
    
    elements.append(img_table)
    elements.append(Spacer(1, 24))
    
    # Add metric explanations
    elements.append(Paragraph("<i>Training Metrics Legend:</i>", styles['Italic']))
    elements.append(Paragraph(
        "• <b>Loss:</b> CrossEntropyLoss with label smoothing", 
        styles['BodyText']
    ))
    elements.append(Paragraph(
        "• <b>Validation Metrics:</b> Calculated on full validation set after each epoch", 
        styles['BodyText']
    ))
    
    return elements

def create_predictions_section(model, test_loader, output_dir, styles):
    """Create section with sample predictions"""
    elements = []
    elements.append(Paragraph("<b>Sample Predictions</b>", styles['Heading2']))
    elements.append(Spacer(1, 12))
    
    # Generate prediction images
    sample_images = generate_organized_predictions(model, test_loader, output_dir)
    
    # Create image grid (3 per row)
    rows = []
    current_row = []
    for img_path in sample_images:
        current_row.append(Image(img_path, width=1.8*inch, height=1.8*inch))
        if len(current_row) == 3:
            rows.append(current_row)
            current_row = []
    if current_row:
        rows.append(current_row)
    
    # Create prediction table
    pred_table = Table(rows, colWidths=[2*inch]*3)
    pred_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    
    elements.append(pred_table)
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(
        "<i>Note: Green labels indicate correct predictions, red indicates errors</i>",
        styles['Italic']
    ))
    
    return elements

def create_combined_loss_plot(metrics):
    """Create professional loss curves plot"""
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'train_loss' in metrics:
        ax.plot(metrics['train_loss'], 
                label='Training Loss', 
                linewidth=2,
                color='#2B3A67',
                alpha=0.9)
        
    if 'val_loss' in metrics:
        ax.plot(metrics['val_loss'], 
                label='Validation Loss', 
                linewidth=2,
                color='#7EBDC2',
                linestyle='--')
        
    ax.set_title('Training and Validation Loss', fontsize=14, pad=15)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    return fig

def create_metrics_plot(metrics):
    """Create validation metrics plot"""
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_styles = {
        'val_f1': {'color': '#3A5A40', 'label': 'F1 Score'},
        'val_precision': {'color': '#7EBDC2', 'label': 'Precision'},
        'val_recall': {'color': '#FF6B35', 'label': 'Recall'}
    }
    
    for metric, style in metric_styles.items():
        if metric in metrics:
            ax.plot(metrics[metric], 
                    label=style['label'], 
                    linewidth=2,
                    color=style['color'])
            
    ax.set_title('Validation Metrics', fontsize=14, pad=15)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    return fig

def generate_organized_predictions(model, test_loader, output_dir, num_samples=9):
    """Generate organized prediction visualizations"""
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
                
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(img)
                ax.set_title(
                    f'True: {y[idx].item()}\nPred: {preds[idx].item()}',
                    fontsize=9,
                    color='#3A5A40' if y[idx] == preds[idx] else '#FF6B35'
                )
                ax.axis('off')
                plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05)
                
                img_path = os.path.join(output_dir, f'pred_{len(samples)}.png')
                plt.savefig(img_path, dpi=120)
                plt.close()
                samples.append(img_path)
    return samples

def cleanup_temp_files(output_dir):
    """Clean temporary visualization files"""
    for f in os.listdir(output_dir):
        if f.startswith('pred_') or f.endswith(('_curves.png', '_metrics.png')):
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

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Evaluate the model
    train_val_metrics = trainer.logged_metrics.copy()  # Save a copy of the metrics
    trainer.test(model, test_loader)
    
    
    # generate pdf report
    report_path = generate_report(
        log_dir=logger.log_dir,
        model=model,
        test_loader=test_loader,
        output_dir=hydra_cfg.runtime.output_dir,
        cfg=cfg  # Pass your Hydra config object
    )

    print(f"Generated comprehensive report at: {report_path}")
    # Save the trained model
    model_path = f"{hydra_cfg.runtime.output_dir}/lora_only_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    import os
    print(f"Model size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()