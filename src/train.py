import torch
import logging
import torch.nn as nn

log = logging.getLogger(__name__)

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        output = model(data)
        
        loss = criterion(output, target)
        
        # Backward
        loss.backward()
        
        # Step
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
    avg_loss = total_loss / len(loader)
    acc = 100. * correct / total
    
    return avg_loss, acc

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
    avg_loss = total_loss / len(loader)
    acc = 100. * correct / total
    
    return avg_loss, acc


def train(model, train_loader, test_loader, criterion, optimizer, run_manager, start_epoch, epochs, device):
    # 7. Training Loop
    for epoch in range(start_epoch, epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        log.info(f"Epoch {epoch}: Train Loss={train_loss:.4f} Acc={train_acc:.2f}%")
        
        # Validation
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        log.info(f"Epoch {epoch}: Val Loss={val_loss:.4f} Acc={val_acc:.2f}%")

        # Save
        run_manager.save_ckpt(model, optimizer, epoch, loss=train_loss)
        
        # Save Best
        run_manager.save_best_ckpt(model, optimizer, epoch, val_acc, mode='max')


def pretrain_geometric_encoder(model, data_loader, device, epochs=50, lr=1e-3, orth_lambda=1.0):
    """
    Pretrains the GeometricEncoder using reconstruction and orthogonality losses.
    
    Args:
        model: The GeometricEncoder instance.
        data_loader: DataLoader returning (x, _) pairs.
        device: torch.device.
        epochs (int): Number of pretraining epochs.
        lr (float): Learning rate.
        orth_lambda (float): Weight for the orthogonality loss.
    
    Returns:
        model: The pretrained model.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss_fn = nn.MSELoss()
    
    model.train()
    log.info(f"Starting GeometricEncoder pretraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss_avg = 0
        
        for batch_idx, (x, _) in enumerate(data_loader):
            # x shape: (Batch, D)
            x = x.to(device)
            
            # Forward
            z, x_recon = model(x)
            
            # Loss 1: Reconstruction (PCA objective)
            # Minimizing reconstruction error maximizes variance captured
            loss_rec = mse_loss_fn(x_recon, x)
            
            # Loss 2: Orthogonality (Stiefel/Grassmannian constraint)
            loss_orth = model.get_orthogonality_loss()
            
            # Total Loss
            loss = loss_rec + (orth_lambda * loss_orth)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss_avg += loss.item()
            
        avg_loss = total_loss_avg / len(data_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info(f"Pretrain Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")

    return model


