import torch
from torch.cuda.amp import autocast
from torch.amp import GradScaler

class EarlyStopping:
  def __init__(self, patience=5, min_delta=0):
      self.patience = patience
      self.min_delta = min_delta
      self.counter = 0
      self.best_loss = None
      self.early_stop = False

  def __call__(self, val_loss):
      if self.best_loss is None:
          self.best_loss = val_loss
      elif val_loss > self.best_loss - self.min_delta:
          self.counter += 1
          print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
          if self.counter >= self.patience:
              self.early_stop = True
      else:
          self.best_loss = val_loss
          self.counter = 0

class Training:
  def __init__(self):
    pass

  def train_step(self, model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = None):
    """Performs a training with model trying to learn on data_loader"""
    train_loss, train_acc = 0, 0
    model.train()

    # Initiate GradScaler
    scaler = GradScaler()

    # Add a loop to loop through training batches
    for batch, (X, y) in enumerate(data_loader):
      # Put data on target device
      X, y = X.to(device), y.to(device)

      with autocast():
        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss # accumulate train loss
        train_acc += accuracy_fn(y_true=y,
                                y_pred=y_pred.argmax(dim=1)) # logits -> prediction labels

      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backward
      scaler.scale(loss).backward()

      # 5. Optimizer step
      scaler.step(optimizer)
      scaler.update()

      # Print out what's happening
      if batch % 400 == 0:
        print(f"Looked at {batch * len(X)}/{len(data_loader.dataset)} samples")

    # Divide total train loss and accuracy by length of train dataloader
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc * 100:.2f}%\n")

    return (train_loss, train_acc)

  def valid_step(self, model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              accuracy_fn,
              device: torch.device = None):
    """Performs a testing loop step on model going over data_loader"""
    valid_loss, valid_acc = 0, 0

    # Put the model on eval mode
    model.eval()

    # Turn on inference mode context manager
    with torch.inference_mode():
      for X, y in data_loader:
        # Send the datat to the target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass (outputs raw logits)
        valid_pred = model(X)

        # 2. Calculate the loss/acc
        valid_loss += loss_fn(valid_pred, y)
        valid_acc += accuracy_fn(y_true=y,
                                y_pred=valid_pred.argmax(dim=1)) # logits -> prediction labels

      # Adjust metrics and print out
      valid_loss /= len(data_loader)
      valid_acc /= len(data_loader)
      print(f"Valid loss: {valid_loss:.5f} | Valid acc: {valid_acc * 100:.2f}%\n")

      return (valid_loss, valid_acc)
