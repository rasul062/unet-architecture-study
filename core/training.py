import torch
from torch.cuda.amp import autocast
from torch.amp import GradScaler

class EarlyStopping:
  """
    Monitors a specific metric (usually validation loss) to stop training when performance plateaus.
    
    This prevents overfitting by terminating the process if the monitored metric does not 
    improve for a specified number of consecutive checks.
  """
  def __init__(self, patience=5, min_delta=0):
      """
        Initializes the early stopping monitor.
        
        Args:
            patience (int): Number of checks to wait before stopping after the last improvement.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
      """
      self.patience = patience
      self.min_delta = min_delta
      self.counter = 0
      self.best_loss = None
      self.early_stop = False

  def __call__(self, val_loss):
      """
        Updates the internal counter based on the latest validation loss.
        
        Args:
            val_loss (float): The current validation loss value to compare against the best recorded loss.
      """
      if self.best_loss is None:
          # First run initialization
          self.best_loss = val_loss
      elif val_loss > self.best_loss - self.min_delta:
          # Performance has not significantly improved
          self.counter += 1
          print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
          if self.counter >= self.patience:
              self.early_stop = True
      else:
          # New best loss found; reset the patience counter
          self.best_loss = val_loss
          self.counter = 0

class Training:
  """
    A utility class encapsulating the execution logic for training and validation cycles.
  """
  def __init__(self):
    """Initializes the training wrapper."""
    pass

  def train_step(self, model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = None):
    """
        Executes a single training epoch with automatic mixed precision (AMP).
        
        Args:
            model (torch.nn.Module): The neural network to train.
            data_loader (torch.utils.data.DataLoader): The dataset iterator for the training set.
            loss_fn (torch.nn.Module): The criterion used to calculate error.
            optimizer (torch.optim.Optimizer): The optimization algorithm.
            accuracy_fn (callable): Metric function to evaluate classification performance.
            device (torch.device): Hardware target (CPU/CUDA) for computation.
        Returns:
            tuple: (Average train loss, Average train accuracy)
    """
    train_loss, train_acc = 0, 0
    model.train()

    # Initiate GradScaler
    scaler = GradScaler()

    # Add a loop to loop through training batches
    for batch, (X, y) in enumerate(data_loader):
      # Put data on target device
      X, y = X.to(device), y.to(device)

      # Autocast context for mixed precision computation
      with autocast():
        # Forward pass
        y_pred = model(X)

        # Calculate loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss # accumulate train loss
        train_acc += accuracy_fn(y_true=y,
                                y_pred=y_pred.argmax(dim=1)) # logits -> prediction labels

      # Optimizer zero grad
      optimizer.zero_grad()

      # Loss backward
      scaler.scale(loss).backward()

      # Optimizer step
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
    """
        Executes a single validation/inference epoch in evaluation mode.
        
        Args:
            model (torch.nn.Module): The neural network to evaluate.
            data_loader (torch.utils.data.DataLoader): The dataset iterator for the validation set.
            loss_fn (torch.nn.Module): The criterion used to calculate error.
            optimizer (torch.optim.Optimizer): Optimizer reference (unused in validation logic).
            accuracy_fn (callable): Metric function to evaluate classification performance.
            device (torch.device): Hardware target (CPU/CUDA) for computation.
        Returns:
            tuple: (Average validation loss, Average validation accuracy)
    """
    valid_loss, valid_acc = 0, 0

    # Put the model on eval mode
    model.eval()

    # Turn on inference mode context manager
    with torch.inference_mode():
      for X, y in data_loader:
        # Send the data to the target device
        X, y = X.to(device), y.to(device)

        # Forward pass (outputs raw logits)
        valid_pred = model(X)

        # Calculate the loss/acc
        valid_loss += loss_fn(valid_pred, y)
        valid_acc += accuracy_fn(y_true=y,
                                y_pred=valid_pred.argmax(dim=1)) # logits -> prediction labels

      # Adjust metrics and print out
      valid_loss /= len(data_loader)
      valid_acc /= len(data_loader)
      print(f"Valid loss: {valid_loss:.5f} | Valid acc: {valid_acc * 100:.2f}%\n")

      return (valid_loss, valid_acc)
