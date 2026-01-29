import torch.nn as nn

class ModelInspector:
    """
    Diagnostic utility for monitoring activation and gradient statistics across neural network layers.
    
    This tool identifies numerical instability, such as dead neurons or gradient 
    abnormalities, by attaching hooks to the model's internal computation graph.
    """
    def __init__(self, model):
        """Initializes the inspector with a target model and storage for statistics."""
        self.model = model
        self.forward_stats = {}
        self.backward_stats = {}
        self.hooks = []

    def _forward_hook(self, name):
        """Creates a hook to capture activation distributions during the forward pass."""
        def hook(module, input, output):
            # Handle potential tuple outputs in complex layer types
            if isinstance(output, tuple): output = output[0]

            # Capture gradient metrics to detect vanishing or exploding gradients
            self.forward_stats[name] = {
                'mean': output.data.mean().item(),
                'std': output.data.std().item(),
                'min': output.data.min().item(),
                'max': output.data.max().item()
            }
        return hook

    def _backward_hook(self, name):
        """Creates a hook to monitor gradient flow and magnitude during backpropagation."""
        def hook(module, grad_input, grad_output):
            # Handle potential tuple outputs in complex layer types
            if isinstance(grad_output, tuple): grad_out = grad_output[0]
            else: grad_out = grad_output

            # Capture gradient metrics to detect vanishing or exploding gradients
            if grad_out is not None:
                self.backward_stats[name] = {
                    'grad_mean': grad_out.data.mean().item(),
                    'grad_max': grad_out.data.abs().max().item()
                }
            else:
                self.backward_stats[name] = "None"
        return hook

    def register_hooks(self):
        """Attaches diagnostic hooks to layers containing learnable parameters (Conv, Linear, BatchNorm)."""
        # Attach to Conv and Linear layers only
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                # Register both forward and backward hooks for comprehensive layer analysis
                self.hooks.append(layer.register_forward_hook(self._forward_hook(name)))
                self.hooks.append(layer.register_full_backward_hook(self._backward_hook(name)))
        print(f"DEBUG: Attached inspectors to {len(self.hooks)//2} layers.")

    def clear_hooks(self):
        """Removes all active hooks to restore original model behavior and free memory."""
        for h in self.hooks: h.remove()
        self.hooks = []

    def print_report(self):
        """Generates a summary report evaluating the numerical stability of the network."""
        print("\n" + "="*60)
        print(f"{'LAYER (Forward)':<30} | {'Mean':<10} | {'Std':<10} | {'Status'}")
        print("-" * 60)

        # Analyze activation health based on mean and standard deviation
        for name, stats in self.forward_stats.items():
            status = "OK"
            # Flag dead layers where all activations have collapsed to zero
            if stats['mean'] == 0 and stats['std'] == 0: status = "DEAD (All Zeros)"
            # Flag exploding activations that may lead to NaN values
            if abs(stats['mean']) > 100: status = "EXPLODING"
            print(f"{name:<30} | {stats['mean']:.4f}     | {stats['std']:.4f}     | {status}")

        print("\n" + "-"*60)
        print(f"{'LAYER (Backward/Gradients)':<30} | {'Grad Mean':<12} | {'Grad Max':<12} | {'Status'}")
        print("-" * 60)
        
        # Reverse order to match gradient flow (Output -> Input)
        for name in reversed(list(self.backward_stats.keys())):
            stats = self.backward_stats[name]
            if stats == "None":
                print(f"{name:<30} | {'None':<12} | {'None':<12} | BROKEN")
                continue

            # Evaluate gradient magnitude against numerical stability thresholds
            status = "OK"
            if stats['grad_max'] == 0: status = "ZERO GRAD"
            elif stats['grad_max'] < 1e-7: status = "VANISHING"
            elif stats['grad_max'] > 100: status = "EXPLODING"

            print(f"{name:<30} | {stats['grad_mean']:.2e}   | {stats['grad_max']:.2e}   | {status}")
        print("="*60 + "\n")
