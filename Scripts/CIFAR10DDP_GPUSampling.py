import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.optim.optimizer import Optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

# -----------------------------------------------------------------------------
# Function: GPU-based sampling from a Levy alpha-stable distribution.
# -----------------------------------------------------------------------------
def sample_levy_alpha_stable(alpha, beta, scale, size, device="cuda"):
    """
    Samples from a Levy alpha-stable distribution using PyTorch for GPU compatibility.
    Ensures numerical stability by using double precision and clipping extreme values.

    Args:
        alpha (float): Stability parameter (0 < alpha <= 2).
        beta (float): Skewness parameter (-1 <= beta <= 1).
        scale (float): Scale parameter (must be positive).
        size (int): Number of samples to generate.
        device (str): Device to use (e.g., "cuda" or "cpu").

    Returns:
        torch.Tensor: Sampled noise of shape (size,), in float32 precision.
    """
    # Use double precision for numerical stability.
    PI = torch.tensor(np.pi, dtype=torch.float64, device=device)
    PI_HALF = PI / 2

    # Generate uniform and exponential random variables.
    U = torch.rand(size, dtype=torch.float64, device=device) * PI - PI_HALF
    W = -torch.log(torch.rand(size, dtype=torch.float64, device=device) + 1e-8)

    # Compute noise samples.
    if alpha != 1:
        numerator = torch.sin(alpha * U)
        denominator = (torch.cos(U) + 1e-8) ** (1 / alpha)  # Avoid division by zero.
        scale_adjustment = (torch.cos(U * (1 - alpha)) / (W + 1e-8)) ** ((1 - alpha) / alpha)
        Z = (numerator / denominator) * scale_adjustment
    else:
        Z = (2 / PI) * ((PI_HALF + beta * U) * torch.tan(U) -
                        beta * torch.log((PI_HALF * W * torch.cos(U)) / (PI_HALF + beta * U + 1e-8) + 1e-8))

    # Scale the noise and convert to float32.
    Z = scale * Z
    Z = Z.float()

    # Clip extreme values to avoid overflow.
    Z = torch.clamp(Z, -1e4, 1e4)  # Adjust bounds as needed.
    assert torch.all(torch.isfinite(Z)), "NaN/Inf detected in noise samples"

    return Z

# -----------------------------------------------------------------------------
# Constants for noise and training hyperparameters.
# -----------------------------------------------------------------------------
LEARNING_RATE = 0.1
CLIP = 10
NOISE_INIT = 0.005
NOISE_DECAY = 0.55
num_epochs = 100

# -----------------------------------------------------------------------------
# Custom Optimiser: AnnealisingOptimiser with momentum support.
# -----------------------------------------------------------------------------
class AnnealisingOptimiser(Optimizer):
    """
    A heavy-tailed optimizer that uses a fixed, non-adaptive update.
    The stability parameter alpha increases on each step according to:
         alpha = 2 - exp(- (k) * step)
    Noise variance decays per step.
    Additionally, this optimizer stores each noise sample (for later plotting).
    """
    def __init__(self,
                 params,
                 lr=1e-3,
                 alpha=1.2,  # initial value; will be updated each step
                 rho=0.05,
                 decay=0.95,
                 noise_init=NOISE_INIT,
                 noise_decay=NOISE_DECAY,
                 weight_decay=0,
                 k=1/5,
                 momentum=0.9,
                 **kwargs):

        defaults = dict(
            lr=lr,
            alpha=alpha,
            rho=rho,
            weight_decay=weight_decay,
            momentum=momentum
        )
        super(AnnealisingOptimiser, self).__init__(params, defaults)

        self.decay = decay
        self.noise_init = noise_init
        self.noise_decay = noise_decay
        self.beta = 0  # symmetric noise (beta = 0)
        self.num_steps = 1  # count of updates (for alpha update)
        self.current_sigma = noise_init  # current noise scale
        self.noise_samples = []  # to store the noise vectors
        self.epoch = 0  # epoch counter for noise decay
        self.k = k

    def update_epoch(self):
        self.epoch += 1

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError("Requires closure for loss computation")
        closure = torch.enable_grad()(closure)
        loss = closure()

        # Update alpha: alpha = 2 - exp(- k * step)
        for group in self.param_groups:
            group['alpha'] = 2 - math.exp(-self.k * self.num_steps)

        # Update noise scale using noise decay (per step)
        self.current_sigma = np.sqrt(self.noise_init) / ((1 + self.num_steps) ** (self.noise_decay / 2))

        # Generate heavy-tailed noise for all parameters using GPU sampling.
        total_elems = sum(p.numel() for group in self.param_groups for p in group['params'])
        device_local = self.param_groups[0]['params'][0].device
        big_noise = sample_levy_alpha_stable(
            alpha=self.param_groups[0]['alpha'],
            beta=self.beta,
            scale=self.current_sigma,
            size=total_elems,
            device=device_local
        )
        # Clip noise to avoid extreme updates.
        big_noise = torch.clamp(big_noise, -CLIP, CLIP)

        # Save noise for later analysis/plotting.
        self.noise_samples.append(big_noise.detach().cpu().numpy())

        idx_start = 0
        for group in self.param_groups:
            lr = group['lr']
            momentum_val = group.get('momentum', 0)
            for p in group['params']:
                if p.grad is None:
                    continue
                # Apply weight decay if needed.
                if group['weight_decay'] != 0:
                    p.grad.data.add_(p.data, alpha=group['weight_decay'])
                # Momentum update.
                if momentum_val != 0:
                    if 'momentum_buffer' not in self.state[p]:
                        buf = p.grad.data.clone()
                    else:
                        buf = self.state[p]['momentum_buffer']
                        buf.mul_(momentum_val).add_(p.grad.data)
                    self.state[p]['momentum_buffer'] = buf
                    grad_update = buf
                else:
                    grad_update = p.grad.data

                elem_count = p.numel()
                noise_slice = big_noise[idx_start:idx_start + elem_count].view_as(p.data)
                idx_start += elem_count
                # Scale noise with a factor that depends on lr and alpha.
                noise_coeff = (lr ** (1 / group['alpha'])) * noise_slice
                p.data.add_(grad_update, alpha=-lr)
                p.data.add_(noise_coeff)
        self.num_steps += 1
        return loss.item()

# -----------------------------------------------------------------------------
# Training function for noise-based optimizers (using closure).
# -----------------------------------------------------------------------------
def train_noise(model, train_loader, optimizer, epoch, train_losses, noise_list, device):
    model.train()
    running_loss = 0.0
    total_batches = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            return loss
        loss_val = optimizer.step(closure=closure)
        running_loss += loss_val
        total_batches += 1
    epoch_loss = running_loss / total_batches
    train_losses.append(epoch_loss)
    noise_list.append(optimizer.current_sigma)
    return epoch_loss

# -----------------------------------------------------------------------------
# Model definition: Modify ResNet18 for CIFAR-10.
# -----------------------------------------------------------------------------
def get_modified_resnet18():
    model = resnet18(num_classes=10)
    # Modify the first convolution and remove maxpool to suit CIFAR-10's 32x32 images.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

# -----------------------------------------------------------------------------
# Main function to run DDP training.
# -----------------------------------------------------------------------------
def main():
    # Initialize the distributed process group.
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    
    # Data preparation.
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform_test)

    # Use DistributedSampler to partition the training data among processes.
    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(),
                                         rank=dist.get_rank(), shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    # -----------------------------------------------------------------------------
    # Model and Optimiser Setup.
    # -----------------------------------------------------------------------------
    model = get_modified_resnet18().to(device)
    # Wrap the model with DDP.
    model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    criterion = nn.CrossEntropyLoss()
    optimizer_anneal = AnnealisingOptimiser(
        model.parameters(),
        lr=LEARNING_RATE,
        alpha=1.7,
        rho=0.05,
        decay=0.95,
        noise_init=NOISE_INIT,
        noise_decay=NOISE_DECAY,
        k=1/5,
        weight_decay=5e-4,
        momentum=0.9
    )
    scheduler_anneal = optim.lr_scheduler.StepLR(optimizer_anneal, step_size=30, gamma=0.1)

    train_losses_anneal = []
    noise_list = []

    # -----------------------------------------------------------------------------
    # Distributed Training Loop.
    # -----------------------------------------------------------------------------
    for epoch in range(num_epochs):
        # Set epoch for the sampler for shuffling.
        train_sampler.set_epoch(epoch)
        epoch_loss = train_noise(model, train_loader, optimizer_anneal, epoch,
                                 train_losses_anneal, noise_list, device)
        
        # Evaluate on test set only on rank 0 to avoid duplicate printing.
        if dist.get_rank() == 0:
            model.eval()
            test_loss = 0
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total_test += targets.size(0)
                    correct_test += predicted.eq(targets).sum().item()
            test_loss /= len(test_loader)
            test_acc = 100. * correct_test / total_test
            print(f"[Rank 0][Anneal] Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {epoch_loss:.4f} || Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
        scheduler_anneal.step()

    # Clean up the distributed process group.
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
