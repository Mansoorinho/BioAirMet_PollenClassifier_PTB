'''
@file    :   ssl_losses.py
@create date : 2025-06-10 13:36:08
@modify date 2026-04-29 12:15:17
@author  :   Mansoor Nabawi
@version :   1.0
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    This file defines the loss functions used for self-supervised learning (SSL) in the BioAirMet project.
    - ContrastiveLoss: An improved implementation of symmetric cross-entropy loss over similarity matrices. It features enhanced numerical stability through log-space temperature parameters, epsilon-normalized embeddings, and logit clamping to prevent NaN/Inf crashes during training.
    - negative_cosine_similarity: A utility function for computing cosine similarity based losses.
    - build_ssl_loss_from_config: A factory function to instantiate SSL losses with support for learnable temperature parameters.
    ]
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist

try:
    import torch.distributed.nn as dist_nn
    HAS_DIST_NN = True
except ImportError:
    HAS_DIST_NN = False
    
class ContrastiveLoss(nn.Module):
    """
    Improved ContrastiveLoss with numerical stability guarantees.
    Fixes the temperature scaling issue that causes NaN/Inf crashes.
    """
    def __init__(self, temperature=0.5, learnable_temperature=False):
        super().__init__()
        # Store temperature in log space for numerical stability
        self.log_temperature = nn.Parameter(
            torch.tensor(np.log(max(temperature, 0.01))),
            requires_grad=learnable_temperature
        )
        
    @property
    def temperature(self):
        """Get temperature with clamping to prevent extreme values."""
        return torch.clamp(self.log_temperature.exp(), min=0.01, max=10.0).item()
        
    def forward(self, image_embeddings, fl_embeddings):
        """
        Forward pass with numerical stability improvements.
        
        Args:
            image_embeddings: (B, D) tensor
            fl_embeddings: (B, D) tensor
            
        Returns:
            scalar loss value
        """
        B = image_embeddings.shape[0]
        
        # Normalize embeddings with epsilon for numerical stability
        z_i = F.normalize(image_embeddings, p=2, dim=-1, eps=1e-8)
        z_j = F.normalize(fl_embeddings, p=2, dim=-1, eps=1e-8)
        
        # Get temperature with safety clamping
        temperature = torch.clamp(self.log_temperature.exp(), min=0.01, max=10.0)
        
        # Compute similarity matrix: (B, B)
        # similarity ∈ [-1, 1] for normalized vectors
        logits = torch.matmul(z_i, z_j.t())  # ← First compute similarity
        
        # THEN scale by temperature (this prevents extreme values)
        logits = logits / temperature
        
        # Clamp logits to prevent overflow in exp()
        # log(2^30) ≈ 20.7, so we clamp to [-50, 50] for safety
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        
        # Check for NaN/Inf before loss computation
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise RuntimeError(
                f"NaN/Inf detected in logits. "
                f"temp={temperature.item():.6f}, "
                f"logits: min={logits.min()}, max={logits.max()}"
            )
        
        # Create labels: each sample i should match with fl_embedding i
        labels = torch.arange(B, device=image_embeddings.device)
        
        # Cross-entropy loss
        # Image as query: logits[i, j] = sim(image_i, fl_j), we want i==j
        loss_i = F.cross_entropy(logits, labels)
        # FL as query: logits.t()[i, j] = sim(fl_i, image_j), we want i==j
        loss_j = F.cross_entropy(logits.t(), labels)
        
        loss = (loss_i + loss_j) / 2
        
        # Final NaN check
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(
                f"NaN/Inf loss detected. loss_i={loss_i}, loss_j={loss_j}, "
                f"temperature={temperature.item()}"
            )

        return loss
    

def negative_cosine_similarity(p, z):
    # z must be detached before calling this!
    return -F.cosine_similarity(p, z.detach(), dim=-1).mean()


class Clip_ContrastiveLoss(nn.Module):
    def __init__(
        self,
        emulate_old_loss: bool = False,
        local_loss: bool = False,
        gather_with_grad: bool = True,  # ✅ better default
    ):
        super().__init__()
        self.emulate_old_loss = emulate_old_loss
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad

    def _get_dist_info(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def gather_features(self, image_features, fl_features, rank, world_size):
        if world_size == 1:
            return image_features, fl_features
        
        # Detach to avoid gradient flow through all_gather
        # image_features = image_features.detach()
        # fl_features = fl_features.detach()

        if self.gather_with_grad:
            assert HAS_DIST_NN
            gathered_img = dist_nn.all_gather(image_features)
            gathered_fl = dist_nn.all_gather(fl_features)
            all_image_features = torch.cat(list(gathered_img), dim=0)
            all_fl_features = torch.cat(list(gathered_fl), dim=0)
        else:
            image_list = [torch.empty_like(image_features) for _ in range(world_size)]
            fl_list  = [torch.empty_like(fl_features) for _ in range(world_size)]

            dist.all_gather(image_list, image_features)
            dist.all_gather(fl_list, fl_features)

            image_list[rank] = image_features
            fl_list[rank] = fl_features

            all_image_features = torch.cat(image_list, dim=0)
            all_fl_features = torch.cat(fl_list, dim=0)

        return all_image_features, all_fl_features

    def forward(self, image_features, fl_features, logit_scale):
        rank, world_size = self._get_dist_info()

        # --- Old loss ---
        if self.emulate_old_loss or world_size == 1:
            logits = logit_scale * (image_features @ fl_features.t())
            labels = torch.arange(logits.size(0), device=logits.device)

            loss_i = F.cross_entropy(logits, labels)
            loss_f = F.cross_entropy(logits.t(), labels)
            return 0.5 * (loss_i + loss_f)

        # --- Distributed ---
        all_image_features, all_fl_features = self.gather_features(
            image_features, fl_features, rank, world_size
        )

        if self.local_loss:
            logits_per_image = logit_scale * (image_features @ all_fl_features.t())
            logits_per_fl  = logit_scale * (fl_features @ all_image_features.t())

            batch_size = image_features.shape[0]
            labels = torch.arange(batch_size, device=image_features.device)
            labels = labels + batch_size * rank
        else:
            logits_per_image = logit_scale * (all_image_features @ all_fl_features.t())
            logits_per_fl  = logits_per_image.t()

            labels = torch.arange(logits_per_image.size(0), device=image_features.device)

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_f = F.cross_entropy(logits_per_fl, labels)

        return 0.5 * (loss_i + loss_f)
    
# Factory function to build SSL loss from config
def build_ssl_loss_from_config(config):
    """
    Builds an SSL loss function instance based on the provided configuration.

    Args:
        config (EasyDict): Configuration object.

    Returns:
        torch.nn.Module: An instance of the specified SSL loss function.
    """
    loss_name = config.train.loss.get('name', 'ContrastiveLoss') # Default to ContrastiveLoss
    
    if loss_name == "ContrastiveLoss":
        temperature = config.train.loss.get('temperature', 0.1)
        learnable_temp = config.train.loss.get('learnable_temperature', False)
        return ContrastiveLoss(temperature=temperature, learnable_temperature=learnable_temp)
    
    else:
        raise ValueError(f"Unknown SSL loss function name: {loss_name}")