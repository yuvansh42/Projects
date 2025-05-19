#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FINAL MULTI-MODAL NEURAL NETWORK PROJECT
========================================
This script implements a state-of-the-art, versatile deep neural network
in PyTorch to process three data modalities:
  1) Tabular/structured data,
  2) Images,
  3) Text sequences (via a Transformer encoder).

Key features in this architecture:
  - Residual connections in both the CNN (ResNet-like) and the MLP.
  - Batch Normalization and Dropout to stabilize training and reduce overfitting.
  - Transformer-based encoder for text input with multi-head self-attention.
  - Late fusion of modality-specific embeddings (concatenation) followed by an MLP.
  - AdamW optimizer with an optional learning rate scheduler.
  - Gradient clipping (optional) to handle exploding gradients in very deep networks.
  - Proper weight initialization (He or Xavier) for newly created layers.

Sections in this script:
  A) Imports and global configurations
  B) Encoders:
       1) TabularEncoder
       2) SimpleResNet (or optionally you can replace with torchvision ResNet)
       3) TextTransformerEncoder
  C) MultiModalNet (fusion of encoders)
  D) Example dataset creation (synthetic) for demonstration
  E) Training and evaluation functions
  F) Main entry point: usage example

Usage:
  1. Ensure you have Python 3.x, PyTorch, torchvision, etc. installed:
       pip install torch torchvision
       # For advanced text: pip install transformers (if you want huggingface usage)
  2. Run this script directly:
       python final_multimodal_project.py
  3. Modify or replace the synthetic data generation with your real dataset loading logic.
  4. Customize hyperparameters, network sizes, etc., to suit your project’s needs.
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torchvision import transforms

# Optional: set a manual seed for reproducibility (comment out if truly random is desired)
# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)

###############################################################################
# A) Global Configurations (Adjust as needed)
###############################################################################

# Device configuration: use GPU if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters / Setup
TABULAR_NUM_FEATURES = 10  # Example: number of features in tabular data
NUM_IMAGE_CHANNELS = 3     # Usually 3 for RGB images
IMAGE_SIZE = 64            # We'll assume 64x64 images in synthetic example
VOCAB_SIZE = 1000          # For text token IDs in synthetic example
MAX_SEQ_LEN = 16           # Synthetic sequence length
EMBED_DIM = 128            # Embedding dimension for text tokens
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
EPOCHS = 5
FUSION_DIM = 256           # Dimension of fused representation
OUTPUT_DIM = 4             # Example: classification into 4 classes

###############################################################################
# B) Definition of Encoder Modules
###############################################################################
#
# We provide three separate encoders for the three modalities:
#  1) TabularEncoder (MLP)
#  2) SimpleResNet (CNN for images)
#  3) TextTransformerEncoder (Transformer for textual input)
#
# Each encoder outputs a feature vector that will later be combined (fused).
###############################################################################


###############################################################################
# B.1) TabularEncoder
###############################################################################

class ResidualBlockMLP(nn.Module):
    """
    A simple residual block for MLP: 
        x -> Linear -> BN -> ReLU -> Linear -> BN + skip connection -> ReLU
    """
    def __init__(self, hidden_dim):
        super(ResidualBlockMLP, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.bn2(out)
        # Add skip connection
        out += identity
        out = self.activation(out)
        return out


class TabularEncoder(nn.Module):
    """
    An MLP-based encoder for tabular data with optional:
      - BatchNorm
      - Residual connections
      - Dropout
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, use_residual=True, dropout=0.1):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Dimension of hidden layers in the MLP.
            num_layers (int): How many linear layers (besides potential residual blocks).
            use_residual (bool): Whether to use residual blocks in between.
            dropout (float): Dropout rate.
        """
        super(TabularEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.dropout = nn.Dropout(dropout)

        # Initial linear layer to go from input_dim -> hidden_dim
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.bn_in = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.ReLU(inplace=True)

        # Residual blocks or linear layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if use_residual:
                self.layers.append(ResidualBlockMLP(hidden_dim))
            else:
                # Simple linear + BN + ReLU block
                block = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True)
                )
                self.layers.append(block)

        # Output dimension will be hidden_dim
        self.output_dim = hidden_dim

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        # Xavier or Kaiming initialization for linear layers
        nn.init.kaiming_normal_(self.fc_in.weight, nonlinearity='relu')
        if self.fc_in.bias is not None:
            nn.init.zeros_(self.fc_in.bias)

    def forward(self, x):
        """
        x shape: (batch_size, input_dim) for tabular data
        """
        # Initial transform
        out = self.fc_in(x)
        out = self.bn_in(out)
        out = self.activation(out)
        out = self.dropout(out)

        # Pass through each block
        for block in self.layers:
            out = block(out)
            out = self.dropout(out)

        return out


###############################################################################
# B.2) SimpleResNet for Images
###############################################################################

class BasicBlock(nn.Module):
    """
    A basic ResNet-like block with two convolutions and a skip connection.
    This is a simplified version for demonstration (kernel_size=3, etc.).
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If stride != 1 or in_channels != out_channels, adjust the skip path:
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SimpleResNet(nn.Module):
    """
    A simplified ResNet-style CNN for image feature extraction.
    The final output is a vector from global average pooling, which will be
    used as the image embedding.
    """
    def __init__(self, layers=[2,2], base_channels=32):
        """
        Args:
            layers (list): Number of blocks in each layer (list of length 2 or 3).
            base_channels (int): Number of filters in the first layer.
        Note: This is not the full ResNet-50 or etc., but demonstrates
              the concept of a deep CNN with residual blocks.
        """
        super(SimpleResNet, self).__init__()
        self.in_channels = base_channels

        # Initial conv/bn
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Construct residual blocks
        self.layer1 = self._make_layer(base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(base_channels*2, layers[1], stride=2)

        # If you want more layers, define self.layer3, self.layer4 similarly

        # We'll do a final global average pooling & flatten
        # The channel dimension after layer2 is base_channels*2
        # The final feature dimension after GAP is base_channels*2
        self.output_dim = base_channels * 2

        # We might add a final linear layer if desired
        # but typically we just pass the pooled feature to the next stage.

        self._init_weights()

    def _init_weights(self):
        # Kaiming initialization for convolution weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, out_channels, blocks, stride):
        """
        Creates a stack of BasicBlock residual blocks.
        """
        layers = []
        # First block can have stride for downsampling
        layers.append(BasicBlock(self.in_channels, out_channels, stride=stride))
        self.in_channels = out_channels
        # The remaining blocks have stride=1
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        x shape: (batch_size, 3, H, W)
        """
        # Initial conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool1(out)

        # Residual blocks
        out = self.layer1(out)
        out = self.layer2(out)

        # Global average pool
        out = F.adaptive_avg_pool2d(out, (1, 1))  # shape: (batch_size, channels, 1, 1)
        out = out.view(out.size(0), -1)  # shape: (batch_size, channels)

        return out


###############################################################################
# B.3) TextTransformerEncoder
###############################################################################

class TextTransformerEncoder(nn.Module):
    """
    A text encoder using PyTorch's built-in TransformerEncoder.
    We'll embed token IDs, apply positional encoding, then pass through
    multi-head self-attention blocks, and finally pool the output to produce
    a fixed-size text representation.
    """
    def __init__(self, vocab_size, embed_dim=128, max_seq_len=32,
                 num_layers=2, num_heads=4, feedforward_dim=256,
                 dropout=0.1):
        """
        Args:
            vocab_size (int): Size of the vocabulary for token embedding.
            embed_dim (int): Dimension of token embeddings.
            max_seq_len (int): Maximum sequence length for positional encoding.
            num_layers (int): Number of transformer encoder layers.
            num_heads (int): Number of attention heads in each layer.
            feedforward_dim (int): Hidden dim of the Transformer feed-forward network.
            dropout (float): Dropout rate.
        """
        super(TextTransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        # Positional encoding buffer (we'll create a matrix of shape [max_seq_len, embed_dim])
        self.pos_encoding = self._create_pos_encoding(max_seq_len, embed_dim)

        # TransformerEncoder requires EncoderLayer objects
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=feedforward_dim,
            dropout=dropout, 
            activation='relu',
            batch_first=True  # Ensures shape is (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )

        # We'll do a simple "CLS pooling" or "mean pooling" to get final embedding
        # For clarity, let's do mean pooling across sequence dimension
        self.output_dim = embed_dim

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)

    def _create_pos_encoding(self, max_len, d_model):
        """
        Standard sinusoidal positional encoding as in "Attention is All You Need".
        Returns a buffer of shape (max_len, d_model).
        """
        pos_enc = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        denominators = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pos_enc[:, 0::2] = torch.sin(positions * denominators)
        pos_enc[:, 1::2] = torch.cos(positions * denominators)
        return nn.Parameter(pos_enc, requires_grad=False)

    def forward(self, token_ids):
        """
        token_ids shape: (batch_size, seq_len), each entry is a token index in [0, vocab_size-1].
        """
        # 1) Embed tokens
        x = self.token_embed(token_ids)  # shape: (batch_size, seq_len, embed_dim)
        seq_len = x.shape[1]

        # 2) Add positional encoding
        # we slice pos_encoding to match actual seq_len
        x = x + self.pos_encoding[:seq_len, :]

        # 3) Pass through the Transformer Encoder
        # If needed, we can supply src_key_padding_mask or attention_mask
        x = self.transformer_encoder(x)  # shape: (batch_size, seq_len, embed_dim)

        # 4) Pool the sequence output to get a fixed-size representation
        # We'll do mean pooling over the seq_len dimension:
        out = torch.mean(x, dim=1)  # shape: (batch_size, embed_dim)

        return out


###############################################################################
# C) MultiModalNet
###############################################################################

class MultiModalNet(nn.Module):
    """
    A flexible multi-modal network that fuses outputs from:
      - A tabular encoder (MLP)
      - An image encoder (CNN)
      - A text encoder (Transformer)
    Then it concatenates these embeddings and processes them through
    a fusion MLP to produce a final output (e.g. for classification).
    """
    def __init__(self, 
                 tabular_encoder: nn.Module, 
                 image_encoder: nn.Module, 
                 text_encoder: nn.Module,
                 fusion_dim=256, 
                 output_dim=4, 
                 dropout=0.2):
        """
        Args:
            tabular_encoder (nn.Module): e.g. TabularEncoder instance
            image_encoder   (nn.Module): e.g. SimpleResNet instance
            text_encoder    (nn.Module): e.g. TextTransformerEncoder instance
            fusion_dim (int): Dimensionality of the hidden layer after concatenation.
            output_dim (int): The size of the final output (e.g. num classes).
            dropout (float): Dropout probability in the fusion MLP.
        """
        super(MultiModalNet, self).__init__()
        self.tabular_enc = tabular_encoder
        self.image_enc = image_encoder
        self.text_enc = text_encoder

        # Compute total embedding size from each encoder
        t_dim = getattr(self.tabular_enc, 'output_dim', 0)
        i_dim = getattr(self.image_enc, 'output_dim', 0)
        x_dim = getattr(self.text_enc, 'output_dim', 0)
        self.concat_dim = t_dim + i_dim + x_dim

        # Fusion MLP
        self.fusion_fc1 = nn.Linear(self.concat_dim, fusion_dim)
        self.bn_fusion = nn.BatchNorm1d(fusion_dim)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.fusion_fc2 = nn.Linear(fusion_dim, output_dim)
        # For classification with cross-entropy, we typically output raw logits
        # and apply softmax in the training loop / or rely on CrossEntropyLoss.

        # Weight init
        nn.init.kaiming_normal_(self.fusion_fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fusion_fc1.bias)
        nn.init.xavier_uniform_(self.fusion_fc2.weight)
        nn.init.zeros_(self.fusion_fc2.bias)

    def forward(self, tabular_input, image_input, text_input):
        """
        Perform forward pass on the three inputs:
            tabular_input: shape (batch_size, T_features)
            image_input:   shape (batch_size, 3, H, W)
            text_input:    shape (batch_size, seq_len)
        Returns:
            logits of shape (batch_size, output_dim)
        """
        # Encode each modality
        tab_feat = self.tabular_enc(tabular_input)  # (B, t_dim)
        img_feat = self.image_enc(image_input)      # (B, i_dim)
        txt_feat = self.text_enc(text_input)        # (B, x_dim)

        # Concatenate
        fused = torch.cat([tab_feat, img_feat, txt_feat], dim=1)  # (B, concat_dim)

        # Pass through fusion MLP
        out = self.fusion_fc1(fused)   # (B, fusion_dim)
        out = self.bn_fusion(out)
        out = self.activation(out)
        out = self.dropout(out)

        logits = self.fusion_fc2(out)  # (B, output_dim)
        return logits


###############################################################################
# D) Synthetic Dataset for Demonstration
###############################################################################
#
# In a real project, you'd have custom Dataset classes that load your actual
# tabular features, images, and text. Here, we generate random data to show
# how the pipeline works end-to-end.
###############################################################################

class SyntheticMultiModalDataset(torch.utils.data.Dataset):
    """
    Generates random samples of:
       - tabular_data: shape (TABULAR_NUM_FEATURES)
       - image_data:   shape (3, IMAGE_SIZE, IMAGE_SIZE)
       - text_data:    shape (MAX_SEQ_LEN)
       - label:        random integer class in [0, OUTPUT_DIM-1]
    for demonstration/training sanity checks.
    """
    def __init__(self, num_samples=1000, 
                 tab_feat_dim=TABULAR_NUM_FEATURES, 
                 img_size=IMAGE_SIZE,
                 vocab_size=VOCAB_SIZE, 
                 seq_len=MAX_SEQ_LEN, 
                 num_classes=OUTPUT_DIM):
        super().__init__()
        self.num_samples = num_samples
        self.tab_feat_dim = tab_feat_dim
        self.img_size = img_size
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_classes = num_classes

        # Optionally, create transforms for synthetic images:
        self.img_transform = transforms.Compose([
            # Here we can do random augmentation if we want
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(5),
            #
