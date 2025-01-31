# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

import logging
import os
import random

import time

from datetime import date
from typing import Dict, Optional

import gin

import torch
import torch.distributed as dist

from generative_recommenders.data.eval import (
    _avg,
    add_to_summary_writer,
    eval_metrics_v2_from_tensors,
    get_eval_state,
)

from generative_recommenders.data.reco_dataset import get_reco_dataset
from generative_recommenders.indexing.utils import get_top_k_module
from generative_recommenders.modeling.sequential.autoregressive_losses import (
    BCELoss,
    InBatchNegativesSampler,
    LocalNegativesSampler,
)
from generative_recommenders.modeling.sequential.embedding_modules import (
    EmbeddingModule,
    LocalEmbeddingModule,
)
from generative_recommenders.modeling.sequential.encoder_utils import (
    get_sequential_encoder,
)
from generative_recommenders.modeling.sequential.features import (
    movielens_seq_features_from_row,
)
from generative_recommenders.modeling.sequential.input_features_preprocessors import (
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
)
from generative_recommenders.modeling.sequential.losses.sampled_softmax import (
    SampledSoftmaxLoss,
)
from generative_recommenders.modeling.sequential.output_postprocessors import (
    L2NormEmbeddingPostprocessor,
    LayerNormEmbeddingPostprocessor,
)
from generative_recommenders.modeling.similarity_utils import get_similarity_function
from generative_recommenders.trainer.data_loader import create_data_loader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

"""
Main training script for generative recommender systems. This module implements a distributed training
pipeline for sequential recommendation models with various embedding, loss, and sampling strategies.

Key components:
- Distributed training setup using PyTorch DDP
- Configurable model architecture (SASRec etc.)
- Multiple negative sampling strategies
- Evaluation pipeline with metrics like NDCG, HR, MRR
- TensorBoard logging and checkpoint saving
"""

def setup(rank: int, world_size: int, master_port: int) -> None:
    """
    Initialize distributed training environment.
    
    Args:
        rank: Process rank/GPU ID
        world_size: Total number of processes/GPUs
        master_port: Port for distributed coordination
    
    References:
        https://pytorch.org/docs/stable/distributed.html
    """
    os.environ["MASTER_ADDR"] = "localhost"
    # Port for distributed coordination between processes
    # References:
        # https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        # Port used for TCP communication between processes during distributed training
        # Should be a free port on the machine that won't conflict with other services
    os.environ["MASTER_PORT"] = str(master_port)

    # initialize the process group
    # https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup() -> None:
    """
    Cleanup distributed training environment.
    
    References:
        https://pytorch.org/docs/stable/distributed.html#torch.distributed.destroy_process_group
"""
    dist.destroy_process_group()


@gin.configurable
def get_weighted_loss(
    main_loss: torch.Tensor,
    aux_losses: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> torch.Tensor:
    """
    Compute weighted combination of main loss and auxiliary losses.
    
    Args:
        main_loss: Primary training loss
        aux_losses: Dictionary of auxiliary loss terms
        weights: Weight coefficients for auxiliary losses
    
    Returns:
        Combined weighted loss
    """
    weighted_loss = main_loss
    for key, weight in weights.items():
        cur_weighted_loss = aux_losses[key] * weight
        weighted_loss = weighted_loss + cur_weighted_loss
    return weighted_loss


@gin.configurable
def train_fn(
    # Process rank in distributed training setup (0 to world_size-1)
    # References:
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
        # Process rank in distributed training setup (0 to world_size-1)
    rank: int,
    # Total number of processes participating in distributed training
    # References:
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
        # Total number of processes participating in distributed training
    world_size: int,
    # Port used for distributed process coordination
    # References:
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
        # Port used for distributed process coordination
        # Should be a free port on the machine that won't conflict with other services
    master_port: int,
    # Name of dataset to use for training (e.g. "ml-20m" for MovieLens 20M)
    dataset_name: str = "ml-20m",
    # Maximum length of input sequence to consider
    # References:
        # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        # Maximum sequence length is needed to:
        # 1. Control memory usage - longer sequences require more memory for attention
        # 2. Improve efficiency - shorter sequences are faster to process
        # 3. Handle variable length data - provides consistent truncation point
        # 4. Match model architecture - transformer models have position embedding limits
    max_sequence_length: int = 200,
    # Ratio for sampling positions in sequence during training
    # References:
# https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        # Positional sampling ratio < 1.0 can be beneficial for:
        # 1. Memory efficiency - sampling fewer positions reduces memory usage during training
        # 2. Training speed - processing fewer positions per sequence is faster
        # 3. Regularization - random sampling of positions acts as a form of data augmentation
        # 4. Handling long sequences - allows training on longer sequences by sampling subset of positions
        # 5. Curriculum learning - can gradually increase ratio to help model learn position dependencies
        # While still allowing model to learn sequential patterns, just from a subset of positions
        # 
        # Positional sampling ratio = 1.0 means all positions are sampled
        # Positional sampling ratio < 1.0 means fewer positions are sampled
        # Positional sampling ratio > 1.0 means more positions are sampled
    positional_sampling_ratio: float = 1.0,
    # Batch size per GPU/process
    # References:
        # Batch size per GPU/process
        # Called "local" because in distributed training each GPU/process has its own batch size
        # The effective global batch size = local_batch_size * world_size (number of GPUs)
        # Should be chosen based on:
        # 1. Individual GPU memory capacity since each GPU processes this many samples
        # 2. Model complexity and parameters per GPU
        # 3. Training speed and efficiency per GPU
        # 4. Regularization effects of per-GPU batch size
        # 5. Data loading and processing capabilities per GPU
    local_batch_size: int = 128,
    # Batch size to use during evaluation
    # References:
        # Batch size to use during evaluation
    eval_batch_size: int = 128,
    # Maximum batch size for user evaluation, optional limit
    # References:
        # Maximum batch size for user evaluation, optional limit
        # Used to limit the number of user evaluations processed in a single batch
        # This is useful for debugging and ensuring that the evaluation process is manageable
        # If not specified, the default behavior is to process all users in a single batch
    eval_user_max_batch_size: Optional[int] = None,
    # Model architecture to use ("SASRec" or "HSTU")
    main_module: str = "SASRec",
    # Whether to use bfloat16 precision for main module
    # References:
        # https://pytorch.org/docs/stable/generated/torch.bfloat16.html
        # Bfloat16 is a 16-bit floating-point format that provides a good compromise between precision and performance
        # It is faster than FP32 but less precise than FP16
    main_module_bf16: bool = False,
    # Probability of dropping units during training
    # References:
        # https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
        # Dropout is a regularization technique that randomly drops units (neurons) in a neural network during training
        # This helps prevent overfitting by forcing the network to learn more robust features
        # The dropout rate is the fraction of units that are dropped
    dropout_rate: float = 0.2,
    # Type of normalization to apply to user embeddings
    # References:
        # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        # https://pytorch.org/docs/stable/generated/torch.nn.L2Norm.html
        # User embeddings are normalized to prevent issues with large values
        # L2 normalization is used to scale the embeddings to unit length
        # Layer normalization is used to scale the embeddings to unit length
        # Both methods help with numerical stability and training stability
    user_embedding_norm: str = "l2_norm",
    # Strategy for sampling negative examples
    # References:
        # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        # Sampling negative examples is a technique used to train models to distinguish between positive and negative examples
        # In a typical recommendation system, positive examples are the items that the user has interacted with (e.g. clicked, rated, etc.)
        # Negative examples are items that the user has not interacted with (e.g. items that the user has not clicked, rated, etc.)
        # The goal is to learn a model that can predict the likelihood of a user interacting with an item based on their historical interactions
        # The sampling strategy determines how negative examples are sampled
        # In-batch sampling is a simple strategy where negative examples are sampled from the same batch as positive examples
        # Local sampling is a more complex strategy where negative examples are sampled from the entire dataset
        # The choice of sampling strategy can impact the model's ability to learn and generalize
        #
        # More details on in-batch sampling:
        # - For each user in a batch, other items in that batch are treated as negative examples
        # - Example: In a batch with users u1,u2,u3 and their positive items i1,i2,i3:
        #   * For user u1: items i2,i3 become negative examples
        #   * For user u2: items i1,i3 become negative examples  
        #   * For user u3: items i1,i2 become negative examples
        # - This is computationally efficient as it reuses items already loaded in memory
        # - Also provides implicit regularization since popular items appear more as negatives
        # - Alternative is sampling from full item set which may give better negatives but is more expensive
    sampling_strategy: str = "in-batch",
    # Loss function module to use for training
    # References:
        # Loss function module to use for training
        #
        # The loss_module parameter is used in train.py around line 500 to determine which loss function to use:
        # - If "BCELoss": Uses binary cross entropy loss
        # - If "SampledSoftmaxLoss": Uses sampled softmax with negative sampling
        #
        # SampledSoftmaxLoss is a loss function that approximates the full softmax by sampling negative examples
        # Instead of computing probabilities over all items (which is computationally expensive), it:
        # 1. Takes the positive (true) item for each user
        # 2. Samples a fixed number of negative items (controlled by num_negatives parameter)
        # 3. Computes softmax only over this smaller subset of items
        # 4. Applies temperature scaling (controlled by temperature parameter) to control prediction sharpness
        #
        # Key benefits:
        # - Much faster training compared to full softmax over all items
        # - Memory efficient since only needs to store embeddings for sampled items
        # - Still provides good model quality with sufficient negative samples
        # - Temperature scaling helps balance exploration vs exploitation
        #
        # The loss can optionally use activation checkpointing to save memory at the cost of recomputation
        #
        # BCELoss is a simpler alternative that uses binary cross entropy to train the model
        # It treats each user-item interaction as an independent binary classification problem
        #
        # Advantages:
        # - Simple and stable training objective
        # - Works well with in-batch negative sampling
        # - Memory efficient as it doesn't require storing full item embedding matrix
        #
        # Disadvantages:
        # - May be less effective than softmax for ranking
        # - Limited to binary decisions rather than full probability distribution
        # - Requires careful negative sampling strategy
    loss_module: str = "SampledSoftmaxLoss",
    # Optional weights for different loss components
    # References:
        # Optional weights for different loss components
        #
        # The loss_weights parameter is used in train.py around line 500 to determine the weights for different loss components
        # This allows for a weighted combination of losses to improve training stability and convergence
        # The weights are a dictionary where the keys are the names of the loss components and the values are the weights
        # The weights are used to scale the loss contributions of each component
        # The sum of the weights should equal 1.0 to ensure that the total loss is a valid probability distribution
    loss_weights: Optional[Dict[str, float]] = {},
    # Number of negative samples per positive example
    # References:
        # Number of negative samples per positive example
        #
        # The num_negatives parameter is used in train.py around line 500 to determine the number of negative samples to use for each positive example
        # This is a hyperparameter that controls the balance between positive and negative examples in the training process
        # A higher number of negative samples can help the model learn to distinguish between positive and negative examples
        # However, too many negative samples can lead to overfitting and poor generalization
        # The optimal number of negative samples depends on the dataset and model complexity
        # In practice, a value between 1 and 10 is often used
    num_negatives: int = 1,
    # Whether to use activation checkpointing in loss computation
    # References:
        # Whether to use activation checkpointing in loss computation
        #
        # Activation checkpointing is a technique used to save memory during training by recomputing activations instead of storing them
        # This is useful for models with a large number of parameters and activations, as it reduces memory usage
        # The loss_activation_checkpoint parameter is used in train.py around line 500 to determine whether to use activation checkpointing in the loss computation
        # This is a boolean parameter that defaults to False
        # If set to True, activation checkpointing will be used in the loss computation
        # If set to False, activation checkpointing will not be used in the loss computation
    loss_activation_checkpoint: bool = False,
    # Whether to apply L2 normalization to item embeddings
    item_l2_norm: bool = False,
    # Temperature scaling factor for loss computation
    # References:
        # The temperature parameter is used in train.py around line 500 to determine the temperature scaling factor for the loss computation
        # This is a hyperparameter that controls the sharpness of the model's predictions
        # A higher temperature results in more uniform predictions, while a lower temperature results in more peaked predictions
        # The optimal temperature depends on the dataset and model complexity
        # In practice, a value between 0.01 and 1.0 is often used
    temperature: float = 0.05,
    # Total number of training epochs
    # References:
        # Total number of training epochs
        #
        # The num_epochs parameter is used in train.py around line 500 to determine the total number of training epochs
        # This is a hyperparameter that controls the number of times the entire dataset is passed through the model during training
        # A higher number of epochs can help the model learn more complex patterns in the data
        # However, too many epochs can lead to overfitting and poor generalization
        # The optimal number of epochs depends on the dataset and model complexity
        #
        # Values used in .gin config files: 100, 101 or 201
        #
        # The value of 101 epochs was chosen empirically to ensure convergence while avoiding overfitting
    num_epochs: int = 101,
    # Base learning rate for optimization
    # References:
        # Base learning rate for optimization
        #
        # The learning_rate parameter is used in train.py around line 500 to determine the base learning rate for the optimizer
        # This is a hyperparameter that controls the step size for updating the model's parameters during training
        # A higher learning rate can help the model learn more quickly, but too high a rate can lead to unstable training
        # A lower learning rate can help the model learn more slowly, but too low a rate can lead to slow convergence
        # The optimal learning rate depends on the dataset and model complexity
        # In practice, a value between 1e-3 and 1e-1 is often used
    learning_rate: float = 1e-3,
    # Number of steps for learning rate warmup
    # References:
        # Number of steps for learning rate warmup
        #
        # The num_warmup_steps parameter is used in train.py around line 500 to determine the number of steps for learning rate warmup
        # This is a hyperparameter that controls the number of steps for the learning rate to increase from 0 to the base learning rate
        # Warmup is used to prevent the learning rate from being too high at the start of training, which can lead to unstable training
        # The optimal number of warmup steps depends on the dataset and model complexity
        # In practice, a value between 1000 and 10000 is often used
    num_warmup_steps: int = 0,
    # L2 regularization coefficient
    # References:
        # L2 regularization coefficient
        #
        # The weight_decay parameter is used in train.py around line 500 to determine the L2 regularization coefficient for the optimizer
        # This is a hyperparameter that controls the strength of L2 regularization
        # L2 regularization adds a penalty term of 0.5 * weight_decay * ||w||^2 to the loss function
        # where ||w||^2 is the squared L2 norm of all model parameters
        # This is implemented in PyTorch's AdamW optimizer at line 1000 in torch/optim/adamw.py
        # References:
            # https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        # This penalty encourages the model's parameters to be small to prevent overfitting
        # The optimal weight decay depends on the dataset and model complexity
        # In practice, a value between 1e-3 and 1e-1 is often used
        #
    weight_decay: float = 1e-3,
    # Method for computing top-k predictions
    top_k_method: str = "MIPSBruteForceTopK",
    # Number of steps between evaluation runs
    # References:
        # Number of steps between evaluation runs
        #
        # The eval_interval parameter is used in train.py around line 500 to determine the number of steps between evaluation runs
        # This is a hyperparameter that controls the frequency of evaluation during training
        # Evaluating the model periodically can help detect overfitting and improve generalization
        # The optimal eval_interval depends on the dataset and model complexity
        # In practice, a value between 100 and 1000 is often used
    eval_interval: int = 100,
    # Number of epochs between full evaluations
    # References:
        # Number of epochs between full evaluations
        #
        # The full_eval_every_n parameter is used in train.py around line 500 to determine the number of epochs between full evaluations
        # This is a hyperparameter that controls the frequency of full evaluations during training
        # Full evaluations can help detect overfitting and improve generalization
        # The optimal full_eval_every_n depends on the dataset and model complexity
        # In practice, a value between 1 and 10 is often used
    full_eval_every_n: int = 1,
    # Number of steps between saving checkpoints
    # References:
        # Number of steps between saving checkpoints
        #
        # The save_ckpt_every_n parameter is used in train.py around line 500 to determine the number of steps between saving checkpoints
        # This is a hyperparameter that controls the frequency of checkpoint saving during training
        # Checkpoints are used to save the model's state and restore it later
        # The optimal save_ckpt_every_n depends on the dataset and model complexity
        # In practice, a value between 1000 and 10000 is often used
    save_ckpt_every_n: int = 1000,
    # Number of iterations for partial evaluation
    # References:
        # Number of iterations for partial evaluation
        #
        # The partial_eval_num_iters parameter is used in train.py around line 500 to determine the number of iterations for partial evaluation
        # This is a hyperparameter that controls the number of iterations for partial evaluation during training
        # Partial evaluation can help detect overfitting and improve generalization
        # The optimal partial_eval_num_iters depends on the dataset and model complexity
        # In practice, a value between 10 and 100 is often used
    partial_eval_num_iters: int = 32,
    # Type of embedding module to use
    # References:
        # Type of embedding module to use
        #
        # The embedding_module_type parameter is used in train.py around line 500 to determine the type of embedding module to use
        # This is a hyperparameter that controls the type of embedding module used in the model
        # The embedding module is responsible for transforming the input data into dense vectors
        # The embedding module can impact the model's ability to learn and generalize
        # The optimal embedding_module_type depends on the dataset and model complexity
        # In practice, a value between "local" and "distributed" is often used
    embedding_module_type: str = "local",
    # Dimension of item embedding vectors
    # References:
        # Dimension of item embedding vectors
        #
        # The item_embedding_dim parameter is used in train.py around line 500 to determine the dimension of item embedding vectors
        # This is a hyperparameter that controls the dimensionality of the item embeddings
        # The item embeddings are used to represent the items in the dataset
        # The item embedding dimension can impact the model's ability to learn and generalize
        # The optimal item_embedding_dim depends on the dataset and model complexity
        # In practice, a value between 100 and 1000 is often used
    item_embedding_dim: int = 240,
    # Type of interaction module to use
    interaction_module_type: str = "",
    # Length of output sequence for generative models
    # References:
        # Length of output sequence for generative models
        #
        # The gr_output_length parameter is used in train.py around line 500 to determine the length of the output sequence for generative models
        # This is a hyperparameter that controls the length of the output sequence for the generative model
        # The output sequence is used to generate the next item in the sequence
        # The optimal gr_output_length depends on the dataset and model complexity
        # In practice, a value between 1 and 10 is often used
    gr_output_length: int = 10,
    # Small constant for numerical stability in L2 norm
    # References:
        # Small constant for numerical stability in L2 norm
        #
        # The l2_norm_eps parameter is used in train.py around line 500 to determine the small constant for numerical stability in L2 norm
        # This is a hyperparameter that controls the small constant for numerical stability in the L2 norm
        # The L2 norm is used to normalize the item embeddings
        # The optimal l2_norm_eps depends on the dataset and model complexity
        # In practice, a value between 1e-6 and 1e-3 is often used
    l2_norm_eps: float = 1e-6,
    # Whether to enable TensorFloat-32 precision
    # References:
        # Whether to enable TensorFloat-32 precision
        #
        # The enable_tf32 parameter is used in train.py around line 500 to determine whether to enable TensorFloat-32 precision
        # This is a hyperparameter that controls whether to enable TensorFloat-32 precision
        # TensorFloat-32 (TF32) is a precision mode for NVIDIA GPUs that provides a balance between performance and accuracy
        # Enabling TF32 can improve the model's performance and accuracy
        # The optimal enable_tf32 depends on the dataset and model complexity
        # In practice, a value between True and False is often used
    enable_tf32: bool = False,
    # Random seed for reproducibility
    # References:
        # Random seed for reproducibility
        #
        # The random_seed parameter is used in train.py around line 500 to determine the random seed for reproducibility
        # This is a hyperparameter that controls the random seed for the random number generator
        # Setting a random seed ensures that the same random initialization is used for each run
        # This can help with reproducibility and debugging
        # The optimal random_seed depends on the dataset and model complexity
        # In practice, a value between 0 and 1000 is often used
    random_seed: int = 42,
) -> None:
    # to enable more deterministic results.
    """
    Main training function implementing the complete training pipeline.
    
    Key steps:
    1. Initialize distributed training
    2. Set up dataset and dataloaders
    3. Configure model components:
       - Embedding module
       - Interaction module
       - Input/output processors
       - Loss functions
       - Negative samplers
    4. Training loop with:
       - Forward/backward passes
       - Periodic evaluation
       - Checkpoint saving
       - TensorBoard logging
    
    Args:
        rank: Process rank for distributed training
        world_size: Total number of processes
        master_port: Port for distributed coordination
        dataset_name: Name of dataset to train on
        max_sequence_length: Maximum length of input sequences
        positional_sampling_ratio: Ratio for positional sampling
        local_batch_size: Per-GPU batch size
        eval_batch_size: Batch size during evaluation
        eval_user_max_batch_size: Maximum batch size for user evaluation
        main_module: Model architecture (e.g. "SASRec")
        main_module_bf16: Whether to use bfloat16 precision
        dropout_rate: Dropout probability
        user_embedding_norm: Type of user embedding normalization
        sampling_strategy: Strategy for negative sampling
        loss_module: Loss function module to use
        loss_weights: Optional weights for different loss components
        num_negatives: Number of negative samples per positive
        loss_activation_checkpoint: Whether to use activation checkpointing
        item_l2_norm: Whether to use L2 normalization on item embeddings
        temperature: Temperature parameter for loss scaling
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        num_warmup_steps: Steps for learning rate warmup
        weight_decay: L2 regularization factor
        top_k_method: Method for computing top-k predictions
        eval_interval: How often to run evaluation
        full_eval_every_n: Epochs between full evaluations
        save_ckpt_every_n: Steps between checkpoints
        partial_eval_num_iters: Number of iterations for partial eval
        embedding_module_type: Type of embedding module
        item_embedding_dim: Dimension of item embeddings
        interaction_module_type: Type of interaction module
        gr_output_length: Output sequence length for generative model
        l2_norm_eps: Epsilon for L2 normalization
        enable_tf32: Whether to enable TF32 precision
        random_seed: Random seed for reproducibility
    """
    
    # === Initialize training environment ===
    random.seed(random_seed)
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cudnn.allow_tf32 = enable_tf32
    logging.info(f"cuda.matmul.allow_tf32: {enable_tf32}")
    logging.info(f"cudnn.allow_tf32: {enable_tf32}")
    logging.info(f"Training model on rank {rank}.")
    setup(rank, world_size, master_port)

    # === Dataset setup ===
    # Get the recommendation dataset with specified parameters:
    # - dataset_name: Name of the dataset to load (e.g. MovieLens)
    # - max_sequence_length: Maximum length of user interaction sequences
    # - chronological: Whether to maintain temporal order of interactions
    # - positional_sampling_ratio: Controls sampling of positions in sequences
    dataset = get_reco_dataset(
        dataset_name=dataset_name,
        max_sequence_length=max_sequence_length,
        chronological=True,
        positional_sampling_ratio=positional_sampling_ratio,
    )

    # === Create data loaders ===
    # Training loader with shuffling enabled
    train_data_sampler, train_data_loader = create_data_loader(
        dataset.train_dataset,
        batch_size=local_batch_size,
        world_size=world_size,
        rank=rank,
        shuffle=True,
        drop_last=world_size > 1,
    )
    # Evaluation loader
    eval_data_sampler, eval_data_loader = create_data_loader(
        dataset.eval_dataset,
        batch_size=eval_batch_size,
        world_size=world_size,
        rank=rank,
        shuffle=True,  # needed for partial eval
        drop_last=world_size > 1,
    )

    model_debug_str = main_module
    # === Model Configuration ===
    # Initialize the embedding module based on the specified type (local or distributed)
    # Currently only supports "local" embeddings stored on a single device
    # The embedding module maps item IDs to dense vector representations
    if embedding_module_type == "local":
        # LocalEmbeddingModule creates embeddings stored on a single device
        # num_items: Total number of unique items in the dataset
        # item_embedding_dim: Dimension of the embedding vectors for each item
        embedding_module: EmbeddingModule = LocalEmbeddingModule(
            # References:
                # LocalEmbeddingModule
                #
                # This is a module that creates embeddings stored on a single device
                # The embedding module maps item IDs to dense vector representations
                # The embedding module is used in the model to represent the items in the dataset
                # The embedding module is created with the following parameters:
                # - num_items: Total number of unique items in the dataset
                # - item_embedding_dim: Dimension of the embedding vectors for each item
            num_items=dataset.max_item_id,
            item_embedding_dim=item_embedding_dim,
        )
    else:
        # Future support for distributed embeddings could be added here
        raise ValueError(f"Unknown embedding_module_type {embedding_module_type}")
    
    # Add embedding module details to model debug string for logging
    model_debug_str += f"-{embedding_module.debug_str()}"

    # Set up interaction module for computing similarities between items
    # The interaction module computes similarity scores between query and item embeddings
    # References:
        # The interaction module is used to compute similarity scores between:
        # - Query embeddings: Represent the user's current context/interests
        # - Item embeddings: Represent candidate items to recommend
        # Common similarity functions include dot product and cosine similarity
        # The similarity scores are used to rank items for recommendations
    interaction_module, interaction_module_debug_str = get_similarity_function(
        # Type of similarity function to use (e.g. "DotProduct", "MoL")
        module_type=interaction_module_type,
        # Dimension of query embeddings (user/context representations)
        query_embedding_dim=item_embedding_dim,
        # Dimension of item embeddings (must match query dimension)
        item_embedding_dim=item_embedding_dim,
    )

    assert (
        user_embedding_norm == "l2_norm" or user_embedding_norm == "layer_norm"
    ), f"Not implemented for {user_embedding_norm}"
    # Configure embedding normalization
    # References:
        # Configure embedding normalization
        #
        # The output_postproc_module is used in train.py around line 631 to configure the embedding normalization
        # This is a hyperparameter that controls the normalization method used on the output embeddings
        # The output embeddings are used to represent the items in the dataset
        # The normalization method can impact the model's ability to learn and generalize
        # The optimal normalization method depends on the dataset and model complexity
        # In practice, a value between "l2_norm" and "layer_norm" is often used
    output_postproc_module = (
        L2NormEmbeddingPostprocessor(
            # References:
                # L2NormEmbeddingPostprocessor
                #
                # This is a module that normalizes the output embeddings using L2 norm
                # The L2 norm is a common normalization method used in generative recommender models
                # The L2 norm is computed as the square root of the sum of the squares of the embedding values
                # The L2 norm is used to ensure that the output embeddings have a consistent scale
                # The L2 norm is applied to the output embeddings to prevent overfitting and improve generalization
            embedding_dim=item_embedding_dim,
            eps=1e-6,
        )
        if user_embedding_norm == "l2_norm"
        else LayerNormEmbeddingPostprocessor(
            # References:
                # LayerNormEmbeddingPostprocessor
                #
                # This is a module that normalizes the output embeddings using layer normalization
                # Layer normalization is a common normalization method used in generative recommender models
                # Layer normalization transforms each embedding to have zero mean and unit variance
                # This ensures each embedding has a consistent scale where values typically fall within [-2,2]
                # By normalizing to this consistent scale, we avoid embeddings with very large or small magnitudes
                # 
                # Layer normalization is applied to the output embeddings to prevent overfitting and improve generalization
            embedding_dim=item_embedding_dim,
            eps=1e-6,
        )
    )

    # Set up positional embeddings
    # References:
        # Set up positional embeddings
        #
        # input_preproc_module contains a LearnablePositionalEmbeddingInputFeaturesPreprocessor that:
        # 1. Takes item embeddings as input and adds learnable positional embeddings
        # 2. Has parameters:
        #    - max_sequence_len: Maximum length of input sequences + output length + 1
        #    - embedding_dim: Dimension of item embeddings
        #    - dropout_rate: Probability of dropout for regularization
        # 3. Forward pass:
        #    - Scales item embeddings by sqrt(embedding_dim) to control variance
        #      This scaling helps maintain stable gradients during training by keeping
        #      the variance of the dot products between embeddings roughly constant
        #      as embedding dimension increases. Without scaling, larger embedding
        #      dimensions would lead to larger dot products and gradient instability.
        #    - Adds position embeddings from learned embedding table
        #    - Applies dropout to combined embeddings
        #    - Creates mask for valid (non-padding) positions
        #    - Returns sequence lengths, processed embeddings, and attention mask
        # 4. Initialization:
        #    - Creates embedding table of size [max_sequence_len x embedding_dim]
        #    - Initializes embeddings using truncated normal distribution
        #    - Sets up dropout layer with specified rate
        #
        # This module is crucial for the transformer architecture to understand sequence order,
        # since self-attention alone has no inherent way to capture position information.
    input_preproc_module = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
        # References:
            # LearnablePositionalEmbeddingInputFeaturesPreprocessor
            #
            # This is a module that takes item embeddings as input and adds learnable positional embeddings
            # The positional embeddings are used to capture the order of items in the sequence
            # The positional embeddings are learned during training and added to the item embeddings
            # The positional embeddings are used to help the model understand the sequence structure
        max_sequence_len=dataset.max_sequence_length + gr_output_length + 1,
        embedding_dim=item_embedding_dim,
        dropout_rate=dropout_rate,
    )

    # Initialize main sequential encoder model
    # References:
        # The sequential encoder is the core model that:
        # 1. Takes preprocessed item sequences as input
        # 2. Processes them through transformer layers
        # 3. Generates predictions for next items
        #
        # Key components:
        # - module_type: Type of transformer architecture (e.g. HSTU, FALCON)
        # - max_sequence_length: Maximum length of input sequences
        # - max_output_length: Maximum length of output predictions (gr_output_length + 1)
        # - embedding_module: Handles item ID to embedding conversion
        # - interaction_module: Core transformer layers for sequence processing
        # - input_preproc_module: Adds positional embeddings to inputs
        # - output_postproc_module: Normalizes output embeddings
        #
        # The model processes sequences by:
        # 1. Converting item IDs to embeddings via embedding_module
        # 2. Adding positional information via input_preproc_module
        # 3. Passing through transformer layers in interaction_module
        # 4. Normalizing outputs via output_postproc_module
        # 5. Computing loss against target items
    model = get_sequential_encoder(
        module_type=main_module,
        max_sequence_length=dataset.max_sequence_length,
        max_output_length=gr_output_length + 1,
        embedding_module=embedding_module,
        interaction_module=interaction_module,
        input_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
        verbose=True,
    )
    # Get debug string representation of model architecture
    model_debug_str = model.debug_str()

    # loss
    # === Loss and Sampling Setup ===
    # Configure loss function (BCE or Sampled Softmax)
    loss_debug_str = loss_module
    if loss_module == "BCELoss":
        # Binary Cross Entropy Loss for Generative Recommenders
        # 
        # BCELoss treats recommendation as a binary classification problem:
        # - For each item in the sequence, predict whether it will be the next item (1) or not (0)
        # - Loss is computed between model predictions and ground truth labels
        # - Temperature parameter scales logits before sigmoid (fixed at 1.0 for BCE)
        #
        # Advantages:
        # - Simple and stable training objective
        # - Works well with in-batch negative sampling
        # - Memory efficient as it doesn't require storing full item embedding matrix
        #
        # Disadvantages: 
        # - May be less effective than softmax for ranking
        # - Limited to binary decisions rather than full probability distribution
        # - Requires careful negative sampling strategy
        #
        # Implementation:
        # - Uses PyTorch's binary_cross_entropy_with_logits under the hood
        #   References:
        #   - https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        # - Handles both positive (target items) and negative samples
        # - Automatically applies sigmoid activation to model outputs
        loss_debug_str = loss_debug_str[:-4]
        assert temperature == 1.0
        ar_loss = BCELoss(temperature=temperature, model=model)
        # References:
            # BCELoss
            #
            # This is a loss function that computes the binary cross entropy loss between model predictions and ground truth labels
            # The loss is computed between model predictions and ground truth labels
            # The temperature parameter scales logits before sigmoid (fixed at 1.0 for BCE)
            #
    elif loss_module == "SampledSoftmaxLoss":
        # Sampled Softmax Loss for Generative Recommenders
        # 
        # Sampled Softmax Loss is a variant of the softmax loss function:
        # - Instead of using the full item embedding matrix, it samples negative items
        # - This reduces memory usage and computational complexity
        # - The loss is computed between model predictions and ground truth labels
        # - Temperature parameter scales logits before softmax (fixed at 1.0 for SSL)
        #
        # While BCELoss treats recommendation as binary classification (item relevant/not),
        # Sampled Softmax Loss models a full probability distribution over items:
        # - Treats each position as multi-class classification over all possible items
        # - Better captures relative preferences between items vs just binary relevance
        # - More suitable when we want to rank items by probability/preference
        #
        # Key differences from BCE:
        # - BCE: p(item is relevant) for each item independently 
        # - SSL: p(item|context) as distribution over all items
        #
        # Context in Sampled Softmax Loss:
        #
        # The context refers to the sequence of previous items and any other relevant 
        # features that help predict the next item. Specifically:
        #
        # - Previous items in the sequence provide temporal context
        # - User features/embeddings provide user preference context
        # - Item features/embeddings provide item characteristic context
        #
        # The model uses this context to compute p(item|context), which represents
        # the probability of each item being the next one given the context.
        #
        # Key differences from BCE:
        # - BCE: p(item is relevant) for each item independently 
        # - SSL: p(item|context) as distribution over all items
        # - SSL provides more fine-grained signal for ranking items
        # - SSL better models competition between items for user attention
        #
        # We offer both loss functions since they have different tradeoffs:
        # - BCE: Faster, simpler, good for binary relevance
        # - SSL: Better for ranking, models full distribution, but more complex
        #
        # Advantages:
        # - More effective for ranking tasks
        # - Can handle larger datasets with fewer negative samples
        # - Memory efficient as it doesn't require storing full item embedding matrix
        #
        # Disadvantages:
        # - Requires careful negative sampling strategy
        # - May be slower than BCELoss for small datasets
        #
        # Implementation:
        # - Uses PyTorch's cross_entropy under the hood
        # - Handles both positive (target items) and negative samples
        # - Automatically applies softmax activation to model outputs
        #
        loss_debug_str = "ssl"
        if temperature != 1.0:
            loss_debug_str += f"-t{temperature}"
        ar_loss = SampledSoftmaxLoss(
            # AR = Autoregressive, since SSL is used for next-item prediction
            # in an autoregressive fashion (predicting next item based on sequence)
            num_to_sample=num_negatives,
            softmax_temperature=temperature,
            model=model,
            activation_checkpoint=loss_activation_checkpoint,
        )
        loss_debug_str += (
            f"-n{num_negatives}{'-ac' if loss_activation_checkpoint else ''}"
        )
    else:
        raise ValueError(f"Unrecognized loss module {loss_module}.")

    # sampling
    # Set up negative sampling strategy
    if sampling_strategy == "in-batch":
        negatives_sampler = InBatchNegativesSampler(
            l2_norm=item_l2_norm,
            l2_norm_eps=l2_norm_eps,
            dedup_embeddings=True,
        )
        sampling_debug_str = (
            f"in-batch{f'-l2-eps{l2_norm_eps}' if item_l2_norm else ''}-dedup"
        )
    elif sampling_strategy == "local":
        negatives_sampler = LocalNegativesSampler(
            num_items=dataset.max_item_id,
            item_emb=model._embedding_module._item_emb,
            all_item_ids=dataset.all_item_ids,
            l2_norm=item_l2_norm,
            l2_norm_eps=l2_norm_eps,
        )
    else:
        raise ValueError(f"Unrecognized sampling strategy {sampling_strategy}.")
    sampling_debug_str = negatives_sampler.debug_str()

    # === Training Setup ===
    # Move model to GPU and wrap with DistributedDataParallel (DDP)
    # 
    # DDP wrapping is needed to:
    # 1. Enable distributed training across multiple GPUs
    # 2. Automatically synchronize gradients between processes
    # 3. Handle gradient reduction and parameter broadcast
    # 4. Optimize communication between GPUs for better performance
    #
    # Without DDP, each GPU would train independently without sharing updates,
    # leading to divergent model parameters across processes.
    device = rank
    if main_module_bf16:
        model = model.to(torch.bfloat16)
    model = model.to(device)
    ar_loss = ar_loss.to(device)
    negatives_sampler = negatives_sampler.to(device)
    model = DDP(model, device_ids=[rank], broadcast_buffers=False)

    # TODO: wrap in create_optimizer.
    # Initialize optimizer
    opt = torch.optim.AdamW(
        # References:
            # Optimizer
            #
            # The optimizer is used in train.py around line 500 to initialize the optimizer for the model
            # This is a hyperparameter that controls the optimization algorithm used to update the model's parameters during training
            # AdamW is a variant of the Adam optimizer that uses weight decay regularization
            # AdamW is a popular choice for training deep learning models, including generative recommender models
            #
            # Parameters:
            # - model.parameters(): This returns an iterator over all the parameters of the model
            # - lr: The learning rate for the optimizer
            # - betas: The beta1 and beta2 parameters for the AdamW optimizer
            # - weight_decay: The weight decay coefficient for the optimizer
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        weight_decay=weight_decay,
    )

    # === Logging Setup ===
    # Configure TensorBoard writer and logging paths
    date_str = date.today().strftime("%Y-%m-%d")
    model_subfolder = f"{dataset_name}-l{max_sequence_length}"
    model_desc = (
        f"{model_subfolder}"
        + f"/{model_debug_str}_{interaction_module_debug_str}_{sampling_debug_str}_{loss_debug_str}"
        + f"{f'-ddp{world_size}' if world_size > 1 else ''}-b{local_batch_size}-lr{learning_rate}-wu{num_warmup_steps}-wd{weight_decay}{'' if enable_tf32 else '-notf32'}-{date_str}"
    )
    if full_eval_every_n > 1:
        model_desc += f"-fe{full_eval_every_n}"
    if positional_sampling_ratio is not None and positional_sampling_ratio < 1:
        model_desc += f"-d{positional_sampling_ratio}"
    # creates subfolders.
    os.makedirs(f"./exps/{model_subfolder}", exist_ok=True)
    os.makedirs(f"./ckpts/{model_subfolder}", exist_ok=True)
    log_dir = f"./exps/{model_desc}"
    if rank == 0:
        writer = SummaryWriter(log_dir=log_dir)
        logging.info(f"Rank {rank}: writing logs to {log_dir}")
    else:
        writer = None
        logging.info(f"Rank {rank}: disabling summary writer")

    last_training_time = time.time()
    torch.autograd.set_detect_anomaly(True)

    batch_id = 0
    epoch = 0
    for epoch in range(num_epochs):
        # Set epoch for samplers
        if train_data_sampler is not None:
            train_data_sampler.set_epoch(epoch)
        if eval_data_sampler is not None:
            eval_data_sampler.set_epoch(epoch)
        model.train()
        for row in iter(train_data_loader):
            # === Process batch ===
            # seq_features contains the user's past interactions with items, including:
            # - past_ids: Tensor of item IDs the user has interacted with
            # - past_ratings: Tensor of ratings the user gave to those items
            # - past_timestamps: Tensor of timestamps when those interactions occurred
            # target_ids and target_ratings are the next items and ratings we want to predict
            seq_features, target_ids, target_ratings = movielens_seq_features_from_row(
                row,
                device=device,
                max_output_length=gr_output_length + 1,
            )

            # === Periodic Evaluation ===
            if (batch_id % eval_interval) == 0:
                model.eval()

                # Compute evaluation metrics
                eval_state = get_eval_state(
                    model=model.module,
                    all_item_ids=dataset.all_item_ids,
                    negatives_sampler=negatives_sampler,
                    top_k_module_fn=lambda item_embeddings, item_ids: get_top_k_module(
                        top_k_method=top_k_method,
                        model=model.module,
                        item_embeddings=item_embeddings,
                        item_ids=item_ids,
                    ),
                    device=device,
                    float_dtype=torch.bfloat16 if main_module_bf16 else None,
                )
                eval_dict = eval_metrics_v2_from_tensors(
                    eval_state,
                    model.module,
                    seq_features,
                    target_ids=target_ids,
                    target_ratings=target_ratings,
                    user_max_batch_size=eval_user_max_batch_size,
                    dtype=torch.bfloat16 if main_module_bf16 else None,
                )
                # Log results
                add_to_summary_writer(
                    writer, batch_id, eval_dict, prefix="eval", world_size=world_size
                )
                logging.info(
                    f"rank {rank}:  batch-stat (eval): iter {batch_id} (epoch {epoch}): "
                    + f"NDCG@10 {_avg(eval_dict['ndcg@10'], world_size):.4f}, "
                    f"HR@10 {_avg(eval_dict['hr@10'], world_size):.4f}, "
                    f"HR@50 {_avg(eval_dict['hr@50'], world_size):.4f}, "
                    + f"MRR {_avg(eval_dict['mrr'], world_size):.4f} "
                )
                model.train()

            # If we don't append the target item ID to the end of each sequence:
            # 1. The model won't have supervision signal for training, since it needs to know what item
            #    comes next in order to learn sequential patterns
            # 2. The loss function won't be able to compare the model's predictions against the actual
            #    next item that the user interacted with
            # 3. The model won't learn the temporal relationships between items in the sequence
            #
            # For example, if a user watched movies [1,2,3,4] in that order, we need to tell the model
            # that after [1,2,3], movie 4 came next. Without appending movie 4, the model would have
            # no way to learn this pattern.
            #
            B, N = seq_features.past_ids.shape  # B=batch size, N=sequence length
            seq_features.past_ids.scatter_(
                # Reference: https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html
                #
                # Append target item ID to end of each user's sequence by:
                # - dim=1: Operate along sequence length dimension
                # - index: Use past_lengths to identify end position for each sequence
                # - src: Target IDs to insert at those positions
                # This gives model supervision signal by showing what item came next
                dim=1,
                index=seq_features.past_lengths.view(-1, 1),  # Position to insert at
                src=target_ids.view(-1, 1),  # Target item IDs to insert
            )

            # === Training Step ===
            # Forward pass
            # Zero out gradients from previous backward pass
            opt.zero_grad()

            # Get embeddings for all items in the sequence by:
            # 1. Accessing the embedding module through model.module._embedding_module
            # 2. Calling get_item_embeddings() to lookup embeddings for each item ID
            # Shape: [batch_size, seq_len, embedding_dim]
            #
            # Reference: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
            #
            input_embeddings = model.module._embedding_module.get_item_embeddings(seq_features.past_ids)

            # Pass sequence through model to get contextual embeddings
            # Input shape: [batch_size, seq_len]
            # Output shape: [batch_size, seq_len, embedding_dim]
            seq_embeddings = model(
                # The model here converts past_length, past_ids, past_embeddings and past_payloads into contextual embeddings to:
                #
                # 1. Capture sequential patterns and dependencies between items in the user's history
                #    - The model learns how previous items influence what comes next
                #    - Items earlier in sequence provide context for later items
                #
                # 2. Create a unified representation that combines:
                #    - past_length: How many items are in the sequence
                #    - past_ids: The actual item IDs in sequence order  
                #    - past_embeddings: The raw item embeddings
                #    - past_payloads: Additional metadata about each interaction
                #
                # 3. Generate embeddings that are "contextualized" by:
                #    - Position of item in sequence (temporal ordering)
                #    - Other items that came before/after
                #    - User's overall interaction patterns
                #
                # This contextual representation helps the model make better predictions
                # by considering the full sequential context rather than just individual items
                past_lengths=seq_features.past_lengths,
                past_ids=seq_features.past_ids,
                past_embeddings=input_embeddings,
                past_payloads=seq_features.past_payloads,
            )  # [B, X]

            supervision_ids = seq_features.past_ids

            if sampling_strategy == "in-batch":
                # get_item_embeddings currently assume 1-d tensor.
                in_batch_ids = supervision_ids.view(-1)
                negatives_sampler.process_batch(
                    ids=in_batch_ids,
                    presences=(in_batch_ids != 0),
                    embeddings=model.module.get_item_embeddings(in_batch_ids),
                )
            else:
                # pyre-fixme[16]: `InBatchNegativesSampler` has no attribute
                #  `_item_emb`.
                negatives_sampler._item_emb = model.module._embedding_module._item_emb

            # Create a mask for valid items in the sequence (non-zero IDs)
            # This mask excludes padding tokens (id=0) from loss computation
            # Ensures training only occurs on actual sequence items
            # Shape of supervision_ids: [batch_size, seq_len]
            #
            # We start from index 1: because in autoregressive training, we predict the next item
            # given the previous items. For each position i, we use items [0:i] to predict item [i].
            # So supervision_ids[:,1:] contains the target items to predict, while
            # seq_embeddings[:,:-1,:] contains the corresponding input contexts.
            #
            ar_mask = supervision_ids[:, 1:] != 0
            loss, aux_losses = ar_loss(
                # In autoregressive training, we predict the next item given previous items
                # For each position i, we use items [0:i] to predict item [i]
                # 
                # [0:i] in PyTorch is array slicing that takes elements from index 0 up to (but not including) index i
                # For example, if tensor=[1,2,3,4] then tensor[0:2] gives [1,2]
                #
                # :-1 slicing on output_embeddings removes the last position since we can't predict beyond sequence
                # 1: slicing on supervision removes first position since we need at least 1 previous item to predict
                #
                # Example for sequence [A, B, C]:
                # output_embeddings[:, :-1] gives [A, B] - use these to predict
                # supervision_ids[:, 1:] gives [B, C] - these are the targets to predict
                lengths=seq_features.past_lengths,  # [B],
                output_embeddings=seq_embeddings[:, :-1, :],  # [B, N-1, D]
                supervision_ids=supervision_ids[:, 1:],  # [B, N-1]
                supervision_embeddings=input_embeddings[:, 1:, :],  # [B, N - 1, D]
                supervision_weights=ar_mask.float(),
                negatives_sampler=negatives_sampler,
                **seq_features.past_payloads,
            )  # [B, N]

            # After this operation, we have 2 loss objects:
            # 1. loss - The original loss tensor with gradients attached
            # 2. main_loss - A detached clone of loss without gradients
            # Reference: https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html
            #
            # We detach() and clone() the loss here to track the original AR loss value
            # before adding auxiliary losses. This allows us to:
            # 1. detach() returns a new Tensor, detached from the current graph, and breaks gradient flow
            # so we can track raw loss without affecting backprop
            # 2. clone() creates a separate copy to avoid modifying the original loss tensor
            # This separated value is used for logging/monitoring the main AR loss independently
            #
            # loss.detach().clone() is functionally the same as loss.clone().detach().
            # Both operations result in a new tensor that:
            # - Shares the same values as the original tensor (loss).
            # - Is detached from the computational graph (no gradient tracking).
            # - Has a separate copy of the data (due to clone()).
            #
            main_loss = loss.detach().clone()

            # Add auxiliary losses to the main loss
            # This allows us to:
            # 1. Combine multiple loss components (e.g. AR loss + auxiliary losses)
            # 2. Track each loss separately for logging/monitoring
            # 3. Gradients are only computed for the main loss, not auxiliary losses
            # 4. We can adjust weights of each loss component if needed
            # 5. This helps in understanding the contribution of each loss term to the total loss
            # 6. Allows us to optimize the main loss while keeping auxiliary losses in check
            # 7. Helps in debugging and understanding the model's behavior
            #
            # Weights for each loss component can be specified in loss_weights dictionary
            # If no weights are provided, default to equal weighting
            loss = get_weighted_loss(loss, aux_losses, weights=loss_weights or {})

            if rank == 0:
                assert writer is not None
                writer.add_scalar("losses/ar_loss", loss, batch_id)
                writer.add_scalar("losses/main_loss", main_loss, batch_id)

            # Backward pass and optimization
            loss.backward()

            # Optional linear warmup.
            if batch_id < num_warmup_steps:
                # Linearly increase learning rate from 0 to learning_rate over num_warmup_steps
                lr_scalar = min(1.0, float(batch_id + 1) / num_warmup_steps)
                for pg in opt.param_groups:
                    pg["lr"] = lr_scalar * learning_rate
                lr = lr_scalar * learning_rate
            else:
                lr = learning_rate

            if (batch_id % eval_interval) == 0:
                logging.info(
                    f" rank: {rank}, batch-stat (train): step {batch_id} "
                    f"(epoch {epoch} in {time.time() - last_training_time:.2f}s): {loss:.6f}"
                )
                last_training_time = time.time()
                if rank == 0:
                    assert writer is not None
                    writer.add_scalar("loss/train", loss, batch_id)
                    writer.add_scalar("lr", lr, batch_id)

            opt.step()

            batch_id += 1

        def is_full_eval(epoch: int) -> bool:
            return (epoch % full_eval_every_n) == 0

        # eval per epoch
        # === Epoch-level Evaluation ===
        model.eval()
        eval_dict_all = None
        eval_start_time = time.time()
        model.eval()
        eval_state = get_eval_state(
            model=model.module,
            all_item_ids=dataset.all_item_ids,
            negatives_sampler=negatives_sampler,
            top_k_module_fn=lambda item_embeddings, item_ids: get_top_k_module(
                top_k_method=top_k_method,
                model=model.module,
                item_embeddings=item_embeddings,
                item_ids=item_ids,
            ),
            device=device,
            float_dtype=torch.bfloat16 if main_module_bf16 else None,
        )
        for eval_iter, row in enumerate(iter(eval_data_loader)):
            seq_features, target_ids, target_ratings = movielens_seq_features_from_row(
                row, device=device, max_output_length=gr_output_length + 1
            )
            eval_dict = eval_metrics_v2_from_tensors(
                eval_state,
                model.module,
                seq_features,
                target_ids=target_ids,
                target_ratings=target_ratings,
                user_max_batch_size=eval_user_max_batch_size,
                dtype=torch.bfloat16 if main_module_bf16 else None,
            )

            if eval_dict_all is None:
                eval_dict_all = {}
                for k, v in eval_dict.items():
                    eval_dict_all[k] = []

            for k, v in eval_dict.items():
                eval_dict_all[k] = eval_dict_all[k] + [v]
            del eval_dict

            if (eval_iter + 1 >= partial_eval_num_iters) and (not is_full_eval(epoch)):
                logging.info(
                    f"Truncating epoch {epoch} eval to {eval_iter + 1} iters to save cost.."
                )
                break

        assert eval_dict_all is not None
        for k, v in eval_dict_all.items():
            eval_dict_all[k] = torch.cat(v, dim=-1)

        ndcg_10 = _avg(eval_dict_all["ndcg@10"], world_size=world_size)
        ndcg_50 = _avg(eval_dict_all["ndcg@50"], world_size=world_size)
        hr_10 = _avg(eval_dict_all["hr@10"], world_size=world_size)
        hr_50 = _avg(eval_dict_all["hr@50"], world_size=world_size)
        mrr = _avg(eval_dict_all["mrr"], world_size=world_size)

        add_to_summary_writer(
            writer,
            batch_id=epoch,
            metrics=eval_dict_all,
            prefix="eval_epoch",
            world_size=world_size,
        )
        if full_eval_every_n > 1 and is_full_eval(epoch):
            add_to_summary_writer(
                writer,
                batch_id=epoch,
                metrics=eval_dict_all,
                prefix="eval_epoch_full",
                world_size=world_size,
            )
        if rank == 0 and epoch > 0 and (epoch % save_ckpt_every_n) == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                },
                f"./ckpts/{model_desc}_ep{epoch}",
            )

        logging.info(
            f"rank {rank}: eval @ epoch {epoch} in {time.time() - eval_start_time:.2f}s: "
            f"NDCG@10 {ndcg_10:.4f}, NDCG@50 {ndcg_50:.4f}, HR@10 {hr_10:.4f}, HR@50 {hr_50:.4f}, MRR {mrr:.4f}"
        )
        last_training_time = time.time()

    if rank == 0:
        # Save final checkpoint and clean up
        # At rank 0 (main process), we:
        # 1. Flush and close the summary writer to ensure metrics are saved
        # 2. Save the final model checkpoint with:
        #    - Current epoch number
        #    - Model state dict (weights/parameters)
        #    - Optimizer state dict (momentum buffers etc)
        # This allows resuming training from this point if needed
        if writer is not None:
            writer.flush()
            writer.close()

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
            },
            f"./ckpts/{model_desc}_ep{epoch}",
        )

    cleanup()
