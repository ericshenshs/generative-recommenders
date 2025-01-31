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

import gin
from generative_recommenders.modeling.sequential.embedding_modules import (
    EmbeddingModule,
)
from generative_recommenders.modeling.sequential.hstu import HSTU
from generative_recommenders.modeling.sequential.input_features_preprocessors import (
    InputFeaturesPreprocessorModule,
)
from generative_recommenders.modeling.sequential.output_postprocessors import (
    OutputPostprocessorModule,
)
from generative_recommenders.modeling.sequential.sasrec import SASRec
from generative_recommenders.modeling.similarity_module import (
    SequentialEncoderWithLearnedSimilarityModule,
)
from generative_recommenders.rails.similarities.module import SimilarityModule


@gin.configurable
def sasrec_encoder(
    max_sequence_length: int,
    max_output_length: int,
    embedding_module: EmbeddingModule,
    similarity_module: SimilarityModule,
    input_preproc_module: InputFeaturesPreprocessorModule,
    output_postproc_module: OutputPostprocessorModule,
    activation_checkpoint: bool,
    verbose: bool,
    ffn_hidden_dim: int = 64,
    ffn_activation_fn: str = "relu",
    ffn_dropout_rate: float = 0.2,
    num_blocks: int = 2,
    num_heads: int = 1,
) -> SequentialEncoderWithLearnedSimilarityModule:
    return SASRec(
        embedding_module=embedding_module,
        max_sequence_len=max_sequence_length,
        max_output_len=max_output_length,
        embedding_dim=embedding_module.item_embedding_dim,
        ffn_hidden_dim=ffn_hidden_dim,
        ffn_activation_fn=ffn_activation_fn,
        ffn_dropout_rate=ffn_dropout_rate,
        num_blocks=num_blocks,
        num_heads=num_heads,
        similarity_module=similarity_module,  # pyre-ignore [6]
        input_features_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
        activation_checkpoint=activation_checkpoint,
        verbose=verbose,
    )


@gin.configurable
def hstu_encoder(
    max_sequence_length: int,
    max_output_length: int,
    embedding_module: EmbeddingModule,
    similarity_module: SimilarityModule,
    input_preproc_module: InputFeaturesPreprocessorModule,
    output_postproc_module: OutputPostprocessorModule,
    activation_checkpoint: bool,
    verbose: bool,
    num_blocks: int = 2,
    num_heads: int = 1,
    dqk: int = 64,
    dv: int = 64,
    linear_dropout_rate: float = 0.0,
    attn_dropout_rate: float = 0.0,
    normalization: str = "rel_bias",
    linear_config: str = "uvqk",
    linear_activation: str = "silu",
    concat_ua: bool = False,
    enable_relative_attention_bias: bool = True,
) -> SequentialEncoderWithLearnedSimilarityModule:
    return HSTU(
        embedding_module=embedding_module,
        similarity_module=similarity_module,  # pyre-ignore [6]
        input_features_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
        max_sequence_len=max_sequence_length,
        max_output_len=max_output_length,
        embedding_dim=embedding_module.item_embedding_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        attention_dim=dqk,
        linear_dim=dv,
        linear_dropout_rate=linear_dropout_rate,
        attn_dropout_rate=attn_dropout_rate,
        linear_config=linear_config,
        linear_activation=linear_activation,
        normalization=normalization,
        concat_ua=concat_ua,
        enable_relative_attention_bias=enable_relative_attention_bias,
        verbose=verbose,
    )


@gin.configurable
def get_sequential_encoder(
    module_type: str,
    max_sequence_length: int,
    max_output_length: int,
    embedding_module: EmbeddingModule,
    interaction_module: SimilarityModule,
    input_preproc_module: InputFeaturesPreprocessorModule,
    output_postproc_module: OutputPostprocessorModule,
    verbose: bool,
    activation_checkpoint: bool = False,
) -> SequentialEncoderWithLearnedSimilarityModule:
    """
    Factory function to create a sequential encoder model based on the specified module type.
    
    A factory function is a design pattern that provides an interface for creating objects
    without explicitly specifying their exact classes. Key aspects:

    1. Encapsulation: Hides object creation logic from the client code
       - Client code doesn't need to know implementation details
       - Creation logic is centralized in one place
       
    2. Flexibility: Allows runtime decisions about which class to instantiate
       - Can choose different implementations based on parameters
       - Easy to add new types without changing client code
       
    3. Consistency: Ensures objects are created in a standardized way
       - Enforces proper initialization and configuration
       - Reduces duplicate object creation code
       
    In this case, this factory function:
    - Takes configuration parameters as input
    - Decides whether to create SASRec or HSTU model
    - Configures the model with provided modules and parameters
    - Returns a fully initialized sequential encoder

    The sequential encoder processes user interaction sequences to generate recommendations:
    1. Takes sequence of user-item interactions as input
    2. Processes through transformer-based architecture
    3. Generates predictions for next items
    
    Args:
        module_type: Type of transformer architecture ("SASRec" or "HSTU")
        max_sequence_length: Maximum length of input sequences
        max_output_length: Maximum length of output predictions
        embedding_module: Handles conversion of item IDs to embeddings
        interaction_module: Core transformer layers for sequence processing
        input_preproc_module: Adds positional embeddings to inputs
        output_postproc_module: Normalizes output embeddings
        verbose: Whether to print model architecture details
        activation_checkpoint: Whether to use gradient checkpointing
        
    Returns:
        SequentialEncoderWithLearnedSimilarityModule: Configured encoder model
        
    The model processes sequences by:
    1. Converting item IDs to embeddings via embedding_module
    2. Adding positional information via input_preproc_module  
    3. Passing through transformer layers in interaction_module
    4. Normalizing outputs via output_postproc_module
    5. Computing similarity scores for next-item prediction
    
    Supports two architectures:
    - SASRec: Self-Attention based Sequential Recommendation
    - HSTU: Hierarchical Sequential Transformer Unit
      
    The two supported encoder architectures are:

    1. SASRec (Self-Attention based Sequential Recommendation):
       - Classic transformer architecture for sequential recommendation
       - Uses self-attention to capture item-item relationships
       - Each layer has:
         - Multi-head self attention
         - Position-wise feed forward network
         - Layer normalization and residual connections
       - Processes full sequence at once
       - Good for capturing long-range dependencies
       - Memory efficient compared to RNN approaches

    2. HSTU (Hierarchical Sequential Transformer Unit):
       - Novel hierarchical transformer architecture
       - Processes sequence in hierarchical chunks
       - Each chunk processed by:
         - Local self-attention within chunk
         - Global attention across chunk summaries
         - Hierarchical position embeddings
       - Benefits:
         - More efficient than full attention
         - Can handle longer sequences
         - Maintains both local and global context
         - Reduces memory and compute requirements
       - Especially suited for long user histories

    Both encoders:
    - Take preprocessed item sequences as input
    - Apply positional embeddings
    - Process through transformer layers
    - Generate contextualized representations
    - Output embeddings for next-item prediction
    - Support flexible sequence lengths
    - Can be trained end-to-end

    The key difference is in how they process sequences:
    - SASRec uses standard full self-attention
    - HSTU uses hierarchical attention for better efficiency
    """
    if module_type == "SASRec":
        model = sasrec_encoder(
            max_sequence_length=max_sequence_length,
            max_output_length=max_output_length,
            embedding_module=embedding_module,
            similarity_module=interaction_module,
            input_preproc_module=input_preproc_module,
            output_postproc_module=output_postproc_module,
            activation_checkpoint=activation_checkpoint,
            verbose=verbose,
        )
    elif module_type == "HSTU":
        model = hstu_encoder(
            max_sequence_length=max_sequence_length,
            max_output_length=max_output_length,
            embedding_module=embedding_module,
            similarity_module=interaction_module,
            input_preproc_module=input_preproc_module,
            output_postproc_module=output_postproc_module,
            activation_checkpoint=activation_checkpoint,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unsupported module_type {module_type}")
    return model
