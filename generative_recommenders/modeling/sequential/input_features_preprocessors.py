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

import abc
import math
from typing import Dict, Tuple

import torch

from generative_recommenders.modeling.initialization import truncated_normal


class InputFeaturesPreprocessorModule(torch.nn.Module):
    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


class LearnablePositionalEmbeddingInputFeaturesPreprocessor(
    InputFeaturesPreprocessorModule
):
    """
    Input features preprocessor that adds learnable positional embeddings to item embeddings to capture sequential order.

    Positional embeddings are crucial because transformer-based models have no inherent way to 
    understand sequence order. Without position information, the model would treat sequences like 
    ["movie1", "movie2", "movie3"] and ["movie2", "movie1", "movie3"] as identical, since 
    self-attention operates on unordered sets.
    
    The positional order is encoded through learnable embeddings:
    1. Each position index (0 to max_sequence_len-1) maps to a unique embedding vector
    2. The position embeddings are initialized using truncated normal distribution
    3. During forward pass, position embeddings are added to the item embeddings
    4. The combined embeddings allow the model to distinguish item order

    The position embeddings are added (element-wise sum) to the item embeddings, not concatenated.
    This preserves the embedding dimension while allowing the model to learn position-specific 
    modifications to each item embedding dimension.

    For example, if we have:
    - Item embedding: [0.1, 0.2, 0.3]
    - Position embedding for pos 0: [0.01, -0.02, 0.04] 
    
    The final embedding would be:
    [0.11, 0.18, 0.34] = [0.1 + 0.01, 0.2 + (-0.02), 0.3 + 0.04]

    This additive approach:
    - Maintains same embedding dimension throughout network
    - Allows position information to modulate each item embedding dimension
    - Is computationally efficient compared to concatenation
    - Follows standard practice in transformer architectures

    For example, given sequence [item1, item2, item3]:
    - Position 0 embedding is added to item1 embedding
    - Position 1 embedding is added to item2 embedding  
    - Position 2 embedding is added to item3 embedding

    This creates position-aware representations where:
    - Same item at different positions has different final embeddings
    - Model can learn position-specific patterns and biases
    - Relative positions are preserved through unique embedding combinations

    The positional embeddings:
    - Assign a unique learnable vector to each position in the sequence
    - Allow the model to distinguish between items appearing at different positions
    - Help capture temporal patterns and dependencies in user behavior
    - Enable the model to learn position-specific item preferences

    For example, in movie recommendations:
    - Recent movies may be more relevant for next-item prediction
    - The order of movies watched can indicate evolving user tastes
    - Some movies are more likely to be watched as first-time vs follow-up content

    The embeddings are learned during training to capture meaningful position information
    that helps improve recommendation accuracy.
    
    This module:
    - Creates learnable embeddings for each position in the sequence
    - Adds position embeddings to item embeddings after scaling
    - Applies dropout for regularization
    - Masks out padding tokens (id=0)
    
    Args:
        max_sequence_len: Maximum length of input sequences
        embedding_dim: Dimension of item and position embeddings
        dropout_rate: Dropout probability applied to embeddings
        
    The forward pass:
    1. Scales item embeddings by sqrt(embedding_dim)
    2. Adds position embeddings based on sequence position
    3. Applies dropout
    4. Masks out padding tokens
    5. Returns sequence lengths, processed embeddings, and attention mask
    
    References:
        - Vaswani et al. "Attention Is All You Need" (2017)
        - Position embeddings help model learn order dependencies
        - Scaling by sqrt(dim) helps maintain variance after adding embeddings
        - Dropout prevents overfitting on position patterns
    """
    def __init__(
        self,
        max_sequence_len: int,
        embedding_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = embedding_dim
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len,
            self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self.reset_state()

    def debug_str(self) -> str:
        return f"posi_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def forward(
        self,
        past_lengths: torch.Tensor,  # [B] sequence lengths
        past_ids: torch.Tensor,      # [B, N] item ids
        past_embeddings: torch.Tensor,  # [B, N, D] item embeddings
        past_payloads: Dict[str, torch.Tensor],  # Additional payload tensors
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass that adds positional embeddings to item embeddings.
        
        The item embeddings and positional embeddings are added in the body of this 
        function using:
        user_embeddings = past_embeddings * (self._embedding_dim**0.5) + self._pos_emb(...)
        
        This creates position-aware embeddings by:
        1. Scaling item embeddings by sqrt(embedding_dim)
        2. Adding learned positional embeddings based on sequence position
        
        Args:
            past_lengths: Tensor of sequence lengths for each batch [B]
            past_ids: Tensor of item IDs for each position [B, N] 
            past_embeddings: Tensor of item embeddings [B, N, D]
            past_payloads: Additional payload tensors
            
        Returns:
            Tuple of:
            - past_lengths: Original sequence lengths [B]
            - user_embeddings: Position-aware embeddings [B, N, D]
              where:
              - B is the batch size (number of sequences)
              - N is the sequence length (number of items per sequence)
              - D is the embedding dimension size
              Called "user_embeddings" because they represent the user's historical
              sequence of items with positional information added. These embeddings
              capture both the items and their order in the user's history.
            - valid_mask: Binary mask for valid (non-padding) positions [B, N, 1]
        """
        B, N = past_ids.size()
        D = past_embeddings.size(-1)

        user_embeddings = past_embeddings * (self._embedding_dim**0.5) + self._pos_emb(
            torch.arange(N, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        # The dropout layer randomly zeros out elements of user_embeddings with probability dropout_rate
        # This means some positions in the sequence will have their embeddings set to 0
        # For example with dropout_rate=0.1:
        # - ~90% of embeddings will keep their original values
        # - ~10% of embeddings will be set to 0 
        # This helps prevent overfitting by forcing the model to not rely too heavily on any single position
        # The dropout is applied independently to each element in each embedding vector
        # During inference/evaluation, dropout is disabled and all embeddings are scaled by (1-dropout_rate)
        user_embeddings = self._emb_dropout(user_embeddings)

        # Create mask for valid (non-padding) positions
        # past_ids contains item IDs, where 0 is reserved as the padding token
        # When loading sequences, shorter sequences are padded with 0s to match max length
        # For example, if max length is 5, a sequence of length 3 would be:
        # [item1, item2, item3, 0, 0]
        # The valid_mask will be: [1.0, 1.0, 1.0, 0.0, 0.0]
        valid_mask = (past_ids != 0).unsqueeze(-1).float()  # [B, N, 1]
        
        # Zero out embeddings at padding positions by multiplying with valid_mask
        user_embeddings *= valid_mask

        # Return values:
        # past_lengths: Original sequence lengths tensor [B] indicating number of actual items
        #   For example: If sequences are [[1,2,3,0,0], [1,2,0,0,0]]
        #   past_lengths would be [3, 2] 
        #   This is determined when loading the data, counting non-zero items
        # user_embeddings: Position-aware item embeddings with dropout applied [B, N, D]
        #   - Combines item embeddings and positional embeddings
        #   - Scaled by sqrt(embedding_dim) 
        #   - Has dropout applied for regularization
        #   - Masked to zero out padding positions
        # valid_mask: Binary mask tensor [B, N, 1] where:
        #   - 1.0 indicates valid item positions (past_ids != 0)
        #   - 0.0 indicates padding positions (past_ids == 0)
        #   - Used to zero out embeddings at padding positions
        return past_lengths, user_embeddings, valid_mask


class LearnablePositionalEmbeddingRatedInputFeaturesPreprocessor(
    InputFeaturesPreprocessorModule
):
    def __init__(
        self,
        max_sequence_len: int,
        item_embedding_dim: int,
        dropout_rate: float,
        rating_embedding_dim: int,
        num_ratings: int,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = item_embedding_dim + rating_embedding_dim
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len,
            self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self._rating_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_ratings,
            rating_embedding_dim,
        )
        self.reset_state()

    def debug_str(self) -> str:
        return f"posir_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )
        truncated_normal(
            self._rating_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = past_ids.size()

        user_embeddings = torch.cat(
            [past_embeddings, self._rating_emb(past_payloads["ratings"].int())],
            dim=-1,
        ) * (self._embedding_dim**0.5) + self._pos_emb(
            torch.arange(N, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (past_ids != 0).unsqueeze(-1).float()  # [B, N, 1]
        user_embeddings *= valid_mask
        return past_lengths, user_embeddings, valid_mask


class CombinedItemAndRatingInputFeaturesPreprocessor(InputFeaturesPreprocessorModule):
    def __init__(
        self,
        max_sequence_len: int,
        item_embedding_dim: int,
        dropout_rate: float,
        num_ratings: int,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = item_embedding_dim
        # Due to [item_0, rating_0, item_1, rating_1, ...]
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len * 2,
            self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self._rating_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_ratings,
            item_embedding_dim,
        )
        self.reset_state()

    def debug_str(self) -> str:
        return f"combir_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )
        truncated_normal(
            self._rating_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def get_preprocessed_ids(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns (B, N * 2,) x int64.
        """
        B, N = past_ids.size()
        return torch.cat(
            [
                past_ids.unsqueeze(2),  # (B, N, 1)
                past_payloads["ratings"].to(past_ids.dtype).unsqueeze(2),
            ],
            dim=2,
        ).reshape(B, N * 2)

    def get_preprocessed_masks(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns (B, N * 2,) x bool.
        """
        B, N = past_ids.size()
        return (past_ids != 0).unsqueeze(2).expand(-1, -1, 2).reshape(B, N * 2)

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = past_ids.size()
        D = past_embeddings.size(-1)

        user_embeddings = torch.cat(
            [
                past_embeddings,  # (B, N, D)
                self._rating_emb(past_payloads["ratings"].int()),
            ],
            dim=2,
        ) * (self._embedding_dim**0.5)
        user_embeddings = user_embeddings.view(B, N * 2, D)
        user_embeddings = user_embeddings + self._pos_emb(
            torch.arange(N * 2, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (
            self.get_preprocessed_masks(
                past_lengths,
                past_ids,
                past_embeddings,
                past_payloads,
            )
            .unsqueeze(2)
            .float()
        )  # (B, N * 2, 1,)
        user_embeddings *= valid_mask
        return past_lengths * 2, user_embeddings, valid_mask
