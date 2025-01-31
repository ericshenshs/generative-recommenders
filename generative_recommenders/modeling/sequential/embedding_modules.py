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

import torch

from generative_recommenders.modeling.initialization import truncated_normal


class EmbeddingModule(torch.nn.Module):
    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass


class LocalEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
    ) -> None:
        """
        LocalEmbeddingModule provides direct item ID to embedding lookup, 
        where each item has its own unique embedding.

        This is different from CategoricalEmbeddingModule which maps items to category embeddings,
        where multiple items that belong to the same category share the same embedding
        by first looking up the category ID for each item,
        then mapping those category IDs to shared embeddings.

        Key differences:
        - LocalEmbeddingModule: Each item ID maps to a unique embedding vector
        - CategoricalEmbeddingModule: Items are grouped by categories, items in same category share embedding
        
        Args:
            num_items: Number of unique items in the dataset
            item_embedding_dim: Dimension of the embedding vectors
        """
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb = torch.nn.Embedding(
            # Add 1 to num_items to account for padding_idx=0, which will be initialized to zeros.
            # The embedding layer maps item IDs to dense vector representations.
            #
            # Add 1 to num_items to reserve index 0 for padding. Valid item IDs will start at 1.
            # This ensures padding_idx=0 has its own dedicated index and doesn't overlap with real items.
            #
            # If we do not use padding:
            # - Cannot batch sequences of different lengths together efficiently
            # - No way to mask out/ignore invalid positions in sequences
            # - Model will try to learn embeddings for padding positions
            # - Gradient updates will be incorrect due to padding contributing
            # - Memory wasted storing/computing embeddings for padding tokens
            # - More complex logic needed to handle variable length sequences
            #
            # We reserve index 0 for padding to handle variable length sequences:
            # - When batching sequences of different lengths, shorter sequences are padded to match longest
            # - Padding tokens should not contribute to model predictions or loss
            # - By using padding_idx=0, the embeddings at index 0 are initialized to zero vectors and won't be updated during training
            # - This effectively makes padding tokens "invisible" to the model's computations
            # 
            # torch.nn.Embedding provides several advantages over a static dictionary:
            # - Embeddings are learnable parameters updated during training
            # - Supports automatic differentiation and backpropagation 
            # - Optimized for efficient parallel lookup on GPU
            # - Handles padding and out-of-bounds indices gracefully
            #
            # Args:
            #   num_items: Number of items in the dataset
            #   item_embedding_dim: Dimension of the item embedding vectors
            #   padding_idx: Index to be initialized to zeros (0 in this case)
            #
            # Reference: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
            #
            num_embeddings=num_items + 1,
            # Dimension of each embedding vector 
            embedding_dim=item_embedding_dim,
            # Index 0 is used for padding and will be initialized to zeros
            # If specified, the entries at padding_idx do not contribute to the gradient; 
            # therefore, the embedding vector at padding_idx is not updated during training, 
            # i.e. it remains as a fixed “pad”. For a newly constructed Embedding, 
            # the embedding vector at padding_idx will default to all zeros, 
            # but can be updated to another value to be used as the padding vector.
            padding_idx=0
        )
        self.reset_params()

    def debug_str(self) -> str:
        return f"local_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class CategoricalEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        item_id_to_category_id: torch.Tensor,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.register_buffer("_item_id_to_category_id", item_id_to_category_id)
        self.reset_params()

    def debug_str(self) -> str:
        return f"cat_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = self._item_id_to_category_id[(item_ids - 1).clamp(min=0)] + 1
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim
