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
from collections import OrderedDict
from typing import List, Tuple

import torch
import torch.nn.functional as F

from generative_recommenders.rails.similarities.module import SimilarityModule

from torch.utils.checkpoint import checkpoint


class NegativesSampler(torch.nn.Module):
    def __init__(self, l2_norm: bool, l2_norm_eps: float) -> None:
        super().__init__()

        self._l2_norm: bool = l2_norm
        self._l2_norm_eps: float = l2_norm_eps

    def normalize_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self._maybe_l2_norm(x)

    def _maybe_l2_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self._l2_norm:
            x = x / torch.clamp(
                torch.linalg.norm(x, ord=2, dim=-1, keepdim=True),
                min=self._l2_norm_eps,
            )
        return x

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def process_batch(
        self,
        ids: torch.Tensor,
        presences: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> None:
        pass

    @abc.abstractmethod
    def forward(
        self,
        positive_ids: torch.Tensor,
        num_to_sample: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            A tuple of (sampled_ids, sampled_negative_embeddings).
        """
        pass


class LocalNegativesSampler(NegativesSampler):
    """
    LocalNegativesSampler is a sampler that samples negative items from the full item set (not just the current batch).
    It uses a stored item embedding table to look up embeddings for randomly sampled negative items.

    The key difference from InBatchNegativesSampler is that this sampler:
    1. Samples negative items randomly from the full item set rather than just items in the batch
    2. Uses a provided item embedding table to look up embeddings for sampled negatives
    3. Does not require processing batches of embeddings since it samples globally
    4. May have higher memory usage since it needs the full item embedding table
    
    InBatchNegativesSampler on the other hand:
    1. Only samples negatives from items that appear in the current batch
    2. Uses pre-computed embeddings passed in process_batch() rather than an embedding table
    3. Requires processing each batch to build the pool of candidate negatives
    4. More memory efficient since it only needs batch embeddings
    """
    def __init__(
        self,
        num_items: int,
        item_emb: torch.nn.Embedding,
        all_item_ids: List[int],
        l2_norm: bool,
        l2_norm_eps: float,
    ) -> None:
        super().__init__(l2_norm=l2_norm, l2_norm_eps=l2_norm_eps)

        self._num_items: int = len(all_item_ids)
        self._item_emb: torch.nn.Embedding = item_emb
        self.register_buffer("_all_item_ids", torch.tensor(all_item_ids))

    def debug_str(self) -> str:
        sampling_debug_str = (
            f"local{f'-l2-eps{self._l2_norm_eps}' if self._l2_norm else ''}"
        )
        return sampling_debug_str

    def process_batch(
        self,
        ids: torch.Tensor,
        presences: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> None:
        pass

    def forward(
        self,
        positive_ids: torch.Tensor,
        num_to_sample: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            A tuple of (sampled_ids, sampled_negative_embeddings).
        """
        # assert torch.max(torch.abs(self._item_emb(positive_ids) - positive_embeddings)) < 1e-4
        output_shape = positive_ids.size() + (num_to_sample,)
        sampled_offsets = torch.randint(
            low=0,
            high=self._num_items,
            size=output_shape,
            dtype=positive_ids.dtype,
            device=positive_ids.device,
        )
        sampled_ids = self._all_item_ids[sampled_offsets.view(-1)].reshape(output_shape)
        return sampled_ids, self.normalize_embeddings(self._item_emb(sampled_ids))


class InBatchNegativesSampler(NegativesSampler):
    def __init__(
        self,
        l2_norm: bool,
        l2_norm_eps: float,
        dedup_embeddings: bool,
    ) -> None:
        """
        Initialize InBatchNegativesSampler for efficient negative sampling.

        This sampler uses other items in the current batch as negative examples,
        which is memory efficient and can work well in practice.

        Args:
            l2_norm: Whether to L2 normalize embeddings
            l2_norm_eps: Epsilon for L2 normalization numerical stability
            dedup_embeddings: Whether to deduplicate embeddings in batch
                            If True, will only use unique items as negatives
                            If False, may sample same item multiple times

        Key features:
        - Uses items from current batch as negatives instead of sampling from full corpus
        - Option to deduplicate embeddings to avoid repeated negatives
        - L2 normalization of embeddings for stable training
        - Memory efficient since no need to store full item embedding matrix
        """
        super().__init__(l2_norm=l2_norm, l2_norm_eps=l2_norm_eps)

        self._dedup_embeddings: bool = dedup_embeddings

    def debug_str(self) -> str:
        sampling_debug_str = (
            f"in-batch{f'-l2-eps{self._l2_norm_eps}' if self._l2_norm else ''}"
        )
        if self._dedup_embeddings:
            sampling_debug_str += "-dedup"
        return sampling_debug_str

    def process_batch(
        self,
        ids: torch.Tensor,
        presences: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> None:
        """
        Args:
           ids: (N') or (B, N) x int64
           presences: (N') or (B, N) x bool
           embeddings: (N', D) or (B, N, D) x float
        """
        assert ids.size() == presences.size()
        assert ids.size() == embeddings.size()[:-1]
        if self._dedup_embeddings:
            valid_ids = ids[presences]
            unique_ids, unique_ids_inverse_indices = torch.unique(
                input=valid_ids, sorted=False, return_inverse=True
            )
            device = unique_ids.device
            unique_embedding_offsets = torch.empty(
                (unique_ids.numel(),),
                dtype=torch.int64,
                device=device,
            )
            unique_embedding_offsets[unique_ids_inverse_indices] = torch.arange(
                valid_ids.numel(), dtype=torch.int64, device=device
            )
            unique_embeddings = embeddings[presences][unique_embedding_offsets, :]
            self._cached_embeddings = self._maybe_l2_norm(unique_embeddings)
            self._cached_ids = unique_ids
        else:
            self._cached_embeddings = self._maybe_l2_norm(embeddings[presences])
            self._cached_ids = ids[presences]

    def get_all_ids_and_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._cached_ids, self._cached_embeddings

    def forward(
        self,
        positive_ids: torch.Tensor,
        num_to_sample: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            A tuple of (sampled_ids, sampled_negative_embeddings,).
        """
        X = self._cached_ids.size(0)
        sampled_offsets = torch.randint(
            low=0,
            high=X,
            size=positive_ids.size() + (num_to_sample,),
            dtype=positive_ids.dtype,
            device=positive_ids.device,
        )
        return (
            self._cached_ids[sampled_offsets],
            self._cached_embeddings[sampled_offsets],
        )


class AutoregressiveLoss(torch.nn.Module):
    @abc.abstractmethod
    def jagged_forward(
        self,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        """
        Variant of forward() when the tensors are already in jagged format.

        Args:
            output_embeddings: [N', D] x float, embeddings for the current
                input sequence.
            supervision_ids: [N'] x int64, (positive) supervision ids.
            supervision_embeddings: [N', D] x float.
            supervision_weights: Optional [N'] x float. Optional weights for
                masking out invalid positions, or reweighting supervision labels.
            negatives_sampler: sampler used to obtain negative examples paired with
                positives.

        Returns:
            (1), loss for the current engaged sequence.
        """
        pass

    @abc.abstractmethod
    def forward(
        self,
        lengths: torch.Tensor,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        """
        Args:
            lengths: [B] x int32 representing number of non-zero elements per row.
            output_embeddings: [B, N, D] x float, embeddings for the current
                input sequence.
            supervision_ids: [B, N] x int64, (positive) supervision ids.
            supervision_embeddings: [B, N, D] x float.
            supervision_weights: Optional [B, N] x float. Optional weights for
                masking out invalid positions, or reweighting supervision labels.
            negatives_sampler: sampler used to obtain negative examples paired with
                positives.

        Returns:
            (1), loss for the current engaged sequence.
        """
        pass


class BCELoss(AutoregressiveLoss):
    def __init__(
        self,
        temperature: float,
        model: SimilarityModule,
    ) -> None:
        """
        Initialize BCELoss module.

        Args:
            temperature: Float scaling factor for logits. Higher values make the model more 
                        confident in its predictions.
            model: SimilarityModule that computes interaction scores between input and target 
                  embeddings.

        BCELoss implements binary cross entropy loss for sequential recommendation.
    
        For each position in the sequence, it:
        1. Computes similarity scores between the input embedding and positive target
        2. Samples negative items and computes similarity scores with those
        The negative sampling process helps create contrastive pairs for training:
        - For each positive item, sample 1 negative item the user hasn't interacted with
        - This provides signal about what items are not relevant for the user
        - The model learns to score positive items higher than negative ones
        3. Applies binary cross entropy loss to encourage high scores for positives 
           and low scores for negatives
        
        This creates a contrastive learning signal that helps the model learn to 
        distinguish between items the user would engage with vs not engage with.
        """
        super().__init__()
        self._temperature: float = temperature
        self._model = model

    def jagged_forward(
        self,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        """
        Forward pass for jagged/variable length sequences.

        This method handles sequences where each batch element can have a different length.
        The input tensors are "jagged" - flattened into 1D tensors where sequential elements 
        are concatenated. Jagged tensors are tensors where the dimensions are not uniform,
        often used in scenarios like sparse data or variable-length sequences.
        
        References "jagged":
        - https://pytorch.org/FBGEMM/fbgemm_gpu-overview/jagged-tensor-ops/JaggedTensorOps.html

        Args:
            output_embeddings: Flattened tensor of shape [N', D] containing embeddings for each 
                position in all sequences, where
                N' is the total number of sequence elements across all batches (flattened)
                D is the embedding dimension for each item
            supervision_ids: Flattened tensor [N'] containing target item IDs for each position
            supervision_embeddings: Flattened tensor [N', D] containing target item embeddings
                supervision_embeddings represents the ground truth/target embeddings that the model
                should learn to predict. The term "supervision" is used because these embeddings
                provide the supervisory signal (labels) for training - they represent the correct
                outputs that guide/supervise the model's learning process through the loss function.
                This is standard terminology in machine learning where "supervised learning" refers
                to training with labeled data.
            supervision_weights: Flattened tensor [N'] containing loss weights for each position
            negatives_sampler: Module for sampling negative items

        Returns:
            Scalar loss averaged over all positions with non-zero weights
        """
        # Validate input tensor shapes match expected dimensions
        # output_embeddings and supervision_embeddings should be [N', D]
        # supervision_ids and supervision_weights should be [N']
        # where N' is total sequence elements across batches
        # and D is embedding dimension
        assert output_embeddings.size() == supervision_embeddings.size()
        assert supervision_ids.size() == supervision_embeddings.size()[:-1]
        assert supervision_ids.size() == supervision_weights.size()

        # Sample negative items for contrastive learning
        # Returns:
        # - sampled_ids: [N', 1] tensor of sampled negative item IDs
        # - sampled_negative_embeddings: [N', 1, D] tensor of embeddings for sampled negatives
        # The negative samples are used to create contrastive pairs with the positive examples
        # for training the model to distinguish between relevant and irrelevant items
        sampled_ids, sampled_negative_embeddings = negatives_sampler(
            positive_ids=supervision_ids,
            num_to_sample=1,
        )

        # Compute logits for positive and negative items
        # Logits are raw prediction scores before applying softmax/sigmoid activation
        # They represent how likely each item is to be the next item in the sequence
        # Higher logit = model thinks that item is more likely to be next
        # 
        # positive_logits: [N'] tensor of logits for positive items (true next items)
        # sampled_negatives_logits: [N'] tensor of logits for sampled negative items
        # sampled_negatives_valid_mask: [N'] tensor indicating if negative samples are valid
        #   A negative sample is considered valid if it is different from the positive item
        #   i.e. sampled_id != supervision_id for that position
        #   Invalid negatives (where sampled_id = supervision_id) are masked out
        # 
        # loss_weights: [N'] tensor of loss weights for each position
        #   Determined by:
        #   1. supervision_weights: Initial weights provided for each position
        #   2. sampled_negatives_valid_mask: Only consider positions where negative samples are valid
        #   Final loss_weights = supervision_weights * sampled_negatives_valid_mask
        #
        # weighted_losses: [N'] tensor of weighted losses for each position
        #   Computed as:
        #   The loss formula combines two binary cross entropy terms:
        #   1. BCE between positive_logits and 1's: -log(sigmoid(positive_logits))
        #   2. BCE between negative_logits and 0's: -log(1 - sigmoid(negative_logits))
        #   The two terms are averaged (*0.5) and weighted by loss_weights:
        #   weighted_losses = loss_weights * 0.5 * (-log(sigmoid(positive_logits)) - log(1 - sigmoid(negative_logits)))
        #   
        #   This combines the loss for positive and negative items, scales by loss_weights, and averages
        #   the result to get a single loss value for each position
        positive_logits = (
            self._model.interaction(
                input_embeddings=output_embeddings,  # [B, D] = [N', D]
                target_ids=supervision_ids.unsqueeze(1),  # [N', 1]
                target_embeddings=supervision_embeddings.unsqueeze(
                    1
                ),  # [N', D] -> [N', 1, D]
            )[0].squeeze(1)
            / self._temperature
        )  # [N']

        sampled_negatives_logits = (
            self._model.interaction(
                input_embeddings=output_embeddings,  # [N', D]
                target_ids=sampled_ids,  # [N', 1]
                target_embeddings=sampled_negative_embeddings,  # [N', 1, D]
            )[0].squeeze(1)
            / self._temperature
        )  # [N']
        sampled_negatives_valid_mask = (
            supervision_ids != sampled_ids.squeeze(1)
        ).float()  # [N']
        loss_weights = supervision_weights * sampled_negatives_valid_mask

        # The loss function implements a binary cross entropy (BCE) loss for sequential recommendation
        # 
        # For each position in the sequence:
        # 1. We have a positive item (the actual next item) and a sampled negative item
        # 2. The model produces logits (scores) for both items
        # 3. We want:
        #    - High probability (sigmoid(logit) close to 1) for positive items
        #    - Low probability (sigmoid(logit) close to 0) for negative items
        #
        # The BCE loss is computed in two parts:
        # 1. BCE between positive logits and target=1: -log(sigmoid(positive_logits))
        #    - This encourages high scores for positive items
        # 2. BCE between negative logits and target=0: -log(1-sigmoid(negative_logits)) 
        #    - This encourages low scores for negative items
        #
        # The losses are:
        # - Weighted by loss_weights to mask invalid positions
        # - Averaged (*0.5) between positive and negative terms
        # - Summed across all positions and normalized by sum of weights
        #
        # This trains the model to score actual next items higher than random negative items
        weighted_losses = (
            (
                F.binary_cross_entropy_with_logits(
                    # Compute BCE loss between positive logits and target=1
                    # positive_logits: scores for actual next items [N']
                    # target=1: we want sigmoid(positive_logits) close to 1
                    # reduction="none": return per-element losses without averaging
                    #
                    # Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html
                    #
                    # This function is used to compute the binary cross-entropy loss between the target and the input logits.
                    # It is a combination of a sigmoid layer and the binary cross-entropy loss, which is more numerically 
                    # stable than using a plain sigmoid followed by a binary cross-entropy loss.
                    input=positive_logits,
                    target=torch.ones_like(positive_logits),
                    reduction="none",
                )
                + F.binary_cross_entropy_with_logits(
                    input=sampled_negatives_logits,
                    target=torch.zeros_like(sampled_negatives_logits),
                    reduction="none",
                )
            )
            * loss_weights
            * 0.5
        )
        return weighted_losses.sum() / loss_weights.sum()

    def forward(
        self,
        lengths: torch.Tensor,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        """
        Args:
            lengths: Tensor of shape [B] containing sequence lengths for each batch element
                B: Batch size - number of sequences in the batch
                Each value indicates how many valid items are in that sequence
                Used to mask out padding and identify valid positions for loss computation
            output_embeddings: Tensor of shape [B, N, D] containing embeddings for the current
                input sequence.
                B: Batch size - number of sequences in the batch
                N: Sequence length - max number of items in each sequence 
                D: Embedding dimension - size of the embedding vectors
            supervision_ids: Tensor of shape [B, N] containing the ground truth (positive) item IDs.
            supervision_embeddings: Tensor of shape [B, N, D] containing embeddings for the ground
                truth items.
            supervision_weights: Optional tensor of shape [B, N] containing weights for masking
                invalid positions or reweighting supervision labels.

                Tensor of shape [B, N] containing weights for each position
                Used to:
                1. Mask out padding positions (weight=0) vs valid positions (weight=1)
                2. Optionally reweight importance of different positions in the sequence
                3. Gets converted to jagged format and multiplies final per-element losses
            negatives_sampler: Sampler object used to obtain negative examples to pair with the
                positive examples.

        Returns:
            Scalar tensor containing the loss value for the current sequence.
          
        # This method computes the autoregressive loss for sequential recommendation
        # by comparing output embeddings against supervision embeddings.
        #
        # The inputs are:
        # - lengths: Batch of sequence lengths indicating valid elements per row
        # - output_embeddings: Model output embeddings for input sequences 
        # - supervision_ids: Ground truth item IDs for next item prediction
        # - supervision_embeddings: Embeddings of ground truth next items
        # - supervision_weights: Optional weights for masking/reweighting
        # - negatives_sampler: Strategy for sampling negative examples
        #
        # The method:
        # 1. Validates input tensor shapes match
        # 2. Converts dense tensors to jagged format using sequence lengths
        # 3. Calls jagged_forward() with converted tensors to compute loss
        #
        # Returns a scalar loss value averaged over the batch
        """
        # Validate input tensor shapes match expected dimensions
        # output_embeddings and supervision_embeddings should be [B,N,D]
        # supervision_ids should be [B,N] (matches embeddings except for last dim)
        # supervision_weights should be [B,N] (matches supervision_ids shape)
        assert output_embeddings.size() == supervision_embeddings.size()
        assert supervision_ids.size() == supervision_embeddings.size()[:-1]

        # Convert dense tensors to jagged format using sequence lengths
        # jagged_id_offsets: Tensor of shape [B] containing cumulative sum of sequence lengths
        # jagged_supervision_ids: Tensor of shape [B,N'] containing supervision IDs in jagged format
        # jagged_supervision_weights: Tensor of shape [B,N'] containing supervision weights in jagged format
        jagged_id_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        # jagged_id_offsets contains the cumulative sum of sequence lengths
        # For example, if lengths = [2,3,1], then jagged_id_offsets = [0,2,5,6]
        # This is used to convert dense tensors to jagged format by indicating
        # the start indices for each sequence in the flattened jagged tensor
    
        # Convert dense tensors to jagged format for efficient processing
        # jagged_supervision_ids: Flattened tensor of supervision IDs
        # - Unsqueeze adds singleton dimension for fbgemm compatibility 
        # - Float conversion needed for fbgemm op
        # - Squeeze removes singleton dimension after conversion
        # - Long conversion restores integer dtype
        jagged_supervision_ids = (
            torch.ops.fbgemm.dense_to_jagged(
                # The dense_to_jagged op takes:
                # 1. Dense input tensor
                # 2. List of offsets tensors indicating sequence boundaries
                # Returns a list of jagged tensors, we take first element [0]
                supervision_ids.unsqueeze(-1).float(), [jagged_id_offsets]
            )[0]
            .squeeze(1)
            .long()
        )
        # jagged_supervision_weights: Flattened tensor of supervision weights
        # - Follows same process as supervision IDs but keeps float dtype
        # - Used to mask padding and optionally reweight positions
        jagged_supervision_weights = torch.ops.fbgemm.dense_to_jagged(
            supervision_weights.unsqueeze(-1),
            [jagged_id_offsets],
        )[0].squeeze(1)
        return self.jagged_forward(
            output_embeddings=torch.ops.fbgemm.dense_to_jagged(
                output_embeddings,
                [jagged_id_offsets],
            )[0],
            supervision_ids=jagged_supervision_ids,
            supervision_embeddings=torch.ops.fbgemm.dense_to_jagged(
                supervision_embeddings,
                [jagged_id_offsets],
            )[0],
            supervision_weights=jagged_supervision_weights,
            negatives_sampler=negatives_sampler,
        )


class BCELossWithRatings(AutoregressiveLoss):
    def __init__(
        self,
        temperature: float,
        model: SimilarityModule,
    ) -> None:
        super().__init__()
        self._temperature: float = temperature
        self._model = model

    def jagged_forward(
        self,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        supervision_ratings: torch.Tensor,
        negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        assert output_embeddings.size() == supervision_embeddings.size()
        assert supervision_ids.size() == supervision_embeddings.size()[:-1]
        assert supervision_ids.size() == supervision_weights.size()

        target_logits = (
            self._model.interaction(
                input_embeddings=output_embeddings,  # [B, D] = [N', D]
                target_ids=supervision_ids.unsqueeze(1),  # [N', 1]
                target_embeddings=supervision_embeddings.unsqueeze(
                    1
                ),  # [N', D] -> [N', 1, D]
            )[0].squeeze(1)
            / self._temperature
        )  # [N', 1]

        weighted_losses = (
            F.binary_cross_entropy_with_logits(
                input=target_logits,
                target=supervision_ratings.to(dtype=target_logits.dtype),
                reduction="none",
            )
        ) * supervision_weights
        return weighted_losses.sum() / supervision_weights.sum()

    def forward(
        self,
        lengths: torch.Tensor,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        supervision_ratings: torch.Tensor,
        negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        """
        Args:
          lengths: [B] x int32 representing number of non-zero elements per row.
          output_embeddings: [B, N, D] x float, embeddings for the current
              input sequence.
          supervision_ids: [B, N] x int64, (positive) supervision ids.
          supervision_embeddings: [B, N, D] x float.
          supervision_weights: Optional [B, N] x float. Optional weights for
              masking out invalid positions, or reweighting supervision labels.
          negatives_sampler: sampler used to obtain negative examples paired with
              positives.
        Returns:
          (1), loss for the current engaged sequence.
        """
        assert output_embeddings.size() == supervision_embeddings.size()
        assert supervision_ids.size() == supervision_embeddings.size()[:-1]
        jagged_id_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        jagged_supervision_ids = (
            torch.ops.fbgemm.dense_to_jagged(
                supervision_ids.unsqueeze(-1).float(), [jagged_id_offsets]
            )[0]
            .squeeze(1)
            .long()
        )
        jagged_supervision_weights = torch.ops.fbgemm.dense_to_jagged(
            supervision_weights.unsqueeze(-1),
            [jagged_id_offsets],
        )[0].squeeze(1)
        return self.jagged_forward(
            output_embeddings=torch.ops.fbgemm.dense_to_jagged(
                output_embeddings,
                [jagged_id_offsets],
            )[0],
            supervision_ids=jagged_supervision_ids,
            supervision_embeddings=torch.ops.fbgemm.dense_to_jagged(
                supervision_embeddings,
                [jagged_id_offsets],
            )[0],
            supervision_weights=jagged_supervision_weights,
            supervision_ratings=torch.ops.fbgemm.dense_to_jagged(
                supervision_ratings.unsqueeze(-1),
                [jagged_id_offsets],
            )[0].squeeze(1),
            negatives_sampler=negatives_sampler,
        )
