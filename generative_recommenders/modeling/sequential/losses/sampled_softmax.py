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

from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from generative_recommenders.modeling.sequential.autoregressive_losses import (
    AutoregressiveLoss,
    NegativesSampler,
)

from torch.utils.checkpoint import checkpoint


class SampledSoftmaxLoss(AutoregressiveLoss):
    """
    Sampled Softmax Loss (SSL) is a loss function used in autoregressive models for 
    next-item prediction. It treats each position in the sequence as a multi-class 
    classification problem over all possible items, allowing the model to learn 
    relative preferences between items.
    """
    def __init__(
        self,
        num_to_sample: int,
        softmax_temperature: float,
        model,
        activation_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            num_to_sample: Number of negative samples to draw for each positive sample.
            softmax_temperature: Temperature parameter for scaling logits before softmax.
                Higher values make the distribution more uniform, lower values make it more peaked.
            model: The model to use for the loss function.
            activation_checkpoint: Whether to use activation checkpointing.
        """
        super().__init__()

        self._num_to_sample: int = num_to_sample
        self._softmax_temperature: float = softmax_temperature
        self._model = model
        self._activation_checkpoint: bool = activation_checkpoint

    def jagged_forward(
        self,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negatives_sampler: NegativesSampler,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            output_embeddings: [B, N, D] x float, embeddings for the current
                input sequence.
                # N represents the sequence length dimension in the tensor shape
                # [B, N, D] where:
                # B = batch size
                # N = sequence length
                # D = embedding dimension
            supervision_ids: [B, N] x int64, (positive) supervision ids.
            supervision_embeddings: [B, N, D] x float.
            supervision_weights: Optional [B, N] x float. Optional weights for
                masking out invalid positions, or reweighting supervision labels.
            negatives_sampler: sampler used to obtain negative examples paired with
                positives.
        """
        assert output_embeddings.size() == supervision_embeddings.size()
        assert supervision_ids.size() == supervision_embeddings.size()[:-1]
        assert supervision_ids.size() == supervision_weights.size()

        sampled_ids, sampled_negative_embeddings = negatives_sampler(
            positive_ids=supervision_ids,
            num_to_sample=self._num_to_sample,
        )
        # We don't normalize for BCE Loss because:
        # 1. BCE Loss operates on raw logits and applies sigmoid internally
        # 2. For sampled softmax, we need normalized embeddings to ensure 
        #    consistent similarity scores across different samples
        # 3. The softmax operation is sensitive to the scale of logits,
        #    so normalized embeddings help maintain stable gradients
        sampled_negative_embeddings = negatives_sampler.normalize_embeddings(
            sampled_negative_embeddings
        )
        positive_embeddings = negatives_sampler.normalize_embeddings(
            supervision_embeddings
        )
        positive_logits, aux_losses = self._model.similarity_fn(
        # similarity_fn computes raw similarity scores between query and item embeddings
            # These scores are called logits because they are the raw, unnormalized values
            # that will be passed into softmax later. The term logits comes from logistic
            # regression where raw scores are transformed via softmax into probabilities.
            #
            # Returns:
            # - positive_logits: [B, 1] tensor of raw similarity scores for positive items
            #   that will be normalized via softmax with negative examples
            # - aux_losses: Dict of auxiliary losses from the similarity computation
            #   Examples:
            #   - L2 regularization loss on embeddings
            #   - Dropout regularization loss
            #   - Attention dropout loss if using attention mechanism
            #   - Any other regularization losses from the similarity computation
            query_embeddings=output_embeddings,  # [B, D] = [N', D]
            item_ids=supervision_ids.unsqueeze(1),  # [N', 1]
            item_embeddings=positive_embeddings.unsqueeze(1),  # [N', D] -> [N', 1, D]
            **kwargs,
        )
        # Temperature scaling in softmax serves multiple purposes:
        # 1. Controls the "sharpness" of probability distribution:
        #    - Higher temperature (>1) makes distribution more uniform/smooth
        #    - Lower temperature (<1) makes distribution more peaked/concentrated
        # 2. Helps with gradient flow during training:
        #    - Very large logits can lead to vanishing gradients in softmax
        #    - Temperature scaling prevents this by scaling down large values
        # 3. Can be used for calibration:
        #    - Models often output overconfident predictions
        #    - Temperature scaling helps calibrate confidence levels
        #
        # Example:
        # softmax([1, 2, 3]) = [0.09, 0.24, 0.67]  # Original
        # softmax([1, 2, 3]/2) = [0.18, 0.27, 0.55]  # Higher temp = smoother
        # softmax([1, 2, 3]*2) = [0.02, 0.12, 0.86]  # Lower temp = sharper
        #
        positive_logits = positive_logits / self._softmax_temperature  # [0]
        sampled_negatives_logits, _ = self._model.similarity_fn(
            # We discard aux_losses for negative samples because:
            # 1. They are already included in the positive sample computation
            # 2. Including them again would double-count regularization terms
            # 3. The same embeddings may appear multiple times in negatives
            #    leading to incorrect weighting of regularization
            query_embeddings=output_embeddings,  # [N', D]
            item_ids=sampled_ids,  # [N', R]
            item_embeddings=sampled_negative_embeddings,  # [N', R, D]
            **kwargs,
        )  # [N', R]  # [0]
        sampled_negatives_logits = torch.where(
            # Mask out logits where sampled negatives match the positive item
            #
            # This prevents a positive item from being treated as its own negative
            # We use a large negative value (-5e4) to effectively zero out the 
            # probability after softmax because:
            #
            # 1. We want to completely exclude these cases from contributing to the loss
            # 2. After softmax, e^(-5e4) ≈ 0, so these items get ~0 probability
            # 3. If we didn't zero them out, the model would get conflicting signals:
            #    - Same item treated as both positive and negative example
            #    - This would harm learning since gradients would partially cancel
            #
            # In general, we use a large negative value to effectively zero out the probability 
            # after softmax. Common choices include:
            #
            # - float('-inf'): Most theoretically correct but can cause numerical issues
            #   because:
            #   1. When computing softmax(x) = exp(x)/sum(exp(x)), if any x is -inf,
            #      we get 0/0 which is undefined
            #   2. During backprop, gradients through -inf become NaN
            #   3. NaN values can then propagate through the network
            # 
            # - -1e9: Large enough to zero probability but avoids numerical instability
            # - -1e4 to -1e5: Common range that works well in practice
            #
            # We use -5e4 here as a good balance between:
            # 1. Being large enough to zero out probability (e^(-5e4) ≈ 0)
            # 2. Avoiding potential numerical instability of float('-inf')
            # 3. Following common practice in other implementations
            supervision_ids.unsqueeze(1) == sampled_ids,  # [N', R]
            -5e4,
            sampled_negatives_logits / self._softmax_temperature,
        )

        # We compute softmax over all logits (positive and negative) to get normalized probabilities.
        # By taking the negative log of the positive item's probability, we get a loss that:
        # - Approaches 0 when the positive probability is close to 1 (desired outcome)
        # - Increases as the positive probability decreases (penalizes wrong predictions)
        # This effectively trains the model to assign high probability to true items.
        jagged_loss = -F.log_softmax(
            # Concatenate positive and negative logits for softmax computation
            # positive_logits: [N', 1] - logits for true/positive items 
            # sampled_negatives_logits: [N', R] - logits for sampled negative items
            #   where R = num_to_sample (number of negative samples per positive)
            # Result: [N', R+1] - combined logits tensor for softmax
            #
            # This concatenates the logits for the positive (true) items with the logits
            # for the sampled negative items along dimension 1 (the sequence length dimension).
            # The result is a tensor containing all logits that will be input to softmax.
            #
            # We concatenate the logits to compute softmax over both positive and negative items
            # together. This is necessary because:
            # 1. Softmax needs to normalize over all possible choices to get valid probabilities
            # 2. The denominator in softmax must include both positive and negative logits
            # 3. This gives us proper relative probabilities between positive/negative items
            #
            # For example, if we have:
            # positive_logit = 5.0
            # negative_logits = [3.0, 2.0]
            #
            # softmax([5.0, 3.0, 2.0]) = [0.67, 0.20, 0.13]
            # The 0.67 probability for positive item properly accounts for competition from negatives
            #
            # If we computed softmax separately:
            # softmax([5.0]) = [1.0]  # Incorrect - ignores negatives
            # softmax([3.0, 2.0]) = [0.62, 0.38]  # Incorrect - ignores positive
            # 
            # 
            # Concatenate and apply softmax along dim=1, then select first element
            # Input shapes:
            # - positive_logits: [N', 1] - logits for true items
            # - sampled_negatives_logits: [N', R] - logits for negative samples
            # After cat: [N', R+1] - combined logits
            # After softmax: [N', R+1] - probabilities
            # After [:, 0]: [N'] - probabilities of true items only
            torch.cat([positive_logits, sampled_negatives_logits], dim=1), dim=1
        )[:, 0]

        return (
            # Apply supervision weights to the jagged loss
            # supervision_weights: [N'] - weights for each position
            # jagged_loss: [N'] - raw loss values before weighting
            # 
            # Multiply the loss by weights to:
            # 1. Mask out invalid positions (weight=0)
            # 2. Reweight importance of different positions
            # 3. Normalize by sum of weights to get per-position average
            #
            # For example:
            # jagged_loss = [1.2, 0.8, 1.5]
            # weights = [1.0, 0.0, 0.5]
            # 
            # weighted_loss = [1.2, 0.0, 0.75]  # Element-wise multiply
            # final_loss = sum(weighted_loss) / sum(weights) # Normalize
            jagged_loss * supervision_weights
        ).sum() / supervision_weights.sum(), aux_losses

    def forward(
        self,
        lengths: torch.Tensor,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negatives_sampler: NegativesSampler,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
            Tuple of (loss for the current engaged sequence, str-keyed aux_losses).
        """
        torch._assert(
            output_embeddings.size() == supervision_embeddings.size(),
            "Invalid supervision embeddings size.",
        )
        torch._assert(
            supervision_ids.size() == supervision_embeddings.size()[:-1],
            "Invalid supervision ids size.",
        )

        jagged_id_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        jagged_supervision_ids = (
            torch.ops.fbgemm.dense_to_jagged(
                supervision_ids.unsqueeze(-1).float(), [jagged_id_offsets]
            )[0]
            .squeeze(1)
            .long()
        )
        if "user_ids" in kwargs:
            # expand to jagged.
            max_length: int = int(lengths.max())
            kwargs["user_ids"] = torch.ops.fbgemm.dense_to_jagged(
                kwargs["user_ids"]
                .unsqueeze(1)
                .expand(-1, max_length)
                .unsqueeze(2),  # (B, max_length, 1)
                [jagged_id_offsets],
            )[0].squeeze(1)

        args = OrderedDict(
            [
                (
                    "output_embeddings",
                    torch.ops.fbgemm.dense_to_jagged(
                        output_embeddings,
                        [jagged_id_offsets],
                    )[0],
                ),
                ("supervision_ids", jagged_supervision_ids),
                (
                    "supervision_embeddings",
                    torch.ops.fbgemm.dense_to_jagged(
                        supervision_embeddings,
                        [jagged_id_offsets],
                    )[0],
                ),
                (
                    "supervision_weights",
                    torch.ops.fbgemm.dense_to_jagged(
                        supervision_weights.unsqueeze(-1),
                        [jagged_id_offsets],
                    )[0].squeeze(1),
                ),
                ("negatives_sampler", negatives_sampler),
            ]
        )
        args.update(kwargs)
        if self._activation_checkpoint:
            return checkpoint(
                self.jagged_forward,
                *args.values(),
                use_reentrant=False,
            )
        else:
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
                supervision_weights=torch.ops.fbgemm.dense_to_jagged(
                    supervision_weights.unsqueeze(-1),
                    [jagged_id_offsets],
                )[0].squeeze(1),
                negatives_sampler=negatives_sampler,
                **kwargs,
            )
