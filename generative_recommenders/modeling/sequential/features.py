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

from typing import Dict, NamedTuple, Optional, Tuple

import torch


class SequentialFeatures(NamedTuple):
    """
    Represents a user's historical interactions with items.
    Contains:
    - past_lengths: (B,) Length of each user's history
    - past_ids: (B, N,) Movie IDs in user histories
    - past_embeddings: (B, N, D) Embeddings for each movie in user history
    - past_payloads: Dict[str, torch.Tensor] Additional metadata about each interaction
    """
    # (B,) x int64. Requires past_lengths[i] > 0 \forall i.
    past_lengths: torch.Tensor
    # (B, N,) x int64. 0 denotes valid ids.
    past_ids: torch.Tensor
    # (B, N, D) x float.
    past_embeddings: Optional[torch.Tensor]
    # Implementation-specific payloads.
    # e.g., past timestamps, past event_types (e.g., clicks, likes), etc.
    past_payloads: Dict[str, torch.Tensor]


def movielens_seq_features_from_row(
    row: Dict[str, torch.Tensor],
    device: int,
    max_output_length: int,
) -> Tuple[SequentialFeatures, torch.Tensor, torch.Tensor]:
    """Converts a row of MovieLens data into sequential features format.

    Args:
        row (Dict[str, torch.Tensor]): Dictionary containing MovieLens data tensors:
            - history_lengths: (B,) Length of each user's history
            - historical_ids: (B, N) Movie IDs in user histories
            - historical_ratings: (B, N) Ratings in user histories
            - historical_timestamps: (B, N) Timestamps in user histories
            - target_ids: (B,) Target movie IDs to predict
            - target_ratings: (B,) Target ratings to predict
            - target_timestamps: (B,) Target timestamps
        device (int): GPU device ID to place tensors on
        max_output_length (int): If > 0, pad histories to this length and include target

    Returns:
        Tuple containing:
        - SequentialFeatures: Named tuple with user history data
        - target_ids: (B, 1) Target movie IDs
        - target_ratings: (B, 1) Target ratings
    """
    historical_lengths = row["history_lengths"].to(device)  # [B]
    historical_ids = row["historical_ids"].to(device)  # [B, N]
    historical_ratings = row["historical_ratings"].to(device)
    historical_timestamps = row["historical_timestamps"].to(device)
    target_ids = row["target_ids"].to(device).unsqueeze(1)  # [B, 1]
    target_ratings = row["target_ratings"].to(device).unsqueeze(1)
    target_timestamps = row["target_timestamps"].to(device).unsqueeze(1)
    if max_output_length > 0:
        B = historical_lengths.size(0)
        historical_ids = torch.cat(
            [
                historical_ids,
                torch.zeros(
                    (B, max_output_length), dtype=historical_ids.dtype, device=device
                ),
            ],
            dim=1,
        )
        historical_ratings = torch.cat(
            [
                historical_ratings,
                torch.zeros(
                    (B, max_output_length),
                    dtype=historical_ratings.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        historical_timestamps = torch.cat(
            [
                historical_timestamps,
                torch.zeros(
                    (B, max_output_length),
                    dtype=historical_timestamps.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        historical_timestamps.scatter_(
            dim=1,
            index=historical_lengths.view(-1, 1),
            src=target_timestamps.view(-1, 1),
        )
        # print(f"historical_ids.size()={historical_ids.size()}, historical_timestamps.size()={historical_timestamps.size()}")
    features = SequentialFeatures(
        past_lengths=historical_lengths,
        past_ids=historical_ids,
        past_embeddings=None,
        past_payloads={
            "timestamps": historical_timestamps,
            "ratings": historical_ratings,
        },
    )
    return features, target_ids, target_ratings
