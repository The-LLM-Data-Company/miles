import torch

from miles.utils.iter_utils import group_by
from miles.utils.types import Sample


def group_normalize_rewards(
    *,
    samples: list[Sample],
    raw_rewards: list[float],
    advantage_estimator: str,
    grpo_std_normalization: bool,
) -> list[float]:
    normalized_rewards: list[float] = [0.0] * len(samples)
    for _group_index, group_items in group_by(list(enumerate(samples)), lambda x: x[1].group_index).items():
        indices = [idx for idx, _sample in group_items]
        group_rewards = torch.tensor([raw_rewards[idx] for idx in indices], dtype=torch.float)
        group_rewards = group_rewards - group_rewards.mean()

        if advantage_estimator in ["grpo", "gspo"] and grpo_std_normalization:
            group_rewards = group_rewards / (group_rewards.std() + 1e-6)

        for idx, val in zip(indices, group_rewards.tolist(), strict=False):
            normalized_rewards[idx] = val

    return normalized_rewards
