from miles.utils.reward_normalization import group_normalize_rewards
from miles.utils.types import Sample


def test_group_reward_normalization_uses_group_index():
    samples = [
        Sample(group_index=0, reward=1.0),
        Sample(group_index=1, reward=2.0),
        Sample(group_index=0, reward=3.0),
        Sample(group_index=1, reward=4.0),
        Sample(group_index=1, reward=6.0),
    ]

    raw_rewards = [s.reward for s in samples]

    normalized = group_normalize_rewards(
        samples=samples,
        raw_rewards=raw_rewards,
        advantage_estimator="grpo",
        grpo_std_normalization=False,
    )

    assert raw_rewards == [1.0, 2.0, 3.0, 4.0, 6.0]

    # group 0 mean = 2 -> [-1, +1]
    # group 1 mean = 4 -> [-2, 0, +2]
    assert normalized == [-1.0, -2.0, 1.0, 0.0, 2.0]


if __name__ == "__main__":
    test_group_reward_normalization_uses_group_index()
