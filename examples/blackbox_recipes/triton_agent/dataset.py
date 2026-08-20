"""verl dataset adapter for the Triton operator recipe."""

from __future__ import annotations

from verl.utils.dataset.rl_dataset import RLHFDataset


class TritonOperatorDataset(RLHFDataset):
    """Assert the recipe fields required by Uni-Agent's task runner."""

    def __getitem__(self, item):
        row = super().__getitem__(item)
        extra_info = row.get("extra_info")
        if not isinstance(extra_info, dict):
            raise ValueError("TritonOperatorDataset requires an extra_info mapping")
        tools_kwargs = extra_info.get("tools_kwargs")
        if not isinstance(tools_kwargs, dict) or not isinstance(tools_kwargs.get("task"), dict):
            raise ValueError("TritonOperatorDataset requires extra_info.tools_kwargs.task")
        row.setdefault("data_source", "triton_operator")
        row.setdefault("reward_model", {"ground_truth": {"uid": extra_info.get("uid")}, "style": "rule"})
        return row
