import pytest


torch = pytest.importorskip("torch")

from temporal_finance.temporal_lora import (  # noqa: E402
    LoRALinear,
    count_trainable_parameters,
    inject_lora_adapters,
    load_lora_state,
    save_lora_state,
)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())
        self.dep_layer = torch.nn.Linear(3, 2)
        self.head = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.head(self.dep_layer(self.transformer(x)))


def test_lora_injection_freezes_base_and_targets_only_requested_modules():
    model = ToyModel()

    report = inject_lora_adapters(
        model,
        rank=2,
        alpha=4,
        dropout=0.0,
        target_prefixes=("transformer", "dep_layer"),
        exclude_keywords=("head",),
    )

    assert report.target_names == ["transformer.0", "dep_layer"]
    assert isinstance(model.transformer[0], LoRALinear)
    assert isinstance(model.dep_layer, LoRALinear)
    assert not isinstance(model.head, LoRALinear)
    assert count_trainable_parameters(model) > 0
    assert all(
        parameter.requires_grad == ("lora_" in name)
        for name, parameter in model.named_parameters()
    )


def test_lora_save_load_round_trip_restores_predictions(tmp_path):
    torch.manual_seed(7)
    model = ToyModel()
    inject_lora_adapters(
        model,
        rank=2,
        alpha=4,
        dropout=0.0,
        target_prefixes=("transformer", "dep_layer"),
        exclude_keywords=("head",),
    )
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.data.fill_(0.2)
    x = torch.ones(2, 4)
    expected = model(x).detach()
    path = tmp_path / "adapter.pt"

    save_lora_state(model, path, metadata={"adapter_name": "slow"})

    torch.manual_seed(7)
    restored = ToyModel()
    inject_lora_adapters(
        restored,
        rank=2,
        alpha=4,
        dropout=0.0,
        target_prefixes=("transformer", "dep_layer"),
        exclude_keywords=("head",),
    )
    metadata = load_lora_state(restored, path)

    assert metadata["adapter_name"] == "slow"
    assert torch.allclose(restored(x), expected)
