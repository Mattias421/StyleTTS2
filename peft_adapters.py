from collections.abc import Callable, Mapping

from peft import LoraConfig, inject_adapter_in_model
from torch import nn


def compatible_lora_targets(
    module: nn.Module,
    predicate: Callable[[str, nn.Module], bool] | None = None,
) -> list[str]:
    targets = []
    for name, submodule in module.named_modules():
        if not name or not isinstance(submodule, (nn.Conv1d, nn.Linear)):
            continue
        # PEFT's Conv1d LoRA branch does not preserve output length for
        # dilated convolutions because its adapter path omits the base dilation.
        if isinstance(submodule, nn.Conv1d) and submodule.dilation != (1,):
            continue
        if predicate is None or predicate(name, submodule):
            targets.append(name)
    return targets


def inject_lora_adapter(
    module: nn.Module,
    config: Mapping,
    *,
    label: str,
    default_adapter_name: str,
    target_predicate: Callable[[str, nn.Module], bool] | None = None,
) -> nn.Module:
    for parameter in module.parameters():
        parameter.requires_grad = False

    targets = config.get("target_modules") or compatible_lora_targets(
        module,
        predicate=target_predicate,
    )
    if not targets:
        raise ValueError(f"No PEFT-compatible Conv1d or Linear modules were found in {label}.")

    lora_config = LoraConfig(
        r=int(config.get("rank", 8)),
        lora_alpha=int(config.get("alpha", 16)),
        lora_dropout=float(config.get("dropout", 0.0)),
        bias="none",
        target_modules=targets,
    )
    adapter_name = str(config.get("adapter_name", default_adapter_name))
    module = inject_adapter_in_model(lora_config, module, adapter_name=adapter_name)

    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in module.parameters())
    if trainable == 0:
        raise RuntimeError(f"PEFT injection did not create trainable adapter parameters in {label}.")
    target_summary = ", ".join(targets[:10])
    if len(targets) > 10:
        target_summary += f", ... ({len(targets)} total)"
    print(
        "%s PEFT targets: %s; trainable parameters: %d/%d (%.3f%%)"
        % (label, target_summary, trainable, total, 100 * trainable / total)
    )
    return module
