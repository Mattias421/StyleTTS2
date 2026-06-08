import unittest

import torch
from peft import LoraConfig, inject_adapter_in_model
from torch import nn
from torch.nn.utils import weight_norm

from utils import merged_peft_state_dict


class WeightNormalizedConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = weight_norm(nn.Conv1d(4, 4, kernel_size=3, padding=1))

    def forward(self, inputs):
        return self.conv(inputs)


class MergedPeftStateDictTest(unittest.TestCase):
    def test_preserves_lora_update_for_weight_normalized_conv(self):
        torch.manual_seed(7)
        model = WeightNormalizedConv().eval()
        model = inject_adapter_in_model(
            LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["conv"],
                bias="none",
            ),
            model,
            adapter_name="test",
        )
        with torch.no_grad():
            model.conv.lora_A["test"].weight.normal_()
            model.conv.lora_B["test"].weight.normal_()

        inputs = torch.randn(2, 4, 8)
        expected = model(inputs)
        state_dict = merged_peft_state_dict(model)
        actual_after_save = model(inputs)

        reloaded = WeightNormalizedConv().eval()
        reloaded.load_state_dict(state_dict, strict=True)
        actual_reloaded = reloaded(inputs)

        torch.testing.assert_close(actual_after_save, expected)
        torch.testing.assert_close(actual_reloaded, expected, rtol=1e-5, atol=1e-5)
        self.assertFalse(any("lora_" in key or "base_layer" in key for key in state_dict))


if __name__ == "__main__":
    unittest.main()
