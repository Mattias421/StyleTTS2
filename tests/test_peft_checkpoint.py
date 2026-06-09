import unittest

import torch
from peft import LoraConfig, inject_adapter_in_model
from torch import nn
from torch.nn.utils import weight_norm

from peft_adapters import compatible_lora_targets, inject_lora_adapter
from utils import merged_peft_state_dict


class WeightNormalizedConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = weight_norm(nn.Conv1d(4, 4, kernel_size=3, padding=1))

    def forward(self, inputs):
        return self.conv(inputs)


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = weight_norm(nn.Conv1d(4, 4, kernel_size=3, padding=1))
        self.dilated = nn.Conv1d(4, 4, kernel_size=3, padding=2, dilation=2)
        self.style = nn.Linear(4, 4)
        self.upsample = nn.ConvTranspose1d(4, 4, kernel_size=2, stride=2)

    def forward(self, inputs):
        return self.style(self.pre(inputs).transpose(1, 2)).transpose(1, 2)


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

    def test_decoder_adapter_targets_supported_layers_and_freezes_base(self):
        model = inject_lora_adapter(
            ToyDecoder(),
            {"rank": 2, "alpha": 4, "dropout": 0.0},
            label="Decoder",
            default_adapter_name="decoder_lora",
        )

        self.assertEqual(compatible_lora_targets(ToyDecoder()), ["pre", "style"])
        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertTrue(trainable_names)
        self.assertTrue(all("lora_" in name for name in trainable_names))
        self.assertFalse(model.dilated.weight.requires_grad)
        self.assertFalse(model.upsample.weight.requires_grad)
        self.assertFalse(model.pre.base_layer.weight_g.requires_grad)
        self.assertFalse(model.style.base_layer.weight.requires_grad)

    def test_merged_decoder_adapter_reloads_without_peft_wrappers(self):
        torch.manual_seed(11)
        model = inject_lora_adapter(
            ToyDecoder().eval(),
            {"rank": 2, "alpha": 4},
            label="Decoder",
            default_adapter_name="decoder_lora",
        )
        with torch.no_grad():
            model.pre.lora_A["decoder_lora"].weight.normal_()
            model.pre.lora_B["decoder_lora"].weight.normal_()
            model.style.lora_A["decoder_lora"].weight.normal_()
            model.style.lora_B["decoder_lora"].weight.normal_()

        inputs = torch.randn(2, 4, 8)
        expected = model(inputs)
        state_dict = merged_peft_state_dict(model)

        reloaded = ToyDecoder().eval()
        reloaded.load_state_dict(state_dict, strict=True)
        torch.testing.assert_close(reloaded(inputs), expected, rtol=1e-5, atol=1e-5)
        self.assertFalse(any("lora_" in key or "base_layer" in key for key in state_dict))


if __name__ == "__main__":
    unittest.main()
