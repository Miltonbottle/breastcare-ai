"""
Final Updated Model Summary Code
Fixes:
1. CPU/GPU device mismatch
2. CUDA tensor mismatch
3. Proper torchinfo usage
4. Clean model summary printing
"""

import torch
from torchinfo import summary

from hybridsegnet import HybridViTGABVSSMUNet


# =========================================================
# Device Configuration
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print(f"Using Device : {device}")
print("=" * 80)


# =========================================================
# Initialize Model
# =========================================================
model = HybridViTGABVSSMUNet(
    in_channels=3,
    num_classes=1,
    encoder_dims=(32, 64, 128, 256),
    encoder_depths=(2, 2, 2, 2),
    encoder_heads=(1, 2, 4, 8),
    dropout=0.1
)

# Move model to device
model = model.to(device)

# Evaluation mode
model.eval()


# =========================================================
# Print Model Summary
# =========================================================
print("\n" + "=" * 80)
print("MODEL SUMMARY")
print("=" * 80)

model_stats = summary(
    model=model,
    input_size=(1, 3, 256, 256),   # (B, C, H, W)
    device=device,
    depth=6,
    verbose=1,
    col_names=[
        "input_size",
        "output_size",
        "num_params",
        "trainable",
        "mult_adds"
    ],
    row_settings=["var_names"]
)

print(model_stats)


# =========================================================
# Verify Forward Pass
# =========================================================
print("\n" + "=" * 80)
print("FORWARD PASS TEST")
print("=" * 80)

with torch.no_grad():

    x = torch.randn(1, 3, 256, 256).to(device)

    y = model(x)

    print(f"Input Shape  : {x.shape}")
    print(f"Output Shape : {y.shape}")

    print("\nForward pass successful!")


# =========================================================
# Parameter Statistics
# =========================================================
print("\n" + "=" * 80)
print("PARAMETER STATISTICS")
print("=" * 80)

total_params = sum(p.numel() for p in model.parameters())

trainable_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

non_trainable_params = total_params - trainable_params

print(f"Total Parameters       : {total_params:,}")
print(f"Trainable Parameters   : {trainable_params:,}")
print(f"Non-trainable Params   : {non_trainable_params:,}")


# =========================================================
# GPU Memory Usage
# =========================================================
if torch.cuda.is_available():

    print("\n" + "=" * 80)
    print("GPU MEMORY USAGE")
    print("=" * 80)

    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2

    print(f"Allocated Memory : {allocated:.2f} MB")
    print(f"Reserved Memory  : {reserved:.2f} MB")


print("\n" + "=" * 80)
print("MODEL SUMMARY COMPLETED SUCCESSFULLY")
print("=" * 80)