# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:38:25 2026

@author: USER
"""


import time
import torch
import torch.nn as nn
import pandas as pd

from thop import profile, clever_format
from fvcore.nn import FlopCountAnalysis, parameter_count_table

from hybridsegnet import HybridViTGABVSSMUNet


# ---------------------------------------------------------
# Device Setup
# ---------------------------------------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# Parameter Count
# ---------------------------------------------------------
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return total_params, trainable_params, non_trainable_params


# ---------------------------------------------------------
# FLOPs using THOP
# ---------------------------------------------------------
def compute_flops_thop(model, input_tensor):
    macs, params = profile(
        model,
        inputs=(input_tensor,),
        verbose=False
    )

    flops = 2 * macs

    macs_fmt, params_fmt = clever_format([macs, params], "%.3f")
    flops_fmt, _ = clever_format([flops, params], "%.3f")

    return macs, flops, params, macs_fmt, flops_fmt, params_fmt


# ---------------------------------------------------------
# FLOPs using FVCore
# ---------------------------------------------------------
def compute_flops_fvcore(model, input_tensor):
    flops = FlopCountAnalysis(model, input_tensor)
    total_flops = flops.total()
    return total_flops


# ---------------------------------------------------------
# Latency and FPS
# ---------------------------------------------------------
@torch.no_grad()
def measure_latency_fps(
    model,
    input_tensor,
    device,
    warmup=50,
    repetitions=200
):
    model.eval()

    for _ in range(warmup):
        _ = model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for _ in range(repetitions):
        _ = model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    total_time = end_time - start_time
    latency = total_time / repetitions
    fps = 1.0 / latency

    return latency, fps


# ---------------------------------------------------------
# GPU Memory
# ---------------------------------------------------------
@torch.no_grad()
def measure_gpu_memory(model, input_tensor, device):
    if device.type != "cuda":
        return None

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    _ = model(input_tensor)

    torch.cuda.synchronize()

    peak_memory = torch.cuda.max_memory_allocated(device)
    peak_memory_mb = peak_memory / (1024 ** 2)

    return peak_memory_mb


# ---------------------------------------------------------
# Energy Profiling using NVIDIA NVML
# ---------------------------------------------------------
class EnergyProfiler:
    def __init__(self, device_index=0):
        self.available = False

        try:
            import pynvml
            self.pynvml = pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.available = True
        except Exception:
            self.available = False

    def get_power_watts(self):
        if not self.available:
            return None

        power_mw = self.pynvml.nvmlDeviceGetPowerUsage(self.handle)
        return power_mw / 1000.0

    def shutdown(self):
        if self.available:
            self.pynvml.nvmlShutdown()


@torch.no_grad()
def measure_energy(
    model,
    input_tensor,
    device,
    warmup=50,
    repetitions=200
):
    if device.type != "cuda":
        return None, None, None

    profiler = EnergyProfiler(device_index=torch.cuda.current_device())

    if not profiler.available:
        return None, None, None

    model.eval()

    for _ in range(warmup):
        _ = model(input_tensor)

    torch.cuda.synchronize()

    power_readings = []

    start_time = time.perf_counter()

    for _ in range(repetitions):
        power = profiler.get_power_watts()
        if power is not None:
            power_readings.append(power)

        _ = model(input_tensor)

    torch.cuda.synchronize()

    end_time = time.perf_counter()

    profiler.shutdown()

    elapsed_time = end_time - start_time

    if len(power_readings) == 0:
        return None, None, None

    avg_power = sum(power_readings) / len(power_readings)
    total_energy = avg_power * elapsed_time
    energy_per_inference = total_energy / repetitions

    return avg_power, total_energy, energy_per_inference


# ---------------------------------------------------------
# Complete Profiling Function
# ---------------------------------------------------------
def profile_model(
    model,
    input_size=(1, 3, 256, 256),
    warmup=50,
    repetitions=200
):
    device = get_device()
    model = model.to(device)
    model.eval()

    input_tensor = torch.randn(*input_size).to(device)

    total_params, trainable_params, non_trainable_params = count_parameters(model)

    macs, flops, thop_params, macs_fmt, flops_fmt, params_fmt = compute_flops_thop(
        model,
        input_tensor
    )

    try:
        fvcore_flops = compute_flops_fvcore(model, input_tensor)
    except Exception:
        fvcore_flops = None

    latency, fps = measure_latency_fps(
        model,
        input_tensor,
        device,
        warmup=warmup,
        repetitions=repetitions
    )

    peak_memory_mb = measure_gpu_memory(
        model,
        input_tensor,
        device
    )

    avg_power, total_energy, energy_per_inference = measure_energy(
        model,
        input_tensor,
        device,
        warmup=warmup,
        repetitions=repetitions
    )

    results = {
        "Device": str(device),
        "Input Size": str(input_size),
        "Total Parameters": total_params,
        "Trainable Parameters": trainable_params,
        "Non-trainable Parameters": non_trainable_params,
        "THOP Params": params_fmt,
        "MACs": macs_fmt,
        "FLOPs": flops_fmt,
        "FVCore FLOPs": fvcore_flops,
        "Latency / Image (ms)": latency * 1000,
        "FPS": fps,
        "Peak GPU Memory (MB)": peak_memory_mb,
        "Average Power (W)": avg_power,
        "Total Energy (J)": total_energy,
        "Energy / Inference (J)": energy_per_inference
    }

    return results


# ---------------------------------------------------------
# Print Results
# ---------------------------------------------------------
def print_results(results):
    print("\n" + "=" * 70)
    print("MODEL PROFILING RESULTS")
    print("=" * 70)

    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.6f}")
        else:
            print(f"{key:30s}: {value}")

    print("=" * 70)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":

    model = HybridViTGABVSSMUNet(
        in_channels=3,
        num_classes=1,
        encoder_dims=(32, 64, 128, 256),
        encoder_depths=(2, 2, 2, 2),
        encoder_heads=(1, 2, 4, 8),
        dropout=0.1
    )

    results = profile_model(
        model=model,
        input_size=(1, 3, 256,256 ),
        warmup=50,
        repetitions=200
    )

    print_results(results)

    df = pd.DataFrame([results])
    df.to_csv("model_profiling_results.csv", index=False)

    print("\nSaved profiling results to: model_profiling_results.csv")