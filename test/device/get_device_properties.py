#!/usr/bin/env python
# -*- coding: utf-8 -*-

import PybindGPU

print("Initializing NVML:")
print(f"{PybindGPU.nvmlInit()=}")

n_device, err = PybindGPU.cudaGetDeviceCount()

for i in range(n_device):
    prop = PybindGPU.cudaDeviceProp(i)
    print(f"Device {i}:")
    print(f" | name = {prop.name()}")
    if not PybindGPU.use_hip:
        print(f" | uuid = {prop.uuid()}")
    print(f" | pciBusID = {prop.pciBusID()}")
    print(f" | pciDeviceID = {prop.pciDeviceID()}")
    print(f" `-pciDomainID = {prop.pciDomainID()}")

    nvml_handle = PybindGPU.nvmlDevice(i)
    sm, mem = nvml_handle.utilization_rates()
    free, total, used = nvml_handle.memory_info()
    print(f"NVML data for Device {i}: {sm=}, {mem=}, {free=}, {total=}, {used=}")

print("Shutting Down NVML:")
print(f"{PybindGPU.nvmlShutdown()=}")