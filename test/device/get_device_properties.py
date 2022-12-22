#!/usr/bin/env python
# -*- coding: utf-8 -*-

import PybindGPU

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
