#!/usr/bin/env python
# -*- coding: utf-8 -*-

import PybindGPU

n_device, err = py_device_properties.cudaGetDeviceCount()

for i in range(n_device):
    prop = py_device_properties.cudaDeviceProp(i)
    print(f"Device {i}:")
    print(f" | name = {prop.name()}")
    print(f" | uuid = {prop.uuid()}")
    print(f" | pciBusID = {prop.pciBusID()}")
    print(f" | pciDeviceID = {prop.pciDeviceID()}")
    print(f" `-pciDomainID = {prop.pciDomainID()}")
