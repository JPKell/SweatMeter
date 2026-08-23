# NVIDIA telemetry fixtures

The `580.173.02-single-*` files were captured on 2026-08-22 from an NVIDIA GeForce RTX
5060 Ti running NVIDIA driver 580.173.02. The GPU UUID was replaced with
`GPU-fixture-0000`; volatile utilization, temperature, power, fan, clock, and memory values
remain representative captured values.

Capture commands:

```text
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader,nounits
nvidia-smi --query-gpu=index,compute_cap --format=csv,noheader,nounits
nvidia-smi --query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,temperature.memory,power.draw,power.limit,fan.speed,clocks.current.sm,clocks.current.memory --format=csv,noheader,nounits
nvidia-smi --query-gpu=index,clocks_throttle_reasons.gpu_idle,clocks_throttle_reasons.applications_clocks_setting,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.hw_power_brake_slowdown,clocks_throttle_reasons.sync_boost,clocks_throttle_reasons.sw_thermal_slowdown --format=csv,noheader,nounits
nvidia-smi
```

The `580.173.02-two-gpu-*` files add a clearly synthetic second device to the captured first
row. They exist to make multi-GPU ordering and per-device throttle behavior deterministic on CI
without claiming access to a two-GPU capture machine.
