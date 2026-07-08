# omniverse-twin-lab

> Personal R&D lab tracking my exploration of digital twin engineering, physical simulation, and Python scripting within the NVIDIA Omniverse / Isaac Sim ecosystem.

---

## Overview

This repository is a collection of self-contained mini-experiments and hands-on labs focused on building and iterating digital twin workflows using **NVIDIA Omniverse** and **OpenUSD**. Each experiment targets a specific concept — from scene construction and physics simulation to synthetic data generation — accumulated incrementally as I deepen my understanding of the platform.

---

## Development Workflow

Scripts are authored externally in **Visual Studio Code** and executed directly inside a running Isaac Sim instance via the **Isaac Sim VS Code Edition** extension.

- **On the Omniverse side**: the `isaacsim.code_editor.vscode` extension bridges the running Kit application to VS Code, exposing a live Python environment.
- **On the VS Code side**: the [Isaac Sim VS Code Edition](https://marketplace.visualstudio.com/items?itemName=nvidia.isaacsim-vscode-edition) extension sends selected code or entire files to that environment for immediate execution and output inspection.

Full setup reference: [Isaac Sim – Visual Studio Code (VS Code)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/development_tools/vscode.html)

---

## Repository Structure

```
omniverse-twin-lab/
├── README.md
├── hello.py              # Smoke test — verifies VS Code ↔ Isaac Sim connection
└── m1/
    └── m1_cube.py        # Lab 1: physics-enabled cube + Replicator synthetic data capture
```

---

## Labs

| # | Folder | Description |
|---|--------|-------------|
| – | `hello.py` | Minimal `carb.log_warn` sanity check for the VS Code integration |
| 1 | `m1/` | Spawns a rigid-body cube, runs a PhysX simulation step, and captures RGB frames using Omniverse Replicator |

---

## Tech Stack

- **NVIDIA Isaac Sim** (Omniverse Kit application)
- **OpenUSD** — `UsdGeom`, `UsdPhysics`, `UsdLux`, `Gf`
- **Omniverse Replicator** — synthetic data generation (`omni.replicator.core`)
- **Python** — async scripting via `omni.kit.app` update loop
- **Visual Studio Code** + Isaac Sim VS Code Edition
