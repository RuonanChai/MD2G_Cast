# MD2G-Cast: A Multicast Control Plane for Scalable Volumetric Media over MoQ (Review Artifact)

## Abstract/Overview
This repository contains a lightweight (Review Artifact) implementation of **MD2G-Cast: A Multicast Control Plane for Scalable Volumetric Media over MoQ**. MD2G-Cast targets scalable distribution of volumetric media by combining:
- a **PPO-based dynamic multicast control plane** that decides when to pull Base vs. Enhanced layers,
- **hierarchical (Base/Enhanced) streaming** for progressive quality delivery, and
- a **Mininet + Open vSwitch** experimentation environment to evaluate end-to-end QoE and system behavior under constrained, heterogeneous network conditions.

To keep the Artifact lightweight, we ship toy datasets and a tiny `video/bbb.mp4` placeholder for end-to-end sanity testing. Full-scale traces and official point-cloud derived data are described below.

## Repository Structure
Even though files are kept minimal for the Review Artifact, the code maps to the following components:
- `mininet_topology/` (role): orchestration of the Mininet topology, relay/publisher/subscriber startup, logging, and QoE evaluation runs.
- `client_logic/` (role): client-side MoQ subscription and PPO/heuristic-driven layer switching with QoE-aware feedback.
- `controllers/` (role): edge controller logic (PPO/heuristic/controller policies) that produces per-user decisions for the Base/Enhanced pulling behavior.

In this Artifact, the concrete entry points are:
- `moq_mininet_eval.py`: Mininet orchestration and experiment runner (mapping to `mininet_topology/`).
- `moq_client_dispatch.py`: client-side MoQ subscription and adaptive layer scheduling (mapping to `client_logic/`).
- `controllers/regional_relay_controller.py`: controller policy logic (mapping to `controllers/`).

## Prerequisites
System-level dependencies:
- **Linux** (tested with a 5.x kernel) and **sudo** access for Mininet.
- **Mininet**
- **Open vSwitch** (`ovs-vsctl`, OVS kernel modules/daemon)
- **Rust** + **Cargo** (for MoQ binaries)
- **Git** (for cloning MoQ)
- **FFmpeg** (for generating fragmented Base/Enhanced MP4 layers)

Python dependencies (inside the experiment environment):
- Python 3
- `numpy`, `pandas`
- (Optional) `torch`: if PyTorch is not available or model weights are missing, the controller falls back to heuristic grouping so the pipeline can still run.

## Quick Start (Toy Example)
This quick start demonstrates an end-to-end run with toy datasets.

1. Build MoQ binaries (moq-relay/moq-sub/hang)
   ```bash
   bash setup.sh
   ```

2. Generate toy datasets (lightweight CSVs)
   ```bash
   python3 generate_toy_data.py
   ```

3. Run Mininet + MoQ experiment (requires sudo)
   ```bash
   sudo python3 moq_mininet_eval.py --clients 10 --strategy md2g
   ```

Notes:
- The first run may take a while because `moq_mininet_eval.py` will create fragmented Base/Enhanced MP4s under `video/` if they are not present yet.
- If you want to point to a full dataset later, set `DATASET_DIR` to your dataset root.

Supported baseline strategies in this Artifact:
`md2g`, `rolling`, `heuristic`, `clustering`.

## Reproducibility Note
For Artifact lightweightness and anonymity review requirements, we do **not** include:
- the full multi-GB tracking datasets used in the paper,
- MPEG V-PCC original point-cloud / derived content used for generating layered media.

After acceptance, the full-scale traces and official V-PCC derived assets will be released. The current repository keeps only what is necessary to run the end-to-end control-plane pipeline with toy data and a tiny placeholder video.

