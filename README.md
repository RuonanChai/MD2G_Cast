
# MD2G-Cast: Relay-Coordinated Multicast for Scalable Volumetric Streaming over MoQ (Review Artifact)

## Abstract/Overview
This repository contains a lightweight Review Artifact for **MD2G-Cast: Relay-Coordinated Multicast for Scalable Volumetric Streaming over MoQ**. MD2G-Cast targets the scalable distribution of volumetric media (e.g., point cloud video) by shifting multicast coordination from the application layer into the MoQ transport plane.

The core contributions demonstrated in this artifact include:
- A **relay-coordinated control plane** that enables MoQ relays to make distribution-aware scheduling decisions based on viewing similarity, device heterogeneity, and network conditions.
- A **PPO-based dynamic grouping and layer allocation policy** that dictates Base vs. Enhanced layer subscriptions.
- A **Mininet + Open vSwitch** experimentation environment to evaluate end-to-end QoE, Time-To-First-Byte (TTFB), and system overhead under constrained, heterogeneous network conditions.

To keep the Artifact lightweight for the double-blind review process, we ship a script to generate toy datasets and utilize a small placeholder video (`video/bbb.mp4`) for end-to-end sanity testing. Full-scale traces and official point-cloud derived data are detailed below.

## Repository Structure
While files are kept minimal for the Review Artifact, the codebase maps to the following core system components:

- `mininet_topology/` (`moq_mininet_eval.py`): Orchestration of the Mininet topology. Handles relay/publisher/subscriber startup, link shaping, logging, and the QoE evaluation loop.
- `client_logic/` (`moq_client_dispatch.py`): Client-side MoQ subscription logic. Manages adaptive layer scheduling and QoE-aware feedback mechanisms.
- `controllers/` (`regional_relay_controller.py`): The edge relay controller logic. Executes the PPO policy (or heuristic baselines) to produce per-user decisions for Base/Enhanced pulling behavior based on simulated runtime state.

## Prerequisites
**System-level dependencies:**
- **Linux** (tested with a 5.x kernel) and **sudo** access for Mininet orchestration.
- **Mininet**
- **Open vSwitch** (`ovs-vsctl` and OVS kernel modules/daemon)
- **Rust** + **Cargo** (required to build the official MoQ binaries)
- **Git**
- **FFmpeg** (for generating fragmented Base/Enhanced MP4 layers from the placeholder video)

**Python dependencies (inside the experiment environment):**
- Python 3.8+
- `numpy`, `pandas`
- (Optional) `torch`: Required for the PPO (`md2g`) and `rolling` strategies. If PyTorch is unavailable or model weights are missing, the controller gracefully falls back to heuristic grouping to ensure the pipeline executes.

## Quick Start (Toy Example)
This quick start demonstrates an end-to-end run of the MD2G-Cast pipeline using toy datasets.

1. **Build MoQ Binaries** (compiles `moq-relay`, `moq-sub`, and `hang` from the official repository):
   ```bash
   bash setup.sh
   ```

2. **Generate Toy Datasets** (creates lightweight CSVs for bandwidth and device profiles):
   ```bash
   python3 generate_toy_data.py
   ```

3. **Run Mininet + MoQ Experiment** (Requires sudo. Orchestrates the topology and runs the evaluation):
   ```bash
   sudo python3 moq_mininet_eval.py --clients 10 --strategy md2g
   ```

**Notes:**
- The initial run may take additional time as `moq_mininet_eval.py` will invoke FFmpeg to transcode and fragment the placeholder video into Base and Enhanced MP4s under the `video/` directory.
- The artifact supports evaluating four baseline strategies via the `--strategy` flag: `md2g`, `rolling`, `heuristic`, and `clustering`.
- To evaluate against a full dataset in the future, set the `DATASET_DIR` variable within `moq_mininet_eval.py` to point to your dataset root.

## Reproducibility Note
To adhere to Artifact lightweightness guidelines and double-blind review requirements, we do **not** include:
- The full multi-GB bandwidth and FoV tracking datasets utilized in the paper's evaluation.
- The original MPEG V-PCC point-cloud assets and the derived heavy volumetric content used for generating layered media.

Upon acceptance, the full-scale traces, trained model weights, and official V-PCC derived assets will be released. The current repository contains the necessary components to validate the end-to-end control-plane pipeline, relay coordination logic, and Mininet evaluation framework using toy data and a tiny placeholder video.
