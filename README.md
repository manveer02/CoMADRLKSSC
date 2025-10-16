# Co-MADRL-KSSC: Cooperative Multi-Agent Deep Reinforcement Learning with Knowledge Sharing and Safety Constraints

This repository contains the official implementation for the paper "Cooperative Multi-Agent Deep Reinforcement Learning with Known State-Space Constraints (Co-MADRL-KSSC)". Our algorithm introduces a novel, principled mechanism for incorporating safety constraints into multi-agent learning, built within a state-of-the-art framework to ensure high performance and stability.

The architecture synergizes three key components:
1.  **Centralized Critic (CTDE):** Provides a stable foundation for effective learning.
2.  **Intrinsic Curiosity Module (ICM):** Drives efficient, systematic exploration in sparse-reward environments.
3.  **KSSC Mechanism:** Implements our novel, cost-penalized advantage function with a learned Lagrangian multiplier to intelligently balance objectives and safety.

---
## Installation

This project uses Python `3.10` or newer. We recommend using a virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/manveer02/CoMADRLKSSC
    cd Co-MADRL-KSSC
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create the virtual environment
    python -m venv venv

    # Activate it (on Linux/macOS)
    source venv/bin/activate

    # Or activate it (on Windows)
    .\venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

---
## Usage

You can run a quick demo to verify the installation or run the full experiment suite to reproduce the paper's results.

### Quick Demo

To run a short training session (200 episodes) with live visualization, use the `train_grid.py` script:

```bash
python train_grid.py --episodes 200 --render
```

### Full Experiment Orchestration

The orchestrator script is designed to run all experiments for the paper, including your main algorithm and the baselines.

Note: This repository does not include baseline implementations. The orchestrator will generate placeholder CSV files for baselines if their corresponding directories are not found.

To run the full suite:

```bash
python orchestrator/run_all_experiments.py --config config/auto_config.yaml
```

### Generating Analysis and Plots

After running the experiments, you can automatically generate all figures, tables, and a summary PDF from the results:

```bash
python analysis/run_analysis.py --results_dir results/
```

This will create an analysis_summary.pdf file with the final learning curves and performance metrics.
