**Co-MADRL-KSSC Auto-Orchestrator**

This bundle contains:
- train_mpe.py : trains our algorithm on PettingZoo MPE
- orchestrator/run_all_experiments.py : orchestration to run ours + baselines (placeholders)
- analysis/run_analysis.py : generates figures and a PDF summary
- config/auto_config.yaml : example config

Run demo (small):
  python train_mpe.py --episodes 20 --num_agents 3 --log_dir results/ours_mpe
Then generate analysis:
  python orchestrator/run_all_experiments.py --config config/auto_config.yaml

Note: Baseline repos are not included. Orchestrator generates placeholder baseline CSVs when repos are absent.
