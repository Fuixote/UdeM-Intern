# Formulations

This directory groups solver and decision-focused-learning entry points by
optimization formulation.

Current layout:

- `cf/stage2_solver.py`: candidate-based `CF-CYCLE + CF-CHAIN` stage-2 solver
- `cf/end2end_gnn.py`: GNN DFL training using the CF black-box solver
- `cf/end2end_reg.py`: MLP/Regression DFL training using the CF black-box solver
- `hybrid/stage2_solver.py`: hybrid `CF-CYCLE + PIEF-CHAIN` solver
- `hybrid/end2end_gnn.py`: GNN DFL training with the hybrid formulation
- `hybrid/end2end_reg.py`: MLP/Regression DFL training with the hybrid formulation
- `pief/stage2_solver.py`: dual-PIEF (`PIEF-CYCLE + PIEF-CHAIN`) solver
- `pief/end2end_gnn.py`: GNN DFL training with the dual-PIEF formulation
- `pief/end2end_reg.py`: MLP/Regression DFL training with the dual-PIEF formulation

Future formulations should follow the same pattern, for example:

- `pief/stage2_solver.py`
- `pief/end2end_gnn.py`
- `pief/end2end_reg.py`



/usr/bin/time -p /home/weikang/miniconda3/envs/KEPs/bin/python formulations/cf/stage2_solver.py \
  --model_path results/2stg_Reg_2026-03-30_151205/best_stage1_model_real.pth \
  --data_dir dataset/processed/2026-03-30_124404 \
  --max_cycle 3 \
  --max_chain 4 \
  --solutions_root /tmp/2stg_reg_speed_compare


/usr/bin/time -p /home/weikang/miniconda3/envs/KEPs/bin/python formulations/hybrid/stage2_solver.py \
  --model_path results/2stg_Reg_2026-03-30_151205/best_stage1_model_real.pth \
  --data_dir dataset/processed/2026-03-30_124404 \
  --max_cycle 3 \
  --max_chain 4 \
  --solutions_root /tmp/2stg_reg_speed_compare


/usr/bin/time -p /home/weikang/miniconda3/envs/KEPs/bin/python formulations/pief/stage2_solver.py \
  --model_path results/2stg_Reg_2026-03-30_151205/best_stage1_model_real.pth \
  --data_dir dataset/processed/2026-03-30_124404 \
  --max_cycle 3 \
  --max_chain 4 \
  --solutions_root /tmp/2stg_reg_speed_compare

