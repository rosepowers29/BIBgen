import sys
sys.path.insert(0, "examples")  # wherever analyze_ablation.py lives
from analyze_ablation import load_val_loss

best = load_val_loss("training/")  # dir containing history_gaussian_*.csv
print(best.filter(like="positional_phi2_s32").sort_values())
