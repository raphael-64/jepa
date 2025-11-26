import json
import numpy as np

with open('checkpoints/metrics.json', 'r') as f:
    m = json.load(f)

train = m['train_loss']
val = m['val_loss']

print("="*70)
print("TRAINING ANALYSIS - JEPA World Model")
print("="*70)
print(f"Total epochs completed: {len(train)}")
print(f"\n{'='*70}")
print("TRAIN LOSS:")
print(f"{'='*70}")
print(f"  Start (epoch 1):  {train[0]:.8f}")
print(f"  End (epoch {len(train)}):    {train[-1]:.8f}")
print(f"  Minimum:          {min(train):.8f} (epoch {train.index(min(train))+1})")
print(f"  Reduction:        {(1 - train[-1]/train[0])*100:.1f}%")
print(f"\n  Last 5 epochs:")
for i, v in enumerate(train[-5:], start=len(train)-4):
    print(f"    Epoch {i}: {v:.8f}")

print(f"\n{'='*70}")
print("VALIDATION LOSS:")
print(f"{'='*70}")
print(f"  Start (epoch 1):  {val[0]:.8f}")
print(f"  End (epoch {len(val)}):    {val[-1]:.8f}")
print(f"  Minimum:          {min(val):.8f} (epoch {val.index(min(val))+1})")
print(f"  Maximum:          {max(val):.8f} (epoch {val.index(max(val))+1})")
print(f"  Reduction:        {(1 - val[-1]/val[0])*100:.1f}%")
print(f"\n  Last 5 epochs:")
for i, v in enumerate(val[-5:], start=len(val)-4):
    print(f"    Epoch {i}: {v:.8f}")

# Statistical analysis
print(f"\n{'='*70}")
print("STATISTICAL ANALYSIS:")
print(f"{'='*70}")
val_recent = val[-10:]
val_mean = np.mean(val_recent)
val_std = np.std(val_recent)
val_cv = (val_std / val_mean) * 100 if val_mean > 0 else 0

print(f"  Val loss (last 10 epochs):")
print(f"    Mean: {val_mean:.8f}")
print(f"    Std:  {val_std:.8f}")
print(f"    CV (coefficient of variation): {val_cv:.1f}%")

# Trend analysis
if len(train) > 20:
    recent_5 = np.mean(train[-5:])
    recent_10 = np.mean(train[-10:-5])
    earlier_10 = np.mean(train[-20:-10])
    
    print(f"\n  Trend analysis:")
    print(f"    Epochs 81-85:  {earlier_10:.8f}")
    print(f"    Epochs 86-90:  {recent_10:.8f}")
    print(f"    Epochs 91-95:  {recent_5:.8f}")
    
    if recent_5 < recent_10 < earlier_10:
        print(f"    ✅ Still decreasing!")
    elif recent_5 > recent_10:
        print(f"    ⚠️  Slight increase in recent epochs")

# Collapse detection
print(f"\n{'='*70}")
print("COLLAPSE DETECTION:")
print(f"{'='*70}")

if min(train) < 1e-6:
    print(f"  ⚠️  WARNING: Train loss extremely small (< 1e-6)")
    print(f"     Minimum: {min(train):.2e}")
    print(f"     This could indicate model collapse to trivial predictions")
    print(f"     RECOMMENDATION: Train decoder to visualize predictions")

if min(val) < 1e-6:
    print(f"  ⚠️  WARNING: Val loss extremely small (< 1e-6)")
    print(f"     Minimum: {min(val):.2e}")

# Overfitting check
if len(val) > 5:
    train_recent = np.mean(train[-5:])
    val_recent_mean = np.mean(val[-5:])
    gap = val_recent_mean - train_recent
    gap_pct = (gap / train_recent * 100) if train_recent > 0 else 0
    
    print(f"\n  Overfitting check (last 5 epochs):")
    print(f"    Train avg: {train_recent:.8f}")
    print(f"    Val avg:   {val_recent_mean:.8f}")
    print(f"    Gap:       {gap:.8f} ({gap_pct:.1f}%)")
    
    if gap > train_recent * 0.3:
        print(f"    ⚠️  Possible overfitting (val >30% higher than train)")
    elif gap < 0:
        print(f"    ⚠️  Val lower than train (unusual, check for bugs)")
    else:
        print(f"    ✅ Gap looks reasonable")

# Final verdict
print(f"\n{'='*70}")
print("VERDICT:")
print(f"{'='*70}")

issues = []
if min(train) < 1e-6:
    issues.append("Losses extremely small - possible collapse")
if val_cv > 50:
    issues.append("High variance in validation loss")
if len(issues) == 0:
    print("  ✅ Training completed successfully!")
    print("  ✅ Losses decreased significantly")
    print("  ✅ Model appears to be learning")
    print("\n  ⚠️  However, losses are very small - recommend:")
    print("     1. Train decoder to visualize predictions")
    print("     2. Check if model learned meaningful representations")
else:
    print("  ⚠️  Potential issues detected:")
    for issue in issues:
        print(f"     - {issue}")

print(f"{'='*70}")

