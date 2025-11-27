"""
Diagnostic script to understand what's going wrong with SAM-Med3D predictions
"""

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path


def load_nifti(path):
    """Load NIFTI file"""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return arr


def analyze_case(case_name, img_dir, gt_dir, ts_dir, pred_dir):
    """Analyze a single case to understand the problem"""
    
    # Load all data
    img_path = Path(img_dir) / f"{case_name}.nii.gz"
    gt_path = Path(gt_dir) / f"{case_name}.nii.gz"
    ts_path = Path(ts_dir) / case_name / "intestinal_tract.nii.gz"
    pred_path = Path(pred_dir) / f"{case_name}.nii.gz"
    
    print(f"\n{'='*80}")
    print(f"CASE: {case_name}")
    print(f"{'='*80}")
    
    # Check if files exist
    for name, path in [("Image", img_path), ("GT", gt_path), ("TS", ts_path), ("Pred", pred_path)]:
        exists = "✓" if path.exists() else "✗"
        print(f"{exists} {name}: {path}")
    
    if not all([img_path.exists(), gt_path.exists(), ts_path.exists(), pred_path.exists()]):
        print("⚠️  Missing files, skipping analysis")
        return
    
    # Load arrays
    img = load_nifti(img_path)
    gt = load_nifti(gt_path)
    ts = load_nifti(ts_path)
    pred = load_nifti(pred_path)
    
    # Basic stats
    print(f"\nSHAPE ANALYSIS:")
    print(f"  Image shape: {img.shape}")
    print(f"  GT shape: {gt.shape}")
    print(f"  TS shape: {ts.shape}")
    print(f"  Pred shape: {pred.shape}")
    
    # Shape mismatch check
    if gt.shape != pred.shape:
        print(f"  ⚠️  WARNING: GT and Prediction shapes don't match!")
        print(f"     This could cause Dice computation issues")
    
    # Foreground analysis
    gt_binary = (gt > 0).astype(np.uint8)
    ts_binary = (ts > 0).astype(np.uint8)
    pred_binary = (pred > 0).astype(np.uint8)
    
    gt_voxels = gt_binary.sum()
    ts_voxels = ts_binary.sum()
    pred_voxels = pred_binary.sum()
    
    print(f"\nFOREGROUND VOXEL COUNT:")
    print(f"  GT:         {gt_voxels:,} voxels")
    print(f"  TS:         {ts_voxels:,} voxels ({100*ts_voxels/gt_voxels:.1f}% of GT)")
    print(f"  Prediction: {pred_voxels:,} voxels ({100*pred_voxels/gt_voxels:.1f}% of GT)")
    
    # Overlap analysis
    tp = np.logical_and(pred_binary, gt_binary).sum()
    fp = np.logical_and(pred_binary, ~gt_binary).sum()
    fn = np.logical_and(~pred_binary, gt_binary).sum()
    
    dice = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\nPERFORMANCE METRICS:")
    print(f"  Dice:      {dice:.4f}")
    print(f"  Precision: {precision:.4f} (how much of prediction is correct)")
    print(f"  Recall:    {recall:.4f} (how much of GT is captured)")
    print(f"\n  True Pos:  {tp:,} voxels")
    print(f"  False Pos: {fp:,} voxels (over-segmentation)")
    print(f"  False Neg: {fn:,} voxels (under-segmentation)")
    
    # Diagnosis
    print(f"\nDIAGNOSIS:")
    if pred_voxels < 0.3 * gt_voxels:
        print(f"  ⚠️  SEVERE UNDER-SEGMENTATION")
        print(f"     Prediction has only {100*pred_voxels/gt_voxels:.1f}% of GT volume")
        print(f"     → Try: More positive clicks, less erosion")
    elif pred_voxels > 2.0 * gt_voxels:
        print(f"  ⚠️  SEVERE OVER-SEGMENTATION")
        print(f"     Prediction has {100*pred_voxels/gt_voxels:.1f}% of GT volume")
        print(f"     → Try: More negative clicks, more erosion")
    elif precision < 0.5:
        print(f"  ⚠️  LOW PRECISION ({precision:.2f})")
        print(f"     Many false positives - including wrong regions")
        print(f"     → Try: More negative clicks near boundaries")
    elif recall < 0.5:
        print(f"  ⚠️  LOW RECALL ({recall:.2f})")
        print(f"     Many false negatives - missing GT regions")
        print(f"     → Try: More positive clicks, less erosion")
    else:
        print(f"  ✓ Reasonable balance, but room for improvement")
        print(f"    → Try: Iterative refinement, adjust click locations")
    
    # TS mask quality
    ts_dice = 2 * np.logical_and(ts_binary, gt_binary).sum() / (ts_binary.sum() + gt_binary.sum())
    print(f"\nTS MASK QUALITY:")
    print(f"  TS vs GT Dice: {ts_dice:.4f}")
    if ts_dice < 0.5:
        print(f"  ⚠️  WARNING: TotalSegmentator mask is poor quality!")
        print(f"     This will generate bad clicks and hurt performance")
    
    # Visualization suggestion
    print(f"\nVISUALIZATION:")
    print(f"  Middle slice index: {img.shape[0] // 2}")
    print(f"  To visualize, use this code:")
    print(f"""
    import matplotlib.pyplot as plt
    slice_idx = {img.shape[0] // 2}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img[slice_idx], cmap='gray')
    axes[0].set_title('Image')
    axes[1].imshow(gt[slice_idx], cmap='Reds', alpha=0.5)
    axes[1].set_title('Ground Truth')
    axes[2].imshow(ts[slice_idx], cmap='Blues', alpha=0.5)
    axes[2].set_title('TotalSegmentator')
    axes[3].imshow(pred[slice_idx], cmap='Greens', alpha=0.5)
    axes[3].set_title('Prediction')
    plt.savefig('diagnosis_{case_name}.png')
    """)
    
    return {
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'gt_voxels': gt_voxels,
        'pred_voxels': pred_voxels,
        'ts_dice': ts_dice
    }


def analyze_dataset(img_dir, gt_dir, ts_dir, pred_dir, num_cases=5):
    """Analyze multiple cases"""
    
    gt_dir_path = Path(gt_dir)
    gt_files = sorted(gt_dir_path.glob("*.nii.gz"))
    
    print(f"\n{'='*80}")
    print(f"ANALYZING {num_cases} CASES FROM DATASET")
    print(f"{'='*80}")
    
    results = []
    for gt_file in gt_files[:num_cases]:
        case_name = gt_file.stem.replace('.nii', '')
        result = analyze_case(case_name, img_dir, gt_dir, ts_dir, pred_dir)
        if result:
            results.append(result)
    
    # Summary
    if results:
        print(f"\n{'='*80}")
        print(f"SUMMARY ACROSS {len(results)} CASES")
        print(f"{'='*80}")
        
        avg_dice = np.mean([r['dice'] for r in results])
        avg_precision = np.mean([r['precision'] for r in results])
        avg_recall = np.mean([r['recall'] for r in results])
        avg_ts_dice = np.mean([r['ts_dice'] for r in results])
        
        print(f"Average Dice:      {avg_dice:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall:    {avg_recall:.4f}")
        print(f"Average TS Quality: {avg_ts_dice:.4f}")
        
        # Overall diagnosis
        print(f"\nOVERALL DIAGNOSIS:")
        if avg_ts_dice < 0.6:
            print(f"  🔴 MAIN ISSUE: TotalSegmentator masks are poor quality!")
            print(f"     TS Dice: {avg_ts_dice:.2f} - clicks are based on bad masks")
            print(f"     → Consider using GT directly for click generation (testing only)")
            print(f"     → Or use a different automatic segmentation method")
        elif avg_recall < 0.5:
            print(f"  🟡 MAIN ISSUE: Under-segmentation")
            print(f"     → Increase positive clicks (40-50)")
            print(f"     → Reduce erosion (0-1 iterations)")
            print(f"     → Check if ROI crop (128³) is cutting off anatomy")
        elif avg_precision < 0.5:
            print(f"  🟡 MAIN ISSUE: Over-segmentation")
            print(f"     → Increase negative clicks (15-20)")
            print(f"     → Increase erosion (3-4 iterations)")
        else:
            print(f"  🟢 Reasonable performance, minor tuning needed")
            print(f"     → Try different click sampling strategies")
            print(f"     → Consider larger ROI crop size (256³)")


if __name__ == "__main__":
    # Configuration
    IMG_DIR = "/data/ibd/data/RAOS/RAOS-Real/CancerImages(Set1)/imagesTr"
    GT_DIR = "/data/ibd/data/RAOS/RAOS-Real/CancerImages(Set1)/labelsTr_intestinal_tract"
    TS_DIR = "/data/ibd/data/RAOS/raos_tr_totalseg_labels"
    PRED_DIR = "./data/raos_pred_sammed3d_improved"
    
    # Analyze first 5 cases
    analyze_dataset(IMG_DIR, GT_DIR, TS_DIR, PRED_DIR, num_cases=5)