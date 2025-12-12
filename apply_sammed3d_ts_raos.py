# -*- encoding: utf-8 -*-

import os
import os.path as osp
from glob import glob

import medim
from tqdm import tqdm
import torch
import numpy as np

from utils.infer_utils_hoda import validate_paired_img_gt_with_ts_clicks

os.environ["CUDA_VISIBLE_DEVICES"] = "6"


if __name__ == "__main__":
    ''' 
    SAM-Med3D with Center-of-Mass Point Prompts from TotalSegmentator/Vista
    
    Strategy:
    1. Load TotalSegmentator/Vista mask as guidance
    2. For each slice with mask content, compute center of mass -> point prompt
    3. Sample additional points from foreground if needed to reach ~50 points
    4. Generate negative points from boundary zone
    5. Feed all points to SAM-Med3D for segmentation
    
    This mimics MedSAM2's slice selection but uses point prompts instead of masks.
    '''
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    
    NUM_POSITIVE_TARGET = 50   # Target number of positive clicks (center-of-mass + random)
    NUM_NEGATIVE = 20          # Negative clicks from boundary zone
    STRIDE = 1                 # Sample every Nth slice (1 = every slice with content)
    EROSION_ITERATIONS = 0     # Erosion before extracting centers (0 = use full mask)
    CROP_SIZE = 128            # Must use 128³ (SAM-Med3D limitation)
    BOUNDARY_DILATION = 5      # Pixels to dilate for negative sampling zone
    
    # ============================================================================
    
    test_data_list = [
        dict(
            img_dir="/data/ibd/data/RAOS/RAOS-Real/CancerImages(Set1)/imagesTr",
            gt_dir="/data/ibd/data/RAOS/RAOS-Real/CancerImages(Set1)/labelsTr_intestinal_tract",
            ts_dir="/data/ibd/data/RAOS/raos_tr_totalseg_labels",  # or raos_tr_vista_labels
            out_dir="./data/raos_pred_sammed3d_com_prompts",
            ckpt_path="./sam_med3d_turbo.pth",
        ),
    ]
    
    print("\n" + "="*80)
    print("SAM-Med3D with CENTER-OF-MASS POINT PROMPTS")
    print("="*80)
    print(f"Target Positive Clicks: {NUM_POSITIVE_TARGET}")
    print(f"  - Center-of-mass per slice with content")
    print(f"  - Additional random foreground samples if needed")
    print(f"Negative Clicks: {NUM_NEGATIVE} (from boundary zone)")
    print(f"Slice Stride: {STRIDE} (1 = every slice with TS content)")
    print(f"Erosion Iterations: {EROSION_ITERATIONS} (0 = no erosion)")
    print(f"ROI Crop Size: {CROP_SIZE}³ (model limitation)")
    print(f"Boundary Dilation: {BOUNDARY_DILATION} pixels")
    print(f"\nStrategy:")
    print(f"  1. Identify slices with TotalSegmentator/Vista content")
    print(f"  2. Compute center-of-mass in each slice -> positive points")
    print(f"  3. Sample additional foreground points if <{NUM_POSITIVE_TARGET}")
    print(f"  4. Sample negative points from dilated boundary zone")
    print(f"  5. Feed all {NUM_POSITIVE_TARGET + NUM_NEGATIVE} points to SAM-Med3D")
    print("="*80 + "\n")
    
    for test_data in test_data_list:
        print(f"Loading model from {test_data['ckpt_path']}...")
        model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=test_data["ckpt_path"])
        
        model = model.to(device)
        model.eval()
        print(f"✅ Model loaded on {device}")
        
        gt_fname_list = sorted(glob(osp.join(test_data["gt_dir"], "*.nii.gz")))
        
        print(f"\nProcessing {len(gt_fname_list)} cases...\n")
        
        dice_scores = []
        
        for gt_fname in tqdm(gt_fname_list, desc="Segmenting cases"):
            case_name = osp.basename(gt_fname).replace(".nii.gz", "")
            
            img_path = osp.join(test_data["img_dir"], f"{case_name}.nii.gz")
            gt_path = gt_fname
            ts_path = osp.join(test_data["ts_dir"], case_name, "intestinal_tract.nii.gz")
            out_path = osp.join(test_data["out_dir"], f"{case_name}.nii.gz")
            
            if not osp.exists(img_path):
                print(f"⚠️  Skipping {case_name}: Image not found")
                continue
            
            if not osp.exists(ts_path):
                print(f"⚠️  Skipping {case_name}: TS mask not found")
                continue
            
            try:
                print(f"\n{'='*60}")
                print(f"Processing: {case_name}")
                print(f"{'='*60}")
                
                dice_score = validate_paired_img_gt_with_ts_clicks(
                    model=model,
                    img_path=img_path,
                    gt_path=gt_path,
                    ts_path=ts_path,
                    output_path=out_path,
                    num_positive_target=NUM_POSITIVE_TARGET,
                    num_negative=NUM_NEGATIVE,
                    stride=STRIDE,
                    erosion_iterations=EROSION_ITERATIONS,
                    crop_size=CROP_SIZE,
                    target_spacing=(1.5, 1.5, 1.5),
                    seed=233,
                    device=device
                )
                dice_scores.append(dice_score)
                
            except Exception as e:
                print(f"❌ Error processing {case_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Summary
        if dice_scores:
            print("\n" + "="*80)
            print("FINAL RESULTS:")
            print("="*80)
            print(f"Cases processed: {len(dice_scores)}")
            print(f"Mean Dice: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
            print(f"Median Dice: {np.median(dice_scores):.4f}")
            print(f"Range: [{np.min(dice_scores):.4f}, {np.max(dice_scores):.4f}]")
            print("="*80)
            
            # Show distribution
            bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
            bin_labels = ["<0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.0"]
            hist, _ = np.histogram(dice_scores, bins=bins)
            print("\nDice Distribution:")
            for label, count in zip(bin_labels, hist):
                pct = 100 * count / len(dice_scores)
                print(f"  {label}: {count} cases ({pct:.1f}%)")
                bar = "█" * int(pct / 2)
                print(f"    {bar}")
        
        print(f"\n✅ Complete! Results saved to {test_data['out_dir']}")


# ============================================================================
# TROUBLESHOOTING GUIDE:
# ============================================================================
#
# If under-segmenting (missing regions):
# 1. Set EROSION_ITERATIONS = 0 (use full TS mask)
# 2. Increase STRIDE to get more slices (try stride=1)
# 3. Increase NUM_POSITIVE_TARGET to 70-100
# 4. Check if TS mask has good coverage
#
# If over-segmenting (including too much):
# 1. Set EROSION_ITERATIONS = 2-3 (more conservative centers)
# 2. Increase NUM_NEGATIVE to 30-40
# 3. Reduce BOUNDARY_DILATION to 3
# 4. Try STRIDE = 2 (fewer positive prompts)
#
# If results are similar to previous approach:
# - The issue may be SAM-Med3D's 128³ limitation cutting off anatomy
# - Compare with MedSAM2 which uses full-resolution propagation
# - Consider using larger patch size if model supports it
#
# Expected behavior:
# - Should generate ~50 positive points distributed along organ
# - Should see printout showing: "Found N slices with mask content"
# - Points should be in (Z, Y, X) coordinate order