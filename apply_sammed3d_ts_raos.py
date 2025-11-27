# -*- encoding: utf-8 -*-

import os
import os.path as osp
from glob import glob

import medim
from tqdm import tqdm
import torch
import numpy as np

from utils.infer_utils_hoda import validate_paired_img_gt_with_ts_clicks

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == "__main__":
    ''' 
    IMPROVED SAM-Med3D INFERENCE:
    
    Key improvements:
    1. Proper coordinate conversion (z,y,x) → (x,y,z) for SAM
    2. Bounding box prompts in addition to point clicks
    3. Optional iterative refinement
    4. More positive clicks for complex anatomy
    '''
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ============================================================================
    # IMPROVED CONFIGURATION - BASED ON DIAGNOSTICS
    # ============================================================================
    
    # Problem: Model only supports 128³ input size (trained at this resolution)
    # Strategy: Keep 128³ but maximize clicks and no erosion
    
    NUM_POSITIVE_CLICKS = 50   # Maximum guidance with many clicks
    NUM_NEGATIVE_CLICKS = 20   # Strong boundary definition
    USE_BOX_PROMPT = False     
    NUM_ITERATIONS = 3         # Multiple refinement passes
    EROSION_ITERATIONS = 0     # NO erosion - use full TS mask
    CROP_SIZE = 128            # Must use 128 (model limitation)
    
    # The 128³ crop might cut off anatomy, but it's a model limitation
    # We compensate with many clicks and iterative refinement
    
    # ============================================================================
    
    test_data_list = [
        dict(
            img_dir="/data/ibd/data/RAOS/RAOS-Real/CancerImages(Set1)/imagesTr",
            gt_dir="/data/ibd/data/RAOS/RAOS-Real/CancerImages(Set1)/labelsTr_intestinal_tract",
            ts_dir="/data/ibd/data/RAOS/raos_tr_totalseg_labels",
            out_dir="./data/raos_pred_sammed3d_improved",
            ckpt_path="./sam_med3d_turbo.pth",
        ),
    ]
    
    print("\n" + "="*80)
    print("IMPROVED SAM-Med3D CONFIGURATION:")
    print("="*80)
    print(f"Positive Clicks: {NUM_POSITIVE_CLICKS}")
    print(f"Negative Clicks: {NUM_NEGATIVE_CLICKS}")
    print(f"Use Bounding Box: {USE_BOX_PROMPT} (disabled - SAM-Med3D box format issues)")
    print(f"Refinement Iterations: {NUM_ITERATIONS}")
    print(f"Erosion Iterations: {EROSION_ITERATIONS}")
    print(f"ROI Crop Size: {CROP_SIZE}³ (model limitation - trained at 128³)")
    print(f"Total prompts: {NUM_POSITIVE_CLICKS + NUM_NEGATIVE_CLICKS} point clicks")
    print(f"Key fix: Coordinates converted from (z,y,x) to (x,y,z) for SAM")
    print(f"\nDiagnostic findings: Severe under-segmentation (Recall: 0.27)")
    print(f"Strategy: Maximum clicks + No erosion + 3x refinement")
    print(f"Note: 128³ crop is a model limitation, may cut off large anatomy")
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
                dice_score = validate_paired_img_gt_with_ts_clicks(
                    model=model,
                    img_path=img_path,
                    gt_path=gt_path,
                    ts_path=ts_path,
                    output_path=out_path,
                    num_positive_clicks=NUM_POSITIVE_CLICKS,
                    num_negative_clicks=NUM_NEGATIVE_CLICKS,
                    use_box_prompt=USE_BOX_PROMPT,
                    num_iterations=NUM_ITERATIONS,
                    erosion_iterations=EROSION_ITERATIONS,
                    crop_size=CROP_SIZE,  # Use larger crop size
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
        
        print(f"\n✅ Complete! Results in {test_data['out_dir']}")


# TROUBLESHOOTING GUIDE:
#
# If Dice is still low (< 0.5):
# 1. Try MORE positive clicks (30-40) - bowel is very complex
# 2. Try NUM_ITERATIONS = 2 or 3 for refinement
# 3. Reduce EROSION_ITERATIONS to 1 for more coverage
# 4. Check TotalSegmentator mask quality visually
#
# If over-segmenting (including too much):
# 1. Increase EROSION_ITERATIONS (3-4)
# 2. Add more negative clicks (10-15)
# 3. Reduce positive clicks (12-15)
#
# If under-segmenting (missing regions):
# 1. Increase positive clicks (30-40)
# 2. Reduce EROSION_ITERATIONS (1)
# 3. Try NUM_ITERATIONS = 2-3
#
# Box prompts are DISABLED - SAM-Med3D has format incompatibility issues