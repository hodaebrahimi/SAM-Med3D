# -*- encoding: utf-8 -*-

import os
import os.path as osp
from glob import glob

import medim
from tqdm import tqdm
import torch
import numpy as np

# Import the sliding window with POINT PROMPTS function
from utils.infer_utils_hoda_point_prompt import validate_with_sliding_window_points

os.environ["CUDA_VISIBLE_DEVICES"] = "5"


if __name__ == "__main__":
    ''' 
    SAM-Med3D with SLIDING WINDOW + POINT PROMPTS
    
    KEY CHANGE FROM PREVIOUS VERSION:
    ✓ Uses POINT PROMPTS (what SAM-Med3D was trained on!)
    ✓ TS mask generates points, not used as dense mask prompt
    ✓ Each patch gets 10 positive + 5 negative points from TS mask
    ✓ Sliding window handles large volumes
    
    This should work MUCH better than dense mask prompts!
    '''
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    
    # Sliding window settings
    PATCH_SIZE = 128           # SAM-Med3D requirement (cannot change)
    MIN_CONTENT = 0.01         # Skip patches with <1% organ content
    
    # Point prompt settings (KEY PARAMETERS!)
    NUM_POSITIVE_PER_PATCH = 10   # Positive points per patch from TS mask
    NUM_NEGATIVE_PER_PATCH = 5    # Negative points per patch (boundary)
    EROSION_ITERATIONS = 0        # Erosion of TS mask before point generation
    
    # Data settings
    TARGET_SPACING = (1.5, 1.5, 1.5)
    
    # ============================================================================
    
    test_data_list = [
        dict(
            img_dir="/data/ibd/data/RAOS/RAOS-Real/CancerImages(Set1)/imagesTr",
            gt_dir="/data/ibd/data/RAOS/RAOS-Real/CancerImages(Set1)/labelsTr_intestinal_tract",
            ts_dir="/data/ibd/data/RAOS/raos_tr_totalseg_labels",
            out_dir="./data/raos_pred_sammed3d_points_sliding",
            ckpt_path="./sam_med3d_turbo.pth",
        ),
    ]
    
    print("\n" + "="*80)
    print("SAM-Med3D: SLIDING WINDOW + POINT PROMPTS (CORRECTED APPROACH)")
    print("="*80)
    print(f"Patch Size: {PATCH_SIZE}³ (model requirement)")
    print(f"Min Content: {MIN_CONTENT} (skip patches <{MIN_CONTENT*100}% organ)")
    print(f"Points per patch: {NUM_POSITIVE_PER_PATCH} positive + {NUM_NEGATIVE_PER_PATCH} negative")
    print(f"Erosion: {EROSION_ITERATIONS} iterations")
    print(f"Target Spacing: {TARGET_SPACING}")
    print(f"\n🔑 KEY DIFFERENCE FROM PREVIOUS VERSION:")
    print(f"  ✗ Previous: Used dense TS mask as prompt (model not trained for this!)")
    print(f"  ✓ Now: Generate POINT prompts from TS mask (what model expects!)")
    print(f"\nWorkflow per patch:")
    print(f"  1. Extract 128³ patch from volume")
    print(f"  2. Extract corresponding TS mask patch")
    print(f"  3. Sample {NUM_POSITIVE_PER_PATCH} points from TS foreground")
    print(f"  4. Sample {NUM_NEGATIVE_PER_PATCH} points from TS boundary")
    print(f"  5. Feed patch + points to SAM-Med3D")
    print(f"  6. Merge predictions with Gaussian weights")
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
        processing_times = []
        
        import time
        
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
                
                start_time = time.time()
                
                dice_score = validate_with_sliding_window_points(
                    model=model,
                    img_path=img_path,
                    gt_path=gt_path,
                    ts_path=ts_path,
                    output_path=out_path,
                    patch_size=PATCH_SIZE,
                    min_content=MIN_CONTENT,
                    num_positive_per_patch=NUM_POSITIVE_PER_PATCH,
                    num_negative_per_patch=NUM_NEGATIVE_PER_PATCH,
                    erosion_iterations=EROSION_ITERATIONS,
                    target_spacing=TARGET_SPACING,
                    seed=233,
                    device=device
                )
                
                elapsed_time = time.time() - start_time
                
                dice_scores.append(dice_score)
                processing_times.append(elapsed_time)
                
                print(f"   Processing time: {elapsed_time:.1f} seconds")
                
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
            print(f"\nAccuracy Metrics:")
            print(f"  Mean Dice: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
            print(f"  Median Dice: {np.median(dice_scores):.4f}")
            print(f"  Range: [{np.min(dice_scores):.4f}, {np.max(dice_scores):.4f}]")
            
            print(f"\nTiming Metrics:")
            print(f"  Mean time: {np.mean(processing_times):.1f} ± {np.std(processing_times):.1f} seconds")
            print(f"  Median time: {np.median(processing_times):.1f} seconds")
            print(f"  Total time: {np.sum(processing_times)/60:.1f} minutes")
            
            print("="*80)
            
            # Show distribution
            bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
            bin_labels = ["<0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.0"]
            hist, _ = np.histogram(dice_scores, bins=bins)
            print("\nDice Distribution:")
            for label, count in zip(bin_labels, hist):
                pct = 100 * count / len(dice_scores)
                print(f"  {label}: {count:2d} cases ({pct:4.1f}%)", end="")
                bar = "█" * int(pct / 2)
                print(f"  {bar}")
            
            # Analysis
            print("\n" + "="*80)
            print("COMPARISON WITH PREVIOUS APPROACHES:")
            print("="*80)
            print("Previous Results:")
            print("  - Dense mask prompts: Dice ~0.10 (FAILED - model not trained for this)")
            print(f"\nCurrent Results (Point Prompts):")
            print(f"  - Mean Dice: {np.mean(dice_scores):.4f}")
            
            if np.mean(dice_scores) > 0.7:
                print("\n✅ Excellent! Point prompts work much better!")
                print("   SAM-Med3D understands point-based guidance")
            elif np.mean(dice_scores) > 0.5:
                print("\n✓ Good improvement! Consider tuning:")
                print(f"   - Increase NUM_POSITIVE_PER_PATCH to 15-20")
                print(f"   - Try EROSION_ITERATIONS = 1-2 for more conservative points")
            elif np.mean(dice_scores) > 0.3:
                print("\n⚠️  Moderate improvement but still not great. Try:")
                print(f"   - Increase points: NUM_POSITIVE_PER_PATCH = 20-30")
                print(f"   - Reduce MIN_CONTENT to 0.005 (process more patches)")
                print(f"   - Check TS mask quality visually")
            else:
                print("\n⚠️  Still low scores. Possible issues:")
                print(f"   - TS baseline quality may be very poor")
                print(f"   - Try comparing with TS→GT Dice as baseline")
                print(f"   - SAM-Med3D may not generalize well to intestinal tracts")
            
            high_quality = sum(1 for d in dice_scores if d >= 0.7)
            print(f"\nHigh quality cases (Dice ≥ 0.7): {high_quality}/{len(dice_scores)}")
        
        print(f"\n✅ Complete! Results saved to {test_data['out_dir']}")


# ============================================================================
# WHY THIS SHOULD WORK BETTER:
# ============================================================================
#
# SAM-Med3D Training:
#   ✓ Trained with POINT prompts (main method)
#   ✓ Trained with box prompts (secondary)
#   ✗ NOT trained with dense external masks as prompts
#
# Previous Approach (mask prompts):
#   - Fed full TS mask as dense prompt
#   - Model confused - never saw external masks during training
#   - Result: Dice ~0.10 (terrible!)
#
# Current Approach (point prompts):
#   - Sample 10 points from TS mask per patch
#   - Use points as prompts (what model expects!)
#   - Model knows how to use point guidance
#   - Expected: Much better results (Dice > 0.5)
#
# ============================================================================
# TUNING GUIDE:
# ============================================================================
#
# If results are good (Dice > 0.7):
#   ✅ You're done! The approach works.
#
# If results are moderate (Dice 0.5-0.7):
#   Try these improvements:
#   1. Increase NUM_POSITIVE_PER_PATCH to 15-20
#   2. Try EROSION_ITERATIONS = 1 (more conservative points)
#   3. Reduce MIN_CONTENT to 0.005 (process more patches)
#
# If results are still low (Dice < 0.5):
#   Check baseline:
#   1. What is TS→GT Dice? (your baseline)
#   2. If TS→GT is already low, SAM-Med3D can't fix poor input
#   3. Try visualizing a few cases to diagnose issues
#
# If much faster processing needed:
#   1. Increase MIN_CONTENT to 0.02 (skip more patches)
#   2. Reduce NUM_POSITIVE_PER_PATCH to 5
#   3. Increase stride to 80 (line 146 in utils file)
# ============================================================================