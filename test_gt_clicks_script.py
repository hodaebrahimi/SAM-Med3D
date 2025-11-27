import os
import os.path as osp
from glob import glob
import medim
from tqdm import tqdm
import torch
import numpy as np

from utils.infer_utils_hoda import validate_with_gt_clicks

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    """
    DIAGNOSTIC TEST: Use REAL GT for click generation
    
    This will tell us if SAM-Med3D can actually segment bowel well
    when given perfect clicks from the ground truth.
    
    If this STILL gives low Dice (~0.4), then SAM-Med3D is just not 
    suitable for this task.
    
    If this gives high Dice (>0.7), then the problem is that TS masks
    are not good enough for generating clicks.
    """
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Simple configuration
    NUM_POSITIVE_CLICKS = 20  # From real GT foreground
    NUM_NEGATIVE_CLICKS = 10  # From real GT background
    CROP_SIZE = 128           # Model requirement
    
    print("\n" + "="*80)
    print("DIAGNOSTIC TEST: Using REAL GT for Click Generation")
    print("="*80)
    print(f"This tests SAM-Med3D's TRUE capability on bowel segmentation")
    print(f"Positive Clicks: {NUM_POSITIVE_CLICKS} (from GT foreground)")
    print(f"Negative Clicks: {NUM_NEGATIVE_CLICKS} (from GT background)")
    print(f"NO erosion, NO TS masks - using pure GT")
    print("="*80 + "\n")
    
    test_data = dict(
        img_dir="/data/ibd/data/RAOS/RAOS-Real/CancerImages(Set1)/imagesTr",
        gt_dir="/data/ibd/data/RAOS/RAOS-Real/CancerImages(Set1)/labelsTr_intestinal_tract",
        out_dir="./data/raos_pred_sammed3d_GT_CLICKS_TEST",
        ckpt_path="./sam_med3d_turbo.pth",
    )
    
    print(f"Loading model...")
    model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=test_data["ckpt_path"])
    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded\n")
    
    gt_fname_list = sorted(glob(osp.join(test_data["gt_dir"], "*.nii.gz")))
    
    # Test on first 10 cases only
    print(f"Testing on first 10 cases (diagnostic run)...\n")
    
    dice_scores = []
    
    for gt_fname in tqdm(gt_fname_list[:10], desc="Testing cases"):
        case_name = osp.basename(gt_fname).replace(".nii.gz", "")
        
        img_path = osp.join(test_data["img_dir"], f"{case_name}.nii.gz")
        gt_path = gt_fname
        out_path = osp.join(test_data["out_dir"], f"{case_name}.nii.gz")
        
        if not osp.exists(img_path):
            print(f"⚠️  Skipping {case_name}: Image not found")
            continue
        
        try:
            dice_score = validate_with_gt_clicks(
                model=model,
                img_path=img_path,
                gt_path=gt_path,
                output_path=out_path,
                num_positive=NUM_POSITIVE_CLICKS,
                num_negative=NUM_NEGATIVE_CLICKS,
                crop_size=CROP_SIZE,
                target_spacing=(1.5, 1.5, 1.5),
                seed=233,
                device=device
            )
            dice_scores.append(dice_score)
            
        except Exception as e:
            print(f"❌ Error: {case_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Results
    if dice_scores:
        print("\n" + "="*80)
        print("DIAGNOSTIC RESULTS (with GT clicks):")
        print("="*80)
        print(f"Cases tested: {len(dice_scores)}")
        print(f"Mean Dice: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
        print(f"Median Dice: {np.median(dice_scores):.4f}")
        print(f"Range: [{np.min(dice_scores):.4f}, {np.max(dice_scores):.4f}]")
        print("="*80)
        
        # Interpretation
        mean_dice = np.mean(dice_scores)
        print("\nINTERPRETATION:")
        if mean_dice > 0.7:
            print(f"✅ GOOD ({mean_dice:.2f}) - SAM-Med3D CAN segment bowel well!")
            print("   Problem is likely TS mask quality or erosion strategy.")
            print("   → Try less erosion or better automatic segmentation")
        elif mean_dice > 0.5:
            print(f"🟡 MODERATE ({mean_dice:.2f}) - SAM-Med3D has limited capability")
            print("   Even with perfect clicks, performance is mediocre.")
            print("   → SAM-Med3D may not be ideal for bowel")
        else:
            print(f"🔴 POOR ({mean_dice:.2f}) - SAM-Med3D cannot handle this task")
            print("   Even with GT clicks, it fails.")
            print("   → Need a different model/approach entirely")
        print("="*80)
    
    print(f"\n✅ Test complete! Results in {test_data['out_dir']}")