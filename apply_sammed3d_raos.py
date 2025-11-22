# -*- encoding: utf-8 -*-

import os
import os.path as osp
from glob import glob

import medim
from tqdm import tqdm
from scipy import ndimage as ndi
import torch

from utils.infer_utils_hoda import validate_paired_img_gt_with_ts_clicks

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def binary_erosion(mask, iterations=2):
    """
    Apply binary erosion to a 3D mask to shrink foreground regions.
    :param mask: 3D numpy array with boolean or binary values.
    :param iterations: Number of erosion iterations.
    :return: Eroded mask as a 3D numpy array.
    """
    struct = ndi.generate_binary_structure(3, 1)  # 3x3x3 cross structure
    eroded_mask = ndi.binary_erosion(mask, structure=struct, iterations=iterations)
    return eroded_mask.astype(mask.dtype)


if __name__ == "__main__":
    ''' 
    This script uses TotalSegmentator eroded masks for click generation
    and real ground truth masks for dice computation.
    '''
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    test_data_list = [
        dict(
            img_dir="/data/ibd/data/RAOS/RAOS-Real/CancerImages(Set1)/imagesTr",
            gt_dir="/data/ibd/data/RAOS/RAOS-Real/CancerImages(Set1)/labelsTr_intestinal_tract",
            ts_dir="/data/ibd/data/RAOS/raos_tr_totalseg_labels",  # TotalSegmentator masks
            out_dir="./data/raos_pred_sammed3d_ts_clicks",
            ckpt_path="./sam_med3d_turbo.pth",
            erosion_iterations=2,  # Number of erosion iterations for TS masks
        ),
    ]
    
    for test_data in test_data_list:
        print(f"Loading model from {test_data['ckpt_path']}...")
        model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=test_data["ckpt_path"])
        
        # Move model to GPU and set to eval mode
        model = model.to(device)
        model.eval()
        print(f"✅ Model loaded on {device}")
        
        gt_fname_list = sorted(glob(osp.join(test_data["gt_dir"], "*.nii.gz")))
        
        print(f"\nProcessing {len(gt_fname_list)} cases...")
        print(f"Ground truth dir: {test_data['gt_dir']}")
        print(f"TotalSegmentator dir: {test_data['ts_dir']}")
        print(f"Output dir: {test_data['out_dir']}\n")
        
        for gt_fname in tqdm(gt_fname_list, desc="Segmenting cases"):
            case_name = osp.basename(gt_fname).replace(".nii.gz", "")
            
            # Paths
            img_path = osp.join(test_data["img_dir"], f"{case_name}.nii.gz")
            gt_path = gt_fname  # Real ground truth for dice computation
            ts_path = osp.join(test_data["ts_dir"], case_name, "intestinal_tract.nii.gz")
            out_path = osp.join(test_data["out_dir"], f"{case_name}.nii.gz")
            
            # Check if files exist
            if not osp.exists(img_path):
                print(f"⚠️  Skipping {case_name}: Image not found at {img_path}")
                continue
            
            if not osp.exists(ts_path):
                print(f"⚠️  Skipping {case_name}: TotalSegmentator mask not found at {ts_path}")
                continue
            
            # Run inference with TotalSegmentator clicks
            try:
                validate_paired_img_gt_with_ts_clicks(
                    model=model,
                    img_path=img_path,
                    gt_path=gt_path,  # Real GT for dice
                    ts_path=ts_path,  # TotalSegmentator for clicks
                    output_path=out_path,
                    num_clicks=100,  # 100 foreground clicks
                    erosion_iterations=test_data.get("erosion_iterations", 2),
                    crop_size=128,
                    target_spacing=(1.5, 1.5, 1.5),
                    seed=233,
                    device=device  # Pass device to function
                )
            except Exception as e:
                print(f"❌ Error processing {case_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n✅ Processing complete! Results saved to {test_data['out_dir']}")