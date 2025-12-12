import copy
import os
import os.path as osp
import numpy as np
import SimpleITK as sitk
import torch
import torchio as tio
from scipy import ndimage as ndi


def binary_erosion(mask, iterations=2):
    """Apply binary erosion. If iterations <= 0, return mask unchanged."""
    if iterations <= 0:
        return mask.astype(mask.dtype)
    struct = ndi.generate_binary_structure(3, 1)
    eroded_mask = ndi.binary_erosion(mask, structure=struct, iterations=iterations)
    return eroded_mask.astype(mask.dtype)


def generate_center_of_mass_clicks(
    ts_mask, 
    num_positive_target=50, 
    num_negative=20,
    stride=1,
    erosion_iterations=0,
    boundary_dilation=5,
    seed=None
):
    """
    Generate point prompts by finding center of mass in each slice with mask content.
    
    Strategy:
    1. Identify slices with non-zero mask pixels (like MedSAM2)
    2. For each slice, compute center of mass -> positive click
    3. Sample additional positive clicks from high-confidence regions if needed
    4. Generate negative clicks from dilated boundary or background
    
    Args:
        ts_mask: (D, H, W) or (H, W, D) binary mask from TotalSegmentator/Vista
        num_positive_target: Target number of positive clicks (~50)
        num_negative: Number of negative clicks
        stride: Slice sampling stride (1 = every slice with content)
        erosion_iterations: Erode mask before extracting centers
        boundary_dilation: Pixels to dilate for negative sampling zone
        seed: Random seed
        
    Returns:
        points_coords: (1, N, 3) tensor in (Z, Y, X) order
        points_labels: (1, N) tensor with 1=positive, 0=negative
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Ensure 3D
    if ts_mask.ndim != 3:
        ts_mask = ts_mask.squeeze()
    
    # Optional erosion to make prompts more conservative
    if erosion_iterations > 0:
        ts_mask = binary_erosion(ts_mask, iterations=erosion_iterations)
    
    ts_binary = ts_mask > 0
    D, H, W = ts_binary.shape
    
    points_list = []
    labels_list = []
    
    # ========================================================================
    # POSITIVE CLICKS: Center of mass per slice
    # ========================================================================
    
    # Find slices with content (same as MedSAM2 seed selection)
    slice_has_content = ts_binary.sum(axis=(1, 2)) > 0
    valid_slice_indices = np.where(slice_has_content)[0][::stride]
    
    print(f"   Found {len(valid_slice_indices)} slices with mask content (stride={stride})")
    
    # Compute center of mass for each valid slice
    for z_idx in valid_slice_indices:
        slice_mask = ts_binary[z_idx, :, :]
        
        if slice_mask.sum() == 0:
            continue
        
        # Compute center of mass in this slice
        y_coords, x_coords = np.where(slice_mask)
        center_y = int(np.mean(y_coords))
        center_x = int(np.mean(x_coords))
        
        # Store as (Z, Y, X) - SAM-Med3D format
        points_list.append([z_idx, center_y, center_x])
        labels_list.append(1)
    
    num_com_clicks = len(points_list)
    print(f"   Generated {num_com_clicks} center-of-mass clicks")
    
    # ========================================================================
    # ADDITIONAL POSITIVE CLICKS: Sample randomly from foreground if needed
    # ========================================================================
    
    if num_com_clicks < num_positive_target:
        num_additional = num_positive_target - num_com_clicks
        
        # Get all foreground voxels
        fg_coords = np.argwhere(ts_binary)  # (N, 3) in (Z, Y, X)
        
        if len(fg_coords) > 0:
            # Sample additional points randomly
            num_sample = min(num_additional, len(fg_coords))
            indices = np.random.choice(len(fg_coords), num_sample, replace=False)
            
            for idx in indices:
                points_list.append(fg_coords[idx])
                labels_list.append(1)
            
            print(f"   Added {num_sample} additional random foreground clicks")
    
    # ========================================================================
    # NEGATIVE CLICKS: Sample from boundary zone or background
    # ========================================================================
    
    if num_negative > 0:
        # Create boundary zone: dilate mask and subtract original
        if boundary_dilation > 0:
            struct = ndi.generate_binary_structure(3, 1)
            dilated_mask = ndi.binary_dilation(
                ts_binary, 
                structure=struct, 
                iterations=boundary_dilation
            )
            boundary_zone = dilated_mask & ~ts_binary
        else:
            # Just use background
            boundary_zone = ~ts_binary
        
        bg_coords = np.argwhere(boundary_zone)
        
        if len(bg_coords) > 0:
            num_sample = min(num_negative, len(bg_coords))
            indices = np.random.choice(len(bg_coords), num_sample, replace=False)
            
            for idx in indices:
                points_list.append(bg_coords[idx])
                labels_list.append(0)
            
            print(f"   Generated {num_sample} negative clicks from boundary zone")
        else:
            print(f"   Warning: No boundary zone found for negative clicks")
    
    # ========================================================================
    # Convert to tensors
    # ========================================================================
    
    if len(points_list) == 0:
        print("   Warning: No clicks generated!")
        return torch.zeros(1, 0, 3), torch.zeros(1, 0, dtype=torch.long)
    
    # Stack as (Z, Y, X) - DO NOT REORDER
    points_coords = torch.tensor(np.array(points_list), dtype=torch.float32).unsqueeze(0)
    points_labels = torch.tensor(labels_list, dtype=torch.long).unsqueeze(0)
    
    total_pos = (points_labels == 1).sum().item()
    total_neg = (points_labels == 0).sum().item()
    print(f"   Total prompts: {total_pos} positive + {total_neg} negative = {total_pos + total_neg}")
    
    return points_coords, points_labels


def sam_model_infer_with_ts_clicks(
    model, 
    roi_image, 
    roi_ts_mask,
    num_positive_target=50,
    num_negative=20,
    stride=1,
    erosion_iterations=0,
    seed=None, 
    device=None
):
    """
    Inference using center-of-mass clicks from TotalSegmentator/Vista mask.
    
    Args:
        model: SAM-Med3D model
        roi_image: (1, 1, D, H, W) preprocessed CT image
        roi_ts_mask: (1, 1, D, H, W) preprocessed TS mask
        num_positive_target: Target number of positive clicks
        num_negative: Number of negative clicks
        stride: Slice sampling stride
        erosion_iterations: Erosion before extracting centers
        seed: Random seed
        device: Torch device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    # Check if mask is empty
    if roi_ts_mask is not None and (roi_ts_mask == 0).all():
        print("   Warning: Empty TS mask")
        return np.zeros_like(roi_image.cpu().numpy().squeeze()), None
    
    with torch.no_grad():
        # Get image embeddings
        input_tensor = roi_image.to(device)
        image_embeddings = model.image_encoder(input_tensor)
        
        # Generate clicks from TS mask
        ts_mask_np = roi_ts_mask[0, 0].cpu().numpy()  # (D, H, W)
        
        points_coords, points_labels = generate_center_of_mass_clicks(
            ts_mask_np,
            num_positive_target=num_positive_target,
            num_negative=num_negative,
            stride=stride,
            erosion_iterations=erosion_iterations,
            seed=seed
        )
        
        if points_coords.shape[1] == 0:
            print("   No valid clicks generated, returning empty mask")
            return np.zeros_like(roi_image.cpu().numpy().squeeze()), None
        
        points_coords = points_coords.to(device)
        points_labels = points_labels.to(device)
        
        # Initialize mask input
        prev_low_res_mask = torch.zeros(
            1, 1,
            roi_image.shape[2] // 4,
            roi_image.shape[3] // 4,
            roi_image.shape[4] // 4,
            device=device, dtype=torch.float
        )
        
        # Single pass through SAM
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=[points_coords, points_labels],
            boxes=None,
            masks=prev_low_res_mask,
        )
        
        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
        )
        
        # Upscale to full resolution
        final_masks_hr = torch.nn.functional.interpolate(
            low_res_masks,
            size=roi_image.shape[-3:],
            mode='trilinear',
            align_corners=False
        )
    
    medsam_seg_prob = torch.sigmoid(final_masks_hr)
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
    medsam_seg_mask = (medsam_seg_prob > 0.5).astype(np.uint8)
    
    return medsam_seg_mask, low_res_masks.detach()


# ============================================================================
# Helper functions (unchanged from original)
# ============================================================================

def read_arr_from_nifti(nii_path, get_meta_info=False):
    sitk_image = sitk.ReadImage(nii_path)
    arr = sitk.GetArrayFromImage(sitk_image)
    if not get_meta_info:
        return arr
    meta_info = {
        "sitk_image_object": sitk_image,
        "sitk_origin": sitk_image.GetOrigin(),
        "sitk_direction": sitk_image.GetDirection(),
        "sitk_spacing": sitk_image.GetSpacing(),
        "original_numpy_shape": arr.shape,
    }
    return arr, meta_info


def get_roi_from_subject(subject_canonical, meta_info, crop_transform, norm_transform):
    meta_info["canonical_subject_shape"] = subject_canonical.spatial_shape
    meta_info["canonical_subject_affine"] = subject_canonical.image.affine.copy()
    padding_params, cropping_params = crop_transform._compute_center_crop_or_pad(subject_canonical)
    subject_cropped = crop_transform(subject_canonical)
    meta_info["padding_params_functional"] = padding_params
    meta_info["cropping_params_functional"] = cropping_params
    meta_info["roi_subject_affine"] = subject_cropped.image.affine.copy()
    
    img3D_roi = subject_cropped.image.data.clone().detach()
    img3D_roi = norm_transform(img3D_roi.squeeze(dim=1))
    img3D_roi = img3D_roi.unsqueeze(dim=1)
    gt3D_roi = subject_cropped.label.data.clone().detach()
    
    def correct_roi_dim(roi_tensor): 
        if roi_tensor.ndim == 3:
            roi_tensor = roi_tensor.unsqueeze(0).unsqueeze(0)
        if roi_tensor.ndim == 4:
            roi_tensor = roi_tensor.unsqueeze(0)
        if roi_tensor.shape[1] != 1:
            roi_tensor = roi_tensor[:, 0:1,...]
        return roi_tensor
    
    img3D_roi = correct_roi_dim(img3D_roi)
    gt3D_roi = correct_roi_dim(gt3D_roi)
    return img3D_roi, gt3D_roi, meta_info


def get_subject_and_meta_info(img_path, label_path):
    _, meta_info = read_arr_from_nifti(img_path, get_meta_info=True)
    subject = tio.Subject(
        image=tio.ScalarImage(img_path),
        label=tio.LabelMap(label_path) 
    )
    return subject, meta_info


def data_preprocess(subject, meta_info, category_index, target_spacing, crop_size=128):
    label_data_for_cat = subject.label.data.clone()
    new_label_data = torch.zeros_like(label_data_for_cat)
    new_label_data[label_data_for_cat == category_index] = 1
    subject.label.set_data(new_label_data)
    
    meta_info["original_subject_affine"] = subject.image.affine.copy()
    meta_info["original_subject_spatial_shape"] = subject.image.spatial_shape
    
    resampler = tio.Resample(target=target_spacing)
    subject_resampled = resampler(subject)
    transform_canonical = tio.ToCanonical()
    subject_canonical = transform_canonical(subject_resampled)
    crop_transform = tio.CropOrPad(mask_name='label', target_shape=(crop_size, crop_size, crop_size))
    norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
    roi_image, roi_label, meta_info = get_roi_from_subject(
        subject_canonical, meta_info, crop_transform, norm_transform
    )
    return roi_image, roi_label, meta_info


def data_postprocess(roi_pred_numpy, meta_info):
    roi_pred_tensor = torch.from_numpy(roi_pred_numpy.astype(np.float32)).unsqueeze(0)
    pred_label_map_roi_space = tio.LabelMap(
        tensor=roi_pred_tensor,
        affine=meta_info["roi_subject_affine"]
    )
    reference_tensor_shape = (1, *meta_info["original_subject_spatial_shape"]) 
    reference_image_original_space = tio.ScalarImage(
        tensor=torch.zeros(reference_tensor_shape),
        affine=meta_info["original_subject_affine"]
    )
    resampler_to_original_grid = tio.Resample(
        target=reference_image_original_space,
        image_interpolation='nearest' 
    )
    pred_resampled_to_original_space = resampler_to_original_grid(pred_label_map_roi_space)
    final_pred_numpy_dhw = pred_resampled_to_original_space.data.squeeze(0).cpu().numpy()
    final_pred_numpy = final_pred_numpy_dhw.astype(np.uint8)
    return final_pred_numpy.transpose(2, 1, 0)


def save_numpy_to_nifti(in_arr: np.array, out_path, meta_info_for_saving):
    out_img = sitk.GetImageFromArray(in_arr)
    original_sitk_image = meta_info_for_saving.get("sitk_image_object")
    if original_sitk_image:
        out_img.SetOrigin(original_sitk_image.GetOrigin())
        out_img.SetDirection(original_sitk_image.GetDirection())
        out_img.SetSpacing(original_sitk_image.GetSpacing())
    else:
        out_img.SetOrigin(meta_info_for_saving["sitk_origin"])
        out_img.SetDirection(meta_info_for_saving["sitk_direction"])
        out_img.SetSpacing(meta_info_for_saving["sitk_spacing"])
    sitk.WriteImage(out_img, out_path)


def compute_dice(pred, gt):
    pred_bool = pred > 0
    gt_bool = gt > 0
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = pred_bool.sum() + gt_bool.sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / union


def validate_paired_img_gt_with_ts_clicks(
    model, 
    img_path, 
    gt_path,
    ts_path,
    output_path,
    num_positive_target=50,
    num_negative=20,
    stride=1,
    erosion_iterations=0,
    crop_size=128, 
    target_spacing=(1.5, 1.5, 1.5), 
    seed=233, 
    device=None
):
    """
    Main validation function using TotalSegmentator mask for click generation.
    
    Args:
        model: SAM-Med3D model
        img_path: Path to CT image
        gt_path: Path to ground truth (for evaluation only)
        ts_path: Path to TotalSegmentator/Vista mask (for click generation)
        output_path: Where to save prediction
        num_positive_target: Target number of positive clicks (~50)
        num_negative: Number of negative clicks
        stride: Slice sampling stride (1 = every slice)
        erosion_iterations: Erosion before extracting centers (0 = no erosion)
        crop_size: ROI crop size (must be 128 for SAM-Med3D)
        target_spacing: Target voxel spacing
        seed: Random seed
        device: Torch device
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    
    # Load GT for evaluation
    gt_arr = read_arr_from_nifti(gt_path)
    _, gt_meta_for_saving = read_arr_from_nifti(gt_path, get_meta_info=True)
    gt_arr_binary = (gt_arr > 0).astype(np.uint8)
    
    # Preprocess IMAGE with TS mask
    subject, meta_info = get_subject_and_meta_info(img_path, ts_path)
    subject_copy = copy.deepcopy(subject)
    meta_info_copy = copy.deepcopy(meta_info)
    
    roi_image, roi_ts_mask, meta_info_processed = data_preprocess(
        subject_copy,
        meta_info_copy,
        category_index=1,
        target_spacing=target_spacing,
        crop_size=crop_size
    )
    
    print(f"   ROI TS mask voxels: {(roi_ts_mask > 0).sum().item()}")
    
    # Run inference with TS-based clicks
    roi_pred_numpy, _ = sam_model_infer_with_ts_clicks(
        model, 
        roi_image, 
        roi_ts_mask=roi_ts_mask,
        num_positive_target=num_positive_target,
        num_negative=num_negative,
        stride=stride,
        erosion_iterations=erosion_iterations,
        seed=seed,
        device=device
    )
    
    # Post-process
    final_pred_binary = data_postprocess(roi_pred_numpy, meta_info_processed)
    final_pred_binary = (final_pred_binary > 0).astype(np.uint8)
    
    # Save
    save_numpy_to_nifti(final_pred_binary, output_path, gt_meta_for_saving)
    
    # Compute dice
    dice_score = compute_dice(final_pred_binary, gt_arr_binary)
    print(f"✅ Dice (using TS center-of-mass clicks): {dice_score:.4f}\n")
    
    return dice_score