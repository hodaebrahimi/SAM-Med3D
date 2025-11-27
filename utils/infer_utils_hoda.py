# Based on the original SAM-Med3D code that works
# Key insight: Keep coordinates in (Z, Y, X) order, not (X, Y, Z)!

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


def generate_gt_based_clicks_simple(gt_mask, num_positive, num_negative, seed=None):
    """
    Generate clicks directly from GT mask (for testing SAM's capability).
    Uses simple random sampling - no EDT.
    Coordinates stay in (Z, Y, X) order as SAM-Med3D expects!
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    if gt_mask.ndim != 3:
        gt_mask = gt_mask.squeeze()
    
    gt_binary = gt_mask > 0
    points_list = []
    labels_list = []
    
    # Positive clicks from foreground
    if num_positive > 0:
        fg_points = torch.argwhere(gt_binary).cpu().numpy()
        if len(fg_points) > 0:
            indices = np.random.choice(len(fg_points), min(num_positive, len(fg_points)), replace=False)
            for idx in indices:
                # Keep in (Z, Y, X) order - don't convert!
                points_list.append(fg_points[idx])
                labels_list.append(1)
    
    # Negative clicks from background
    if num_negative > 0:
        bg_points = torch.argwhere(~gt_binary).cpu().numpy()
        if len(bg_points) > 0:
            indices = np.random.choice(len(bg_points), min(num_negative, len(bg_points)), replace=False)
            for idx in indices:
                # Keep in (Z, Y, X) order - don't convert!
                points_list.append(bg_points[idx])
                labels_list.append(0)
    
    if len(points_list) == 0:
        return torch.zeros(1, 0, 3), torch.zeros(1, 0, dtype=torch.long)
    
    # Stack as (Z, Y, X) - DO NOT REORDER
    points_coords = torch.tensor(np.array(points_list), dtype=torch.float32).unsqueeze(0)
    points_labels = torch.tensor(labels_list, dtype=torch.long).unsqueeze(0)
    
    return points_coords, points_labels


def sam_model_infer_simple(model, roi_image, roi_gt, num_positive=10, num_negative=5, 
                          seed=None, device=None):
    """Simplified inference - all clicks at once, based directly on GT"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    if roi_gt is not None and (roi_gt == 0).all():
        print("Warning: Empty GT mask")
        return np.zeros_like(roi_image.cpu().numpy().squeeze()), None
    
    with torch.no_grad():
        input_tensor = roi_image.to(device)
        image_embeddings = model.image_encoder(input_tensor)
        
        # Generate all clicks from GT
        points_coords, points_labels = generate_gt_based_clicks_simple(
            roi_gt[0, 0],
            num_positive=num_positive,
            num_negative=num_negative,
            seed=seed
        )
        
        points_coords = points_coords.to(device)
        points_labels = points_labels.to(device)
        
        print(f"      Generated {(points_labels==1).sum().item()} pos + {(points_labels==0).sum().item()} neg clicks")
        
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


# Keep all the helper functions from original code
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


def validate_with_gt_clicks(model, img_path, gt_path, output_path,
                            num_positive=10, num_negative=5,
                            crop_size=128, target_spacing=(1.5, 1.5, 1.5), 
                            seed=233, device=None):
    """
    TEST VERSION: Use real GT for click generation to see SAM's true capability
    This isolates whether the problem is SAM or the TS mask quality
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    
    # Load GT
    gt_arr = read_arr_from_nifti(gt_path)
    _, gt_meta_for_saving = read_arr_from_nifti(gt_path, get_meta_info=True)
    gt_arr_binary = (gt_arr > 0).astype(np.uint8)
    
    # Preprocess with REAL GT
    subject, meta_info = get_subject_and_meta_info(img_path, gt_path)
    subject_copy = copy.deepcopy(subject)
    meta_info_copy = copy.deepcopy(meta_info)
    
    roi_image, roi_gt, meta_info_processed = data_preprocess(
        subject_copy,
        meta_info_copy,
        category_index=1,
        target_spacing=target_spacing,
        crop_size=crop_size
    )
    
    print(f"   ROI GT voxels: {(roi_gt > 0).sum().item()}")
    
    # Run inference with GT-based clicks
    roi_pred_numpy, _ = sam_model_infer_simple(
        model, 
        roi_image, 
        roi_gt=roi_gt,
        num_positive=num_positive,
        num_negative=num_negative,
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
    print(f"✅ Dice (using GT clicks): {dice_score:.4f}\n")
    
    return dice_score