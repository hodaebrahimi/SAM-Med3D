import copy
import os
import os.path as osp

import edt
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torchio as tio
from scipy import ndimage as ndi


def binary_erosion(mask, iterations=2):
    """
    Apply binary erosion to a 3D mask to shrink foreground regions.
    """
    struct = ndi.generate_binary_structure(3, 1)
    eroded_mask = ndi.binary_erosion(mask, structure=struct, iterations=iterations)
    return eroded_mask.astype(mask.dtype)


def random_sample_next_click(prev_mask, gt_mask, method='random', force_positive=True):
    """
    Randomly sample one click from ground-truth mask and previous seg mask

    Arguments:
        prev_mask: (torch.Tensor) [H,W,D] previous mask that SAM-Med3D predict
        gt_mask: (torch.Tensor) [H,W,D] ground-truth mask for this image
        force_positive: (bool) If True, only generate positive (foreground) clicks
    """
    def ensure_3D_data(roi_tensor):
        if roi_tensor.ndim != 3:
            roi_tensor = roi_tensor.squeeze()
        assert roi_tensor.ndim == 3, "Input tensor must be 3D"
        return roi_tensor

    prev_mask = ensure_3D_data(prev_mask)
    gt_mask = ensure_3D_data(gt_mask)

    prev_mask = prev_mask > 0
    true_masks = gt_mask > 0

    if not true_masks.any():
        raise ValueError("Cannot find true value in the ground-truth!")

    fn_masks = torch.logical_and(true_masks, torch.logical_not(prev_mask))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), prev_mask)

    if method.lower() == 'random':
        if force_positive:
            # Only use false negatives (missed regions) or any true region if no errors
            if fn_masks.any():
                to_point_mask = fn_masks
            else:
                # If no false negatives, sample from any true region
                to_point_mask = true_masks
            
            all_points = torch.argwhere(to_point_mask)
            point = all_points[np.random.randint(len(all_points))]
            is_positive = True  # Always positive
        else:
            # Original behavior with both positive and negative clicks
            to_point_mask = torch.logical_or(fn_masks, fp_masks)

            if not to_point_mask.any():
                all_points = torch.argwhere(true_masks)
                point = all_points[np.random.randint(len(all_points))]
                is_positive = True
            else:
                all_points = torch.argwhere(to_point_mask)
                point = all_points[np.random.randint(len(all_points))]
                is_positive = bool(fn_masks[point[0], point[1], point[2]])

        sampled_point = point.clone().detach().reshape(1, 1, 3)
        sampled_label = torch.tensor([[int(is_positive)]], dtype=torch.long)

        return sampled_point, sampled_label

    elif method.lower() == 'ritm':
        fn_mask_single = F.pad(fn_masks[None, None], (1, 1, 1, 1, 1, 1), "constant", value=0).to(torch.uint8)[0, 0]
        fp_mask_single = F.pad(fp_masks[None, None], (1, 1, 1, 1, 1, 1), "constant", value=0).to(torch.uint8)[0, 0]

        fn_mask_dt = torch.tensor(edt.edt(fn_mask_single.cpu().numpy(), black_border=True, parallel=4))[1:-1, 1:-1, 1:-1]
        fp_mask_dt = torch.tensor(edt.edt(fp_mask_single.cpu().numpy(), black_border=True, parallel=4))[1:-1, 1:-1, 1:-1]

        fn_max_dist = torch.max(fn_mask_dt)
        fp_max_dist = torch.max(fp_mask_dt)

        if force_positive:
            # Always use false negatives for positive clicks
            is_positive = True
            dt = fn_mask_dt
            max_dist = fn_max_dist
        else:
            # Original behavior
            is_positive = fn_max_dist > fp_max_dist
            dt = fn_mask_dt if is_positive else fp_mask_dt
            max_dist = max(fn_max_dist, fp_max_dist)

        to_point_mask = (dt > (max_dist / 2.0))
        all_points = torch.argwhere(to_point_mask)

        if len(all_points) == 0:
            if force_positive:
                # Fallback to any true region
                all_points = torch.argwhere(true_masks)
                if len(all_points) > 0:
                    point = all_points[np.random.randint(len(all_points))]
                else:
                    point = torch.tensor([gt_mask.shape[0] // 2, gt_mask.shape[1] // 2, gt_mask.shape[2] // 2])
                is_positive = True
            else:
                point = torch.tensor([gt_mask.shape[0] // 2, gt_mask.shape[1] // 2, gt_mask.shape[2] // 2])
                is_positive = False
        else:
            point = all_points[np.random.randint(len(all_points))]
            if force_positive:
                is_positive = True
            else:
                is_positive = bool(fn_masks[point[0], point[1], point[2]])

        sampled_point = point.clone().detach().reshape(1, 1, 3)
        sampled_label = torch.tensor([[int(is_positive)]], dtype=torch.long)

        return sampled_point, sampled_label

    else:
        raise ValueError(f"Unsupported method: {method}. Choose 'ritm' or 'random'.")


def sam_model_infer(model,
                    roi_image,
                    roi_gt=None,
                    prompt_generator=random_sample_next_click,
                    prev_low_res_mask=None,
                    num_clicks=1,
                    force_positive=True,
                    device=None):
    '''
    Inference for SAM-Med3D, inputs prompt points with its labels
    
    Args:
        force_positive: If True, only generate positive (foreground) clicks
        device: torch.device to use for inference
    '''
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    # Don't move model here - assume it's already on the correct device

    if roi_gt is not None and (roi_gt == 0).all() and num_clicks > 0:
        print("Warning: roi_gt is empty. Prediction will be empty.")
        return np.zeros_like(roi_image.cpu().numpy().squeeze()), None

    with torch.no_grad():
        input_tensor = roi_image.to(device)
        image_embeddings = model.image_encoder(input_tensor)

        points_coords = torch.zeros(1, 0, 3, device=device)
        points_labels = torch.zeros(1, 0, device=device)
        
        current_prev_mask_for_click_generation = torch.zeros_like(roi_image, device=device)[:,0,...]
        
        if prev_low_res_mask is None:
            prev_low_res_mask = torch.zeros(1, 1, roi_image.shape[2] // 4,
                                           roi_image.shape[3] // 4,
                                           roi_image.shape[4] // 4, device=device, dtype=torch.float)

        for click_idx in range(num_clicks):
            if roi_gt is not None:
                # Pass force_positive to the prompt generator
                new_points_co, new_points_la = prompt_generator(
                    current_prev_mask_for_click_generation.squeeze(0).cpu(),
                    roi_gt[0, 0].cpu(),
                    force_positive=force_positive
                )
                new_points_co, new_points_la = new_points_co.to(device), new_points_la.to(device)
            else:
                if points_coords.shape[1] == 0:
                    center_z = roi_image.shape[2] // 2
                    center_y = roi_image.shape[3] // 2
                    center_x = roi_image.shape[4] // 2
                    new_points_co = torch.tensor([[[center_x, center_y, center_z]]], device=device, dtype=torch.float)
                    new_points_la = torch.tensor([[1]], device=device, dtype=torch.int64)
                else:
                    print("Warning: No ground truth for subsequent click generation.")
                    break 
            
            points_coords = torch.cat([points_coords, new_points_co], dim=1)
            points_labels = torch.cat([points_labels, new_points_la], dim=1)

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
            
            prev_low_res_mask = low_res_masks.detach()

            current_prev_mask_for_click_generation = F.interpolate(low_res_masks,
                                   size=roi_image.shape[-3:],
                                   mode='trilinear',
                                   align_corners=False)
            current_prev_mask_for_click_generation = torch.sigmoid(current_prev_mask_for_click_generation) > 0.5

        final_masks_hr = F.interpolate(low_res_masks,
                                       size=roi_image.shape[-3:],
                                       mode='trilinear',
                                       align_corners=False)

    medsam_seg_prob = torch.sigmoid(final_masks_hr)
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
    medsam_seg_mask = (medsam_seg_prob > 0.5).astype(np.uint8)

    return medsam_seg_mask, low_res_masks.detach()


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
    """
    Applies CropOrPad and ZNormalization to the canonical subject.
    """
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
    """
    Preprocess subject: extract category, resample, canonicalize, crop/pad, normalize
    """
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
    """
    Map ROI prediction back to original image grid
    """
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
    """Saves a NumPy array to NIFTI using SimpleITK, restoring original metadata."""
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


def get_category_list_and_zero_mask(label_path):
    """Get unique categories and create zero mask"""
    arr, meta = read_arr_from_nifti(label_path, get_meta_info=True)
    unique_label = np.unique(arr)
    unique_fg_labels = [int(l) for l in unique_label if l != 0]
    return unique_fg_labels, np.zeros(meta["original_numpy_shape"], dtype=np.uint8)


def compute_dice(pred, gt):
    """Compute Dice coefficient"""
    pred_bool = pred > 0
    gt_bool = gt > 0
    
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = pred_bool.sum() + gt_bool.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return 2.0 * intersection / union


def validate_paired_img_gt_with_ts_clicks(model, img_path, gt_path, ts_path, output_path, 
                                          num_clicks=1, erosion_iterations=2,
                                          crop_size=128, target_spacing=(1.5, 1.5, 1.5), 
                                          seed=233, device=None):
    """
    Validate using TotalSegmentator masks for click generation and real GT for dice computation.
    
    Args:
        model: SAM-Med3D model
        img_path: Path to input CT image
        gt_path: Path to ground truth mask (for dice computation only)
        ts_path: Path to TotalSegmentator mask (for click generation)
        output_path: Path to save prediction
        num_clicks: Number of clicks to use
        erosion_iterations: Number of erosion iterations for TS foreground mask
        crop_size: ROI crop size
        target_spacing: Target spacing for resampling
        seed: Random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(osp.dirname(output_path), exist_ok=True)
    
    # Load GT for dice computation and getting categories
    exist_categories, final_pred_numpy_original_grid = get_category_list_and_zero_mask(gt_path)
    _, gt_meta_for_saving = read_arr_from_nifti(gt_path, get_meta_info=True)
    
    # Load and validate real GT array for final dice computation
    gt_arr = read_arr_from_nifti(gt_path)
    
    # Ensure GT is binary (0 and 1 only)
    unique_gt_values = np.unique(gt_arr)
    print(f"   GT unique values before binarization: {unique_gt_values}")
    gt_arr_binary = (gt_arr > 0).astype(np.uint8)
    print(f"   GT unique values after binarization: {np.unique(gt_arr_binary)}")
    
    for category_index in exist_categories:
        # Load original TS mask
        ts_arr_original = read_arr_from_nifti(ts_path)
        ts_binary_original = (ts_arr_original == category_index).astype(np.uint8)
        
        # Create ERODED version for foreground
        ts_binary_eroded = binary_erosion(ts_binary_original, iterations=erosion_iterations)
        
        print(f"   TS foreground voxels (original): {ts_binary_original.sum()}")
        print(f"   TS foreground voxels (eroded): {ts_binary_eroded.sum()}")
        
        # Create combined mask: eroded foreground (1) + original background (0)
        ts_combined = ts_binary_eroded.astype(np.uint8)  # Eroded foreground
        
        # Save the combined TS mask to a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_file:
            tmp_ts_path = tmp_file.name
        
        # Save combined mask with same metadata as original TS
        ts_meta = read_arr_from_nifti(ts_path, get_meta_info=True)[1]
        save_numpy_to_nifti(ts_combined, tmp_ts_path, ts_meta)
        
        # Now create subject with the image and temporary combined TS mask
        subject_for_clicks, meta_info = get_subject_and_meta_info(img_path, tmp_ts_path)
        subject_for_clicks_copy = copy.deepcopy(subject_for_clicks)
        meta_info_for_clicks = copy.deepcopy(meta_info)
        
        # Preprocess with combined TS mask (category_index=1 since we already made it binary)
        roi_image, roi_ts_combined, meta_info_processed = data_preprocess(
            subject_for_clicks_copy,
            meta_info_for_clicks,
            category_index=1,  # Binary mask
            target_spacing=target_spacing,
            crop_size=crop_size
        )

        # Run inference with combined TS clicks (eroded FG + original BG)
        roi_pred_numpy, _ = sam_model_infer(
            model, 
            roi_image, 
            roi_gt=roi_ts_combined,  # Use eroded FG + original BG for clicks
            num_clicks=num_clicks,
            prev_low_res_mask=None,
            force_positive=True  # Only positive clicks
        )

        # Post-process to original grid
        cls_pred_original_grid = data_postprocess(roi_pred_numpy, meta_info_processed)
        final_pred_numpy_original_grid[cls_pred_original_grid == 1] = category_index
        
        # Clean up temporary file
        try:
            os.remove(tmp_ts_path)
        except:
            pass

    # Ensure prediction is binary
    final_pred_binary = (final_pred_numpy_original_grid > 0).astype(np.uint8)
    print(f"   Prediction unique values: {np.unique(final_pred_binary)}")
    
    # Save prediction
    save_numpy_to_nifti(final_pred_binary, output_path, gt_meta_for_saving)
    
    # Compute dice with REAL ground truth (both binary)
    dice_score = compute_dice(final_pred_binary, gt_arr_binary)
    print(f"✅ Saved to {output_path}")
    print(f"   Dice (vs real GT): {dice_score:.4f}")
    
    return dice_score


# Keep original function for backward compatibility
def validate_paired_img_gt(model, img_path, gt_path, output_path, num_clicks=1, 
                          crop_size=128, target_spacing=(1.5, 1.5, 1.5), seed=233):
    """Original function using GT for both clicks and dice"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(osp.dirname(output_path), exist_ok=True)
    
    exist_categories, final_pred_numpy_original_grid = get_category_list_and_zero_mask(gt_path)
    _, gt_meta_for_saving = read_arr_from_nifti(gt_path, get_meta_info=True)
    subject, meta_info = get_subject_and_meta_info(img_path, gt_path)

    for category_index in exist_categories:
        category_specific_subject = copy.deepcopy(subject)
        category_specific_meta_info = copy.deepcopy(meta_info)
        
        roi_image, roi_label, meta_info = data_preprocess(
            category_specific_subject,
            category_specific_meta_info,
            category_index=category_index,
            target_spacing=target_spacing,
            crop_size=crop_size
        )

        roi_pred_numpy, _ = sam_model_infer(
            model, roi_image, roi_gt=roi_label, 
            num_clicks=num_clicks,
            prev_low_res_mask=None
        )

        cls_pred_original_grid = data_postprocess(roi_pred_numpy, meta_info)
        final_pred_numpy_original_grid[cls_pred_original_grid == 1] = category_index

    save_numpy_to_nifti(final_pred_numpy_original_grid, output_path, gt_meta_for_saving)