import copy
import os
import os.path as osp
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torchio as tio
from scipy import ndimage as ndi


def binary_erosion(mask, iterations=2):
    """Apply binary erosion. If iterations <= 0, return mask unchanged."""
    if iterations <= 0:
        return mask.astype(mask.dtype)
    struct = ndi.generate_binary_structure(3, 1)
    eroded_mask = ndi.binary_erosion(mask, structure=struct, iterations=iterations)
    return eroded_mask.astype(mask.dtype)


def generate_points_from_patch_mask(mask_patch, num_positive=10, num_negative=5, seed=None):
    """
    Generate point prompts from a mask patch for SAM-Med3D.
    Returns points in (Z, Y, X) coordinate order.
    
    Args:
        mask_patch: (128, 128, 128) numpy array
        num_positive: Number of positive points to sample from foreground
        num_negative: Number of negative points to sample from background
        seed: Random seed
    
    Returns:
        points_coords: (1, N, 3) tensor in (Z, Y, X) order
        points_labels: (1, N) tensor with 1=positive, 0=negative
    """
    if seed is not None:
        np.random.seed(seed)
    
    mask_binary = mask_patch > 0
    
    points_list = []
    labels_list = []
    
    # Positive points from foreground
    if num_positive > 0:
        fg_coords = np.argwhere(mask_binary)  # (N, 3) in (Z, Y, X)
        if len(fg_coords) > 0:
            num_sample = min(num_positive, len(fg_coords))
            indices = np.random.choice(len(fg_coords), num_sample, replace=False)
            for idx in indices:
                points_list.append(fg_coords[idx])  # Keep (Z, Y, X) order
                labels_list.append(1)
    
    # Negative points from background
    if num_negative > 0:
        # Sample from regions near the foreground (boundary zone)
        if mask_binary.sum() > 0:
            # Dilate to create boundary zone
            struct = ndi.generate_binary_structure(3, 1)
            dilated = ndi.binary_dilation(mask_binary, structure=struct, iterations=5)
            boundary_zone = dilated & ~mask_binary
            
            bg_coords = np.argwhere(boundary_zone)
            if len(bg_coords) > 0:
                num_sample = min(num_negative, len(bg_coords))
                indices = np.random.choice(len(bg_coords), num_sample, replace=False)
                for idx in indices:
                    points_list.append(bg_coords[idx])  # Keep (Z, Y, X) order
                    labels_list.append(0)
    
    if len(points_list) == 0:
        # Fallback: center point as positive
        center = mask_patch.shape[0] // 2
        points_list.append([center, center, center])
        labels_list.append(1)
    
    points_coords = torch.tensor(np.array(points_list), dtype=torch.float32).unsqueeze(0)
    points_labels = torch.tensor(labels_list, dtype=torch.long).unsqueeze(0)
    
    return points_coords, points_labels


def adaptive_patch_extraction(volume, ts_mask, patch_size=128, min_content=0.01):
    """
    Adaptively extract patches based on mask content.
    Only extract patches that contain significant mask content.
    
    Args:
        volume: (1, C, D, H, W) volume tensor
        ts_mask: (1, 1, D, H, W) mask tensor
        patch_size: Size of patches
        min_content: Minimum fraction of voxels in patch that must contain mask
    
    Returns:
        patches: List of volume patches
        mask_patches: List of corresponding mask patches
        coordinates: List of patch positions
    """
    D, H, W = volume.shape[-3:]
    
    patches = []
    mask_patches = []
    coordinates = []
    
    # Calculate stride (with overlap)
    stride = patch_size // 2  # 50% overlap
    
    for d_start in range(0, D, stride):
        for h_start in range(0, H, stride):
            for w_start in range(0, W, stride):
                # Calculate end positions
                d_end = min(d_start + patch_size, D)
                h_end = min(h_start + patch_size, H)
                w_end = min(w_start + patch_size, W)
                
                # Adjust start if we're at the boundary
                if d_end - d_start < patch_size:
                    d_start = max(0, D - patch_size)
                    d_end = D
                if h_end - h_start < patch_size:
                    h_start = max(0, H - patch_size)
                    h_end = H
                if w_end - w_start < patch_size:
                    w_start = max(0, W - patch_size)
                    w_end = W
                
                # Extract mask patch
                mask_patch = ts_mask[..., d_start:d_end, h_start:h_end, w_start:w_end]
                
                # Check if patch has sufficient content
                content_fraction = mask_patch.sum().item() / (patch_size ** 3)
                
                if content_fraction > min_content:
                    vol_patch = volume[..., d_start:d_end, h_start:h_end, w_start:w_end]
                    
                    # Pad if necessary (at boundaries)
                    if vol_patch.shape[-3:] != (patch_size, patch_size, patch_size):
                        pad_d = patch_size - vol_patch.shape[-3]
                        pad_h = patch_size - vol_patch.shape[-2]
                        pad_w = patch_size - vol_patch.shape[-1]
                        
                        vol_patch = F.pad(
                            vol_patch,
                            (0, pad_w, 0, pad_h, 0, pad_d),
                            mode='constant',
                            value=0
                        )
                        mask_patch = F.pad(
                            mask_patch,
                            (0, pad_w, 0, pad_h, 0, pad_d),
                            mode='constant',
                            value=0
                        )
                    
                    patches.append(vol_patch)
                    mask_patches.append(mask_patch)
                    coordinates.append((d_start, h_start, w_start))
    
    if len(patches) == 0:
        # Fallback: use center patch if no patches meet criteria
        d_start = max(0, (D - patch_size) // 2)
        h_start = max(0, (H - patch_size) // 2)
        w_start = max(0, (W - patch_size) // 2)
        
        d_end = min(d_start + patch_size, D)
        h_end = min(h_start + patch_size, H)
        w_end = min(w_start + patch_size, W)
        
        vol_patch = volume[..., d_start:d_end, h_start:h_end, w_start:w_end]
        mask_patch = ts_mask[..., d_start:d_end, h_start:h_end, w_start:w_end]
        
        # Pad to patch_size
        pad_d = patch_size - vol_patch.shape[-3]
        pad_h = patch_size - vol_patch.shape[-2]
        pad_w = patch_size - vol_patch.shape[-1]
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            vol_patch = F.pad(vol_patch, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)
            mask_patch = F.pad(mask_patch, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)
        
        patches = [vol_patch]
        mask_patches = [mask_patch]
        coordinates = [(d_start, h_start, w_start)]
    
    return patches, mask_patches, coordinates


def merge_predictions_3d(predictions, coordinates, original_shape, patch_size=128):
    """
    Merge overlapping patch predictions using Gaussian-weighted averaging.
    
    Args:
        predictions: List of (patch_size, patch_size, patch_size) predictions
        coordinates: List of (d_start, h_start, w_start) positions
        original_shape: (D, H, W) shape of original volume
        patch_size: Size of patches
    
    Returns:
        merged: (D, H, W) merged prediction
    """
    D, H, W = original_shape
    
    # Accumulation arrays
    prediction_sum = np.zeros((D, H, W), dtype=np.float32)
    weight_sum = np.zeros((D, H, W), dtype=np.float32)
    
    # Create gaussian weight for blending
    center = patch_size // 2
    grid = np.mgrid[0:patch_size, 0:patch_size, 0:patch_size].astype(np.float32)
    distances = np.sqrt(
        (grid[0] - center)**2 + 
        (grid[1] - center)**2 + 
        (grid[2] - center)**2
    )
    max_dist = np.sqrt(3 * (center**2))
    gaussian_weight = np.exp(-distances**2 / (2 * (max_dist/3)**2))
    
    # Accumulate weighted predictions
    for pred, (d_start, h_start, w_start) in zip(predictions, coordinates):
        d_end = min(d_start + patch_size, D)
        h_end = min(h_start + patch_size, H)
        w_end = min(w_start + patch_size, W)
        
        # Actual patch size (may be smaller at boundaries)
        actual_d = d_end - d_start
        actual_h = h_end - h_start
        actual_w = w_end - w_start
        
        # Crop prediction and weight if at boundary
        pred_crop = pred[:actual_d, :actual_h, :actual_w]
        weight_crop = gaussian_weight[:actual_d, :actual_h, :actual_w]
        
        prediction_sum[d_start:d_end, h_start:h_end, w_start:w_end] += pred_crop * weight_crop
        weight_sum[d_start:d_end, h_start:h_end, w_start:w_end] += weight_crop
    
    # Avoid division by zero
    weight_sum[weight_sum == 0] = 1.0
    
    # Weighted average
    merged = prediction_sum / weight_sum
    
    return merged


def segment_volume_with_sliding_window_points(
    model,
    volume_full,
    ts_mask_full,
    patch_size=128,
    min_content=0.01,
    num_positive_per_patch=10,
    num_negative_per_patch=5,
    erosion_iterations=0,
    device=None
):
    """
    Segment a large volume using sliding window + POINT PROMPTS.
    
    Key difference from mask prompt version:
    - Generates point prompts from TS mask for each patch
    - Uses points (what SAM-Med3D was trained on)
    - No dense mask prompts
    
    Args:
        model: SAM-Med3D model
        volume_full: (1, 1, D, H, W) full preprocessed volume
        ts_mask_full: (1, 1, D, H, W) full TS mask (used to generate points)
        patch_size: Size of patches (128 for SAM-Med3D)
        min_content: Minimum mask content to process patch
        num_positive_per_patch: Positive points per patch
        num_negative_per_patch: Negative points per patch
        erosion_iterations: Erosion of TS mask before point generation
        device: Torch device
    
    Returns:
        final_prediction: (D, H, W) binary segmentation
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    D, H, W = volume_full.shape[-3:]
    
    # Check if volume fits in single patch
    if D <= patch_size and H <= patch_size and W <= patch_size:
        print(f"   Volume ({D}×{H}×{W}) fits in single patch")
        # Just use the full volume with padding
        pad_d = max(0, patch_size - D)
        pad_h = max(0, patch_size - H)
        pad_w = max(0, patch_size - W)
        
        vol_padded = F.pad(volume_full, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)
        ts_padded = F.pad(ts_mask_full, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)
        
        # Apply erosion if requested
        if erosion_iterations > 0:
            ts_np = ts_padded[0, 0].cpu().numpy()
            ts_np = binary_erosion(ts_np, iterations=erosion_iterations)
            ts_padded = torch.from_numpy(ts_np).unsqueeze(0).unsqueeze(0)
        
        # Single inference with point prompts
        with torch.no_grad():
            vol_padded = vol_padded.to(device)
            
            # Generate point prompts from TS mask
            ts_np = ts_padded[0, 0].cpu().numpy()
            points_coords, points_labels = generate_points_from_patch_mask(
                ts_np, num_positive_per_patch, num_negative_per_patch
            )
            points_coords = points_coords.to(device)
            points_labels = points_labels.to(device)
            
            print(f"   Generated {(points_labels==1).sum().item()} pos + {(points_labels==0).sum().item()} neg points")
            
            # Encode image
            image_embeddings = model.image_encoder(vol_padded)
            
            # Initialize empty mask prompt
            prev_low_res_mask = torch.zeros(
                1, 1,
                patch_size // 4,
                patch_size // 4,
                patch_size // 4,
                device=device, dtype=torch.float
            )
            
            # Encode prompts (POINTS, not dense mask)
            sparse_emb, dense_emb = model.prompt_encoder(
                points=[points_coords, points_labels],
                boxes=None,
                masks=prev_low_res_mask
            )
            
            # Decode
            low_res_pred, _ = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb
            )
            
            # Upscale
            final_pred = F.interpolate(
                low_res_pred,
                size=(patch_size, patch_size, patch_size),
                mode='trilinear',
                align_corners=False
            )
            
            pred_prob = torch.sigmoid(final_pred).cpu().numpy()[0, 0]
        
        # Crop back to original size
        return (pred_prob[:D, :H, :W] > 0.5).astype(np.uint8)
    
    # Volume is large - use sliding window
    print(f"   Volume ({D}×{H}×{W}) > {patch_size}³ - using sliding window with point prompts")
    
    # Apply erosion to full mask if requested
    if erosion_iterations > 0:
        ts_np = ts_mask_full[0, 0].cpu().numpy()
        ts_np = binary_erosion(ts_np, iterations=erosion_iterations)
        ts_mask_full = torch.from_numpy(ts_np).unsqueeze(0).unsqueeze(0)
        print(f"   Applied {erosion_iterations} erosion iterations to TS mask")
    
    # Extract patches adaptively
    vol_patches, mask_patches, coordinates = adaptive_patch_extraction(
        volume_full, ts_mask_full, patch_size, min_content
    )
    
    print(f"   Processing {len(vol_patches)} patches with point prompts...")
    
    # Process each patch
    predictions = []
    with torch.no_grad():
        for i, (vol_patch, mask_patch) in enumerate(zip(vol_patches, mask_patches)):
            vol_patch = vol_patch.to(device)
            mask_np = mask_patch[0, 0].cpu().numpy()
            
            # Generate point prompts from this patch's TS mask
            points_coords, points_labels = generate_points_from_patch_mask(
                mask_np, 
                num_positive_per_patch, 
                num_negative_per_patch,
                seed=233 + i  # Different seed per patch
            )
            points_coords = points_coords.to(device)
            points_labels = points_labels.to(device)
            
            # Encode image
            image_embeddings = model.image_encoder(vol_patch)
            
            # Initialize empty mask prompt (SAM-Med3D still needs this input)
            prev_low_res_mask = torch.zeros(
                1, 1,
                patch_size // 4,
                patch_size // 4,
                patch_size // 4,
                device=device, dtype=torch.float
            )
            
            # Encode prompts (POINTS are the main guidance)
            sparse_emb, dense_emb = model.prompt_encoder(
                points=[points_coords, points_labels],
                boxes=None,
                masks=prev_low_res_mask
            )
            
            # Decode
            low_res_pred, _ = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb
            )
            
            # Upscale
            pred_hr = F.interpolate(
                low_res_pred,
                size=(patch_size, patch_size, patch_size),
                mode='trilinear',
                align_corners=False
            )
            
            pred_prob = torch.sigmoid(pred_hr).cpu().numpy()[0, 0]
            predictions.append(pred_prob)
            
            if (i + 1) % 10 == 0:
                print(f"      Processed {i + 1}/{len(vol_patches)} patches")
    
    # Merge predictions
    print(f"   Merging {len(predictions)} patch predictions...")
    merged = merge_predictions_3d(
        predictions, coordinates, (D, H, W), patch_size
    )
    
    return (merged > 0.5).astype(np.uint8)


# ============================================================================
# Helper functions (same as before)
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


def get_roi_from_subject(subject_canonical, meta_info, norm_transform):
    """Modified to NOT crop - we'll handle sizing with sliding window"""
    meta_info["canonical_subject_shape"] = subject_canonical.spatial_shape
    meta_info["canonical_subject_affine"] = subject_canonical.image.affine.copy()
    
    img3D = subject_canonical.image.data.clone().detach()
    img3D = norm_transform(img3D.squeeze(dim=1))
    img3D = img3D.unsqueeze(dim=1)
    
    label3D = subject_canonical.label.data.clone().detach()
    
    def correct_roi_dim(roi_tensor): 
        if roi_tensor.ndim == 3:
            roi_tensor = roi_tensor.unsqueeze(0).unsqueeze(0)
        if roi_tensor.ndim == 4:
            roi_tensor = roi_tensor.unsqueeze(0)
        if roi_tensor.shape[1] != 1:
            roi_tensor = roi_tensor[:, 0:1,...]
        return roi_tensor
    
    img3D = correct_roi_dim(img3D)
    label3D = correct_roi_dim(label3D)
    
    return img3D, label3D, meta_info


def get_subject_and_meta_info(img_path, label_path):
    _, meta_info = read_arr_from_nifti(img_path, get_meta_info=True)
    subject = tio.Subject(
        image=tio.ScalarImage(img_path),
        label=tio.LabelMap(label_path) 
    )
    return subject, meta_info


def data_preprocess_no_crop(subject, meta_info, category_index, target_spacing):
    """Modified preprocessing that doesn't crop to 128³"""
    label_data_for_cat = subject.label.data.clone()
    new_label_data = torch.zeros_like(label_data_for_cat)
    # Convert all non-zero labels to 1 (combines all intestinal tract labels)
    new_label_data[label_data_for_cat > 0] = 1
    subject.label.set_data(new_label_data)
    
    meta_info["original_subject_affine"] = subject.image.affine.copy()
    meta_info["original_subject_spatial_shape"] = subject.image.spatial_shape
    
    # Resample to target spacing
    resampler = tio.Resample(target=target_spacing)
    subject_resampled = resampler(subject)
    
    # Canonicalize
    transform_canonical = tio.ToCanonical()
    subject_canonical = transform_canonical(subject_resampled)
    
    # NO CROPPING - sliding window will handle size
    norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
    full_image, full_label, meta_info = get_roi_from_subject(
        subject_canonical, meta_info, norm_transform
    )
    
    return full_image, full_label, meta_info


def data_postprocess(roi_pred_numpy, meta_info):
    """Postprocess prediction back to original space"""
    roi_pred_tensor = torch.from_numpy(roi_pred_numpy.astype(np.float32)).unsqueeze(0)
    pred_label_map = tio.LabelMap(
        tensor=roi_pred_tensor,
        affine=meta_info["canonical_subject_affine"]
    )
    
    reference_tensor_shape = (1, *meta_info["original_subject_spatial_shape"]) 
    reference_image = tio.ScalarImage(
        tensor=torch.zeros(reference_tensor_shape),
        affine=meta_info["original_subject_affine"]
    )
    
    resampler = tio.Resample(
        target=reference_image,
        image_interpolation='nearest' 
    )
    
    pred_resampled = resampler(pred_label_map)
    final_pred_numpy = pred_resampled.data.squeeze(0).cpu().numpy()
    final_pred_numpy = final_pred_numpy.astype(np.uint8)
    
    return final_pred_numpy.transpose(2, 1, 0)


def save_numpy_to_nifti(in_arr, out_path, meta_info_for_saving):
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


def validate_with_sliding_window_points(
    model,
    img_path,
    gt_path,
    ts_path,
    output_path,
    patch_size=128,
    min_content=0.01,
    num_positive_per_patch=10,
    num_negative_per_patch=5,
    erosion_iterations=0,
    target_spacing=(1.5, 1.5, 1.5),
    seed=233,
    device=None
):
    """
    Complete validation pipeline with sliding window + POINT PROMPTS.
    
    Key difference: Uses TS mask to generate points, not as dense prompt.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    
    # Load GT for evaluation only
    gt_arr = read_arr_from_nifti(gt_path)
    _, gt_meta_for_saving = read_arr_from_nifti(gt_path, get_meta_info=True)
    gt_arr_binary = (gt_arr > 0).astype(np.uint8)
    
    # Preprocess with TS mask (NO CROPPING)
    subject, meta_info = get_subject_and_meta_info(img_path, ts_path)
    subject_copy = copy.deepcopy(subject)
    meta_info_copy = copy.deepcopy(meta_info)
    
    full_image, full_ts_mask, meta_info_processed = data_preprocess_no_crop(
        subject_copy,
        meta_info_copy,
        category_index=1,
        target_spacing=target_spacing
    )
    
    print(f"   Full volume shape: {full_image.shape}")
    print(f"   TS mask voxels: {(full_ts_mask > 0).sum().item()}")
    
    # Run sliding window inference with POINT PROMPTS
    roi_pred_numpy = segment_volume_with_sliding_window_points(
        model=model,
        volume_full=full_image,
        ts_mask_full=full_ts_mask,
        patch_size=patch_size,
        min_content=min_content,
        num_positive_per_patch=num_positive_per_patch,
        num_negative_per_patch=num_negative_per_patch,
        erosion_iterations=erosion_iterations,
        device=device
    )
    
    # Post-process
    final_pred_binary = data_postprocess(roi_pred_numpy, meta_info_processed)
    final_pred_binary = (final_pred_binary > 0).astype(np.uint8)
    
    # Save
    save_numpy_to_nifti(final_pred_binary, output_path, gt_meta_for_saving)
    
    # Evaluate against GT
    dice_score = compute_dice(final_pred_binary, gt_arr_binary)
    print(f"✅ Dice (sliding window + point prompts vs GT): {dice_score:.4f}\n")
    
    return dice_score