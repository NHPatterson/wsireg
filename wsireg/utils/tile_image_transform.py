import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import dask.array as da
import numpy as np
import SimpleITK as sitk
import zarr

from wsireg.utils.shape_utils import prepare_pt_transformation_data
from wsireg.utils.tform_utils import ELX_TO_ITK_INTERPOLATORS


def tile_pad_output_size(y_size, x_size, tile_size=512):
    y_size, x_size = (
        np.ceil(np.asarray((y_size, x_size)) / tile_size) * tile_size
    ).astype(int)
    return y_size, x_size


def generate_tile_coords(output_size_xy, output_spacing, tile_size=512):

    image = sitk.Image(
        np.asarray(output_size_xy).astype(np.int32).tolist(), sitk.sitkUInt8
    )
    image.SetSpacing(output_spacing)
    x_size, y_size = image.GetSize()
    tile_coordinate_data = []
    for y in range(0, y_size, tile_size):
        for x in range(0, x_size, tile_size):
            tile_origin = image.TransformIndexToPhysicalPoint([x, y])
            tile_data = {
                "tile_index": (x, y),
                "tile_origin": tile_origin,
                "output_size": (tile_size, tile_size),
                "output_spacing": output_spacing,
            }
            tile_coordinate_data.append(tile_data)

    return tile_coordinate_data


def add_tile_location_on_moving(
    wsireg_transforms, tile_coordinate_data, source_res=1
):

    pt_transform, target_res = prepare_pt_transformation_data(
        wsireg_transforms, compute_inverse=False
    )

    # ordering is to transform moving to fixed
    # reversing goes from fixed to moving
    pt_transform = pt_transform[::-1]

    for tile in tile_coordinate_data:
        x0 = tile["tile_index"][0]
        y0 = tile["tile_index"][1]
        x1 = x0 + tile["output_size"][0]
        y1 = y0 + tile["output_size"][1]
        tile_corners = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

        tformed_pts = []
        for pt in tile_corners:
            pt = pt * target_res
            for idx, t in enumerate(pt_transform):
                if idx == 0:
                    t_pt = t.itk_transform.TransformPoint(pt)
                else:
                    t_pt = t.itk_transform.TransformPoint(t_pt)
            t_pt = np.array(t_pt)
            t_pt /= source_res
            tformed_pts.append(t_pt)

        transformed_tile_pts = np.stack(tformed_pts)
        tile.update({"tile_on_moving": transformed_tile_pts})

    return tile_coordinate_data


def pad_transform_tile_corners(
    x_size, y_size, transformed_tile_pts, padding=128
):
    x_min = np.floor(np.min(transformed_tile_pts[:, 0]))
    y_min = np.floor(np.min(transformed_tile_pts[:, 1]))
    x_max = np.ceil(np.max(transformed_tile_pts[:, 0]))
    y_max = np.ceil(np.max(transformed_tile_pts[:, 1]))

    if (x_min - padding) < 0:
        x_min = 0
    else:
        x_min -= padding

    if (y_min - padding) < 0:
        y_min = 0
    else:
        y_min -= padding

    if (x_max + padding) > x_size:
        x_max = x_size
    else:
        if (x_max + padding) > 0:
            x_max += padding
        else:
            x_max = x_size

    if (y_max + padding) > y_size:
        y_max = y_size
    else:
        if (y_max + padding) > 0:
            y_max += padding
        else:
            y_max = y_size

    return int(x_min), int(x_max), int(y_min), int(y_max)


def wsireg_tile_transform_to_resampler(
    tile_data, final_transform_interpolator="FinalNearestNeighborInterpolator"
):
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputOrigin(tile_data["tile_origin"])
    resampler.SetSize(tile_data["output_size"])
    resampler.SetOutputSpacing(tile_data["output_spacing"])
    interpolator = ELX_TO_ITK_INTERPOLATORS.get(final_transform_interpolator)
    resampler.SetInterpolator(interpolator)
    return resampler


def determine_moving_tile_padding(itk_transforms):
    transform_linearity = [t.is_linear for t in itk_transforms]
    if all(transform_linearity):
        tile_padding = 256
    else:
        nl_idxs = np.where(np.array(transform_linearity) == 0)[0]
        for nl_idx in nl_idxs:
            output_spacing = itk_transforms[nl_idx].output_spacing[0]
            grid_spacing = itk_transforms[
                nl_idx
            ].itk_transform.GetFixedParameters()[4:6]
            tile_padding = (np.max(grid_spacing) / output_spacing) * 12
    return tile_padding


def itk_transform_tiles(
    tform_reg_im,
    y_size,
    x_size,
    tile_coordinate_data,
    itk_transform_composite,
    ch_idx=0,
    tile_size=512,
    tile_padding=128,
    use_multiprocessing=False,
):

    zstr = zarr.TempStore()
    zgrp = zarr.open(zstr)
    if tform_reg_im.is_rgb:
        resample_zarray = zgrp.create_dataset(
            'temparray',
            shape=(y_size, x_size, tform_reg_im.im_dims[-1]),
            chunks=(tile_size, tile_size),
            dtype=tform_reg_im.im_dtype,
        )
    else:
        resample_zarray = zgrp.create_dataset(
            'temparray',
            shape=(y_size, x_size),
            chunks=(tile_size, tile_size),
            dtype=tform_reg_im.im_dtype,
        )
    if tform_reg_im.is_rgb:
        y_size_moving = tform_reg_im.im_dims[0]
        x_size_moving = tform_reg_im.im_dims[1]
    else:
        y_size_moving = tform_reg_im.im_dims[1]
        x_size_moving = tform_reg_im.im_dims[2]

    def transform_tile(tile):
        tile_resampler = wsireg_tile_transform_to_resampler(tile)
        x_min, x_max, y_min, y_max = pad_transform_tile_corners(
            x_size_moving,
            y_size_moving,
            tile["tile_on_moving"],
            padding=tile_padding,
        )

        x_idx_start = tile["tile_index"][0]
        x_idx_end = x_idx_start + tile["output_size"][0]
        y_idx_start = tile["tile_index"][1]
        y_idx_end = y_idx_start + tile["output_size"][1]

        if x_min == 0 and x_max == 0:
            x_max = 1
        if y_min == 0 and y_max == 0:
            y_max = 1

        if tform_reg_im.is_rgb:
            image = tform_reg_im.image[y_min:y_max, x_min:x_max, :]
            image = sitk.GetImageFromArray(image, isVector=True)
            image.SetSpacing((tform_reg_im.image_res, tform_reg_im.image_res))
            image.SetOrigin(
                image.TransformIndexToPhysicalPoint([int(x_min), int(y_min)])
            )

        elif tform_reg_im.n_ch == 1:
            image = tform_reg_im.image[y_min:y_max, x_min:x_max]
            image = sitk.GetImageFromArray(image, isVector=False)
            image.SetSpacing((tform_reg_im.image_res, tform_reg_im.image_res))
            image.SetOrigin(
                image.TransformIndexToPhysicalPoint([int(x_min), int(y_min)])
            )
        else:
            image = tform_reg_im.image[ch_idx, y_min:y_max, x_min:x_max]
            image = sitk.GetImageFromArray(image, isVector=False)
            image.SetSpacing((tform_reg_im.image_res, tform_reg_im.image_res))
            image.SetOrigin(
                image.TransformIndexToPhysicalPoint([int(x_min), int(y_min)])
            )

        tile_resampler.SetTransform(itk_transform_composite)
        tile_resampled = tile_resampler.Execute(image)

        if tform_reg_im.is_rgb:
            resample_zarray[
                y_idx_start:y_idx_end, x_idx_start:x_idx_end, :
            ] = sitk.GetArrayFromImage(tile_resampled)
        else:
            resample_zarray[
                y_idx_start:y_idx_end, x_idx_start:x_idx_end
            ] = sitk.GetArrayFromImage(tile_resampled)

    if use_multiprocessing:
        max_workers = multiprocessing.cpu_count() - 1
        with ThreadPoolExecutor(max_workers) as executor:
            executor.map(transform_tile, tile_coordinate_data)
    else:
        for tile in tile_coordinate_data:
            transform_tile(tile)

    return resample_zarray


def zarr_to_tiles(z, tile_coordinate_data, is_rgb):
    for tile in tile_coordinate_data:
        x_idx_start = tile["tile_index"][0]
        x_idx_end = x_idx_start + tile["output_size"][0]
        y_idx_start = tile["tile_index"][1]
        y_idx_end = y_idx_start + tile["output_size"][1]
        if is_rgb:
            yield z[y_idx_start:y_idx_end, x_idx_start:x_idx_end, :]
        else:
            yield z[y_idx_start:y_idx_end, x_idx_start:x_idx_end]


def subres_zarr_to_tiles(z, tile_size, is_rgb):
    for y in range(0, z.shape[0], tile_size):
        for x in range(0, z.shape[1], tile_size):
            if is_rgb:
                yield z[y : y + tile_size, x : x + tile_size, :].compute()
            else:
                yield z[y : y + tile_size, x : x + tile_size].compute()


def compute_sub_res(
    tform_reg_im, resampled_zarray, ds_factor, tile_size, is_rgb
):
    if is_rgb:
        resampling_axis = {0: ds_factor, 1: ds_factor, 2: 1}
        tiling = (tile_size, tile_size, 3)
    else:
        resampling_axis = {0: ds_factor, 1: ds_factor}
        tiling = (tile_size, tile_size)

    resampled_zarray_subres = da.coarsen(
        np.mean,
        da.from_zarr(resampled_zarray),
        resampling_axis,
        trim_excess=True,
    )
    resampled_zarray_subres = resampled_zarray_subres.astype(
        tform_reg_im.im_dtype
    )
    resampled_zarray_subres = resampled_zarray_subres.rechunk(tiling)
    orig_shape = resampled_zarray_subres.shape

    y_size_pyr, x_size_pyr = tile_pad_output_size(
        resampled_zarray_subres.shape[0],
        resampled_zarray_subres.shape[1],
        tile_size=tile_size,
    )
    y_pad = y_size_pyr - resampled_zarray_subres.shape[0]
    x_pad = x_size_pyr - resampled_zarray_subres.shape[1]
    if is_rgb:
        padding = ((0, y_pad), (0, x_pad), (0, 0))
    else:
        padding = ((0, y_pad), (0, x_pad))

    resampled_zarray_subres = da.pad(resampled_zarray_subres, padding)
    return resampled_zarray_subres, orig_shape
