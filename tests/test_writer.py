from pathlib import Path
import os
import random
import string

import numpy as np
import pytest
from tifffile import imread
import dask.array as da

from wsireg.reg_images.loader import reg_image_loader
from wsireg.reg_images.merge_reg_image import MergeRegImage
from wsireg.reg_transforms.reg_transform_seq import RegTransformSeq
from wsireg.writers.merge_ome_tiff_writer import MergeOmeTiffWriter
from wsireg.writers.ome_tiff_writer import OmeTiffWriter
from wsireg.writers.tiled_ome_tiff_writer import OmeTiffTiledWriter

HERE = os.path.dirname(__file__)
TFORM_FP = os.path.join(HERE, "fixtures/complex_linear_reg_transform.json")


def gen_project_name_str():
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(10))


@pytest.mark.usefixtures("complex_transform")
def test_OmeTiffWriter_by_plane(complex_transform, tmp_path):
    reg_image = reg_image_loader(np.ones((1024, 1024), dtype=np.uint8), 1)
    # composite_transform, _, final_transform = prepare_wsireg_transform_data(
    #     complex_transform
    # )
    rts = RegTransformSeq(complex_transform)

    ometiffwriter = OmeTiffWriter(reg_image, reg_transform_seq=rts)
    by_plane_fp = ometiffwriter.write_image_by_plane(
        gen_project_name_str(),
        output_dir=str(tmp_path),
    )
    by_plane_image = reg_image_loader(by_plane_fp, 2)
    assert by_plane_image.shape == (1, 1024, 1024)


@pytest.mark.usefixtures("complex_transform")
def test_OmeTiffWriter_by_tile(complex_transform, tmp_path):
    reg_image = reg_image_loader(np.ones((4096, 4096), dtype=np.uint8), 0.5)
    rts = RegTransformSeq(complex_transform)

    ometiffwriter = OmeTiffTiledWriter(reg_image, reg_transform_seq=rts)

    by_tile_fp = ometiffwriter.write_image_by_tile(
        gen_project_name_str(),
        output_dir=str(tmp_path),
        zarr_temp_dir=tmp_path,
    )
    by_tile_image = reg_image_loader(by_tile_fp, 2)
    assert by_tile_image.shape == (1, 1024, 1024)


@pytest.mark.usefixtures("simple_transform_affine")
def test_OmeTiffWriter_compare_tile_plane(simple_transform_affine, tmp_path):
    reg_image = reg_image_loader(np.ones((1024, 1024), dtype=np.uint8), 1)
    rts = RegTransformSeq(simple_transform_affine)
    ometifftilewriter = OmeTiffTiledWriter(reg_image, reg_transform_seq=rts)
    ometiffwriter = OmeTiffWriter(reg_image, reg_transform_seq=rts)

    by_tile_fp = ometifftilewriter.write_image_by_tile(
        gen_project_name_str(),
        output_dir=str(tmp_path),
        zarr_temp_dir=tmp_path,
    )
    by_tile_image = reg_image_loader(by_tile_fp, 2)

    by_plane_fp = ometiffwriter.write_image_by_plane(
        gen_project_name_str(),
        output_dir=str(tmp_path),
    )
    by_plane_image = reg_image_loader(by_plane_fp, 2)

    ch0_im_tile = by_tile_image.read_single_channel(0)
    ch0_im_plane = by_plane_image.read_single_channel(0)

    assert by_plane_image.shape == by_tile_image.shape
    assert np.array_equal(ch0_im_tile, ch0_im_plane)


@pytest.mark.usefixtures("simple_transform_affine")
def test_OmeTiffWriter_compare_tile_plane_rgb(
    simple_transform_affine, tmp_path
):
    reg_image = reg_image_loader(
        np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8), 1
    )

    rts = RegTransformSeq(simple_transform_affine)
    ometiffwriter = OmeTiffWriter(reg_image, reg_transform_seq=rts)
    ometiletiffwriter = OmeTiffTiledWriter(reg_image, reg_transform_seq=rts)
    by_tile_fp = ometiletiffwriter.write_image_by_tile(
        gen_project_name_str(),
        output_dir=str(tmp_path),
        zarr_temp_dir=tmp_path,
    )

    by_plane_fp = ometiffwriter.write_image_by_plane(
        gen_project_name_str(),
        output_dir=str(tmp_path),
    )
    Path(by_tile_fp).as_posix()
    im_tile = imread(by_tile_fp)
    im_plane = imread(by_plane_fp)
    assert np.array_equal(im_tile, im_plane)


@pytest.mark.usefixtures("simple_transform_affine_nl")
def test_OmeTiffWriter_compare_tile_plane_rgb_nl(
    simple_transform_affine_nl, tmp_path
):
    reg_image = reg_image_loader(
        np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8), 1
    )

    rts = RegTransformSeq(simple_transform_affine_nl)
    ometiffwriter = OmeTiffWriter(reg_image, reg_transform_seq=rts)
    ometiletiffwriter = OmeTiffTiledWriter(reg_image, reg_transform_seq=rts)

    by_tile_fp = ometiletiffwriter.write_image_by_tile(
        gen_project_name_str(),
        output_dir=str(tmp_path),
    )

    by_plane_fp = ometiffwriter.write_image_by_plane(
        gen_project_name_str(),
        output_dir=str(tmp_path),
    )

    im_tile = imread(by_tile_fp)
    im_plane = imread(by_plane_fp)

    assert np.array_equal(im_tile, im_plane)


@pytest.mark.usefixtures("simple_transform_affine")
def test_OmeTiffWriter_compare_tile_plane_mc(
    simple_transform_affine, tmp_path
):
    reg_image = reg_image_loader(
        np.random.randint(0, 255, (3, 1024, 1024), dtype=np.uint8), 1
    )
    rts = RegTransformSeq(simple_transform_affine)
    ometiffwriter = OmeTiffWriter(reg_image, reg_transform_seq=rts)
    ometiletiffwriter = OmeTiffTiledWriter(reg_image, reg_transform_seq=rts)

    by_tile_fp = ometiletiffwriter.write_image_by_tile(
        gen_project_name_str(),
        output_dir=str(tmp_path),
    )

    by_plane_fp = ometiffwriter.write_image_by_plane(
        gen_project_name_str(),
        output_dir=str(tmp_path),
    )

    im_tile = imread(by_tile_fp)
    im_plane = imread(by_plane_fp)

    assert np.array_equal(im_tile, im_plane)


@pytest.mark.usefixtures("simple_transform_affine_nl")
def test_OmeTiffWriter_compare_tile_plane_mc_nl(
    simple_transform_affine_nl, tmp_path
):
    reg_image = reg_image_loader(
        np.random.randint(0, 255, (3, 1024, 1024), dtype=np.uint8), 1
    )
    rts = RegTransformSeq(simple_transform_affine_nl)
    ometiffwriter = OmeTiffWriter(reg_image, reg_transform_seq=rts)
    ometiletiffwriter = OmeTiffTiledWriter(reg_image, reg_transform_seq=rts)

    by_tile_fp = ometiletiffwriter.write_image_by_tile(
        gen_project_name_str(),
        output_dir=str(tmp_path),
    )

    by_plane_fp = ometiffwriter.write_image_by_plane(
        gen_project_name_str(),
        output_dir=str(tmp_path),
    )

    im_tile = imread(by_tile_fp)
    im_plane = imread(by_plane_fp)

    assert np.array_equal(im_tile, im_plane)


def test_OmeTiffWriter_compare_tile_plane_mc_nl_large(tmp_path):
    im_array = da.from_array(
        np.random.randint(0, 255, (2, 2**13, 2**13), dtype=np.uint8),
        chunks=(1, 1024, 1024),
    )
    reg_image = reg_image_loader(im_array, 0.5)

    rts = RegTransformSeq(TFORM_FP)
    ometiffwriter = OmeTiffWriter(reg_image, reg_transform_seq=rts)
    ometiletiffwriter = OmeTiffTiledWriter(reg_image, reg_transform_seq=rts)

    by_tile_fp = ometiletiffwriter.write_image_by_tile(
        gen_project_name_str(),
        output_dir=str(tmp_path),
    )

    by_plane_fp = ometiffwriter.write_image_by_plane(
        gen_project_name_str(),
        output_dir=str(tmp_path),
    )

    im_tile = imread(by_tile_fp)
    im_plane = imread(by_plane_fp)

    assert np.array_equal(im_tile, im_plane)


def test_OmeTiffWriter_compare_tile_plane_rgb_nl_large(tmp_path):
    im_array = da.from_array(
        np.random.randint(0, 255, (2**13, 2**13, 3), dtype=np.uint8),
        chunks=(1024, 1024, 3),
    )
    reg_image = reg_image_loader(im_array, 0.5)

    rts = RegTransformSeq(TFORM_FP)
    ometiffwriter = OmeTiffWriter(reg_image, reg_transform_seq=rts)
    ometiletiffwriter = OmeTiffTiledWriter(reg_image, reg_transform_seq=rts)

    by_tile_fp = ometiletiffwriter.write_image_by_tile(
        gen_project_name_str(),
        output_dir=str(tmp_path),
    )

    by_plane_fp = ometiffwriter.write_image_by_plane(
        gen_project_name_str(),
        output_dir=str(tmp_path),
    )

    im_tile = imread(by_tile_fp)
    im_plane = imread(by_plane_fp)

    assert np.array_equal(im_tile, im_plane)


@pytest.mark.usefixtures("simple_transform_affine_nl")
def test_MergeOmeTiffWriter_mc(simple_transform_affine_nl, tmp_path):
    reg_image1 = np.random.randint(0, 255, (3, 1024, 1024), dtype=np.uint8)
    reg_image2 = np.random.randint(0, 255, (3, 1024, 1024), dtype=np.uint8)
    reg_image3 = np.random.randint(0, 255, (3, 1024, 1024), dtype=np.uint8)

    mreg_image = MergeRegImage(
        [reg_image1, reg_image2, reg_image3],
        [1, 1, 1],
        channel_names=[["1", "2", "3"], ["1", "2", "3"], ["1", "2", "3"]],
    )
    rts = RegTransformSeq(simple_transform_affine_nl)
    merge_ometiffwriter = MergeOmeTiffWriter(
        mreg_image, reg_transform_seqs=[rts, rts, rts]
    )

    by_plane_fp = merge_ometiffwriter.merge_write_image_by_plane(
        "merge_testimage_by_plane",
        ["1", "2", "3"],
        output_dir=str(tmp_path),
    )

    im_plane = imread(by_plane_fp)

    reg_image1_loaded = reg_image_loader(reg_image1, 1)

    ometiffwriter = OmeTiffWriter(reg_image1_loaded, reg_transform_seq=rts)

    by_plane_fp_s1 = ometiffwriter.write_image_by_plane(
        "testimage_by_plane_s1",
        output_dir=str(tmp_path),
    )

    im_plane_s1 = imread(by_plane_fp_s1)

    reg_image2_loaded = reg_image_loader(reg_image2, 1)

    ometiffwriter = OmeTiffWriter(reg_image2_loaded, reg_transform_seq=rts)

    by_plane_fp_s2 = ometiffwriter.write_image_by_plane(
        "testimage_by_plane_s2",
        output_dir=str(tmp_path),
    )

    im_plane_s2 = imread(by_plane_fp_s2)

    reg_image3_loaded = reg_image_loader(reg_image3, 1)

    ometiffwriter = OmeTiffWriter(reg_image3_loaded, reg_transform_seq=rts)

    by_plane_fp_s3 = ometiffwriter.write_image_by_plane(
        "testimage_by_plane_s3",
        output_dir=str(tmp_path),
    )

    im_plane_s3 = imread(by_plane_fp_s3)

    assert im_plane.shape[0] == 9
    assert np.array_equal(im_plane[0:3, :, :], im_plane_s1)
    assert np.array_equal(im_plane[3:6, :, :], im_plane_s2)
    assert np.array_equal(im_plane[6:9, :, :], im_plane_s3)


@pytest.mark.usefixtures("simple_transform_affine_nl")
def test_MergeOmeTiffWriter_mix_merge(simple_transform_affine_nl, tmp_path):

    reg_image1 = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint16)
    reg_image2 = np.random.randint(0, 255, (3, 1024, 1024), dtype=np.uint16)
    reg_image3 = np.random.randint(0, 255, (3, 1024, 1024), dtype=np.uint8)

    mreg_image = MergeRegImage(
        [reg_image1, reg_image2, reg_image3],
        [1, 1, 1],
        channel_names=[["1", "2", "3"], ["1", "2", "3"], ["1", "2", "3"]],
    )

    rts = RegTransformSeq(simple_transform_affine_nl)
    merge_ometiffwriter = MergeOmeTiffWriter(
        mreg_image, reg_transform_seqs=[rts, rts, rts]
    )

    by_plane_fp = merge_ometiffwriter.merge_write_image_by_plane(
        "merge_testimage_by_plane_mix",
        ["1", "2", "3"],
        output_dir=str(tmp_path),
    )

    im_plane = imread(by_plane_fp)

    assert im_plane.shape[0] == 9
