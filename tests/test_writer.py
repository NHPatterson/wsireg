from pathlib import Path

import numpy as np
import pytest
from tifffile import imread

from wsireg.reg_images.loader import reg_image_loader
from wsireg.reg_images.merge_reg_image import MergeRegImage
from wsireg.reg_transform_seq import RegTransformSeq
from wsireg.writers.merge_ome_tiff_writer import MergeOmeTiffWriter
from wsireg.writers.ome_tiff_writer import OmeTiffWriter


@pytest.fixture(scope="session")
def data_out_dir(tmpdir_factory):
    out_dir = tmpdir_factory.mktemp("output")
    return out_dir


@pytest.mark.usefixtures("complex_transform")
def test_OmeTiffWriter_by_plane(complex_transform, data_out_dir):
    reg_image = reg_image_loader(np.ones((1024, 1024), dtype=np.uint8), 1)
    # composite_transform, _, final_transform = prepare_wsireg_transform_data(
    #     complex_transform
    # )
    rts = RegTransformSeq(complex_transform)

    ometiffwriter = OmeTiffWriter(reg_image, reg_transform_seq=rts)
    by_plane_fp = ometiffwriter.write_image_by_plane(
        "testimage_by_plane",
        output_dir=str(data_out_dir),
    )
    by_plane_image = reg_image_loader(by_plane_fp, 2)
    assert by_plane_image.im_dims == (1, 1024, 1024)


@pytest.mark.usefixtures("complex_transform")
def test_OmeTiffWriter_by_tile(complex_transform, data_out_dir):
    reg_image = reg_image_loader(np.ones((1024, 1024), dtype=np.uint8), 1)
    rts = RegTransformSeq(complex_transform)

    ometiffwriter = OmeTiffWriter(reg_image, reg_transform_seq=rts)
    by_tile_fp = ometiffwriter.write_image_by_tile(
        "testimage_by_plane",
        output_dir=str(data_out_dir),
    )
    by_tile_image = reg_image_loader(by_tile_fp, 2)
    assert by_tile_image.im_dims == (1, 1024, 1024)


@pytest.mark.usefixtures("complex_transform")
def test_OmeTiffWriter_compare_tile_plane(complex_transform, data_out_dir):
    reg_image = reg_image_loader(np.ones((1024, 1024), dtype=np.uint8), 1)
    rts = RegTransformSeq(complex_transform)
    ometiffwriter = OmeTiffWriter(reg_image, reg_transform_seq=rts)
    by_tile_fp = ometiffwriter.write_image_by_tile(
        "testimage_by_tile",
        output_dir=str(data_out_dir),
    )
    by_tile_image = reg_image_loader(by_tile_fp, 2)
    by_plane_fp = ometiffwriter.write_image_by_plane(
        "testimage_by_plane",
        output_dir=str(data_out_dir),
    )
    by_plane_image = reg_image_loader(by_plane_fp, 2)

    ch0_im_tile = by_tile_image.read_single_channel(0)
    ch0_im_plane = by_plane_image.read_single_channel(0)

    assert by_plane_image.im_dims == by_tile_image.im_dims
    assert np.array_equal(ch0_im_tile, ch0_im_plane)


# tests if tile padding works appropriately
@pytest.mark.usefixtures("complex_transform_larger_padded")
def test_OmeTiffWriter_by_tile_nondiv(
    complex_transform_larger_padded, data_out_dir
):
    reg_image = reg_image_loader(np.ones((1024, 1024), dtype=np.uint8), 1)

    rts = RegTransformSeq(complex_transform_larger_padded)

    ometiffwriter = OmeTiffWriter(reg_image, reg_transform_seq=rts)

    by_tile_fp = ometiffwriter.write_image_by_tile(
        "testimage_by_tile_nd",
        output_dir=str(data_out_dir),
    )
    by_tile_image = reg_image_loader(by_tile_fp, 2)

    assert by_tile_image.im_dims == (
        1,
        int(np.ceil(2099 / 512) * 512),
        int(np.ceil(3099 / 512) * 512),
    )


@pytest.mark.usefixtures("complex_transform_larger")
def test_OmeTiffWriter_compare_tile_plane_nondiv(
    complex_transform_larger, data_out_dir
):
    reg_image = reg_image_loader(np.ones((1024, 1024), dtype=np.uint8), 1)

    rts = RegTransformSeq(complex_transform_larger)
    ometiffwriter = OmeTiffWriter(reg_image, reg_transform_seq=rts)

    by_tile_fp = ometiffwriter.write_image_by_tile(
        "testimage_by_tile",
        output_dir=str(data_out_dir),
    )
    by_tile_image = reg_image_loader(by_tile_fp, 2)
    by_plane_fp = ometiffwriter.write_image_by_plane(
        "testimage_by_plane",
        output_dir=str(data_out_dir),
    )
    by_plane_image = reg_image_loader(by_plane_fp, 2)

    ch0_im_tile = by_tile_image.read_single_channel(0)
    ch0_im_plane = by_plane_image.read_single_channel(0)
    assert np.array_equal(
        ch0_im_tile[0:3098, 0:2098], ch0_im_plane[0:3098, 0:2098]
    )


@pytest.mark.usefixtures("simple_transform_affine")
def test_OmeTiffWriter_compare_tile_plane_rgb(
    simple_transform_affine, data_out_dir
):
    reg_image = reg_image_loader(
        np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8), 1
    )

    rts = RegTransformSeq(simple_transform_affine)
    ometiffwriter = OmeTiffWriter(reg_image, reg_transform_seq=rts)

    by_tile_fp = ometiffwriter.write_image_by_tile(
        "testimage_by_tile",
        output_dir=str(data_out_dir),
    )

    by_plane_fp = ometiffwriter.write_image_by_plane(
        "testimage_by_plane",
        output_dir=str(data_out_dir),
    )
    Path(by_tile_fp).as_posix()
    im_tile = imread(by_tile_fp)
    im_plane = imread(by_plane_fp)

    assert np.array_equal(im_tile, im_plane)


@pytest.mark.usefixtures("simple_transform_affine_nl")
def test_OmeTiffWriter_compare_tile_plane_rgb_nl(
    simple_transform_affine_nl, data_out_dir
):
    reg_image = reg_image_loader(
        np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8), 1
    )

    rts = RegTransformSeq(simple_transform_affine_nl)
    ometiffwriter = OmeTiffWriter(reg_image, reg_transform_seq=rts)

    by_tile_fp = ometiffwriter.write_image_by_tile(
        "testimage_by_tile",
        output_dir=str(data_out_dir),
    )

    by_plane_fp = ometiffwriter.write_image_by_plane(
        "testimage_by_plane",
        output_dir=str(data_out_dir),
    )

    im_tile = imread(by_tile_fp)
    im_plane = imread(by_plane_fp)

    assert np.array_equal(im_tile[0:2099, 0:3099, :], im_plane)


@pytest.mark.usefixtures("simple_transform_affine")
def test_OmeTiffWriter_compare_tile_plane_mc(
    simple_transform_affine, data_out_dir
):
    reg_image = reg_image_loader(
        np.random.randint(0, 255, (3, 1024, 1024), dtype=np.uint8), 1
    )
    rts = RegTransformSeq(simple_transform_affine)
    ometiffwriter = OmeTiffWriter(reg_image, reg_transform_seq=rts)

    by_tile_fp = ometiffwriter.write_image_by_tile(
        "testimage_by_tile",
        output_dir=str(data_out_dir),
    )

    by_plane_fp = ometiffwriter.write_image_by_plane(
        "testimage_by_plane",
        output_dir=str(data_out_dir),
    )

    im_tile = imread(by_tile_fp)
    im_plane = imread(by_plane_fp)

    assert np.array_equal(im_tile, im_plane)


@pytest.mark.usefixtures("simple_transform_affine_nl")
def test_OmeTiffWriter_compare_tile_plane_mc_nl(
    simple_transform_affine_nl, data_out_dir
):
    reg_image = reg_image_loader(
        np.random.randint(0, 255, (3, 1024, 1024), dtype=np.uint8), 1
    )
    rts = RegTransformSeq(simple_transform_affine_nl)
    ometiffwriter = OmeTiffWriter(reg_image, reg_transform_seq=rts)

    by_tile_fp = ometiffwriter.write_image_by_tile(
        "testimage_by_tile",
        output_dir=str(data_out_dir),
    )

    by_plane_fp = ometiffwriter.write_image_by_plane(
        "testimage_by_plane",
        output_dir=str(data_out_dir),
    )

    im_tile = imread(by_tile_fp)
    im_plane = imread(by_plane_fp)

    assert np.array_equal(im_tile[:, 0:2099, 0:3099], im_plane)


#
# @pytest.mark.usefixtures("simple_transform_affine_nl_large_output")
# def test_OmeTiffWriter_tile_large_rgb_from_disk(
#     simple_transform_affine_nl_large_output, data_out_dir
# ):
#
#     large_rgb_fp = Path(str(data_out_dir)) / "largergb.tif"
#     imwrite(
#         large_rgb_fp,
#         np.random.randint(0, 255, (16384, 16384, 3), dtype=np.uint8),
#         compression=("jpeg", 5),
#         tile=(512, 512),
#     )
#     reg_image = reg_image_loader(str(large_rgb_fp), 0.8)
#
#     (
#         composite_transform,
#         itk_transforms,
#         final_transform,
#     ) = prepare_wsireg_transform_data(simple_transform_affine_nl_large_output)
#
#     ometiffwriter = OmeTiffWriter(reg_image)
#
#     by_tile_fp = ometiffwriter.write_image_by_tile(
#         "testimage_by_tile",
#         final_transform=final_transform,
#         itk_transforms=itk_transforms,
#         composite_transform=composite_transform,
#         output_dir=str(data_out_dir),
#     )
#
#     by_plane_fp = ometiffwriter.write_image_by_plane(
#         "testimage_by_plane",
#         final_transform=final_transform,
#         composite_transform=composite_transform,
#         output_dir=str(data_out_dir),
#     )
#
#     im_tile = imread(by_tile_fp)
#     im_plane = imread(by_plane_fp)
#
#     assert np.array_equal(
#         im_tile[0:18011, 0:14020, :], im_plane[0:18011, 0:14020, :]
#     )
#
#
# @pytest.mark.usefixtures("simple_transform_affine_nl_large_output")
# def test_OmeTiffWriter_tile_large_rgb_from_disk_plane(
#     simple_transform_affine_nl_large_output, data_out_dir
# ):
#     large_rgb_fp = Path(str(data_out_dir)) / "largergb.tif"
#     imwrite(
#         large_rgb_fp,
#         np.random.randint(0, 255, (3, 16384, 16384), dtype=np.uint8),
#         compression=("jpeg", 5),
#         tile=(512, 512),
#     )
#     reg_image = reg_image_loader(str(large_rgb_fp), 0.8)
#
#     (
#         composite_transform,
#         itk_transforms,
#         final_transform,
#     ) = prepare_wsireg_transform_data(simple_transform_affine_nl_large_output)
#
#     ometiffwriter = OmeTiffWriter(reg_image)
#
#     by_plane_fp = ometiffwriter.write_image_by_plane(
#         "testimage_by_plane",
#         final_transform=final_transform,
#         composite_transform=composite_transform,
#         output_dir=str(data_out_dir),
#     )


@pytest.mark.usefixtures("simple_transform_affine_nl")
def test_MergeOmeTiffWriter_mc(simple_transform_affine_nl, data_out_dir):
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
        mreg_image, reg_seq_transforms=[rts, rts, rts]
    )

    by_plane_fp = merge_ometiffwriter.merge_write_image_by_plane(
        "merge_testimage_by_plane",
        ["1", "2", "3"],
        output_dir=str(data_out_dir),
    )

    im_plane = imread(by_plane_fp)

    reg_image1_loaded = reg_image_loader(reg_image1, 1)

    ometiffwriter = OmeTiffWriter(reg_image1_loaded, reg_transform_seq=rts)

    by_plane_fp_s1 = ometiffwriter.write_image_by_plane(
        "testimage_by_plane_s1",
        output_dir=str(data_out_dir),
    )

    im_plane_s1 = imread(by_plane_fp_s1)

    reg_image2_loaded = reg_image_loader(reg_image2, 1)

    ometiffwriter = OmeTiffWriter(reg_image2_loaded, reg_transform_seq=rts)

    by_plane_fp_s2 = ometiffwriter.write_image_by_plane(
        "testimage_by_plane_s2",
        output_dir=str(data_out_dir),
    )

    im_plane_s2 = imread(by_plane_fp_s2)

    reg_image3_loaded = reg_image_loader(reg_image3, 1)

    ometiffwriter = OmeTiffWriter(reg_image3_loaded, reg_transform_seq=rts)

    by_plane_fp_s3 = ometiffwriter.write_image_by_plane(
        "testimage_by_plane_s3",
        output_dir=str(data_out_dir),
    )

    im_plane_s3 = imread(by_plane_fp_s3)

    assert im_plane.shape[0] == 9
    assert np.array_equal(im_plane[0:3, :, :], im_plane_s1)
    assert np.array_equal(im_plane[3:6, :, :], im_plane_s2)
    assert np.array_equal(im_plane[6:9, :, :], im_plane_s3)


@pytest.mark.usefixtures("simple_transform_affine_nl")
def test_MergeOmeTiffWriter_mix_merge(
    simple_transform_affine_nl, data_out_dir
):

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
        mreg_image, reg_seq_transforms=[rts, rts, rts]
    )

    by_plane_fp = merge_ometiffwriter.merge_write_image_by_plane(
        "merge_testimage_by_plane_mix",
        ["1", "2", "3"],
        output_dir=str(data_out_dir),
    )

    im_plane = imread(by_plane_fp)

    assert im_plane.shape[0] == 9
