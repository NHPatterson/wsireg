import itk
import SimpleITK as sitk


def itk_image_to_sitk_image(image):
    origin = tuple(image.GetOrigin())
    spacing = tuple(image.GetSpacing())
    direction = itk.GetArrayFromMatrix(image.GetDirection()).flatten()
    image = sitk.GetImageFromArray(
        itk.GetArrayFromImage(image),
        isVector=image.GetNumberOfComponentsPerPixel() > 1,
    )
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    return image


def sitk_image_to_itk_image(image, cast_to_float32=False):
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    # direction = image.GetDirection()
    is_vector = image.GetNumberOfComponentsPerPixel() > 1
    if cast_to_float32 is True:
        image = sitk.Cast(image, sitk.sitkFloat32)
        image = sitk.GetArrayFromImage(image)
    else:
        image = sitk.GetArrayFromImage(image)

    image = itk.GetImageFromArray(image, is_vector=is_vector)
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    # image.SetDirection(direction)
    return image
