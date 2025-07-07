# -*- coding: utf-8 -*-
""" tf1/contrib/image.py """

from typing import Optional

import tensorflow as tf

# from tensorflow_addons.image import utils as img_utils
from tf1.tf_addons.image import utils as img_utils
# from tensorflow_addons.utils.types import TensorLike
from tf1.tf_addons.utils.types import TensorLike


__all__ = [
    'transform',
]


_IMAGE_DTYPES = {
    tf.dtypes.uint8,
    tf.dtypes.int32,
    tf.dtypes.int64,
    tf.dtypes.float16,
    tf.dtypes.float32,
    tf.dtypes.float64,
}


def transform(
    images: TensorLike,
    transforms: TensorLike,
    interpolation: str = "nearest",
    fill_mode: str = "constant",
    output_shape: Optional[list] = None,
    name: Optional[str] = None,
    fill_value: TensorLike = 0.0,
) -> tf.Tensor:
    """Applies the given transform(s) to the image(s).

    Args:
      images: A tensor of shape (num_images, num_rows, num_columns,
        num_channels) (NHWC), (num_rows, num_columns, num_channels) (HWC), or
        (num_rows, num_columns) (HW).
      transforms: Projective transform matrix/matrices. A vector of length 8 or
        tensor of size N x 8. If one row of transforms is
        [a0, a1, a2, b0, b1, b2, c0, c1], then it maps the *output* point
        `(x, y)` to a transformed *input* point
        `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
        where `k = c0 x + c1 y + 1`. The transforms are *inverted* compared to
        the transform mapping input points to output points. Note that
        gradients are not backpropagated into transformation parameters.
      interpolation: Interpolation mode.
        Supported values: "nearest", "bilinear".
      fill_mode: Points outside the boundaries of the input are filled according
        to the given mode (one of `{'constant', 'reflect', 'wrap', 'nearest'}`).
        - *reflect*: `(d c b a | a b c d | d c b a)`
          The input is extended by reflecting about the edge of the last pixel.
        - *constant*: `(k k k k | a b c d | k k k k)`
          The input is extended by filling all values beyond the edge with the
          same constant value k = 0.
        - *wrap*: `(a b c d | a b c d | a b c d)`
          The input is extended by wrapping around to the opposite edge.
        - *nearest*: `(a a a a | a b c d | d d d d)`
          The input is extended by the nearest pixel.
      fill_value: a float represents the value to be filled outside the
        boundaries when `fill_mode` is "constant".
      output_shape: Output dimesion after the transform, [height, width].
        If None, output is the same size as input image.

      name: The name of the op.

    Returns:
      Image(s) with the same type and shape as `images`, with the given
      transform(s) applied. Transformed coordinates outside of the input image
      will be filled with zeros.

    Raises:
      TypeError: If `image` is an invalid type.
      ValueError: If output shape is not 1-D int32 Tensor.

    Source: https://github.com/tensorflow/addons/blob/master/tensorflow_addons/image/transform_ops.py
    """
    with tf.name_scope(name or "transform"):
        image_or_images = tf.convert_to_tensor(images, name="images")
        transform_or_transforms = tf.convert_to_tensor(
            transforms, name="transforms", dtype=tf.dtypes.float32
        )
        if image_or_images.dtype.base_dtype not in _IMAGE_DTYPES:
            raise TypeError("Invalid dtype %s." % image_or_images.dtype)
        images = img_utils.to_4D_image(image_or_images)
        original_ndims = img_utils.get_ndims(image_or_images)

        if output_shape is None:
            output_shape = tf.shape(images)[1:3]

        output_shape = tf.convert_to_tensor(
            output_shape, tf.dtypes.int32, name="output_shape"
        )

        if not output_shape.get_shape().is_compatible_with([2]):
            raise ValueError(
                "output_shape must be a 1-D Tensor of 2 elements: "
                "new_height, new_width"
            )

        if len(transform_or_transforms.get_shape()) == 1:
            transforms = transform_or_transforms[None]
        elif transform_or_transforms.get_shape().ndims is None:
            raise ValueError("transforms rank must be statically known")
        elif len(transform_or_transforms.get_shape()) == 2:
            transforms = transform_or_transforms
        else:
            transforms = transform_or_transforms
            raise ValueError(
                "transforms should have rank 1 or 2, but got rank %d"
                % len(transforms.get_shape())
            )

        fill_value = tf.convert_to_tensor(
            fill_value, dtype=tf.float32, name="fill_value"
        )
        output = tf.raw_ops.ImageProjectiveTransformV3(
            images=images,
            transforms=transforms,
            output_shape=output_shape,
            interpolation=interpolation.upper(),
            fill_mode=fill_mode.upper(),
            fill_value=fill_value,
        )
        return img_utils.from_4D_image(output, original_ndims)
