"""Create the background mask of the texture atlas image of a mesh.

Given an input texture image

  path/to/texture.png

the following outputs are generated:

  path/to/texture-background_mask.png:
    Binary mask with 1 for foreground chart, 0 for background.
  path/to/texture-background_mask-mesh.png:
    Mesh wireframe overlaid over the texture image.
  path/to/texture-background_mask-fill.png:
    Foreground mask overlaid over the texture image.
"""

import argparse
import logging
import pathlib
import copy
import numpy as np
from PIL import Image, ImageDraw


import data


logger = logging.getLogger(__name__)


def make_background_mask(texture, texture_vertices, texture_faces):
    """Create the background mask of a texture atlas image.

    Args:
        texture: (h, w, 3) Texture image with RGB values in [0, 1].
        texture_vertices: (n, 2) Array of UV coordinates of the vertices of the
            mesh in the mapped 2D plane.
        texture_faces: (f, 3) Array of faces as triplets of vertex indices.

    Returns:
        mask: (h, w) Boolean mask of the of the texture atlas. False if
            background, True if foreground.
        overlay: (h, w, 3), floats in [0, 1]. Foreground of the texture atlas
            overlaid in translucent white over the texture image.
    """
    image_size = texture.shape[:2]

    vertices = texture_vertices[:]
    vertices[..., 1] = 1.0 - vertices[..., 1]
    vertices *= image_size
    triangle_vertices = vertices[texture_faces]

    mode = "1"  # for a 1-bit black and white image
    mask_image = Image.new(mode, image_size)
    mask_draw = ImageDraw.Draw(mask_image)

    # PIL's ImageDraw.polygon() expects [(x, y), (x, y)...] or [x, y, x, y...]
    # as input points. For the former, the inner container must be a tuple. For
    # both, the outer container cannot be a np.ndarray. So convert the data to
    # the latter format in batch for overall faster drawing.
    n_triangles = len(triangle_vertices)
    flattened_triangle_vertices = triangle_vertices.reshape(n_triangles, -1)
    for coords in flattened_triangle_vertices.tolist():
        mask_draw.polygon(coords, fill=1)
    mask = np.asarray(mask_image)

    # Highlight the foreground regions with a white overlay (blending white
    # with the original colour).
    overlay = copy.deepcopy(texture)
    overlay[mask] = 0.5 * (overlay[mask] + 1.0)

    return mask, overlay


def make_partial_mask(texture, background):
    """Create the mask of the missing foreground regions of the texture atlas.

    Args:
        texture: (h, w, 3) Texture image with RGB values in [0, 1].
        background: (h, w) Boolean mask of the background of the texture atlas.
            I.e. foreground is True, background id False.

    Returns:
        mask: (h, w) Boolean mask of the missing regions of the foreground of
            the texture atlas. False if missing, True otherwise.
        overlay: (h, w, 3) Missing regions overlaid over the texture image.
    """
    mask = np.ones(texture.shape[:2], dtype=bool)
    overlay = copy.deepcopy(texture)

    threshold_notcolored = 0.01
    notcolored = np.sqrt((texture ** 2).sum(axis=-1)) <= threshold_notcolored
    missing = notcolored & background

    mask[missing] = False
    overlay[missing] = 1

    return mask, overlay


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", help="mesh file")
    parser.add_argument("-d", "--debug", default=False, action="store_true",
                        help="output debug messages and images")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    logger.info(f"loading {args.mesh}")
    mesh = data.load_mesh(args.mesh)

    logger.info("generating background mask")
    background_mask, background_overlay = make_background_mask(
        mesh.texture,
        mesh.texcoords,
        mesh.texture_indices,
    )

    background_mask_u8 = (255 * background_mask).astype("uint8")
    background_overlay_u8 = (255 * background_overlay).astype("uint8")

    in_path = pathlib.Path(args.mesh)
    out_dir = in_path.parent
    stem = in_path.stem

    name_mask = f"{stem}-background_mask.png"
    name_overlay = f"{stem}-background_mask-overlay.png"
    Image.fromarray(background_mask_u8).save(out_dir / name_mask)
    Image.fromarray(background_overlay_u8).save(out_dir / name_overlay)

    if mesh.texcoords.max() > 1:
        logger.info("rescaling texcoords to [0, 1]")
        size = mesh.texture.shape[:2]
        mesh.texcoords = mesh.texcoords / size

    logger.info("generating partial mask")
    partial_mask, partial_overlay = make_partial_mask(mesh.texture,
                                                      background_mask)

    partial_mask_u8 = (255 * partial_mask).astype("uint8")
    partial_overlay_u8 = (255 * partial_overlay).astype("uint8")

    name_partial_mask = f"{stem}-partial_mask.png"
    name_partial_overlay = f"{stem}-partial_mask-overlay.png"
    Image.fromarray(partial_mask_u8).save(out_dir / name_partial_mask)
    Image.fromarray(partial_overlay_u8).save(out_dir / name_partial_overlay)


if __name__ == "__main__":
    main()
