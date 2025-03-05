import time
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np


def concat_img_h(img_ls: list[np.ndarray]) -> np.ndarray:
    """Concatenate images horizontally.

    Args:
        img_ls (list[np.ndarray]): List of images to concatenate.

    Returns:
        np.ndarray: Concatenated image.
    """
    # Get the maximum height
    max_h = max(img.shape[0] for img in img_ls)

    # Resize images to the maximum height
    img_ls = [
        cv2.resize(img, (int(img.shape[1] * max_h / img.shape[0]), max_h))
        for img in img_ls
    ]

    # Concatenate images
    return cv2.hconcat(img_ls)


def concat_img_v(img_ls: list[np.ndarray]) -> np.ndarray:
    """Concatenate images vertically.

    Args:
        img_ls (list[np.ndarray]): List of images to concatenate.

    Returns:
        np.ndarray: Concatenated image.
    """
    # Get the maximum width
    max_w = max(img.shape[1] for img in img_ls)

    # Resize images to the maximum width
    img_ls = [
        cv2.resize(img, (max_w, int(img.shape[0] * max_w / img.shape[1])))
        for img in img_ls
    ]

    # Concatenate images
    return cv2.vconcat(img_ls)


def plot_2d_traj(
    img: np.ndarray, trajs: np.ndarray, radius: int = 3, total_len: Optional[int] = None
) -> np.ndarray:
    """Plot 2D trajectories on the image as colored dots.

    This implementation minimizes Python loops by vectorizing operations.
    Args:
        img (np.ndarray): (H, W, 3) image in BGR format.
        trajs (np.ndarray): (n_trajs, n_steps, 2) trajectory array (col, row).
        radius (int): Radius of the plotted dot. If 0, points are single pixels.
        total_len (Optional[int]): Total length of the trajectory. If provided, the
            color of the trajectory is based on the relative position in the trajectory.
            If None, the total len = trajs.shape[1].

    Returns:
        np.ndarray: Modified image with trajectories drawn.
    """
    assert trajs.ndim == 3 and trajs.shape[2] == 2
    n_trajs, n_steps, _ = trajs.shape
    total_len = n_steps if total_len is None else total_len
    # Flatten trajectories to (N, 2) where N = n_trajs * n_steps
    points = trajs.reshape(-1, 2)  # (N, 2)

    # Compute colormap for all points
    cmap = plt.get_cmap("plasma")
    if n_steps > 1:
        rel_positions = np.linspace(0, 1, total_len)[:n_steps]
    else:
        rel_positions = np.array([0.5])  # Single step, pick a middle color

    # Repeat relative positions for each trajectory
    rel_positions = np.tile(rel_positions, n_trajs)
    rgba_colors = cmap(rel_positions)  # (N, 4)
    # Convert RGBA to BGR [0,255]
    bgr_colors = (rgba_colors[:, [2, 1, 0]] * 255).astype(np.uint8)  # (N, 3)

    # If radius == 0, we can directly write pixels without loops
    if radius == 0:
        rows = points[:, 1].astype(int)
        cols = points[:, 0].astype(int)

        valid_mask = (
            (rows >= 0) & (rows < img.shape[0]) & (cols >= 0) & (cols < img.shape[1])
        )
        img[rows[valid_mask], cols[valid_mask]] = bgr_colors[valid_mask]
        return img

    # For radius > 0, we need to draw a filled circle at each point.
    # We'll vectorize this as much as possible:
    # 1. Create a mask of the circle and get the coordinates of all "inside" pixels.
    y_indices, x_indices = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    circle_mask = (x_indices**2 + y_indices**2) <= radius**2
    # Extract relative offsets from the circle_mask
    circle_coords = np.column_stack(np.where(circle_mask))  # (P, 2)
    # Adjust to have offsets around (0,0)
    circle_coords[:, 0] -= radius
    circle_coords[:, 1] -= radius

    # circle_coords[:,0] = dy, circle_coords[:,1] = dx
    # We have N points and P offsets per point.
    P = circle_coords.shape[0]

    # Compute all row and column indices for all points at once
    # rows_for_all_points: shape (N, P)
    # cols_for_all_points: shape (N, P)
    rows_for_all_points = points[:, 1, None] + circle_coords[:, 0]
    cols_for_all_points = points[:, 0, None] + circle_coords[:, 1]

    rows_for_all_points = rows_for_all_points.astype(int)
    cols_for_all_points = cols_for_all_points.astype(int)

    # Flatten these arrays to do a single large assignment
    rows_all_flat = rows_for_all_points.ravel()  # (N*P,)
    cols_all_flat = cols_for_all_points.ravel()  # (N*P,)

    # Repeat colors for each pixel in the circle
    # Each point has the same circle shape, so we replicate bgr_colors by P
    colors_all_flat = np.repeat(bgr_colors, P, axis=0)  # (N*P, 3)

    # Validate indices
    valid_mask = (
        (rows_all_flat >= 0)
        & (rows_all_flat < img.shape[0])
        & (cols_all_flat >= 0)
        & (cols_all_flat < img.shape[1])
    )

    img[rows_all_flat[valid_mask], cols_all_flat[valid_mask]] = colors_all_flat[
        valid_mask
    ]

    return img


def test_plot_2d_traj() -> None:
    # Create a random image
    img = (np.ones((512, 512, 3)) * 255).astype(np.uint8)
    # trajs = np.array([[[10, 10], [20, 20], [30, 30]], [[50, 50], [60, 60], [70, 70]]])
    trajs = np.random.randint(0, 512, (1000, 20, 2))
    start_time = time.time()
    img = plot_2d_traj(img, trajs)
    print(f"Time taken: {time.time() - start_time:.3f}s")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == "__main__":
    test_plot_2d_traj()
