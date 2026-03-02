"""
    The file contains functions to generate data.
    n_samples = 10000 points getting sampled by default. Return format: X_feasible, X_infeasible, X, feasible.
    Refer as data_generation.generate_nonconvex_data().
    - 2D datasets:
        1. blob_with_bite()
        2. concentric_circles()
        3. star_shaped()
        4. two_moons() (disjoint set)
    - 3D datasets:
        1. torus()
        2. sphere_with_bite()
        3. spherical_shell()
        4. disconnected_spherical_shells()
    - Safety Gym data: safety_gym_data()
"""
import numpy as np
from sklearn.datasets import make_moons
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from tqdm import trange
import os

np.random.seed(42)

def generate_nonconvex_data(shape_name, n_samples=10000):
    """
    Generate nonconvex toy data of a specified shape.
    shape_name (str): The name of the shape to generate.
                          Available shapes are the keys of SHAPE_GENERATORS.
    n_samples (int): The total number of points to sample.

    Returns:
        X_feasible: Points that are feasible.
        X_infeasible: Points that are infeasible.
        X: All sampled points.
        feasible: Boolean mask of feasibility.
    """
    generator = SHAPE_GENERATORS.get(shape_name)
    if generator is None:
        raise ValueError(f"Unknown shape: '{shape_name}'. Available shapes: {list(SHAPE_GENERATORS.keys())}")
    return generator(n_samples=n_samples)

# Safety Gym data
def safety_gym_data(
    n_samples=10000,
    dataset_path="data/dataset_pointgoal2.npz",
    balance=True,
    random_state=42
):
    """
    Load Safety Gym (obs, act) pairs collected offline.

    Expects an .npz file with keys: X_feasible, X_infeasible, X_all, feasible_mask
    (as saved by your collect_pointgoal2_dataset.py script).

    Args:
        n_samples: number of samples to return (subsampled from the file)
        dataset_path: path to .npz created by the collector script
        balance: if True, draw a class-balanced subset
        random_state: RNG seed for reproducibility

    Returns:
        X_feasible: feasible subset
        X_infeasible: infeasible subset
        X: all returned samples (subset)
        feasible: boolean mask for X
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Safety Gym dataset not found at '{dataset_path}'. "
            f"Run the collector first, e.g.: "
            f"python collect_pointgoal2_dataset.py --n_samples 250000 --out {dataset_path}"
        )
    D = np.load(dataset_path)
    X_all_full = D["X_all"].astype(np.float32)
    feasible_full = D["feasible_mask"].astype(bool)

    N = X_all_full.shape[0]
    rng = np.random.default_rng(random_state)

    if n_samples is None or n_samples >= N:
        idx = np.arange(N)
    else:
        idx_f = np.where(feasible_full)[0]
        idx_i = np.where(~feasible_full)[0]
        if balance:
            k_f = min(n_samples // 2, len(idx_f))
            k_i = min(n_samples - k_f, len(idx_i))
            sel_f = rng.choice(idx_f, size=k_f, replace=False)
            sel_i = rng.choice(idx_i, size=k_i, replace=False)
            idx = np.concatenate([sel_f, sel_i])
        else:
            idx = rng.choice(N, size=n_samples, replace=False)
        rng.shuffle(idx)

    X = X_all_full[idx]
    feasible = feasible_full[idx]

    X_feasible = X[feasible]
    X_infeasible = X[~feasible]

    return X_feasible, X_infeasible, X, feasible

def blob_with_bite(n_samples=10000):
    """
    Generate a toy 2D nonconvex constraint set resembling a blob with a bite.

    1. Sample points uniformly within a disk of radius R.
    2. Define a 'bite' region as a circle (with given center and radius).
    3. Mark points as infeasible if they are outside the moon.

    Returns:
        X_feasible: Points that are feasible.
        X_infeasible: Points that are infeasible.
        X: All sampled points.
        feasible: Boolean mask of feasibility.

    Note: points are infeasible by default.
    """
    R = 2.0
    r = R * np.sqrt(np.random.uniform(0, 1, n_samples))
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    x = np.random.uniform(-2*R, 2*R, n_samples)
    y = np.random.uniform(-1.5*R, 1.5*R, n_samples)
    X = np.stack([x, y], axis=1)

    circle_center = np.array([0.0, 0.0])
    circle_radius = 2.0
    dist_to_circle = np.linalg.norm(X - circle_center, axis=1)

    bite_center = np.array([1.0, 0.0])
    bite_radius = 1.0
    dist_to_bite = np.linalg.norm(X - bite_center, axis=1)

    feasible = (dist_to_bite >= bite_radius) & (dist_to_circle <= circle_radius)
    X_feasible = X[feasible]
    X_infeasible = X[~feasible]

    return X_feasible, X_infeasible, X, feasible

def concentric_circles(n_samples=10000):
    """
    Generate a toy 2D nonconvex constraint set resembling an annulus (ring).

    1. Sample points uniformly within a bounding box covering the shape.
    2. Define two concentric circles with radii R_inner and R_outer.
    3. Mark points as feasible if they are between the inner and outer circles.

    Returns:
        X_feasible: Points that are feasible.
        X_infeasible: Points that are infeasible.
        X: All sampled points.
        feasible: Boolean mask of feasibility.
    """
    R_inner = 1.0
    R_outer = 2.0
    center = np.array([0.0, 0.0])

    x_min, x_max = -R_outer, R_outer
    y_min, y_max = -R_outer, R_outer

    x = np.random.uniform(x_min, x_max, n_samples)
    y = np.random.uniform(y_min, y_max, n_samples)
    X = np.stack([x, y], axis=1)

    dist_to_center = np.linalg.norm(X - center, axis=1)

    feasible = (dist_to_center >= R_inner) & (dist_to_center <= R_outer)

    X_feasible = X[feasible]
    X_infeasible = X[~feasible]

    return X_feasible, X_infeasible, X, feasible

def star_shaped(n_samples=10000):
    """
    Generate a toy 2D nonconvex constraint set resembling a star.

    1. Sample points uniformly within a bounding box covering the shape.
    2. Define the boundary of a star shape in polar coordinates: r(theta).
    3. Mark points as feasible if their distance from the center is less than or equal to r(theta) for their angle theta.

    Returns:
        X_feasible: Points that are feasible.
        X_infeasible: Points that are infeasible.
        X: All sampled points.
        feasible: Boolean mask of feasibility.
    """
    num_points = 5
    R_outer = 2.0
    R_inner = 1.0
    center = np.array([0.0, 0.0])

    max_r = R_outer
    x_min, x_max = -max_r, max_r
    y_min, y_max = -max_r, max_r

    x = np.random.uniform(x_min, x_max, n_samples)
    y = np.random.uniform(y_min, y_max, n_samples)
    X = np.stack([x, y], axis=1)

    dist_from_center = np.linalg.norm(X - center, axis=1)
    angle = np.arctan2(X[:, 1], X[:, 0])
    angle = np.where(angle < 0, angle + 2*np.pi, angle)

    shifted_angle = angle + np.pi / num_points
    angle_in_segment = shifted_angle % (2*np.pi / num_points)

    relative_angle = angle_in_segment / (np.pi / num_points)

    r_ref = np.zeros_like(dist_from_center)
    half_segment_angle = np.pi / num_points
    segment_factor = relative_angle
    r_ref = np.where(segment_factor <= 1,
                     R_outer - (R_outer - R_inner) * segment_factor,
                     R_inner + (R_outer - R_inner) * (segment_factor - 1)
                    )
    feasible = dist_from_center <= r_ref

    X_feasible = X[feasible]
    X_infeasible = X[~feasible]

    return X_feasible, X_infeasible, X, feasible


def two_moons(n_samples=10000):
    """
    Generate a toy 2D disjoint constraint set resembling two moons.
    Returns:
        X_feasible: Points generated by make_moons.
        X_infeasible: Randomly sampled points outside the moons.
        X: Union of feasible and infeasible points.
        feasible: Boolean mask indicating which points in X are feasible.
    """
    n_feasible_samples = n_samples // 2
    n_infeasible_samples = n_samples - n_feasible_samples

    X_feasible_core, labels = make_moons(n_samples=n_feasible_samples, noise=0.05, random_state=42)

    x_min, y_min = X_feasible_core.min(axis=0) - 0.5
    x_max, y_max = X_feasible_core.max(axis=0) + 0.5
    X_infeasible_random = np.random.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(n_infeasible_samples, 2))
    X = np.vstack([X_feasible_core, X_infeasible_random])
    feasible_mask_feasible_core = np.ones(n_feasible_samples, dtype=bool)
    feasible_mask_infeasible_random = np.zeros(n_infeasible_samples, dtype=bool)
    feasible = np.concatenate([feasible_mask_feasible_core, feasible_mask_infeasible_random])
    shuffle_indices = np.random.permutation(n_samples)
    X = X[shuffle_indices]
    feasible = feasible[shuffle_indices]

    X_feasible = X[feasible]
    X_infeasible = X[~feasible]

    return X_feasible, X_infeasible, X, feasible

# 3D datasets

def torus(n_samples=10000):
    """
    Generate a toy 3D constraint set resembling a torus.
    Returns:
        X_feasible: Points that are feasible.
        X_infeasible: Points that are infeasible.
        X: All sampled points.
        feasible: Boolean mask of feasibility.
    """
    R_major = 2.0
    r_minor = 0.5
    tolerance = 0.1
    max_extent = R_major + r_minor
    x_min, x_max = -max_extent, max_extent
    y_min, y_max = -max_extent, max_extent
    z_min, z_max = -r_minor, r_minor

    padding = tolerance * 2 
    x_min -= padding; x_max += padding
    y_min -= padding; y_max += padding
    z_min -= padding; z_max += padding


    x = np.random.uniform(x_min, x_max, n_samples)
    y = np.random.uniform(y_min, y_max, n_samples)
    z = np.random.uniform(z_min, z_max, n_samples)
    X = np.stack([x, y, z], axis=1)
    dist_to_center_circle_plane = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    dist_from_center_circle = np.abs(dist_to_center_circle_plane - R_major)
    dist_to_closest_point_on_torus_surface = np.sqrt(dist_from_center_circle**2 + X[:, 2]**2)

    feasible = np.abs(dist_to_closest_point_on_torus_surface - r_minor) <= tolerance

    X_feasible = X[feasible]
    X_infeasible = X[~feasible]

    return X_feasible, X_infeasible, X, feasible


def sphere_with_bite(n_samples=10000):
    """
    Generate a toy 3D nonconvex constraint set resembling a sphere with a bite.
    Returns:
        X_feasible: Points that are feasible.
        X_infeasible: Points that are infeasible.
        X: All sampled points.
        feasible: Boolean mask of feasibility.
    """
    R_main = 2.0
    bite_center = np.array([1.0, 0.0, 0.0])
    bite_radius = 1.0
    main_center = np.array([0.0, 0.0, 0.0])

    max_extent = R_main + bite_radius
    x_min, x_max = -max_extent, max_extent
    y_min, y_max = -max_extent, max_extent
    z_min, z_max = -max_extent, max_extent


    x = np.random.uniform(x_min, x_max, n_samples)
    y = np.random.uniform(y_min, y_max, n_samples)
    z = np.random.uniform(z_min, z_max, n_samples)
    X = np.stack([x, y, z], axis=1)

    dist_to_main = np.linalg.norm(X - main_center, axis=1)
    dist_to_bite = np.linalg.norm(X - bite_center, axis=1)

    feasible = (dist_to_main <= R_main) & (dist_to_bite >= bite_radius)

    X_feasible = X[feasible]
    X_infeasible = X[~feasible]

    return X_feasible, X_infeasible, X, feasible


def spherical_shell(n_samples=10000):
    """
    Generate a toy 3D nonconvex constraint set resembling a spherical shell.
    Returns:
        X_feasible: Points that are feasible.
        X_infeasible: Points that are infeasible.
        X: All sampled points.
        feasible: Boolean mask of feasibility.
    """
    R_inner = 1.0
    R_outer = 2.0
    center = np.array([0.0, 0.0, 0.0])

    x_min, x_max = -R_outer, R_outer
    y_min, y_max = -R_outer, R_outer
    z_min, z_max = -R_outer, R_outer

    x = np.random.uniform(x_min, x_max, n_samples)
    y = np.random.uniform(y_min, y_max, n_samples)
    z = np.random.uniform(z_min, z_max, n_samples)
    X = np.stack([x, y, z], axis=1)

    dist_to_center = np.linalg.norm(X - center, axis=1)

    feasible = (dist_to_center >= R_inner) & (dist_to_center <= R_outer)

    X_feasible = X[feasible]
    X_infeasible = X[~feasible]

    return X_feasible, X_infeasible, X, feasible


def disconnected_spherical_shells(n_samples=10000):
    """
    Generate a toy 3D disjoint constraint set resembling two disconnected spherical shells.
    Returns:
        X_feasible: Points within either shell.
        X_infeasible: Randomly sampled points outside the shells.
        X: Union of feasible and infeasible points.
        feasible: Boolean mask indicating which points in X are feasible.
    """
    R_inner = 1.0
    R_outer = 1.5
    center1 = np.array([-2.0, 0.0, 0.0])
    center2 = np.array([2.0, 0.0, 0.0])
    n_feasible_samples = n_samples // 2
    n_infeasible_samples = n_samples - n_feasible_samples

    r1 = np.random.uniform(R_inner, R_outer, n_feasible_samples // 2)
    phi1 = np.random.uniform(0, 2*np.pi, n_feasible_samples // 2)
    theta1 = np.random.uniform(0, np.pi, n_feasible_samples // 2) # spherical coordinates
    x1 = r1 * np.sin(theta1) * np.cos(phi1) + center1[0]
    y1 = r1 * np.sin(theta1) * np.sin(phi1) + center1[1]
    z1 = r1 * np.cos(theta1) + center1[2]
    X_shell1 = np.stack([x1, y1, z1], axis=1)

    r2 = np.random.uniform(R_inner, R_outer, n_feasible_samples - n_feasible_samples // 2)
    phi2 = np.random.uniform(0, 2*np.pi, n_feasible_samples - n_feasible_samples // 2)
    theta2 = np.random.uniform(0, np.pi, n_feasible_samples - n_feasible_samples // 2)
    x2 = r2 * np.sin(theta2) * np.cos(phi2) + center2[0]
    y2 = r2 * np.sin(theta2) * np.sin(phi2) + center2[1]
    z2 = r2 * np.cos(theta2) + center2[2]
    X_shell2 = np.stack([x2, y2, z2], axis=1)

    X_feasible_core = np.vstack([X_shell1, X_shell2])

    max_coord = np.max([np.max(np.abs(X_shell1)), np.max(np.abs(X_shell2))]) + R_outer
    x_min, x_max = -max_coord, max_coord
    y_min, y_max = -max_coord, max_coord
    z_min, z_max = -max_coord, max_coord

    X_infeasible_random = np.random.uniform(low=[x_min, y_min, z_min], high=[x_max, y_max, z_max], size=(n_infeasible_samples, 3))

    X = np.vstack([X_feasible_core, X_infeasible_random])

    feasible_mask_feasible_core = np.ones(X_feasible_core.shape[0], dtype=bool)
    feasible_mask_infeasible_random = np.zeros(X_infeasible_random.shape[0], dtype=bool)
    feasible = np.concatenate([feasible_mask_feasible_core, feasible_mask_infeasible_random])

    shuffle_indices = np.random.permutation(X.shape[0])
    X = X[shuffle_indices]
    feasible = feasible[shuffle_indices]

    X_feasible = X[feasible]
    X_infeasible = X[~feasible]

    return X_feasible, X_infeasible, X, feasible

# Hyperspherical shells
def hyperspherical_shell_nd(n_samples=10000, dim=3, R_inner=1.0, R_outer=2.0, random_state=42):
    """
    Generate points in R^dim and mark as feasible those with radius in [R_inner, R_outer].
    For dim=2, this reduces to an annulus; for dim=3, a spherical shell.
    """
    rng = np.random.default_rng(random_state)
    directions = rng.standard_normal(size=(n_samples, dim)).astype(np.float32)
    norms = np.linalg.norm(directions, axis=1, keepdims=True).astype(np.float32)
    norms = np.where(norms == 0.0, 1.0, norms)
    unit_dirs = directions / norms
    u = rng.random(size=(n_samples, 1)).astype(np.float32)
    radii_sampled = (u ** (1.0 / dim)) * R_outer
    X = unit_dirs * radii_sampled
    radii = np.linalg.norm(X, axis=1)
    feasible = (radii >= R_inner) & (radii <= R_outer)
    X_feasible = X[feasible]
    X_infeasible = X[~feasible]
    return X_feasible, X_infeasible, X, feasible

def hyperspherical_shell_3d(n_samples=10000):
    return hyperspherical_shell_nd(n_samples=n_samples, dim=3, R_inner=1.0, R_outer=2.0)

def hyperspherical_shell_5d(n_samples=10000):
    return hyperspherical_shell_nd(n_samples=n_samples, dim=5, R_inner=1.0, R_outer=2.0)

def hyperspherical_shell_10d(n_samples=10000):
    return hyperspherical_shell_nd(n_samples=n_samples, dim=10, R_inner=1.0, R_outer=2.0)

def hyperspherical_shell_50d(n_samples=10000):
    return hyperspherical_shell_nd(n_samples=n_samples, dim=50, R_inner=1.0, R_outer=2.0)

def generate_dimensional_experiments(
    dims=(3, 5, 10, 50),
    base_n_samples=10000,
    growth_factor=2.0,
    R_inner=1.0,
    R_outer=2.0,
    random_state=42
):
    """
    Generate experiments for a set of dimensions with exponentially increasing sample sizes.
    Returns a dict: dim -> (X_feasible, X_infeasible, X_all, feasible_mask, n_samples_used)
    Example: with base_n_samples=10_000 and growth_factor=2:
        3D: 10k, 5D: 20k, 10D: 40k, 50D: 80k
    """
    results = {}
    for i, d in enumerate(dims):
        n = int(round(base_n_samples * (growth_factor ** i)))
        Xf, Xi, X, feas = hyperspherical_shell_nd(
            n_samples=n, dim=d, R_inner=R_inner, R_outer=R_outer, random_state=random_state + i
        )
        results[d] = (Xf, Xi, X, feas, n)
    return results

def check_feasibility_blob_with_bite(X):
    """Check feasibility for the 'blob_with_bite' shape."""
    circle_center = np.array([0.0, 0.0])
    circle_radius = 2.0
    dist_to_circle = np.linalg.norm(X - circle_center, axis=1)

    bite_center = np.array([1.0, 0.0])
    bite_radius = 1.0
    dist_to_bite = np.linalg.norm(X - bite_center, axis=1)

    return (dist_to_bite >= bite_radius) & (dist_to_circle <= circle_radius)

def check_feasibility_concentric_circles(X):
    """Check feasibility for the 'concentric_circles' shape."""
    R_inner = 1.0
    R_outer = 2.0
    center = np.array([0.0, 0.0])
    dist_to_center = np.linalg.norm(X - center, axis=1)
    return (dist_to_center >= R_inner) & (dist_to_center <= R_outer)

def check_feasibility_star_shaped(X):
    """Check feasibility for the 'star_shaped' shape."""
    num_points = 5
    R_outer = 2.0
    R_inner = 1.0
    center = np.array([0.0, 0.0])

    dist_from_center = np.linalg.norm(X - center, axis=1)
    angle = np.arctan2(X[:, 1], X[:, 0])
    angle = np.where(angle < 0, angle + 2 * np.pi, angle)

    shifted_angle = angle + np.pi / num_points
    angle_in_segment = shifted_angle % (2 * np.pi / num_points)
    relative_angle = angle_in_segment / (np.pi / num_points)

    r_ref = np.where(relative_angle <= 1,
                     R_outer - (R_outer - R_inner) * relative_angle,
                     R_inner + (R_outer - R_inner) * (relative_angle - 1))
    return dist_from_center <= r_ref

def build_two_moons_oracle(n_ref=60000, noise=0.05, radius_mult=3.0):
    """
    Build a nearest-neighbor-based oracle for the two_moons shape that
    mirrors the way feasible points are generated via sklearn.make_moons.

    A point is considered feasible if it lies within radius_mult * noise
    of some point on a dense reference moons dataset.
    """
    X_ref, _ = make_moons(n_samples=n_ref, noise=noise, random_state=42)
    nn = NearestNeighbors(n_neighbors=1).fit(X_ref)
    thresh = radius_mult * noise

    def is_on_moons(points_np):
        dists, _ = nn.kneighbors(points_np)
        return dists[:, 0] <= thresh

    return is_on_moons

def check_feasibility_two_moons(X):
    """
    Check feasibility for the 'two_moons' shape using a dataset-based
    oracle: points close to the canonical sklearn.make_moons manifold
    are treated as feasible.
    """
    oracle = build_two_moons_oracle()
    return oracle(X)

def check_feasibility_torus(X):
    """Check feasibility for the 'torus' shape."""
    R_major = 2.0
    r_minor = 0.5
    tolerance = 0.1
    dist_to_center_circle_plane = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    dist_from_center_circle = np.abs(dist_to_center_circle_plane - R_major)
    dist_to_closest_point_on_torus_surface = np.sqrt(dist_from_center_circle**2 + X[:, 2]**2)
    return np.abs(dist_to_closest_point_on_torus_surface - r_minor) <= tolerance

def check_feasibility_sphere_with_bite(X):
    """Check feasibility for the 'sphere_with_bite' shape."""
    R_main = 2.0
    bite_center = np.array([1.0, 0.0, 0.0])
    bite_radius = 1.0
    main_center = np.array([0.0, 0.0, 0.0])
    dist_to_main = np.linalg.norm(X - main_center, axis=1)
    dist_to_bite = np.linalg.norm(X - bite_center, axis=1)
    return (dist_to_main <= R_main) & (dist_to_bite >= bite_radius)

def check_feasibility_spherical_shell(X):
    """Check feasibility for the 'spherical_shell' shape."""
    R_inner = 1.0
    R_outer = 2.0
    center = np.array([0.0, 0.0, 0.0])
    dist_to_center = np.linalg.norm(X - center, axis=1)
    return (dist_to_center >= R_inner) & (dist_to_center <= R_outer)

def check_feasibility_disconnected_spherical_shells(X):
    """Check feasibility for the 'disconnected_spherical_shells' shape."""
    R_inner = 1.0
    R_outer = 1.5
    center1 = np.array([-2.0, 0.0, 0.0])
    center2 = np.array([2.0, 0.0, 0.0])
    dist_to_center1 = np.linalg.norm(X - center1, axis=1)
    shell1_feasible = (dist_to_center1 >= R_inner) & (dist_to_center1 <= R_outer)
    dist_to_center2 = np.linalg.norm(X - center2, axis=1)
    shell2_feasible = (dist_to_center2 >= R_inner) & (dist_to_center2 <= R_outer)
    return shell1_feasible | shell2_feasible

def check_feasibility_hyperspherical_shell_nd(X, R_inner=1.0, R_outer=2.0):
    """Check feasibility for generalized hyperspherical shell in any dimension."""
    radii = np.linalg.norm(X, axis=1)
    return (radii >= R_inner) & (radii <= R_outer)

def check_feasibility_hyperspherical_shell_3d(X):
    return check_feasibility_hyperspherical_shell_nd(X, R_inner=1.0, R_outer=2.0)

def check_feasibility_hyperspherical_shell_5d(X):
    return check_feasibility_hyperspherical_shell_nd(X, R_inner=1.0, R_outer=2.0)

def check_feasibility_hyperspherical_shell_10d(X):
    return check_feasibility_hyperspherical_shell_nd(X, R_inner=1.0, R_outer=2.0)

def check_feasibility_hyperspherical_shell_50d(X):
    return check_feasibility_hyperspherical_shell_nd(X, R_inner=1.0, R_outer=2.0)

def check_feasibility(X, shape_name):
    """
    Checks the feasibility of points for a given shape.
    Args:
        X: A numpy array of points (n_samples, n_dims).
        shape_name: The name of the shape to check against.
    Returns:
        A boolean numpy array indicating feasibility for each point.
    Note:
        `two_moons` is supported via an analytical approximation.
    """
    checker = SHAPE_CHECKERS.get(shape_name)
    if checker is None:
        raise ValueError(f"Unknown shape: '{shape_name}'. Available shapes: {list(SHAPE_CHECKERS.keys())}")
    return checker(X)

# Generating data points
SHAPE_GENERATORS = {
    'blob_with_bite': blob_with_bite,
    'concentric_circles': concentric_circles,
    'star_shaped': star_shaped,
    'two_moons': two_moons,
    'torus': torus,
    'sphere_with_bite': sphere_with_bite,
    'spherical_shell': spherical_shell,
    'disconnected_spherical_shells': disconnected_spherical_shells,
    'hyperspherical_shell_3d': hyperspherical_shell_3d,
    'hyperspherical_shell_5d': hyperspherical_shell_5d,
    'hyperspherical_shell_10d': hyperspherical_shell_10d,
    'hyperspherical_shell_50d': hyperspherical_shell_50d,
    'safety_gym': safety_gym_data
}

# Feasibility checks
SHAPE_CHECKERS = {
    'blob_with_bite': check_feasibility_blob_with_bite,
    'concentric_circles': check_feasibility_concentric_circles,
    'star_shaped': check_feasibility_star_shaped,
    'two_moons': check_feasibility_two_moons,
    'torus': check_feasibility_torus,
    'sphere_with_bite': check_feasibility_sphere_with_bite,
    'spherical_shell': check_feasibility_spherical_shell,
    'disconnected_spherical_shells': check_feasibility_disconnected_spherical_shells,
    'hyperspherical_shell_3d': check_feasibility_hyperspherical_shell_3d,
    'hyperspherical_shell_5d': check_feasibility_hyperspherical_shell_5d,
    'hyperspherical_shell_10d': check_feasibility_hyperspherical_shell_10d,
    'hyperspherical_shell_50d': check_feasibility_hyperspherical_shell_50d,
}