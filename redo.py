import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
from gudhi.wasserstein import wasserstein_distance
from scipy.spatial import cKDTree
from itertools import combinations
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import TwoSlopeNorm
import time
from joblib import Parallel, delayed

# Try to import CPH2d, but make it optional
try:
    from CPH2d import ChiralPersistentHomology2D
except ImportError:
    ChiralPersistentHomology2D = None
    print("Warning: CPH2d module not found. Some functionality may be limited.")


def compute_rmax_for_luminosity_connectivity(luminosity, percentile=40):
    """
    Compute rmax based on L distances using sampling.
    
    Parameters:
    - luminosity: array of luminosity values
    - percentile: percentile to use for rmax calculation
    
    Returns:
    - rmax value (float)
    """
    n = len(luminosity)
    n_samples = min(5000, max(1000, n * 3))
    indices = np.random.choice(n, size=(2, n_samples), replace=True)
    sample_dists = np.abs(luminosity[indices[0]] - luminosity[indices[1]])
    non_zero = sample_dists[sample_dists > 0]
    if len(non_zero) == 0:
        return 0.1
    return float(np.percentile(non_zero, percentile))


def prepare_topology_2d(data_2d, luminosity=None, eps=None, rmax=None, max_alpha=None):
    """
    Pipeline: Build AlphaComplex from spatial coordinates 
    Uses luminosity only for handedness classification.
    NO FILTERING - includes ALL simplices regardless of size.
    H0 and H1 are combined into single persistence diagrams (no distinction).
    
    Parameters:
    - data_2d: (N, 2) array of spatial coordinates (x, y)
    - luminosity: (N,) array of L values. Used for handedness only.
    - max_alpha: deprecated, kept for compatibility (ignored)
    - eps: filtration value (if None, uses maximum alpha value from complex)
    - rmax: deprecated, kept for compatibility (ignored)
    
    Returns dict with: st, eps, edges, tris, left_tris, right_tris, st_L, st_R, L, R
    where L and R are combined H0+H1 persistence diagrams (no distinction between dimensions).
    Also returns L0, L1, R0, R1 for backwards compatibility.
    """
    start_time = time.time()
    
    if luminosity is None:
        raise ValueError("luminosity (L) is required")
    
    n = len(data_2d)
    
    # Build AlphaComplex from 2D spatial coordinates 
    alpha_complex = gd.AlphaComplex(points=data_2d)
    st = alpha_complex.create_simplex_tree()
    
    # Extract edges and triangles with alpha values 
    skeleton_data = list(st.get_skeleton(2))
    if not skeleton_data:
        return {
            "st": st, "eps": 0.0, "edges": [], "tris": [],
            "left_tris": [], "right_tris": [], "st_L": gd.SimplexTree(), "st_R": gd.SimplexTree(),
            "L": [], "R": [], "L0": [], "L1": [], "R0": [], "R1": [],
        }
    
    # Separate simplices and filtrations (alpha values)
    simplices = [s for s, _ in skeleton_data]
    filtrations = np.array([f for _, f in skeleton_data], dtype=float)
    
    # NO FILTERING - use ALL simplices regardless of alpha value
    simplex_lengths = np.array([len(s) for s in simplices], dtype=int)
    
    # Extract edges (dim=1) and triangles (dim=2) 
    # Convert to Python ints to ensure type consistency
    edges = [tuple(int(x) for x in np.sort(s)) for s, L in zip(simplices, simplex_lengths) if L == 2]
    tris = [tuple(int(x) for x in np.sort(s)) for s, L in zip(simplices, simplex_lengths) if L == 3]
    
    # Store alpha values for edges and triangles
    edges_alpha = [float(filtrations[i]) for i, L in enumerate(simplex_lengths) if L == 2]
    tris_alpha = [float(filtrations[i]) for i, L in enumerate(simplex_lengths) if L == 3]
    
    # Compute eps as maximum filtration value if not provided
    if eps is None:
        eps = float(np.max(filtrations)) if len(filtrations) > 0 else 0.0
    
    print(f"Simplices: {len(edges)} edges, {len(tris)} triangles (NO FILTERING - all simplices included)")
    
    # Split triangles by handedness, vectorized
    if len(tris) == 0:
        left_tris, right_tris = [], []
    else:
        tris_array = np.asarray(tris, dtype=int)
        # Compute oriented area 
        a = data_2d[tris_array[:, 0]]
        b = data_2d[tris_array[:, 1]]
        c = data_2d[tris_array[:, 2]]
        ab = b - a
        ac = c - a
        oriented_area = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
        
        # Classify by handedness 
        is_right_handed = oriented_area > 0
        # Convert to tuples of Python ints to match tri_to_alpha keys
        right_tris = [tuple(int(x) for x in tri) for tri in tris_array[is_right_handed].tolist()]
        left_tris = [tuple(int(x) for x in tri) for tri in tris_array[~is_right_handed].tolist()]
    
    # Build Left/Right simplex trees 
    st_L, st_R = gd.SimplexTree(), gd.SimplexTree()
    
    # Insert ALL edges into both complexes (edges appear in both)
    edge_to_alpha = {edge: alpha for edge, alpha in zip(edges, edges_alpha)}
    for edge, alpha in edge_to_alpha.items():
        st_L.insert(edge, filtration=alpha)
        st_R.insert(edge, filtration=alpha)
    
    # Insert triangles with their alpha values 
    # tri is already a tuple, so we can use it directly as key
    tri_to_alpha = {tri: alpha for tri, alpha in zip(tris, tris_alpha)}
    
    if left_tris:
        for tri in left_tris:
            # All triangles should be in tri_to_alpha since we're not filtering
            # tri is already a tuple, use it directly
            alpha = tri_to_alpha.get(tri, eps)
            st_L.insert(tri, filtration=alpha)
    
    if right_tris:
        for tri in right_tris:
            # All triangles should be in tri_to_alpha since we're not filtering
            # tri is already a tuple, use it directly
            alpha = tri_to_alpha.get(tri, eps)
            st_R.insert(tri, filtration=alpha)
    
    # Persistence computation
    st_L.persistence(homology_coeff_field=2, min_persistence=0.0)
    st_R.persistence(homology_coeff_field=2, min_persistence=0.0)
    L0 = st_L.persistence_intervals_in_dimension(0)
    L1 = st_L.persistence_intervals_in_dimension(1)
    R0 = st_R.persistence_intervals_in_dimension(0)
    R1 = st_R.persistence_intervals_in_dimension(1)
    
    # Combine H0 and H1 into single diagrams (treat them the same)
    def _combine_diagrams(d0, d1):
        """Combine two persistence diagrams into one"""
        d0_arr = np.array(d0, dtype=float) if len(d0) > 0 else np.zeros((0, 2))
        d1_arr = np.array(d1, dtype=float) if len(d1) > 0 else np.zeros((0, 2))
        if d0_arr.size == 0:
            return d1_arr.tolist() if d1_arr.size > 0 else []
        if d1_arr.size == 0:
            return d0_arr.tolist() if d0_arr.size > 0 else []
        # Reshape if needed
        if d0_arr.ndim == 1:
            d0_arr = d0_arr.reshape(-1, 2)
        if d1_arr.ndim == 1:
            d1_arr = d1_arr.reshape(-1, 2)
        # Combine and return as list of lists
        combined = np.vstack([d0_arr, d1_arr])
        return combined.tolist()
    
    L = _combine_diagrams(L0, L1)
    R = _combine_diagrams(R0, R1)
    
    elapsed = time.time() - start_time
    if elapsed > 10:
        print(f"Warning: Function took {elapsed:.2f}s (target: <10s)")
    else:
        print(f"Completed in {elapsed:.2f}s")

    return {
        "st": st, "eps": float(eps),
        "edges": edges, "tris": tris,
        "left_tris": left_tris, "right_tris": right_tris,
        "st_L": st_L, "st_R": st_R,
        "L": L, "R": R,  # Combined H0+H1 diagrams
        "L0": L0, "L1": L1, "R0": R0, "R1": R1,  # Keep for backwards compatibility
    }


def plot_pd_gudhi_split(L, R, fname_prefix="pd_left_right", title_prefix="Persistence diagram"):
    """
    Plot separate images for Left and Right complexes.
    Accepts combined persistence diagrams (H0+H1 combined, no distinction).
    
    Parameters:
    - L: combined persistence diagram for left complex
    - R: combined persistence diagram for right complex
    - fname_prefix: prefix for output filename
    - title_prefix: prefix for plot title
    """
    def _finite_np(dgm):
        d = np.array(dgm, dtype=float)
        if d.size == 0: return d.reshape(0,2)
        if d.ndim == 1: d = d.reshape(-1,2)
        return d[np.isfinite(d).all(axis=1)]

    # Process combined diagrams (H0+H1 together)
    Lf = _finite_np(L)
    Rf = _finite_np(R)

    # Determine common limits (shared across all diagrams for fair comparison)
    def _lim(*arrs):
        mx = 0.0
        for a in arrs:
            if a.size:
                mx = max(mx, float(a.max()))
        return (mx*1.02) if mx > 0 and np.isfinite(mx) else 1.0

    lim = _lim(Lf, Rf)

    def _plot_single(ax, d, label, lim, color):
        xs = np.linspace(0.0, lim, 200)
        ax.plot(xs, xs, color="k", linewidth=1)
        ax.fill_between(xs, xs, lim, color="0.7", alpha=0.25)
        if d.size:
            ax.scatter(d[:,0], d[:,1], s=20, alpha=0.9, label=f"{label} (H0+H1)", color=color, edgecolors="none")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("Birth"); ax.set_ylabel("Death")
        ax.set_title(f"{title_prefix} — {label}")
        ax.legend(loc="lower right")

    # Left and Right complexes (combined H0+H1, no distinction)
    # Colors match simplicial complex: Left=red, Right=blue
    fig, axes = plt.subplots(1, 2, figsize=(9,4))
    _plot_single(axes[0], Lf, "Left", lim, "red")
    _plot_single(axes[1], Rf, "Right", lim, "blue")
    plt.tight_layout()
    plt.savefig(f"{fname_prefix}_left_right.png", dpi=160)
    plt.show()
    print("Saved", f"{fname_prefix}_left_right.png")


def plot_triangles_filled(ax, data_2d, edges, triangles, color, title):
    """
    Helper function to plot edges and triangles with proper filling.
    Shows ALL edges (connecting all points) and filled triangles.
    
    Parameters:
    - ax: matplotlib axes
    - data_2d: (N, 2) array of spatial coordinates
    - edges: list of edge tuples
    - triangles: list of triangle tuples
    - color: color for triangles
    - title: plot title
    """
    # Plot data points
    ax.scatter(data_2d[:, 0], data_2d[:, 1], c='black', marker='o', s=1, alpha=0.6)
    
    # Plot ALL edges (edges appear in both left and right diagrams)
    for edge in edges:
        if isinstance(edge, (tuple, list)) and len(edge) == 2:
            i, j = int(edge[0]), int(edge[1])
            a = data_2d[i]
            b = data_2d[j]
            # Draw edge as line
            ax.plot([a[0], b[0]], [a[1], b[1]], color='black', linewidth=0.2, alpha=0.4)
    
    # Plot and fill each triangle
    for tri in triangles:
        # Handle both tuples and lists
        if isinstance(tri, (tuple, list)) and len(tri) == 3:
            # Get vertex indices
            i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
            # Get coordinates
            a = data_2d[i]
            b = data_2d[j]
            c = data_2d[k]
            # Fill triangle (Ax.fill with color and alpha=0.3)
            ax.fill([a[0], b[0], c[0]], [a[1], b[1], c[1]], color=color, alpha=0.3)
    
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_combined_triangles(ax, data_2d, edges, left_tris, right_tris, title):
    """
    Helper function to plot both left and right triangles together
    Shows all edges, left triangles in red, and right triangles in blue.
    
    Parameters:
    - ax: matplotlib axes
    - data_2d: (N, 2) array of spatial coordinates
    - edges: list of edge tuples
    - left_tris: list of left-handed triangle tuples
    - right_tris: list of right-handed triangle tuples
    - title: plot title
    """
    # Plot data points
    ax.scatter(data_2d[:, 0], data_2d[:, 1], c='black', marker='o', s=1, alpha=0.6)
    
    # Plot ALL edges
    for edge in edges:
        if isinstance(edge, (tuple, list)) and len(edge) == 2:
            i, j = int(edge[0]), int(edge[1])
            a = data_2d[i]
            b = data_2d[j]
            # Draw edge as line
            ax.plot([a[0], b[0]], [a[1], b[1]], color='black', linewidth=0.2, alpha=0.4)
    
    # Plot and fill left-handed triangles (red)
    for tri in left_tris:
        if isinstance(tri, (tuple, list)) and len(tri) == 3:
            i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
            a = data_2d[i]
            b = data_2d[j]
            c = data_2d[k]
            ax.fill([a[0], b[0], c[0]], [a[1], b[1], c[1]], color='red', alpha=0.3)
    
    # Plot and fill right-handed triangles (blue)
    for tri in right_tris:
        if isinstance(tri, (tuple, list)) and len(tri) == 3:
            i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
            a = data_2d[i]
            b = data_2d[j]
            c = data_2d[k]
            ax.fill([a[0], b[0], c[0]], [a[1], b[1], c[1]], color='blue', alpha=0.3)
    
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def auto_plot_handedness_2d(
    data_2d,
    luminosity=None,
    handedness="left",
    rmax=None,
    eps=None,
    max_alpha=None,
    title_prefix="2D complex",
    out=None,
    face_alpha=0.30,
    separate_plots=True,
):
    """
    Pipeline: form simplicial complex using AlphaComplex, plot with proper triangle filling.
    Creates three plots when separate_plots=True: left-only, right-only, and combined
    
    Parameters:
    - data_2d: (N, 2) array of spatial coordinates
    - luminosity: (N,) array of luminosity values
    - handedness: "left" or "right" (only used if separate_plots=False)
    - rmax: deprecated, kept for compatibility
    - eps: filtration value
    - max_alpha: deprecated, kept for compatibility
    - title_prefix: prefix for plot titles
    - out: output filename prefix (optional)
    - face_alpha: transparency for triangles
    - separate_plots: If True, creates three separate figures:
        1. Left-handed triangles only (red)
        2. Right-handed triangles only (blue)
        3. Combined Alpha Complex (both left and right together)
      If False, creates single plot with chosen handedness
    
    Returns dict with topology results
    """
    if luminosity is None:
        raise ValueError("luminosity (L) is required")
    res = prepare_topology_2d(data_2d, luminosity=luminosity, eps=eps, rmax=rmax, max_alpha=max_alpha)
    
    st = res["st"]
    eps = res["eps"]
    edges = res["edges"]
    tris = res["tris"]
    left_tris = res["left_tris"]
    right_tris = res["right_tris"]

    # Convert to lists if needed
    if isinstance(left_tris, np.ndarray):
        left_tris = left_tris.tolist()
    if isinstance(right_tris, np.ndarray):
        right_tris = right_tris.tolist()

    if separate_plots:
        # Create three plots: left-only, right-only, and combined 
        # Left-handed triangles only (with ALL edges)
        fig_left, ax_left = plt.subplots(figsize=(8, 8))
        plot_triangles_filled(ax_left, data_2d, edges, left_tris, 'red', 
                             f"{title_prefix} — Left-handed Triangles (#{len(left_tris)} triangles, #{len(edges)} edges)")
        plt.tight_layout()
        if out:
            out_left = out.replace('.png', '_left.png') if out else None
            if out_left:
                plt.savefig(out_left, dpi=160, bbox_inches='tight')
        plt.show()
        
        # Right-handed triangles only (with ALL edges)
        fig_right, ax_right = plt.subplots(figsize=(8, 8))
        plot_triangles_filled(ax_right, data_2d, edges, right_tris, 'blue', 
                             f"{title_prefix} — Right-handed Triangles (#{len(right_tris)} triangles, #{len(edges)} edges)")
        plt.tight_layout()
        if out:
            out_right = out.replace('.png', '_right.png') if out else None
            if out_right:
                plt.savefig(out_right, dpi=160, bbox_inches='tight')
        plt.show()
        
        # Combined plot: both left and right triangles together 
        fig_combined, ax_combined = plt.subplots(figsize=(8, 8))
        plot_combined_triangles(ax_combined, data_2d, edges, left_tris, right_tris,
                               f"{title_prefix} — Alpha Complex (L={len(left_tris)}, R={len(right_tris)}, E={len(edges)})")
        plt.tight_layout()
        if out:
            out_combined = out.replace('.png', '_combined.png') if out else None
            if out_combined:
                plt.savefig(out_combined, dpi=160, bbox_inches='tight')
        plt.show()
    else:
        # Single plot with chosen handedness (backward compatibility)
        chosen = left_tris if handedness == "left" else right_tris
        color  = "red" if handedness == "left" else "blue"
        label  = "Left-handed" if handedness == "left" else "Right-handed"
        
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_triangles_filled(ax, data_2d, edges, chosen, color, 
                             f"{title_prefix} — {label} (#{len(chosen)} triangles, #{len(edges)} edges)")
        plt.tight_layout()
        if out:
            plt.savefig(out, dpi=160, bbox_inches='tight')
        plt.show()

    return {
        "st": st,
        "eps": eps,
        "edges": edges,
        "tris": tris,
        "left_tris": left_tris,
        "right_tris": right_tris,
    }


def split_by_handedness(data_2d, tris):
    """
    Vectorized handedness split without Python loops.
    
    Parameters:
    - data_2d: (N, 2) array of spatial coordinates
    - tris: list of triangle tuples
    
    Returns:
    - left_tris: list of left-handed triangles
    - right_tris: list of right-handed triangles
    """
    if not tris:
        return [], []
    tris_array = np.asarray(tris, dtype=int)
    n_points = len(data_2d)
    # Filter out triangles with invalid indices (out of bounds)
    valid_mask = (tris_array[:, 0] < n_points) & (tris_array[:, 1] < n_points) & (tris_array[:, 2] < n_points)
    tris_array = tris_array[valid_mask]
    if len(tris_array) == 0:
        return [], []
    a = data_2d[tris_array[:, 0]]
    b = data_2d[tris_array[:, 1]]
    c = data_2d[tris_array[:, 2]]
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    left = tris_array[det > 0].tolist()
    right = tris_array[det < 0].tolist()
    return left, right


def _finite_np(diag):
    """
    Extract finite points from a persistence diagram.
    
    Parameters:
    - diag: persistence diagram (list of [birth, death] pairs)
    
    Returns:
    - numpy array of finite points
    """
    d = np.array(diag, dtype=float)
    if d.size == 0:
        return d.reshape(0, 2)
    if d.ndim == 1:
        d = d.reshape(-1, 2)
    mask = np.isfinite(d).all(axis=1)   # drop essential (∞) points
    return d[mask]


def subtract_diagrams_combined(dgmL, dgmR, order=1, internal_p=1, tol_decimals=None):
    """
    Combined, vectorized routine that:
      - filters finite points
      - optionally rounds to tol_decimals and computes set-style differences (vectorized)
      - computes Wasserstein matching and returns vectorized deltas and unmatched
    
    Parameters:
    - dgmL: left persistence diagram
    - dgmR: right persistence diagram
    - order: Wasserstein order (default: 1)
    - internal_p: internal p-norm (default: 1)
    - tol_decimals: if not None, rounds to this many decimals for set comparisons
    
    Returns dict with keys:
      L_finite, R_finite, W, pairs (i,j,db,dd), L_unmatched, R_unmatched,
      and if tol_decimals is not None: L_only_set, R_only_set
    """
    # finite extractor (vectorized)
    def _finite_np(dgm):
        d = np.array(dgm, dtype=float)
        if d.size == 0:
            return d.reshape(0, 2)
        if d.ndim == 1:
            d = d.reshape(-1, 2)
        m = np.isfinite(d).all(axis=1)
        return d[m]

    L = _finite_np(dgmL)
    R = _finite_np(dgmR)

    out = {"L_finite": L, "R_finite": R}

    # optional set-style differences via rounding (vectorized, no Python loops)
    if tol_decimals is not None:
        if L.size:
            Lr = np.round(L, tol_decimals)
        else:
            Lr = L
        if R.size:
            Rr = np.round(R, tol_decimals)
        else:
            Rr = R
        # use structured dtype so rows are hashable for set-like ops via numpy
        def _as_struct(a):
            if a.size == 0:
                return a.view([("b", float), ("d", float)])
            return a.view([("b", a.dtype), ("d", a.dtype)]).reshape(-1)
        Ls = _as_struct(Lr)
        Rs = _as_struct(Rr)
        # set differences via numpy
        L_only_struct = np.setdiff1d(Ls, Rs, assume_unique=False)
        R_only_struct = np.setdiff1d(Rs, Ls, assume_unique=False)
        L_only = L_only_struct.view(Lr.dtype).reshape(-1, 2) if L_only_struct.size else np.zeros((0,2))
        R_only = R_only_struct.view(Rr.dtype).reshape(-1, 2) if R_only_struct.size else np.zeros((0,2))
        out.update({"L_only_set": L_only, "R_only_set": R_only})

    # ============================================================================
    # WASSERSTEIN SUBTRACTION SECTION
    # ============================================================================
    # This computes the optimal transport (Wasserstein) matching between the two
    # persistence diagrams L and R, then performs "subtraction" by computing
    # the differences between matched points.
    #
    # The Wasserstein distance finds the optimal matching that minimizes the
    # total cost of moving mass from diagram L to diagram R.
    # Each point can be matched to:
    #   1. Another point (i >= 0, j >= 0): both points exist in both diagrams
    #   2. The diagonal (i >= 0, j == -1): point exists only in L
    #   3. The diagonal (j >= 0, i == -1): point exists only in R
    # ============================================================================
    
    # Edge case: both diagrams empty
    if L.size == 0 and R.size == 0:
        out.update({
            "W": 0.0,
            "pairs": [],
            "L_unmatched": np.zeros((0,2)),
            "R_unmatched": np.zeros((0,2)),
        })
        return out

    # STEP 1: Compute Wasserstein distance and get optimal matching
    # This is the CORE Wasserstein subtraction operation:
    # - wasserstein_distance computes the W_p distance (p=order) between diagrams
    # - matching=True returns the optimal matching: list of (i, j) pairs
    #   where i is index in L, j is index in R (or -1 if matched to diagonal)
    # - The matching minimizes total transport cost between persistence points
    W, matching = wasserstein_distance(L, R, order=order, internal_p=internal_p, matching=True)
    matching = np.asarray(matching, dtype=int)
    
    # STEP 2: Extract matching indices
    # matching is shape (n_matches, 2) where each row is [i, j]
    i_idx = matching[:,0]  # indices into L diagram
    j_idx = matching[:,1]  # indices into R diagram (or -1 for diagonal)
    
    # STEP 3: Categorize matches into three types (vectorized boolean masks)
    both = (i_idx >= 0) & (j_idx >= 0)   # L[i] matched to R[j] (both exist)
    l_only = (i_idx >= 0) & (j_idx == -1) # L[i] matched to diagonal (only in L)
    r_only = (j_idx >= 0) & (i_idx == -1) # R[j] matched to diagonal (only in R)

    # STEP 4: Compute subtraction deltas for matched pairs
    # For pairs where both points exist (both=True), compute:
    #   Δbirth = L[i].birth - R[j].birth
    #   Δdeath = L[i].death - R[j].death
    # This is the "subtraction" of matched persistence points (vectorized, no loops)
    pairs = []
    if np.any(both):
        Li = L[i_idx[both]]  # matched points from L diagram
        Rj = R[j_idx[both]]   # matched points from R diagram
        deltas = Li - Rj      # element-wise subtraction: [Δbirth, Δdeath] for each match
        # Stack indices and deltas: each row is [i, j, Δbirth, Δdeath]
        pairs = np.column_stack([i_idx[both], j_idx[both], deltas[:,0], deltas[:,1]]).tolist()

    # STEP 5: Extract unmatched points (those matched to diagonal)
    # These are persistence points that appear in only one diagram
    L_unmatched = L[i_idx[l_only]] if np.any(l_only) else np.zeros((0,2))  # Points only in L
    R_unmatched = R[j_idx[r_only]] if np.any(r_only) else np.zeros((0,2))  # Points only in R

    out.update({
        "W": float(W),
        "pairs": pairs,
        "L_unmatched": L_unmatched,
        "R_unmatched": R_unmatched,
    })
    return out


# ============================================================================
# EXAMPLE USAGE / MAIN SECTION
# ============================================================================

if __name__ == "__main__":
    # Generate 3D data: (x, y, L) where L is luminosity
    # Points are connected based on closest L 
    N_POINTS = 1000
    data_3d = np.random.uniform(-1, 1, (N_POINTS, 3))  # (x, y, L)
    data_2d = data_3d[:, :2]  # Spatial coordinates (x, y) for plotting 
    luminosity = data_3d[:, 2]  # Luminosity L - used for connectivity
    
    # Compute topology
    print("Computing topology...")
    res = prepare_topology_2d(data_2d, luminosity=luminosity, rmax=None)
    print(f"\nResults (connect by closest L, {N_POINTS} points):")
    print(f"  eps={res['eps']:.5g}")
    print(f"  edges={len(res['edges'])}, tris={len(res['tris'])} (gaps OK)")
    print(f"  left_tris={len(res['left_tris'])}, right_tris={len(res['right_tris'])}")
    print(f"  L (combined H0+H1)={len(res['L'])}, R (combined H0+H1)={len(res['R'])}")
    
    # Extract results
    L = res["L"]
    R = res["R"]
    edges = res["edges"]
    tris = res["tris"]
    eps = res["eps"]
    st = res["st"]
    left_tris = res["left_tris"]
    right_tris = res["right_tris"]
    
    # Plot persistence diagrams
    print("\nPlotting persistence diagrams...")
    plot_pd_gudhi_split(L, R, fname_prefix="pd_left_right", title_prefix="PD — Left vs Right")
    
    # Plot simplicial complexes
    print("\nPlotting simplicial complexes...")
    auto_plot_handedness_2d(data_2d, luminosity=luminosity, rmax=None, max_alpha=None,
                            title_prefix="2D Alpha Complex", out=None, separate_plots=True)
    
    # Compute Wasserstein distance
    print("\nComputing Wasserstein distance...")
    Ln = _finite_np(L)
    Rn = _finite_np(R)
    W = wasserstein_distance(Ln, Rn, order=1, internal_p=1)
    
    # Get triangle info for display
    tri_simplices = [(tuple(s), float(f)) for s, f in st.get_skeleton(2) if len(s) == 3]
    tri_indices = [t for t, _ in tri_simplices]
    
    print({
        "triangles_total": len(tri_indices),
        "left_triangles": len(left_tris),
        "right_triangles": len(right_tris),
        "Wasserstein_W1": float(W),  # Combined H0+H1
    })
    
    # Wasserstein subtraction
    print("\nComputing Wasserstein subtraction...")
    res_combined = subtract_diagrams_combined(L, R, order=1, internal_p=1, tol_decimals=9)
    print("Combined H0+H1 Results (no distinction):")
    print(f"  Wasserstein W1 = {res_combined['W']:.6g}")
    print(f"  Matched pairs = {len(res_combined['pairs'])}")
    print(f"  L-only = {len(res_combined['L_unmatched'])}")
    print(f"  R-only = {len(res_combined['R_unmatched'])}")
    print(f"  Set-style L-only = {len(res_combined.get('L_only_set', []))}")
    print(f"  Set-style R-only = {len(res_combined.get('R_only_set', []))}")
    
    if res_combined["pairs"]:
        pairs_arr = np.array(res_combined["pairs"], dtype=float)
        delta = pairs_arr[:, 2:4]  # Extract last two columns (db, dd)
        print(f"\nDelta (birth, death differences): shape {delta.shape}")
    
    print("\nDone!")

