import MDAnalysis
from MDAnalysis.analysis import distances
import numpy as np
from scipy.spatial.distance import pdist
import mdtraj as md
from scipy.stats import linregress


def compute_end_to_end_distance(
    md_analysis_u: MDAnalysis.Universe, residues: MDAnalysis.ResidueGroup
):
    """
    Calculates the distance between the CA atoms of the
    first and last residues across the trajectory.
    """
    # Inside compute_end_to_end_distance
    all_ca = md_analysis_u.select_atoms("name CA")

    start_ca = all_ca[0:1]
    end_ca = all_ca[-1:]

    # Ensure single atom selections
    assert (
        len(start_ca) == 1
    ), f"Start residue CA atom not found. Available atoms names: {residues[0].atoms.names}"
    assert (
        len(end_ca) == 1
    ), f"End residue CA atom not found.Available atoms names: {residues[0].atoms.names}"

    all_distances = []
    for ts in md_analysis_u.trajectory:
        resA, resB, dist = distances.dist(start_ca, end_ca)
        all_distances.append(dist)

    return np.array(all_distances)


def compute_maximum_diameter(md_analysis_u):
    """
    Calculates the maximum distance between any two CA atoms
    (Dmax) for each frame in the trajectory.
    """
    # 1. Pre-select CA atoms to avoid overhead inside the loop
    ca_atoms = md_analysis_u.select_atoms("name CA")

    all_dmax = []

    # 2. Iterate through trajectory
    for ts in md_analysis_u.trajectory:
        # Get the coordinates for the current frame
        coords = ca_atoms.positions

        # pdist computes the distance between every pair of atoms (n*(n-1)/2 distances)
        # np.max gives us the largest of those distances
        dmax = np.max(pdist(coords))
        all_dmax.append(dmax)

    return np.array(all_dmax)


def compute_gyration_tensor_properties(md_traj):
    results = {
        "gyration_eigenvalues_l1": [],
        "gyration_eigenvalues_l2": [],
        "gyration_eigenvalues_l3": [],
        "gyration_l1_per_l2": [],
        "gyration_l1_per_l3": [],
        "gyration_l2_per_l3": [],
        "radius_of_gyration": [],
        "asphericity": [],
        "normalized_acylindricity": [],
        "rel_shape_anisotropy": [],
        "prolateness": [],
    }

    protein_traj = md_traj.atom_slice(md_traj.topology.select("protein"))

    eigvals = md.principal_moments(protein_traj)

    for one_set_of_eigvals in eigvals:
        l1, l2, l3 = one_set_of_eigvals
        results["gyration_eigenvalues_l1"].append(l1)
        results["gyration_eigenvalues_l2"].append(l2)
        results["gyration_eigenvalues_l3"].append(l3)
        results["gyration_l1_per_l2"].append(l1 / l2 if l2 > 0 else 0)
        results["gyration_l1_per_l3"].append(l1 / l3 if l3 > 0 else 0)
        results["gyration_l2_per_l3"].append(l2 / l3 if l3 > 0 else 0)

        rg2 = (
            l1 + l2 + l3
        )  # The square of the radius of gyration and also the trace of the matrix
        rg = np.sqrt(rg2)
        results["radius_of_gyration"].append(rg)

        asphericity = ((l1 - l2) ** 2 + (l1 - l3) ** 2 + (l2 - l3) ** 2) / (
            2 * (rg2**2)
        )
        results["asphericity"].append(asphericity)

        numerator_p = (2 * l1 - l2 - l3) * (2 * l2 - l1 - l3) * (2 * l3 - l1 - l2)
        denominator_p = 2 * (l1**2 + l2**2 + l3**2 - l1 * l2 - l1 * l3 - l2 * l3) ** 1.5
        prolateness = numerator_p / denominator_p
        results["prolateness"].append(prolateness)

        normalized_acylindricity = (l2 - l3) / rg2
        results["normalized_acylindricity"].append(normalized_acylindricity)

        # Assuming l1, l2, l3 are predefined numeric values
        numerator_kappa = 3 * (l1 * l2 + l2 * l3 + l3 * l1)
        denominator_kappa = (l1 + l2 + l3) ** 2
        rel_shape_anisotropy = 1 - (numerator_kappa / denominator_kappa)
        results["rel_shape_anisotropy"].append(rel_shape_anisotropy)

    return {key: np.array(value) for key, value in results.items()}


def compute_scaling_exponent(md_traj, min_s=5, max_s_fraction=0.4):
    ca_mask = md_traj.topology.select("name CA")
    traj_ca = md_traj.atom_slice(ca_mask)
    N = len(ca_mask)

    pairs = np.array(
        [[i, j] for i in range(N) for j in range(i + min_s, int(N * max_s_fraction))]
    )
    separations = pairs[:, 1] - pairs[:, 0]

    dists_sq = md.compute_distances(traj_ca, pairs, periodic=True) ** 2
    r2_mean = dists_sq.mean(0)  # <<R²(s)>>

    unique_seps, r2_binned = [], []
    for s in np.unique(separations):
        mask = separations == s
        if np.sum(mask) > 3:
            unique_seps.append(s)
            r2_binned.append(r2_mean[mask].mean())

    unique_seps, r2_binned = np.array(unique_seps), np.array(r2_binned)
    log_s = np.log(unique_seps)
    log_r = 0.5 * np.log(r2_binned)  # log(R_rms(s))

    valid = np.isfinite(log_r) & (unique_seps < N * 0.3)

    slope, _, r_value, _, _ = linregress(log_s[valid], log_r[valid])
    return slope
