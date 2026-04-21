import numpy as np
from MDAnalysis.analysis import distances
from prody import Ensemble
from prody.measure import superpose


def compute_distance_fluctuations(md_analysis_u, protein_atoms):
    """Identifies rigid vs flexible pairs via SD of distance matrix."""
    ca = protein_atoms.select_atoms("name CA")
    n_ca = len(ca)
    sum_sq_dist = np.zeros((n_ca, n_ca))
    sum_dist = np.zeros((n_ca, n_ca))
    n_frames = len(md_analysis_u.trajectory)

    for ts in md_analysis_u.trajectory:
        d = distances.distance_array(ca.positions, ca.positions)
        sum_dist += d
        sum_sq_dist += d**2

    # Variance = E[X^2] - (E[X])^2
    mean_dist = sum_dist / n_frames
    variance = (sum_sq_dist / n_frames) - (mean_dist**2)
    return np.sqrt(np.maximum(variance, 0))  # Return Standard Deviation


def compute_dccm(md_analysis_u, protein_atoms):
    ca = protein_atoms.select_atoms("name CA")
    n_frames = len(md_analysis_u.trajectory)
    n_atoms = len(ca)

    # 1. Coordinate Extraction
    coords = np.zeros((n_frames, n_atoms, 3))
    for i, ts in enumerate(md_analysis_u.trajectory):
        coords[i] = ca.positions

    # 2. Alignment via ProDy
    ensemble = Ensemble("Protein")
    ensemble.setCoords(coords[0])  # Reference is frame 0
    ensemble.addCoordset(coords)
    superpose(ensemble, ensemble)
    aligned_coords = ensemble.getCoordsets()

    # 3. Fluctuation Calculation
    mean_coords = aligned_coords.mean(axis=0)
    fluctuations = aligned_coords - mean_coords

    # 4. Vectorized DCCM (The Fast Way)
    # Resulting shape: (n_atoms, n_atoms)
    dot_products = np.einsum("fai,fbi->ab", fluctuations, fluctuations) / n_frames

    # 5. Normalization (C_ij = <ΔRi·ΔRj> / sqrt(<ΔRi²><ΔRj²>))
    variances = np.diag(dot_products)
    # Using outer product to create the denominator matrix
    norm_matrix = np.sqrt(np.outer(variances, variances)) + 1e-12
    dccm = dot_products / norm_matrix

    return dccm


def compute_contact_map(md_analysis_u, protein_atoms, cutoff=8.0):
    """Calculates average contact frequency matrix."""
    ca = protein_atoms.select_atoms("name CA")
    n_res = len(ca)
    contact_sum = np.zeros((n_res, n_res))

    for ts in md_analysis_u.trajectory:
        d = distances.distance_array(ca.positions, ca.positions)
        contact_sum += (d < cutoff).astype(int)

    return contact_sum / len(md_analysis_u.trajectory)
