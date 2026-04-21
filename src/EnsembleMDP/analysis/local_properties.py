import mdtraj as md
import numpy as np
import logging
from scipy.stats import circmean, circstd


def remove_problematic_frames(traj, frame_idx):
    """
    Removes specific frames from an md.Trajectory object.

    Parameters:
    - traj: The md.Trajectory object.
    - frame_idx: An integer or a list/set of integers representing the frames to remove.

    Returns:
    - A new md.Trajectory object without the problematic frames.
    """
    # Ensure frame_idx is a set for fast lookup and to handle single integers
    if isinstance(frame_idx, (int, np.integer)):
        to_remove = {frame_idx}
    else:
        to_remove = set(frame_idx)

    # Create a list of all frame indices EXCEPT those in the remove set
    all_indices = np.arange(traj.n_frames)
    keep_indices = [i for i in all_indices if i not in to_remove]

    if not keep_indices:
        print(f"Warning: Removing frames {to_remove} left the trajectory empty!")
        return None

    # Use MDTraj's built-in slicing to create the new trajectory
    # This is highly optimized and preserves topology
    clean_traj = traj[keep_indices]

    return clean_traj


def compute_secondary_structure_propensities(md_traj):
    """
    Computes the SS8 (DSSP) propensity of each secondary structure element
    per residue over the trajectory using MDTraj.
    """
    traj = md_traj.atom_slice(md_traj.top.select("protein"))

    # 2. Compute DSSP with full 8-class output
    # Output shape: (n_frames, n_residues)
    dssp = md.compute_dssp(traj, simplified=False)

    # 3. Map DSSP symbols to SS8 labels (space -> C)
    mapping = {
        "H": "H",  # alpha helix
        "G": "G",  # 3-10 helix
        "I": "I",  # pi helix
        "E": "E",  # beta strand
        "B": "B",  # beta bridge
        "T": "T",  # turn
        "S": "S",  # bend
        " ": "C",  # coil
    }

    ss8_labels = ["H", "G", "I", "E", "B", "T", "S", "C"]
    n_frames, n_residues = dssp.shape

    # 4. Compute per-residue propensities
    propensities = {k: np.zeros(n_residues) for k in ss8_labels}

    for f in range(n_frames):
        for i, code in enumerate(dssp[f]):
            propensities[mapping[code]][i] += 1

    # Normalize by number of frames
    for k in propensities:
        propensities[k] /= n_frames

    # 5. Sanity check on residue indexing (as in your original code)
    resids = np.array([res.resSeq for res in traj.topology.residues])
    assert np.all(
        np.diff(resids) > 0
    ), "Residues are not in order or contain duplicates!"
    assert np.all(np.diff(resids) == 1), f"Gap detected in residue sequence: {resids}"

    return {f"ss_propensity_{k}": v for k, v in propensities.items()}


def compute_dihedral_distribution(md_traj):
    """
    Helper: Computes dense Phi and Psi arrays aligned to residue indices.
    Returns Dictionary with 'phi' and 'psi' arrays of shape (n_frames, n_residues).
    """
    n_frames = md_traj.n_frames
    n_res = md_traj.n_residues

    # 1. Compute raw dihedrals (returns indices and angles)
    # indices: (n_angles, 4), angles: (n_frames, n_angles)
    phi_ind, phi_vals = md.compute_phi(md_traj)
    psi_ind, psi_vals = md.compute_psi(md_traj)

    # 2. Prepare dense arrays filled with NaNs
    phi_dense = np.full((n_frames, n_res), np.nan)
    psi_dense = np.full((n_frames, n_res), np.nan)

    # 3. Map Phi angles to correct residue index
    # Phi is defined by C(i-1)-N(i)-CA(i)-C(i). The 'residue' is index i.
    # MDTraj returns indices for atoms. We grab the residue of the 2nd atom (Nitrogen) or 3rd (CA).
    for col_idx, atoms in enumerate(phi_ind):
        # atoms[2] is the Alpha Carbon of residue i
        res_idx = md_traj.topology.atom(atoms[2]).residue.index
        phi_dense[:, res_idx] = phi_vals[:, col_idx]

    # 4. Map Psi angles to correct residue index
    # Psi is defined by N(i)-CA(i)-C(i)-N(i+1). The 'residue' is index i.
    for col_idx, atoms in enumerate(psi_ind):
        # atoms[1] is the Alpha Carbon of residue i
        res_idx = md_traj.topology.atom(atoms[1]).residue.index
        psi_dense[:, res_idx] = psi_vals[:, col_idx]

    return {"phi": phi_dense, "psi": psi_dense}


def compute_dihedral_sequence_tracks(md_traj, bins=60):
    """
    Computes entropy, circular mean, and circular std for phi/psi.
    Returns a dictionary of sequence tracks (1D arrays of length n_res).
    """
    # Use helper to get dense arrays
    results = compute_dihedral_distribution(md_traj)
    phi_array = results["phi"]
    psi_array = results["psi"]

    n_frames, n_res = phi_array.shape

    # Initialize containers
    tracks = {
        "phi_mean": [],
        "phi_std": [],
        "phi_entropy": [],
        "psi_mean": [],
        "psi_std": [],
        "psi_entropy": [],
    }

    # Loop over every residue column
    for i in range(n_res):
        for name, data in [("phi", phi_array[:, i]), ("psi", psi_array[:, i])]:
            # Remove NaNs for calculation (e.g., N-term Phi / C-term Psi)
            valid_data = data[~np.isnan(data)]

            if len(valid_data) == 0:
                # If no data (terminals), append NaN
                tracks[f"{name}_mean"].append(np.nan)
                tracks[f"{name}_std"].append(np.nan)
                tracks[f"{name}_entropy"].append(np.nan)
                continue

            # 1. Circular Mean & Std
            c_mean = circmean(valid_data, low=-np.pi, high=np.pi)
            c_std = circstd(valid_data, low=-np.pi, high=np.pi)

            # 2. Entropy
            counts, _ = np.histogram(valid_data, bins=bins, range=[-np.pi, np.pi])
            p = counts / len(valid_data)  # Normalize by valid frames
            p = p[p > 0]
            entropy = -np.sum(p * np.log(p))

            tracks[f"{name}_mean"].append(c_mean)
            tracks[f"{name}_std"].append(c_std)
            tracks[f"{name}_entropy"].append(entropy)

    # Convert lists to numpy arrays
    return {k: np.array(v) for k, v in tracks.items()}


def compute_residue_sasa(md_traj, n_sphere_points):
    """
    Returns a 1D numpy array of time-averaged SASA values
    in the order of the protein sequence.
    """
    # MaxASA values in nm^2 (Tien et al. 2013)
    TIEN_MAX_SASA = {
        "ALA": 1.21,
        "ARG": 2.48,
        "ASN": 1.87,
        "ASP": 1.87,
        "CYS": 1.48,
        "GLN": 2.14,
        "GLU": 2.14,
        "GLY": 0.97,
        "HIS": 2.16,
        "ILE": 1.95,
        "LEU": 1.91,
        "LYS": 2.30,
        "MET": 2.03,
        "PHE": 2.28,
        "PRO": 1.54,
        "SER": 1.43,
        "THR": 1.63,
        "TRP": 2.64,
        "TYR": 2.55,
        "VAL": 1.65,
    }

    bad_frames = check_trajectory_correctness(md_traj)
    if (
        bad_frames
    ):  # SASA will crash if two atoms are superposed and this can sometimes happen in some frame because of cg2all backmapping approximation
        logging.debug("Some overlapping atoms detected, removing problematic frames")
        md_traj = remove_problematic_frames(md_traj, bad_frames)
        logging.debug(f"{md_traj.n_frames} remaining frames")

    # Select only the protein (standard practice to avoid membrane/solvent interference)
    protein_indices = md_traj.topology.select("protein")
    t_prot = md_traj.atom_slice(protein_indices)

    # Compute SASA per residue
    # Returns shape (n_frames, n_residues)
    sasa_per_frame_per_res = md.shrake_rupley(
        t_prot, n_sphere_points=n_sphere_points, mode="residue"
    )
    logging.debug("Shrake Rupley algorithm finished")

    # Calculate Mean (Average) and Std (Fluctuations) for both metrics
    # Absolute SASA (nm^2)
    avg_abs_sasa = np.mean(sasa_per_frame_per_res, axis=0)
    std_abs_sasa = np.std(sasa_per_frame_per_res, axis=0)

    # Compute Relative Solvent Accessibility (RSA)
    residue_names = [res.name[:3].upper() for res in t_prot.topology.residues]
    max_vals = np.array([TIEN_MAX_SASA.get(name, 1.0) for name in residue_names])

    # This divides every frame's SASA by the MaxSASA for that residue type
    rsa_per_frame = sasa_per_frame_per_res / max_vals

    # Relative SASA (RSA - normalized 0 to 1)
    avg_rel_sasa = np.mean(rsa_per_frame, axis=0)
    std_rel_sasa = np.std(rsa_per_frame, axis=0)

    return {
        "sasa_abs_mean": avg_abs_sasa,  # Physical area (nm^2)
        "sasa_abs_std": std_abs_sasa,  # Area fluctuations
        "sasa_rel_mean": avg_rel_sasa,  # Normalized exposure (0-1)
        "sasa_rel_std": std_rel_sasa,  # Relative flexibility
    }


def check_trajectory_correctness(traj, distance_threshold=0.0005):
    """
    Checks all frames for common issues that cause SASA algorithms to hang.

    Parameters:
    - traj: md.Trajectory object
    - distance_threshold: Minimum allowed distance between atoms in nm (default 0.1 Angstrom)

    Returns:
    - list of problematic frame indices
    """
    problematic_frames = []
    n_atoms = traj.n_atoms

    logging.info(f"Checking {traj.n_frames} frames for {n_atoms} atoms...")

    for i in range(traj.n_frames):
        coords = traj.xyz[i]

        # 1. Check for NaNs or Infinities
        if not np.isfinite(coords).all():
            print(f"Frame {i}: Contains NaN or Inf coordinates.")
            problematic_frames.append(i)
            continue

        # 2. Check for Extreme Coordinates (Spatial Bloat)
        # If coordinates are > 10,000nm, neighbor searches often overflow
        if np.max(np.abs(coords)) > 10000:
            print(f"Frame {i}: Extreme coordinate values detected (>10,000nm).")
            problematic_frames.append(i)
            continue

        # 3. Check for Atom Overlap (Singularities)
        # We use a fast bounding-box check before doing expensive pdist
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)

        # If all atoms are in the exact same spot (collapse)
        if np.allclose(min_coords, max_coords):
            print(f"Frame {i}: All atoms have collapsed to a single point.")
            problematic_frames.append(i)
            continue

        # Optional: More intensive check for any two atoms sharing a coordinate
        # This can be slow for 28k atoms, so we check just the first 1000 atoms
        # as a proxy, or use a spatial KDTree for speed.
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(coords)
            # Find pairs within distance_threshold
            pairs = tree.query_pairs(r=distance_threshold)
            if pairs:
                print(f"Frame {i}: Detected {len(pairs)} overlapping atom pairs.")
                problematic_frames.append(i)
                continue
        except ImportError:
            pass

    return problematic_frames
