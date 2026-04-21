# EnsembleMDP (Ensemble-based Molecular Dynamics Properties)

**EnsembleMDP** is a high-level analysis framework designed to extract structural and dynamical fingerprints from protein conformational ensembles, specifically optimized for the characterization of folded and intrinsically disordered proteins (IDPs).

---

## Quick Start: How to Use It

The package uses the `ProteinAnalyzer` object as the unified entry point. It handles file validation, topology correction, and frame subsampling automatically.

### Phase 1: Backmapping (Optional)
If starting from coarse-grained data (e.g., **CALVADOS**, **Martini**, or generative models like **IDPFold2**), use the backmapper to reconstruct all-atom detail.

```python
from EnsembleMDP.backmapping.backmapper import CGBackmapper

backmapper = CGBackmapper(top_file="cg.pdb", traj_file="cg.xtc", cg_model="CalphaBasedModel")
backmapper.run(preprocess=True) # Outputs aa_topology.pdb and aa_traj.dcd
```

### Phase 2: Analysis
Load the all-atom data into the analyzer to compute the property suite.

```python
from pathlib import Path
from EnsembleMDP.analysis import ProteinAnalyzer

# Initialize entry point
analyzer = ProteinAnalyzer(pdb_path=Path("aa_topology.pdb"), xtc_path=Path("aa_traj.dcd"))

# Compute all properties (Global, Local, and Pairwise)
results = analyzer.compute_all(contact_cutoff=8.0)
```

---

## Computed Properties

Properties are classified into three levels of resolution: **Global**, **Local**, and **Pairwise**.

### 1. Global Properties
These metrics describe the overall dimensions, shape, and polymer-scaling behavior of the entire protein ensemble. For per-frame metrics, the library reports the mean and standard deviation across the trajectory.

* **Size Descriptors:** Includes the **Radius of Gyration ($R_g$)**, **End-to-End Distance ($R_{ee}$)**, and **Maximum Diameter**.
* **Shape Descriptors:** Derived from the eigenvalues ($\lambda_i$) of the gyration tensor:
    * **Asphericity ($b$):** Measures deviation from a spherical shape.
    * **Prolateness ($S$):** Distinguishes between elongated (prolate) and disk-like (oblate) shapes.
    * **Relative Shape Anisotropy ($\kappa^2$):** A robust measure of overall asymmetry.
* **Scaling Exponent ($\nu$):** Characterizes solvent quality via the power law relationship $R_g \approx R_0 L^\nu$.
    * **$\nu \approx 0.33$:** Folded/Globular state.
    * **$\nu \approx 0.588$:** Disordered/Expanded state (Self-Avoiding Walk).

---

### 2. Local Properties
Residue-specific metrics that provide a "track" of properties along the sequence ($L$).

* **Secondary Structure Propensity:** The statistical likelihood of each residue to occupy one of the 8 standard DSSP states (H, G, I, E, B, T, S, C).
* **Dihedral Entropy ($S_{dih}$):** Quantifies backbone flexibility by analyzing the joint probability distribution of the **$\phi$ (phi)** and **$\psi$ (psi)** torsion angles.
* **Solvent Accessible Surface Area (SASA):** Measures residue exposure. The library computes both the mean and the variance of SASA to identify "gatekeeper" residues that toggle between buried and exposed states.

---

### 3. Pairwise Properties
Matrix-based metrics ($L \times L$) that characterize the spatial and dynamical coupling between pairs of residues.

* **Distance Fluctuation ($CP_{ij}$):** Also known as **Communication Propensity**. It measures the stability of the distance between residues. Low fluctuation indicates high mechanical coupling, identifying potential allosteric signaling pathways.
    $$\mathrm{CP}_{ij} = \sqrt{\langle (d_{ij} - \langle d_{ij} \rangle)^2 \rangle}$$
* **Dynamic Cross-Correlation (DCCM):** Represents the correlated displacements of atoms across the trajectory.
* **Contact Map Frequency:** The probability $P_{ij}$ that two residues are within a specific spatial cutoff (default $8$ Å).

---

## Property Summary Table

| Category | Property Name | Shape/Unit | Description |
| :--- | :--- | :--- | :--- |
| **Global** | $R_g, R_{ee}$, Max Diam | Mean & Std | Overall dimensions and compactness |
| **Global** | $b, S, \kappa^2$ | Mean & Std | Symmetry and shape classification |
| **Global** | Scaling Exponent ($\nu$) | Scalar | Solvent quality and ensemble state |
| **Local** | Secondary Structure | $(L, 8)$ | Propensity for each of the 8 DSSP states |
| **Local** | Dihedral Entropy | $(L, 2)$ | Flexibility of $\phi$ and $\psi$ backbone angles |
| **Local** | SASA | $(L, 2)$ | Residue exposure (Mean and Variance) |
| **Pairwise** | Distance Fluctuation | $(L, L)$ | Mechanical coupling and communication |
| **Pairwise** | DCCM | $(L, L)$ | Correlated atomic motions |
| **Pairwise** | Contact Frequency | $(L, L)$ | Probability of spatial proximity |

> **Note:** For IDPs, the **Standard Deviation** of global properties is often as critical as the **Mean**, as it quantifies the structural heterogeneity of the ensemble. `EnsembleMDP` captures both by default.