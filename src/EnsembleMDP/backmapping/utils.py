import os
import mdtraj as md


def prepare_system_dir(system_name, input_dir, output_dir):
    """
    Phase 1: Converts input PDB into a working directory with DCD/PDB topologies.
    """
    pdb_path = os.path.join(input_dir, f"{system_name}.pdb")
    traj = md.load(pdb_path)

    save_dir = os.path.join(output_dir, system_name)
    os.makedirs(save_dir, exist_ok=True)

    traj.save_dcd(os.path.join(save_dir, "traj.dcd"))
    traj[0].save_pdb(os.path.join(save_dir, "topology.pdb"))

    return save_dir
