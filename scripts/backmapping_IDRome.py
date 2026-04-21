import os
import argparse
import multiprocessing as mp
from tqdm import tqdm
from EnsembleMDP.backmapping.backmapper import CGBackmapper
from EnsembleMDP.backmapping.utils import prepare_system_dir


def run_backmapping(system, output_dir, model, device, batch_size, num_proc):
    """Orchestrates the class instantiation and execution for a specific system."""
    save_dir = os.path.join(output_dir, system)

    backmapper = CGBackmapper(
        top_file=os.path.join(save_dir, "topology.pdb"),
        traj_file=os.path.join(save_dir, "traj.dcd"),
        cg_model=model,
        device=device,
        cg2all_batch_size=batch_size,
        cg2all_nb_proc=num_proc,
    )

    print(f"\n--- Backmapping: {system} ---")
    backmapper.run(preprocess=True, out_format="dcd")


def main():
    parser = argparse.ArgumentParser(description="EnsembleMDP Backmapping CLI")
    parser.add_argument("--input_dir", "-i", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    parser.add_argument("--model", "-m", type=str, default="CalphaBasedModel")
    parser.add_argument("--num_proc", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Directory Preparation (Parallel)
    existing_systems = [
        f.replace(".pdb", "") for f in os.listdir(args.input_dir) if f.endswith(".pdb")
    ]
    print("Step 1: Preparing system directories...")

    with mp.Pool(processes=os.cpu_count()) as pool:
        # Using a helper to pass multiple args to prepare_system_dir
        prepare_tasks = [(s, args.input_dir, args.output_dir) for s in existing_systems]
        list(
            tqdm(
                pool.starmap(prepare_system_dir, prepare_tasks),
                total=len(existing_systems),
            )
        )

    # 2. Backmapping (Sequential reconstruction per system)
    # Note: cg2all is internally parallel, so we loop systems sequentially to avoid OOM or CPU thrashing.
    print("\nStep 2: Reconstructing all-atom trajectories...")
    system_names = [
        f
        for f in os.listdir(args.output_dir)
        if os.path.isdir(os.path.join(args.output_dir, f))
    ]

    for system in system_names:
        run_backmapping(
            system,
            args.output_dir,
            args.model,
            args.device,
            args.batch_size,
            args.num_proc,
        )


if __name__ == "__main__":
    main()
