import h5py
from pathlib import Path
import numpy as np


def save_all_properties(results, output_file: Path):
    with h5py.File(output_file, "w") as hf:
        for key, value in results.items():
            # Convert lists/tuples to numpy arrays for HDF5 compatibility
            if isinstance(value, (list, tuple)):
                value = np.array(value)

            if isinstance(value, np.ndarray) and value.ndim > 0:
                hf.create_dataset(
                    key,
                    data=value,
                    dtype="float32",
                    compression="gzip",
                    compression_opts=4,  # Ranges from 0-9
                )
            else:
                # Scalars cannot be compressed in HDF5
                hf.create_dataset(key, data=value)

    print(f"Successfully saved data to {output_file}")


def load_all_properties(filename: Path):
    """
    Loads properties from an HDF5 file and returns them as a dictionary.
    """
    results = {}

    if not filename.exists():
        raise FileNotFoundError(f"No file found at {filename}")

    with h5py.File(filename, "r") as hf:
        for key in hf.keys():
            # .[()] or .flatten() is the standard way to load
            # the dataset content into a numpy array
            results[key] = hf[key][()]

    return results
