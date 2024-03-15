import subprocess
from pathlib import Path


def run_prepare_data(final_genotype_data_path: str, output_folder: str) -> Path:
    command = [
        "plink_pipelines",
        "--raw_data_path",
        final_genotype_data_path,
        "--output_folder",
        output_folder,
        "--output_format",
        "disk",
        "--array_chunk_size",
        "1000",
    ]

    subprocess.run(command, check=True)

    return Path(output_folder, "processed/full_inds/full_chrs/encoded_outputs")
