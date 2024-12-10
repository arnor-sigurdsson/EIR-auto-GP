import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
from aislib.misc_utils import ensure_path_exists, get_logger
from eir.setup.input_setup_modules.setup_omics import read_bim
from rich.console import Console
from rich.table import Table

logger = get_logger(name=__name__)


def log_overlap(
    df_bim_prd: pd.DataFrame,
    df_bim_exp: pd.DataFrame,
    output_path: Optional[str | Path] = None,
) -> None:
    logger.info("Analyzing SNP overlap between original training data and new cohort.")

    console = Console()

    exact_overlap = df_bim_prd.merge(
        df_bim_exp, on=["CHR_CODE", "BP_COORD", "ALT", "REF"]
    )

    flipped_overlap = df_bim_prd.merge(
        df_bim_exp,
        left_on=["CHR_CODE", "BP_COORD", "REF", "ALT"],
        right_on=["CHR_CODE", "BP_COORD", "ALT", "REF"],
    )

    total_overlap = pd.concat([exact_overlap, flipped_overlap]).drop_duplicates(
        subset=["CHR_CODE", "BP_COORD"]
    )

    orig_count = len(df_bim_exp)
    new_count = len(df_bim_prd)
    exact_match_count = len(exact_overlap)
    flipped_match_count = len(flipped_overlap)
    total_match_count = len(total_overlap)

    orig_coverage = (total_match_count / orig_count * 100) if orig_count > 0 else 0
    new_coverage = (total_match_count / new_count * 100) if new_count > 0 else 0
    exact_ratio = (
        (exact_match_count / total_match_count * 100) if total_match_count > 0 else 0
    )
    flipped_ratio = (
        (flipped_match_count / total_match_count * 100) if total_match_count > 0 else 0
    )

    missing_orig_snps = orig_count - total_match_count
    unused_new_snps = new_count - total_match_count
    missing_orig_percent = (
        (missing_orig_snps / orig_count * 100) if orig_count > 0 else 0
    )

    coverage_color = (
        "green" if orig_coverage > 75 else "yellow" if orig_coverage > 50 else "red"
    )

    match_quality = (
        "High (>90% SNPs recovered)"
        if orig_coverage > 90
        else (
            "Medium (>75% SNPs recovered)"
            if orig_coverage > 75
            else "Low (<75% SNPs recovered)"
        )
    )

    table = Table(
        title="SNP Overlap Analysis Report",
        title_style="bold blue",
        caption="* Percentage relative to new cohort's total SNP count",
        caption_style="italic",
    )

    table.add_column("Metric", style="dim", width=50)
    table.add_column("Value", style="bold")

    results = {
        "coverage_metrics": {
            "Original Training SNPs Recovered (Main)": f"[{coverage_color}]"
            f"{orig_coverage:.1f}%[/]",
            "New Cohort SNPs Usable (of total)": f"{new_coverage:.1f}%*",
        },
        "matching_details": {
            "Exact REF/ALT Matches": f"{exact_match_count:,} ({exact_ratio:.1f}%)",
            "Flipped REF/ALT Matches": f"{flipped_match_count:,} "
            f"({flipped_ratio:.1f}%)",
            "Total Matching SNPs": f"{total_match_count:,}",
        },
        "dataset_info": {
            "Original Training SNPs": f"{orig_count:,}",
            "New Cohort SNPs": f"{new_count:,}",
        },
        "gap_analysis": {
            "Missing Original SNPs": f"{missing_orig_snps:,} "
            f"({missing_orig_percent:.1f}%)",
            "Unused New Cohort SNPs": f"{unused_new_snps:,}",
        },
        "quality": {
            "Match Type Quality": match_quality,
        },
    }

    for section, metrics in results.items():
        table.add_row(section.replace("_", " ").upper(), "")
        for name, value in metrics.items():
            table.add_row(f"  {name}", value)
        table.add_row("", "")

    console.print("\n")
    console.print(table)
    console.print("\n")

    if output_path:
        try:
            coverage_quality = (
                "High"
                if orig_coverage > 75
                else "Medium" if orig_coverage > 50 else "Low"
            )

            output_text = [
                "SNP Overlap Analysis Results",
                "=========================\n",
                "Coverage Metrics:",
                f"- Original Training SNPs Recovered: {orig_coverage:.1f}%",
                "  * This is the primary metric indicating how many of the original",
                "    training SNPs we can utilize in the new cohort",
                f"- New Cohort SNPs Usable: {new_coverage:.1f}%",
                "  * Percentage of new cohort SNPs that match the training set\n",
                "Matching Details:",
                f"- Total Matching SNPs: {total_match_count:,}",
                f"  * Exact REF/ALT Matches: {exact_match_count:,} "
                f"({exact_ratio:.1f}%)",
                f"  * Flipped REF/ALT Matches: {flipped_match_count:,} "
                f"({flipped_ratio:.1f}%)\n",
                "Dataset Information:",
                f"- Original Training SNPs: {orig_count:,}",
                f"- New Cohort SNPs: {new_count:,}\n",
                "Gap Analysis:",
                f"- Missing Original SNPs: {missing_orig_snps:,}"
                f"({missing_orig_percent:.1f}%)",
                f"- Unused New Cohort SNPs: {unused_new_snps:,}\n",
                "Quality Assessment:",
                f"- Coverage Quality: {coverage_quality}",
                f"- Match Type Quality: {match_quality}",
                "\nNote: Percentages are relative to the original training SNP set",
                "unless otherwise noted.",
            ]

            ensure_path_exists(path=output_path, is_folder=False)
            with open(output_path, "w") as f:
                f.write("\n".join(output_text))

            logger.info(f"Detailed analysis saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results to {output_path}: {str(e)}")


def get_experiment_bim_file(experiment_folder: Path) -> Path:
    bim_file = experiment_folder / "meta" / "snps.bim"
    assert bim_file.exists()

    return bim_file


def read_bim_and_cast_dtypes(bim_file_path: Path | str) -> pd.DataFrame:

    dtypes = {
        "CHR_CODE": str,
        "VAR_ID": str,
        "POS_CM": float,
        "BP_COORD": int,
        "ALT": str,
        "REF": str,
    }

    df_bim = read_bim(bim_file_path=str(bim_file_path))

    df_bim = df_bim.astype(dtypes)

    return df_bim


def validate_executable_exists_in_path(executable: str) -> None:
    if shutil.which(executable) is None:
        msg = (
            f"{executable} is not installed or not in the path. "
            f"Please install {executable} and try again."
        )
        raise ValueError(msg)
