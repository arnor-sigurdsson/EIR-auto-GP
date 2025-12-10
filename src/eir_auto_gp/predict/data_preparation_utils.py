from pathlib import Path

import pandas as pd
from aislib.misc_utils import ensure_path_exists, get_logger
from eir.setup.input_setup_modules.setup_omics import read_bim
from rich.console import Console
from rich.table import Table

logger = get_logger(name=__name__)


def log_overlap(
    df_bim_prd: pd.DataFrame,
    df_bim_exp: pd.DataFrame,
    output_path: str | Path | None = None,
) -> None:
    logger.info("Analyzing SNP overlap including strand/ambiguity checks...")

    console = Console()

    def get_comp(s):
        return s.astype(str).str.translate(str.maketrans("ATGC", "TACG"))

    df_prd = df_bim_prd.copy()
    df_exp = df_bim_exp.copy()

    on_cols = ["CHR_CODE", "BP_COORD"]
    merged = pd.merge(df_prd, df_exp, on=on_cols, suffixes=("_new", "_ref"))

    is_palindromic = (
        ((merged["REF_new"] == "A") & (merged["ALT_new"] == "T"))
        | ((merged["REF_new"] == "T") & (merged["ALT_new"] == "A"))
        | ((merged["REF_new"] == "C") & (merged["ALT_new"] == "G"))
        | ((merged["REF_new"] == "G") & (merged["ALT_new"] == "C"))
    )

    ref_ref_comp = get_comp(merged["REF_ref"])
    alt_ref_comp = get_comp(merged["ALT_ref"])

    is_exact = (merged["REF_new"] == merged["REF_ref"]) & (
        merged["ALT_new"] == merged["ALT_ref"]
    )

    is_reversal = (merged["REF_new"] == merged["ALT_ref"]) & (
        merged["ALT_new"] == merged["REF_ref"]
    )

    is_strand = (merged["REF_new"] == ref_ref_comp) & (
        merged["ALT_new"] == alt_ref_comp
    )

    is_strand_rev = (merged["REF_new"] == alt_ref_comp) & (
        merged["ALT_new"] == ref_ref_comp
    )

    valid_mask = ~is_palindromic

    count_exact = (is_exact & valid_mask).sum()
    count_reversal = (is_reversal & valid_mask).sum()
    count_strand = (is_strand & valid_mask).sum()
    count_strand_rev = (is_strand_rev & valid_mask).sum()
    count_ambiguous = is_palindromic.sum()

    total_matches = count_exact + count_reversal + count_strand + count_strand_rev

    matched_mask = is_exact | is_reversal | is_strand | is_strand_rev
    count_mismatch = (valid_mask & ~matched_mask).sum()

    orig_count = len(df_exp)
    new_count = len(df_prd)

    orig_coverage = (total_matches / orig_count * 100) if orig_count > 0 else 0
    new_coverage = (total_matches / new_count * 100) if new_count > 0 else 0
    exact_ratio = (count_exact / total_matches * 100) if total_matches > 0 else 0
    reversal_ratio = (count_reversal / total_matches * 100) if total_matches > 0 else 0
    strand_ratio = (count_strand / total_matches * 100) if total_matches > 0 else 0
    strand_rev_ratio = (
        (count_strand_rev / total_matches * 100) if total_matches > 0 else 0
    )

    missing_orig_snps = orig_count - total_matches
    unused_new_snps = new_count - total_matches
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
            "Total Matching SNPs": f"{total_matches:,}",
            "  - Exact Matches (A/G -> A/G)": f"{count_exact:,} ({exact_ratio:.1f}%)",
            "  - Allele Reversals (A/G -> G/A)": f"{count_reversal:,} "
            f"({reversal_ratio:.1f}%)",
            "  - Strand Flips (A/G -> T/C)": f"{count_strand:,} ({strand_ratio:.1f}%)",
            "  - Strand Flip + Reversal (A/G -> C/T)": f"{count_strand_rev:,} "
            f"({strand_rev_ratio:.1f}%)",
        },
        "dataset_info": {
            "Original Training SNPs": f"{orig_count:,}",
            "New Cohort SNPs": f"{new_count:,}",
            "Positional Overlaps (before filtering)": f"{len(merged):,}",
        },
        "excluded_snps": {
            "Ambiguous/Palindromic SNPs (A/T, C/G)": f"[red]{count_ambiguous:,}[/]",
            "Allele Mismatches (different variants)": f"[red]{count_mismatch:,}[/]",
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
                else "Medium"
                if orig_coverage > 50
                else "Low"
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
                f"- Total Matching SNPs: {total_matches:,}",
                f"  * Exact Matches (A/G -> A/G): {count_exact:,} ({exact_ratio:.1f}%)",
                f"  * Allele Reversals (A/G -> G/A): {count_reversal:,} "
                f"({reversal_ratio:.1f}%)",
                f"  * Strand Flips (A/G -> T/C): "
                f"{count_strand:,} ({strand_ratio:.1f}%)",
                f"  * Strand Flip + Reversal (A/G -> C/T): {count_strand_rev:,} "
                f"({strand_rev_ratio:.1f}%)\n",
                "Dataset Information:",
                f"- Original Training SNPs: {orig_count:,}",
                f"- New Cohort SNPs: {new_count:,}",
                f"- Positional Overlaps (before filtering): {len(merged):,}\n",
                "Excluded SNPs:",
                f"- Ambiguous/Palindromic SNPs (A/T, C/G): {count_ambiguous:,}",
                f"- Allele Mismatches (different variants): {count_mismatch:,}\n",
                "Gap Analysis:",
                f"- Missing Original SNPs: {missing_orig_snps:,} "
                f"({missing_orig_percent:.1f}%)",
                f"- Unused New Cohort SNPs: {unused_new_snps:,}\n",
                "Quality Assessment:",
                f"- Coverage Quality: {coverage_quality}",
                f"- Match Type Quality: {match_quality}",
                "\nNote: Percentages in matching details are"
                " relative to total matches.",
                "Coverage percentages are relative to the original training SNP set",
                "unless otherwise noted.",
            ]

            ensure_path_exists(path=Path(output_path), is_folder=False)
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
