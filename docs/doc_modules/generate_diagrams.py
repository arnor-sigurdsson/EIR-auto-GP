import os
import subprocess
import tempfile
from pathlib import Path

from aislib.misc_utils import get_logger

logger = get_logger(name=__name__)


def generate_all():
    output_root = Path("docs/source/_static/diagrams/feature_selection_methods")

    generate_gwas_method_diagram(
        output_file=str(output_root / "gwas_method_diagram.png")
    )
    generate_dl_method_diagram(output_file=str(output_root / "dl_method_diagram.png"))
    generate_gwas_then_dl_method_diagram(
        output_file=str(output_root / "gwas_then_dl_method_diagram.png")
    )
    generate_dl_gwas_method_diagram(
        output_file=str(output_root / "dl_gwas_method_diagram.png")
    )
    generate_gwas_bo_method_diagram(
        output_file=str(output_root / "gwas_bo_method_diagram.png")
    )
    generate_none_method_diagram(
        output_file=str(output_root / "none_method_diagram.png")
    )


def generate_mermaid_diagram(mermaid_code: str, output_file: str):
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".mmd"
    ) as temp_file:
        temp_file.write(mermaid_code)
        temp_file_path = temp_file.name

    try:
        subprocess.run(
            ["mmdc", "-i", temp_file_path, "-o", output_file],
            check=True,
        )
        logger.info(f"Diagram saved to {output_file}")
    except subprocess.CalledProcessError as e:
        logger.error(f"An error occurred: {e}")
    finally:
        os.remove(temp_file_path)


def generate_dl_method_diagram(output_file: str):
    mermaid_code = """
        graph TD
            subgraph "Initial Training Phase"
                A[Start] --> B[Train on full SNP set]
                B --> C{"Evaluate N times (Initial Training Runs)"}
                C --> D[Compute SNP Attributions]
            end

            subgraph "Bayesian Optimization Loop"
                E[SNP Attributions] --> F["Manually Prime Optimizer X times (Manual Priming Runs)"]
                F --> G{Enter Optimization Runs?}
                G -->|Yes| H[Optimizer]
                H -->|Ask for SNP Fraction| I["Bayesian Optimization (Optimization Runs)"]
                I -->|Tell Results| H
                I --> J[Select optimal SNP count]
                J --> K[Train model with selected SNPs]
                K --> L[Trained Model Validation Performance]
                L --> M{"Total Runs (N + X + Optimization Runs) == --folds?"}
                M -->|Yes| N[End]
                M -->|No| O[Feedback]
                O -->|Validation Performance and SNP Attributions| I
            end

            D --> E
    """
    generate_mermaid_diagram(mermaid_code=mermaid_code, output_file=output_file)


def generate_gwas_method_diagram(output_file: str):
    mermaid_code = """
        graph TD
            subgraph "GWAS Filtering"
                A[Start] --> B[Perform GWAS]
                B --> C[Filter SNPs based on p-value]
            end

            subgraph "Model Training Phase"
                C --> D{"Train model on filtered SNPs for --folds times"}
                D --> E[End]
            end
    """
    generate_mermaid_diagram(mermaid_code=mermaid_code, output_file=output_file)


def generate_gwas_then_dl_method_diagram(output_file: str):
    mermaid_code = """
        graph TD
            subgraph "GWAS Pre-filtering"
                A[Start] --> B[GWAS based on p-value]
                B --> C[Filtered SNP set]
            end

            subgraph "Initial Training Phase"
                C --> D[Train on filtered SNP set]
                D --> E{"Evaluate N times (Initial Training Runs)"}
                E --> F[Compute SNP Attributions]
            end

            subgraph "Bayesian Optimization Loop"
                G[SNP Attributions] --> H["Manually Prime Optimizer X times (Manual Priming Runs)"]
                H --> I{Enter Optimization Runs?}
                I -->|Yes| J[Optimizer]
                J -->|Ask for SNP Fraction| K["Bayesian Optimization (Optimization Runs)"]
                K -->|Tell Results| J
                K --> L[Select optimal SNP count]
                L --> M[Train model with selected SNPs]
                M --> N[Trained Model Validation Performance]
                N --> O{"Total Runs (N + X + Optimization Runs) == --folds?"}
                O -->|Yes| P[End]
                O -->|No| Q[Feedback]
                Q -->|Validation Performance and SNP Attributions| K
            end

            F --> G
    """
    generate_mermaid_diagram(mermaid_code=mermaid_code, output_file=output_file)


def generate_dl_gwas_method_diagram(output_file: str):
    mermaid_code = """
        graph TD
            subgraph "Initial Training Phase"
                A[Start] --> B[Train on full SNP set]
                B --> C{"Evaluate N times (Initial Training Runs)"}
                C --> D[Compute SNP Attributions]
            end

            subgraph "Bayesian Optimization Loop"
                E[SNP Attributions] --> F["Manually Prime Optimizer X times (Manual Priming Runs)"]
                F --> G[Sort SNPs by DL Attributions and GWAS P-values]
                G --> H{Enter Optimization Runs?}
                H -->|Yes| I[Optimizer]
                I -->|Ask for SNP Fraction| J["Bayesian Optimization (Optimization Runs)"]
                J -->|Tell Results| I
                J --> K[Select optimal SNP count]
                K --> L[Train model with selected SNPs]
                L --> M[Trained Model Validation Performance]
                M --> N{"Total Runs (N + X + Optimization Runs) == --folds?"}
                N -->|Yes| O[End]
                N -->|No| P[Feedback]
                P -->|Validation Performance and SNP Attributions| J
            end

            D --> E
    """
    generate_mermaid_diagram(mermaid_code=mermaid_code, output_file=output_file)


def generate_gwas_bo_method_diagram(output_file: str):
    mermaid_code = """
        graph TD
            subgraph "Setup Phase"
                A[Start] --> B["Define Upper Bound from GWAS P-Value Threshold"]
                B --> C["Manually Prime Optimizer X times (Manual Priming Runs)"]
            end

            subgraph "Bayesian Optimization Loop"
                C --> D{Enter Optimization Runs?}
                D -->|Yes| E[Optimizer]
                E -->|Ask for SNP Fraction with Upper Bound Constraint| F["Bayesian Optimization (Optimization Runs)"]
                F -->|Tell Results| E
                F --> G[Select optimal SNP count]
                G --> H[Train model with selected SNPs]
                H --> I[Trained Model Validation Performance]
                I --> J{"Total Runs (X + Optimization Runs) == --folds?"}
                J -->|Yes| K[End]
                J -->|No| L[Feedback]
                L -->|Validation Performance| F
            end
    """
    generate_mermaid_diagram(mermaid_code=mermaid_code, output_file=output_file)


def generate_none_method_diagram(output_file: str):
    mermaid_code = """
        graph TD
            A[Start] --> B[Train model with full SNP set]
            B --> C{"Train model --folds times"}
            C --> D[End]
    """
    generate_mermaid_diagram(mermaid_code=mermaid_code, output_file=output_file)


if __name__ == "__main__":
    generate_all()
