"""
Reproduction de la Figure 2A de Kelliher et al. (2016).

Ce script charge les gènes périodiques de S. cerevisiae,
calcule les z-scores d'expression et génère une heatmap.

Usage
-----
python figure2A.py \
    --input data/pgen.1006453.s002.xlsx \
    --output output/figure2A_Kelliher2016.png
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

LOGGER = logging.getLogger(__name__)

XLSX_HEADER_ROW = 2
COL_FIGURE2A = "Figure2A_order_peaktime"

TIME_POINTS_MIN: list[int] = list(range(0, 250, 5))

VMIN = -1.5
VMAX = 1.5

FIGURE_WIDTH_IN = 4.5
FIGURE_HEIGHT_IN = 10.0

COLORMAP_COLORS: list[tuple[float, str]] = [
    (0.0, "#00FFFF"),
    (0.3, "#007070"),
    (0.5, "#000000"),
    (0.7, "#707000"),
    (1.0, "#FFFF00"),
]


def build_colormap(
    color_stops: list[tuple[float, str]],
) -> LinearSegmentedColormap:
    """
    Construit une colormap continue.

    Parameters
    ----------
    color_stops :
        Liste des points de contrôle.

    Returns
    -------
    LinearSegmentedColormap
        Colormap générée.
    """
    positions = [position for position, _ in color_stops]
    colors = [color for _, color in color_stops]

    return LinearSegmentedColormap.from_list(
        "cyan_black_yellow",
        list(zip(positions, colors, strict=False)),
        N=512,
    )


def load_and_select_periodic_genes(
    xlsx_path: Path,
    header_row: int,
) -> pd.DataFrame:
    """
    Charge les gènes périodiques.

    Parameters
    ----------
    xlsx_path :
        Fichier Excel.
    header_row :
        Ligne d'entête.

    Returns
    -------
    pd.DataFrame
        DataFrame filtré.
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {xlsx_path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataframe_genes = pd.read_excel(
            xlsx_path,
            header=header_row,
        )

    if COL_FIGURE2A not in dataframe_genes.columns:
        raise ValueError(
            f"Colonne manquante : {COL_FIGURE2A}"
        )

    periodic_genes = dataframe_genes[
        dataframe_genes[COL_FIGURE2A].notna()
    ].copy()

    periodic_genes = periodic_genes.sort_values(
        COL_FIGURE2A
    ).reset_index(drop=True)

    gene_count = len(periodic_genes)

    if gene_count != 1246:
        raise ValueError(
            f"1246 gènes attendus, obtenu : {gene_count}"
        )

    return periodic_genes


def compute_zscore(
    expression_matrix: np.ndarray,
) -> np.ndarray:
    """
    Calcule un z-score ligne par ligne.

    Parameters
    ----------
    expression_matrix :
        Matrice d'expression.

    Returns
    -------
    np.ndarray
        Matrice normalisée.
    """
    means = expression_matrix.mean(axis=1, keepdims=True)

    standard_deviations = expression_matrix.std(
        axis=1,
        ddof=1,
        keepdims=True,
    )

    standard_deviations[standard_deviations == 0] = 1.0

    return (
        expression_matrix - means
    ) / standard_deviations


def plot_figure2a(
    zscore_matrix: np.ndarray,
    time_points: list[int],
    cmap: LinearSegmentedColormap,
    output_path: Path,
) -> None:
    """
    Génère la heatmap.

    Parameters
    ----------
    zscore_matrix :
        Matrice z-score.
    time_points :
        Temps en minutes.
    cmap :
        Colormap.
    output_path :
        Image de sortie.
    """
    figure, axis = plt.subplots(
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN)
    )

    image = axis.imshow(
        zscore_matrix,
        aspect="auto",
        cmap=cmap,
        vmin=VMIN,
        vmax=VMAX,
        interpolation="nearest",
        origin="upper",
    )

    tick_minutes = [0, 50, 100, 150, 200]

    tick_positions = [
        (
            minute / time_points[-1]
        ) * (len(time_points) - 1)
        for minute in tick_minutes
    ]

    axis.set_xticks(tick_positions)
    axis.set_xticklabels(
        [str(value) for value in tick_minutes]
    )

    axis.set_xlabel("Time (minutes)")
    axis.set_yticks([])

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )

    colorbar.set_label("Z-score")

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    LOGGER.info(
        "Figure sauvegardée : %s",
        output_path,
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse les arguments CLI.

    Returns
    -------
    argparse.Namespace
        Arguments utilisateur.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def main() -> None:
    """
    Point d'entrée principal.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    arguments = parse_arguments()

    periodic_dataframe = (
        load_and_select_periodic_genes(
            arguments.input,
            XLSX_HEADER_ROW,
        )
    )

    time_columns = sorted(
        [
            column
            for column in periodic_dataframe.columns
            if column in TIME_POINTS_MIN
        ]
    )

    expression_matrix = (
        periodic_dataframe[time_columns]
        .to_numpy(dtype=float)
    )

    zscore_matrix = compute_zscore(
        expression_matrix
    )

    colormap = build_colormap(
        COLORMAP_COLORS
    )

    plot_figure2a(
        zscore_matrix=zscore_matrix,
        time_points=time_columns,
        cmap=colormap,
        output_path=arguments.output,
    )


if __name__ == "__main__":
    main()
