# print("\n... PIP INSTALLS STARTING ...\n")
# print("\n... PIP INSTALLS COMPLETE ...\n")

print("\n... IMPORTS STARTING ...\n")
print("\n\tVERSION INFORMATION")

# Competition Specific Import
# TBD

import pandas as pd; pd.options.mode.chained_assignment = None; pd.set_option('display.max_columns', None); import pandas;
import numpy as np; print(f"\t\t– NUMPY VERSION: {np.__version__}")
import sklearn; print(f"\t\t– SKLEARN VERSION: {sklearn.__version__}")
from sklearn.metrics import roc_curve, auc, roc_auc_score
import cv2
print(f"\t\t– CV2 VERSION: {cv2.__version__}")

# For modelling and dataset
import lightgbm as lgb
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import OrdinalEncoder

# Built-In Imports (mostly don't worry about these)
from typing import Iterable, Any, Callable, Generator

from dataclasses import dataclass
from collections import Counter
from datetime import datetime
from zipfile import ZipFile
from glob import glob
import subprocess
import warnings
import requests
import textwrap
import hashlib
import imageio
import IPython
import urllib
import zipfile
import pickle
import random
import shutil
import string
import h5py
import json
import copy
import math
import time
import gzip
import ast
import sys
import io
import gc
import re
import os

# Visualization Imports (overkill)
from IPython.core.display import HTML, Markdown
import matplotlib.pyplot as plt
from matplotlib import animation, rc; rc('animation', html='jshtml')
from tqdm.notebook import tqdm; tqdm.pandas();
import plotly.graph_objects as go
import plotly.express as px
import plotly
import seaborn as sns
from PIL import Image, ImageEnhance, ImageColor; Image.MAX_IMAGE_PIXELS = 5_000_000_000;
import matplotlib; print(f"\t\t– MATPLOTLIB VERSION: {matplotlib.__version__}");
from colorama import Fore, Style, init; init()
import PIL


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple.

    Args:
        hex_color (str): The hex color string, starting with '#'.

    Returns:
        tuple: A tuple of RGB values.
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def clr_print(text: str, color: str = "#42BFBA", bold: bool = True) -> None:
    """Print the given text with the specified color and bold formatting.

    Args:
        text (str): The text to format.
        color (str): The hex color code to apply. Defaults to "#752F55".
        bold (bool): Whether to apply bold formatting. Defaults to True.
    """
    _text = text.replace('\n', '<br>')
    rgb = hex_to_rgb(color)
    color_style = f"color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]});"
    bold_style = "font-weight: bold;" if bold else ""
    style = f"{color_style} {bold_style}"


def seed_it_all(seed=7):
    """ Attempt to be Reproducible """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # tf.random.set_seed(seed)


seed_it_all()

# Create a Seaborn color palette
nb_palette = sns.color_palette(palette='tab20')

# Create colors for class labels
LABELS = ["Benign", "Malignant"]
COLORS = ['#66c2a5', '#fc8d62']
CLR_MAP_I2C = {i: c for i, c in enumerate(COLORS)}
CLR_MAP_S2C = {l: c for l, c in zip(LABELS, COLORS)}
LBL_MAP_I2S = {i: l for i, l in enumerate(LABELS)}
LBL_MAP_S2I = {v: k for k, v in LBL_MAP_I2S.items()}

# Is this notebook being run on the backend for scoring re-submission
IS_DEBUG = False if os.getenv('KAGGLE_IS_COMPETITION_RERUN') else True
print(f"IS DEBUG: {IS_DEBUG}")

# Plot the palette
clr_print("\n... NOTEBOOK COLOUR PALETTE ...")
sns.palplot(nb_palette, size=0.5)

print("\n\n... IMPORTS COMPLETE ...\n")

"""2024 ISIC Challenge primary prize scoring metric

Given a list of binary labels, an associated list of prediction 
scores ranging from [0,1], this function produces, as a single value, 
the partial area under the receiver operating characteristic (pAUC) 
above a given true positive rate (TPR).
https://en.wikipedia.org/wiki/Partial_Area_Under_the_ROC_Curve.

(c) 2024 Nicholas R Kurtansky, MSKCC
"""


class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, min_tpr: float = 0.80) -> float:
    '''
    2024 ISIC Challenge metric: pAUC

    Given a solution file and submission file, this function returns the
    the partial area under the receiver operating characteristic (pAUC)
    above a given true positive rate (TPR) = 0.80.
    https://en.wikipedia.org/wiki/Partial_Area_Under_the_ROC_Curve.

    (c) 2024 Nicholas R Kurtansky, MSKCC

    Args:
        solution: ground truth pd.DataFrame of 1s and 0s
        submission: solution dataframe of predictions of scores ranging [0, 1]

    Returns:
        Float value range [0, max_fpr]
    '''

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    # check submission is numeric
    if not pandas.api.types.is_numeric_dtype(submission.values):
        raise ParticipantVisibleError('Submission target column must be numeric')

    # rescale the target. set 0s to 1s and 1s to 0s (since sklearn only has max_fpr)
    v_gt = abs(np.asarray(solution.values) - 1)

    # flip the submissions to their compliments
    v_pred = -1.0 * np.asarray(submission.values)

    max_fpr = abs(1 - min_tpr)

    # using sklearn.metric functions: (1) roc_curve and (2) auc
    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)

    # Add a single point at max_fpr by linear interpolation
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)

    #     # Equivalent code that uses sklearn's roc_auc_score
    #     v_gt = abs(np.asarray(solution.values)-1)
    #     v_pred = np.array([1.0 - x for x in submission.values])
    #     max_fpr = abs(1-min_tpr)
    #     partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    #     # change scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
    #     # https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range
    #     partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)

    return (partial_auc)


def flatten_l_o_l(nested_list):
    """ Flatten a list of lists into a single list.

    Args:
        nested_list (Iterable):
            – A list of lists (or iterables) to be flattened.

    Returns:
        A flattened list containing all items from the input list of lists.
    """
    return [item for sublist in nested_list for item in sublist]


def print_ln(symbol="-", line_len=110, newline_before=False, newline_after=False):
    """ Print a horizontal line of a specified length and symbol.

    Args:
        symbol (str, optional):
            – The symbol to use for the horizontal line
        line_len (int, optional):
            – The length of the horizontal line in characters
        newline_before (bool, optional):
            – Whether to print a newline character before the line
        newline_after (bool, optional):
            – Whether to print a newline character after the line

    Returns:
        None; A divider with pre/post new-lines (optional) is printed
    """
    if newline_before: print();
    print(symbol * line_len)
    if newline_after: print();


def display_hr(newline_before=False, newline_after=False):
    """ Renders a HTML <hr>

    Args:
        newline_before (bool, optional):
            – Whether to print a newline character before the line
        newline_after (bool, optional):
            – Whether to print a newline character after the line

    Returns:
        None; A divider with pre/post new-lines (optional) is printed
    """
    if newline_before: print();
    print(HTML("<hr>"))
    if newline_after: print();


def wrap_text(text, width=88):
    """Wrap text to a specified width.

    Args:
        text (str):
            - The text to wrap.
        width (int):
            - The maximum width of a line. Default is 88.

    Returns:
        str: The wrapped text.
    """
    return textwrap.fill(text, width)


def wrap_text_by_paragraphs(text, width=88):
    """Wrap text by paragraphs to a specified width.

    Args:
        text (str):
            - The text containing multiple paragraphs to wrap.
        width (int):
            - The maximum width of a line. Default is 88.

    Returns:
        str: The wrapped text with preserved paragraph separation.
    """
    paragraphs = text.split('\n')  # Assuming paragraphs are separated by newlines
    wrapped_paragraphs = [textwrap.fill(paragraph, width) for paragraph in paragraphs]
    return '\n\n'.join(wrapped_paragraphs)


def load_img_from_hdf5(
        isic_id: str,
        file_path: str = "isic-2024-challenge/train-image.hdf5",
        n_channels: int = 3
):
    """
    Load an image from the HDF5 dataset file by specifying an ISIC ID.

    The ISIC ID is expected to be in the form 'ISIC_#######'.

    Args:
        isic_id (str): The ID of the image to load.
        file_path (str): The path to the HDF5 file.
        n_channels (int): Number of channels (3 for RGB, 1 for grayscale).

    Returns:
        np.ndarray: The loaded image.

    Raises:
        KeyError: If the ISIC ID is not found in the HDF5 file.
        ValueError: If the image data cannot be decoded.

    Example Usage:
        img = load_img_from_hdf5('ISIC_0000000')
    """

    # Handle the case where the isic_id is passed incorrectly
    if not isic_id.lower().startswith("isic"):
        isic_id = f"ISIC_{int(str(isic_id).split('_', 1)[-1]):>07}"

    # Open the HDF5 file in read mode
    with h5py.File(file_path, 'r') as hf:

        # Retrieve the image data from the HDF5 dataset using the provided ISIC ID
        try:
            image_data = hf[isic_id][()]
        except KeyError:
            raise KeyError(f"ISIC ID {isic_id} not found in HDF5 file.")

        # Convert the binary data to a numpy array
        image_array = np.frombuffer(image_data, np.uint8)

        # Decode the image from the numpy array
        if n_channels == 3:
            # Load the image as a color image (BGR) and convert to RGB
            image = cv2.cvtColor(cv2.imdecode(image_array, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        else:
            # Load the image as a grayscale image
            image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

        # If the image failed to load for some reason (problems decoding) ...
        if image is None:
            raise ValueError(f"Could not decode image for ISIC ID: {isic_id}")

        return image


plt.figure(figsize=(6, 6))
plt.title("ISIC_0015670", fontweight="bold")
plt.imshow(load_img_from_hdf5("ISIC_0015670"))
plt.show()

METADATA_COL2DESC = {
    "isic_id": "Unique identifier for each image case.",
    "target": "Binary class label indicating if the lesion is benign (0) or malignant (1).",
    "patient_id": "Unique identifier for each patient.",
    "age_approx": "Approximate age of the patient at the time of imaging.",
    "sex": "Sex of the patient (male or female).",
    "anatom_site_general": "General location of the lesion on the patient's body (e.g., upper extremity, posterior torso).",
    "clin_size_long_diam_mm": "Maximum diameter of the lesion in millimeters.",
    "image_type": "Type of image captured, as defined in the ISIC Archive.",
    "tbp_tile_type": "Lighting modality of the 3D Total Body Photography (TBP) source image.",
    "tbp_lv_A": "Color channel A inside the lesion; related to the green-red axis in LAB color space.",
    "tbp_lv_Aext": "Color channel A outside the lesion; related to the green-red axis in LAB color space.",
    "tbp_lv_B": "Color channel B inside the lesion; related to the blue-yellow axis in LAB color space.",
    "tbp_lv_Bext": "Color channel B outside the lesion; related to the blue-yellow axis in LAB color space.",
    "tbp_lv_C": "Chroma value inside the lesion, indicating color purity.",
    "tbp_lv_Cext": "Chroma value outside the lesion, indicating color purity.",
    "tbp_lv_H": "Hue value inside the lesion, representing the type of color (e.g., red, brown) in LAB color space.",
    "tbp_lv_Hext": "Hue value outside the lesion, representing the type of color (e.g., red, brown) in LAB color space.",
    "tbp_lv_L": "Luminance value inside the lesion; related to lightness in LAB color space.",
    "tbp_lv_Lext": "Luminance value outside the lesion; related to lightness in LAB color space.",
    "tbp_lv_areaMM2": "Area of the lesion in square millimeters.",
    "tbp_lv_area_perim_ratio": "Ratio of the lesion's perimeter to its area, indicating border jaggedness.",
    "tbp_lv_color_std_mean": "Mean color irregularity within the lesion, calculated as the variance of colors.",
    "tbp_lv_deltaA": "Average contrast in color channel A between inside and outside the lesion.",
    "tbp_lv_deltaB": "Average contrast in color channel B between inside and outside the lesion.",
    "tbp_lv_deltaL": "Average contrast in luminance between inside and outside the lesion.",
    "tbp_lv_deltaLB": "Combined contrast between the lesion and its immediate surrounding skin.",
    "tbp_lv_deltaLBnorm": "Normalized contrast between the lesion and its immediate surrounding skin in LAB color space.",
    "tbp_lv_eccentricity": "Eccentricity of the lesion, indicating how elongated it is.",
    "tbp_lv_location": "Detailed anatomical location of the lesion, dividing body parts further (e.g., Left Arm - Upper).",
    "tbp_lv_location_simple": "Simplified anatomical location of the lesion (e.g., Left Arm).",
    "tbp_lv_minorAxisMM": "Smallest diameter of the lesion in millimeters.",
    "tbp_lv_nevi_confidence": "Confidence score (0-100) from a neural network classifier estimating the probability that the lesion is a nevus.",
    "tbp_lv_norm_border": "Normalized border irregularity score on a scale of 0-10.",
    "tbp_lv_norm_color": "Normalized color variation score on a scale of 0-10.",
    "tbp_lv_perimeterMM": "Perimeter of the lesion in millimeters.",
    "tbp_lv_radial_color_std_max": "Color asymmetry score within the lesion, based on color variance in concentric rings.",
    "tbp_lv_stdL": "Standard deviation of luminance within the lesion.",
    "tbp_lv_stdLExt": "Standard deviation of luminance outside the lesion.",
    "tbp_lv_symm_2axis": "Measure of asymmetry of the lesion's border about a secondary axis.",
    "tbp_lv_symm_2axis_angle": "Angle of the secondary axis of symmetry for the lesion's border.",
    "tbp_lv_x": "X-coordinate of the lesion in the 3D TBP model.",
    "tbp_lv_y": "Y-coordinate of the lesion in the 3D TBP model.",
    "tbp_lv_z": "Z-coordinate of the lesion in the 3D TBP model.",
    "attribution": "Source or institution responsible for the image.",
    "copyright_license": "Type of copyright license for the image.",
    "lesion_id": "Unique identifier for lesions that were manually tagged as lesions of interest.",
    "iddx_full": "Full classified diagnosis of the lesion.",
    "iddx_1": "First-level diagnosis of the lesion (e.g., Benign, Malignant).",
    "iddx_2": "Second-level diagnosis providing more specific details about the lesion.",
    "iddx_3": "Third-level diagnosis with further classification details.",
    "iddx_4": "Fourth-level diagnosis with additional specificity.",
    "iddx_5": "Fifth-level diagnosis, providing the most detailed classification.",
    "mel_mitotic_index": "Mitotic index of invasive malignant melanomas, indicating cell division rate.",
    "mel_thick_mm": "Thickness in millimeters of melanoma invasion.",
    "tbp_lv_dnn_lesion_confidence": "Lesion confidence score (0-100) from a deep neural network classifier."
}

METADATA_COL2NAME = {
    "isic_id": "Unique Case Identifier",
    "target": "Binary Lession Classification",
    "patient_id": "Unique Patient Identifier",
    "age_approx": "Approximate Age",
    "sex": "Sex",
    "anatom_site_general": "General Anatomical Location",
    "clin_size_long_diam_mm": "Clinical Size (Longest Diameter in mm)",
    "image_type": "Image Type",
    "tbp_tile_type": "TBP Tile Type",
    "tbp_lv_A": "Color Channel A Inside Lesion",
    "tbp_lv_Aext": "Color Channel A Outside Lesion",
    "tbp_lv_B": "Color Channel B Inside Lesion",
    "tbp_lv_Bext": "Color Channel B Outside Lesion",
    "tbp_lv_C": "Chroma Inside Lesion",
    "tbp_lv_Cext": "Chroma Outside Lesion",
    "tbp_lv_H": "Hue Inside Lesion",
    "tbp_lv_Hext": "Hue Outside Lesion",
    "tbp_lv_L": "Luminance Inside Lesion",
    "tbp_lv_Lext": "Luminance Outside Lesion",
    "tbp_lv_areaMM2": "Lesion Area (mm²)",
    "tbp_lv_area_perim_ratio": "Area-to-Perimeter Ratio",
    "tbp_lv_color_std_mean": "Mean Color Irregularity",
    "tbp_lv_deltaA": "Delta A (Inside vs. Outside)",
    "tbp_lv_deltaB": "Delta B (Inside vs. Outside)",
    "tbp_lv_deltaL": "Delta L (Inside vs. Outside)",
    "tbp_lv_deltaLB": "Delta LB (Contrast)",
    "tbp_lv_deltaLBnorm": "Normalized Delta LB (Contrast)",
    "tbp_lv_eccentricity": "Eccentricity",
    "tbp_lv_location": "Detailed Anatomical Location",
    "tbp_lv_location_simple": "Simplified Anatomical Location",
    "tbp_lv_minorAxisMM": "Smallest Diameter (mm)",
    "tbp_lv_nevi_confidence": "Nevus Confidence Score",
    "tbp_lv_norm_border": "Normalized Border Irregularity",
    "tbp_lv_norm_color": "Normalized Color Variation",
    "tbp_lv_perimeterMM": "Lesion Perimeter (mm)",
    "tbp_lv_radial_color_std_max": "Radial Color Standard Deviation",
    "tbp_lv_stdL": "Standard Deviation of Luminance (Inside)",
    "tbp_lv_stdLExt": "Standard Deviation of Luminance (Outside)",
    "tbp_lv_symm_2axis": "Symmetry (Second Axis)",
    "tbp_lv_symm_2axis_angle": "Symmetry Angle (Second Axis)",
    "tbp_lv_x": "X-Coordinate",
    "tbp_lv_y": "Y-Coordinate",
    "tbp_lv_z": "Z-Coordinate",
    "attribution": "Image Source",
    "copyright_license": "Copyright License",
    "lesion_id": "Unique Lesion Identifier",
    "iddx_full": "Full Diagnosis",
    "iddx_1": "First Level Diagnosis",
    "iddx_2": "Second Level Diagnosis",
    "iddx_3": "Third Level Diagnosis",
    "iddx_4": "Fourth Level Diagnosis",
    "iddx_5": "Fifth Level Diagnosis",
    "mel_mitotic_index": "Mitotic Index (Melanoma)",
    "mel_thick_mm": "Thickness of Melanoma (mm)",
    "tbp_lv_dnn_lesion_confidence": "Lesion Confidence Score"
}



# ROOT PATHS
WORKING_DIR = "/kaggle/working"
INPUT_DIR = "/kaggle/input"
COMPETITION_DIR = os.path.join(INPUT_DIR, "isic-2024-challenge")

# IMAGE DIRS
TRAIN_IMAGE_DIR = os.path.join(COMPETITION_DIR, "train-image", "image")
TEST_IMAGE_DIR = os.path.join(COMPETITION_DIR, "test-image", "image")

# FILE PATHS
TRAIN_METADATA_CSV = os.path.join(COMPETITION_DIR, "train-metadata.csv")
TEST_METADATA_CSV = os.path.join(COMPETITION_DIR, "test-metadata.csv")
TRAIN_IMAGE_HDF5 = os.path.join(COMPETITION_DIR, "train-image.hdf5")
TEST_IMAGE_HDF5 = os.path.join(COMPETITION_DIR, "test-image.hdf5")
SS_CSV_PATH = os.path.join(COMPETITION_DIR, "sample_submission.csv")


# DEFINE COMPETITION DATAFRAMES
clr_print("\n\n... SAMPLE SUBMISSION DATAFRAME ...\n\n")
ss_df = pd.read_csv(SS_CSV_PATH)
print(ss_df.head())

clr_print("\n\n... TRAIN METADATA DATAFRAME ...\n\n")
train_df = pd.read_csv(TRAIN_METADATA_CSV)
print(train_df.head())

clr_print("\n\n... TEST METADATA DATAFRAME ...\n\n")
test_df = pd.read_csv(TEST_METADATA_CSV)
print(test_df.head())

clr_print("\n\n... HDF5 (DATASET) PATHS ...\n\n")
print(f"\t--> {TRAIN_IMAGE_HDF5}")
print(f"\t--> {TEST_IMAGE_HDF5}\n")

for _c in train_df.columns:
    display_hr(True, True)
    clr_print(f"COLUMN NAME         : <code>'{_c}'</code>")
    clr_print(f"HUMAN READABLE NAME : <span style='color: black !important;'>'{METADATA_COL2NAME.get(_c)}'</span>")
    clr_print(f"COLUMN DESCRIPTION  : <span style='color: black !important;'>'{METADATA_COL2DESC.get(_c)}'</span>")

display_hr(True, True)


def plot_nan_heatmap(
        df: pd.DataFrame,
        figsize: tuple = (17, 8),
        cmap: str = 'magma_r',
        title: str = 'NaN Values in DataFrame',
        x_tick_rotation=60,
        show_cbar: bool = False,
        show_yticklabels: bool = False
    ) -> None:
        """Create a heatmap to visualize NaN values in a DataFrame.

        Args:
            df (pd.DataFrame):
                The input DataFrame to visualize.
            figsize (tuple[int], optional):
                Figure size as a tuple of (width, height)
            cmap (str, optional):
                Colormap to use for the heatmap
            title (str, optional):
                Title for the heatmap.
            x_tick_rotation (int, optional):
                Rotation angle for x-axis tick labels.
            show_cbar (bool, optional):
                Whether to show the color bar.
            show_yticklabels (bool, optional):
                Whether to show y-axis tick labels.

        Returns:
            None;
                The function displays the plot using plt.show().
        """

        # Setup the figure
        plt.figure(figsize=figsize)

        # Create the heatmap
        sns.heatmap(df.isna(), cbar=show_cbar, yticklabels=show_yticklabels, cmap=cmap)

        # Update the title/labels
        plt.title(title, fontweight="bold")
        plt.xlabel('Columns', fontweight="bold")
        plt.ylabel('Rows', fontweight="bold")

        # Rotate x-axis labels
        plt.xticks(rotation=x_tick_rotation, ha='right')

        # Adjust the bottom margin to prevent label cutoff
        plt.tight_layout()

        # Render
        plt.show()

        # Print NaN counts per column
        nan_counts = df.isna().sum().sort_values(ascending=False)

        # clr_print()
        clr_print("NaN counts per column:")
        print(nan_counts[nan_counts])
        clr_print("Features with 0 NaN values:")

