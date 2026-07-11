import pandas as pd


FEATURES = [
    "koi_period",
    "koi_time0bk",
    "koi_impact",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_model_snr",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "koi_kepmag",
]


def prepare_input(values):
    """
    Convert user input dictionary to a DataFrame
    with the same feature order used during training.
    """
    df = pd.DataFrame([values])
    return df[FEATURES]