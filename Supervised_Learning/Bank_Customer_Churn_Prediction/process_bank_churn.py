from typing import Tuple, List, Optional, Dict
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# BASIC UTILITIES

def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train and validation sets using stratification.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    target_col : str
        Name of the target column.
    test_size : float
        Validation set size.
    random_state : int
        Random seed.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        train_df, valid_df
    """
    train_df, valid_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col]
    )
    return train_df, valid_df


def extract_inputs_targets(
    df: pd.DataFrame,
    input_cols: List[str],
    target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract input features and target column.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
    """
    X = df[input_cols].copy()
    y = df[target_col].copy()
    return X, y


# FEATURE ENGINEERING


def map_columns(
    df: pd.DataFrame,
    map_cols: List[str],
    map_codes: Dict[str, int]
) -> pd.DataFrame:
    """
    Map categorical columns to numeric using provided dictionary.

    Returns
    -------
    pd.DataFrame
    """
    df_copy = df.copy()
    for col in map_cols:
        df_copy[col] = df_copy[col].map(map_codes)
    return df_copy


def fit_encoder(
    df: pd.DataFrame,
    categorical_cols: List[str]
) -> OneHotEncoder:
    """
    Fit OneHotEncoder on categorical columns.

    Returns
    -------
    OneHotEncoder
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(df[categorical_cols])
    return encoder


def transform_encoder(
    df: pd.DataFrame,
    encoder: OneHotEncoder,
    categorical_cols: List[str]
) -> pd.DataFrame:
    """
    Transform categorical features using fitted encoder.

    Returns
    -------
    pd.DataFrame
    """
    encoded_array = encoder.transform(df[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    return pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)


def fit_scaler(
    df: pd.DataFrame,
    numeric_cols: List[str]
) -> StandardScaler:
    """
    Fit StandardScaler on numeric columns.

    Returns
    -------
    StandardScaler
    """
    scaler = StandardScaler()
    scaler.fit(df[numeric_cols])
    return scaler


def transform_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler,
    numeric_cols: List[str]
) -> pd.DataFrame:
    """
    Scale numeric features using fitted scaler.

    Returns
    -------
    pd.DataFrame
    """
    scaled_array = scaler.transform(df[numeric_cols])
    return pd.DataFrame(scaled_array, columns=numeric_cols, index=df.index)


# MAIN PREPROCESS


def preprocess_data(
    raw_df: pd.DataFrame,
    target_col: str,
    input_cols: List[str],
    map_cols: List[str],
    map_codes: Dict[str, int],
    encoder_cols: List[str],
    scaler_numeric: bool = True
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    List[str],
    Optional[StandardScaler],
    OneHotEncoder
]:
    """
    Full preprocessing pipeline for training and validation data.

    Returns
    -------
    X_train : pd.DataFrame
    train_targets : pd.Series
    X_val : pd.DataFrame
    val_targets : pd.Series
    input_cols : List[str]
    scaler : Optional[StandardScaler]
    encoder : OneHotEncoder
    """

    # Split
    train_df, valid_df = split_data(raw_df, target_col)

    # Extract
    X_train_raw, train_targets = extract_inputs_targets(train_df, input_cols, target_col)
    X_val_raw, val_targets = extract_inputs_targets(valid_df, input_cols, target_col)

    # Detect numeric columns
    numeric_cols = X_train_raw.select_dtypes("number").columns.tolist()

    # Map binary columns
    X_train_mapped = map_columns(X_train_raw, map_cols, map_codes)
    X_val_mapped = map_columns(X_val_raw, map_cols, map_codes)

    # Fit encoder
    encoder = fit_encoder(X_train_mapped, encoder_cols)

    X_train_encoded = transform_encoder(X_train_mapped, encoder, encoder_cols)
    X_val_encoded = transform_encoder(X_val_mapped, encoder, encoder_cols)

    # Scale numeric if required
    scaler = None
    if scaler_numeric:
        scaler = fit_scaler(X_train_mapped, numeric_cols)
        X_train_numeric = transform_scaler(X_train_mapped, scaler, numeric_cols)
        X_val_numeric = transform_scaler(X_val_mapped, scaler, numeric_cols)
    else:
        X_train_numeric = X_train_mapped[numeric_cols]
        X_val_numeric = X_val_mapped[numeric_cols]

    # Combine features
    X_train = pd.concat([X_train_numeric, X_train_encoded], axis=1)
    X_val = pd.concat([X_val_numeric, X_val_encoded], axis=1)

    input_features = X_train.columns.tolist()

    return {
        "X_train": X_train,
        "y_train": train_targets,
        "X_val": X_val,
        "y_val": val_targets,
        "feature_names": input_features,
        "scaler": scaler,
        "encoder": encoder
}


# NEW DATA PROCESS

def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    map_cols: List[str],
    map_codes: Dict[str, int],
    encoder_cols: List[str],
    encoder: OneHotEncoder,
    scaler: Optional[StandardScaler] = None
) -> pd.DataFrame:
    """
    Preprocess new unseen data using already fitted encoder and scaler.

    Useful for test.csv or inference stage.

    Returns
    -------
    pd.DataFrame
    """

    X_raw = new_df[input_cols].copy()

    # Scale numeric if scaler exists
    numeric_cols = X_raw.select_dtypes("number").columns.tolist()

    if scaler is not None:
        X_numeric = transform_scaler(X_raw, scaler, numeric_cols)
    else:
        X_numeric = X_raw[numeric_cols]

    # Map
    X_mapped = map_columns(X_raw, map_cols, map_codes)

    # Encode
    X_encoded = transform_encoder(X_mapped, encoder, encoder_cols)

    # Combine
    X_final = pd.concat([X_numeric, X_encoded], axis=1)

    return X_final