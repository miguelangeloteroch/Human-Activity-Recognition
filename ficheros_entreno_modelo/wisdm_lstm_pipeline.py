"""
Pipeline para depurar el dataset WISDM y entrenar un modelo LSTM con validacion cruzada.

Pasos principales:
- Carga y limpieza del crudo (elimino el ; final, normalizo etiquetas y ordeno por usuario/tiempo).
- Winsorizacion de outliers en variables numericas.
- Imputacion de missing con IterativeImputer + RandomForest en caso de ser necesario.
- Normalizacion estandar.
- Construccion de ventanas temporales para LSTM.
- Grid search grande de hiperparametros con 4 folds y metricas accuracy/recall(F1 macro).

Ejemplo de ejecucion rapida:
python wisdm_lstm_pipeline.py --data-path WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt --max-grid 8 --epochs 6
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, regularizers, utils

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATA_PATH = DATA_DIR / "WISDM_ar_latest" / "WISDM_ar_v1.1" / "WISDM_ar_v1.1_raw.txt"
BASE_FEATURES = ["x", "y", "z", "accel_mag"]
TIMESTAMP_PREFIX = "timestamp_prev_"

# Reproducibilidad
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


@dataclass
class DatasetSplits:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder
    feature_names: List[str]
    scaler: StandardScaler


@dataclass
class CVResult:
    params: Dict
    accuracy: float
    recall: float
    f1: float
    history: Dict


def resolve_project_path(path_str: str) -> Path:
    """Resuelve rutas relativas usando el directorio raiz del proyecto."""
    raw_path = Path(path_str).expanduser()
    if raw_path.is_absolute():
        return raw_path
    return PROJECT_ROOT / raw_path


def load_raw_wisdm(path: Path) -> pd.DataFrame:
    """Carga el fichero crudo o un CSV procesado (con headers) y aplica limpieza basica."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el dataset en {path}")

    base_cols = ["user", "activity", "timestamp", "x", "y", "z"]
    first_line = ""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            first_line = fh.readline().strip().lstrip("\ufeff").lower()
    except OSError:
        first_line = ""
    has_header = first_line.startswith("user") or TIMESTAMP_PREFIX in first_line

    if has_header:
        df = pd.read_csv(path, engine="python")
        df.columns = [str(col).strip() for col in df.columns]
        missing = [col for col in base_cols if col not in df.columns]
        if not missing:
            df["z"] = df["z"].astype(str).str.replace(";", "", regex=False)
            df["activity"] = df["activity"].astype(str).str.strip().str.lower()
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            for axis in ["x", "y", "z"]:
                df[axis] = pd.to_numeric(df[axis], errors="coerce")
            prev_cols = [col for col in df.columns if col.startswith(TIMESTAMP_PREFIX)]
            for col in prev_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=base_cols)
            df = df.drop_duplicates()
            df = df.sort_values(["user", "timestamp"]).reset_index(drop=True)
            return df
        # Si faltan columnas clave, caemos al modo crudo original.

    df = pd.read_csv(
        path,
        header=None,
        names=base_cols,
        engine="python",
        on_bad_lines="skip",  # dataset contiene lineas corruptas con mas columnas
    )
    df["z"] = df["z"].astype(str).str.replace(";", "", regex=False)
    df["z"] = pd.to_numeric(df["z"], errors="coerce")

    df["activity"] = df["activity"].str.strip().str.lower()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    df = df.dropna(subset=base_cols)
    df = df.drop_duplicates()
    df = df.sort_values(["user", "timestamp"]).reset_index(drop=True)
    return df


def remap_activity_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Remapea actividades a 3 clases: walking, jogging, stationary.
    
    Estrategia para stairs: se mapea a 'walking' porque representa
    locomoción activa y tiene patrones similares de aceleración.
    
    Args:
        df: DataFrame con columna 'activity' que contiene las actividades originales.
        
    Returns:
        DataFrame con columna 'activity' remapeada a 3 clases.
    """
    df = df.copy()
    # Normalizar: lowercase, sin espacios ni underscores
    df["activity"] = df["activity"].str.lower().str.replace(r"[\s_]+", "", regex=True)
    
    # Mapeo a 3 clases
    mapping = {
        "walking": "walking",
        "jogging": "jogging",
        "sitting": "stationary",
        "standing": "stationary",
        "lyingdown": "stationary",
        "stairs": "walking",      # Locomoción activa
        "upstairs": "walking",
        "downstairs": "walking",
    }
    df["activity"] = df["activity"].map(mapping)
    
    # Eliminar filas con actividades no reconocidas
    before = len(df)
    df = df.dropna(subset=["activity"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"Advertencia: {dropped} filas eliminadas por actividad no reconocida.")
    
    return df


def winsorize_features(df: pd.DataFrame, feature_cols: Iterable[str], limits: Tuple[float, float]) -> pd.DataFrame:
    """Winsoriza features numéricas con limites bajos/altos proporcionados."""
    df_out = df.copy()
    for col in feature_cols:
        df_out[col] = winsorize(df_out[col], limits=limits)
    return df_out


def impute_missing(df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    """Imputa missings numericos usando RandomForest si existen valores faltantes."""
    df_out = df.copy()
    if df_out[feature_cols].isna().sum().sum() == 0:
        return df_out

    estimator = RandomForestRegressor(
        n_estimators=80,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    imputer = IterativeImputer(
        estimator=estimator,
        random_state=RANDOM_SEED,
        max_iter=10,
        initial_strategy="median",
    )
    df_out[feature_cols] = imputer.fit_transform(df_out[feature_cols])
    return df_out


def normalize_features(df: pd.DataFrame, feature_cols: Iterable[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    df_out = df.copy()
    df_out[feature_cols] = scaler.fit_transform(df_out[feature_cols])
    return df_out, scaler


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Devuelve la lista de features base + cualquier timestamp_prev_* detectado."""
    feature_cols = BASE_FEATURES.copy()

    def sort_key(col: str) -> Tuple[int, int | str]:
        suffix = col[len(TIMESTAMP_PREFIX) :]
        try:
            return (0, int(suffix))
        except ValueError:
            return (1, suffix)

    prev_cols = [col for col in df.columns if col.startswith(TIMESTAMP_PREFIX)]
    prev_cols_sorted = sorted(prev_cols, key=sort_key)
    feature_cols.extend(prev_cols_sorted)
    return feature_cols


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Crea variables derivadas y prepara lista de features."""
    df_out = df.copy()
    df_out["accel_mag"] = np.sqrt(df_out["x"] ** 2 + df_out["y"] ** 2 + df_out["z"] ** 2)
    feature_cols = get_feature_columns(df_out)
    return df_out, feature_cols


def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = 50,
    step: int = 25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Genera ventanas temporales con etiqueta mayoritaria dentro de la ventana."""
    sequences: List[np.ndarray] = []
    labels: List[str] = []

    for _, user_df in df.groupby("user"):
        feats = user_df[feature_cols].values
        acts = user_df["activity"].values
        for start in range(0, len(user_df) - window_size + 1, step):
            end = start + window_size
            window_feats = feats[start:end]
            window_acts = acts[start:end]
            majority_label = pd.Series(window_acts).mode().iloc[0]
            sequences.append(window_feats)
            labels.append(majority_label)

    X = np.stack(sequences)
    y = np.array(labels)
    return X, y


def prepare_splits(
    df: pd.DataFrame, window_size: int, step: int
) -> DatasetSplits:
    df_feat, feature_cols = engineer_features(df)
    extra_feature_count = sum(col.startswith(TIMESTAMP_PREFIX) for col in feature_cols)
    if extra_feature_count:
        print(
            f"Detectadas {extra_feature_count} columnas timestamp_prev_* -> "
            f"{[col for col in feature_cols if col.startswith(TIMESTAMP_PREFIX)]}"
        )
    else:
        print("No se detectaron columnas timestamp_prev_*; se usan solo las features base.")

    # Winsorizar outliers
    df_feat = winsorize_features(df_feat, feature_cols, limits=(0.01, 0.01))

    # Imputacion missing (solo si existe)
    df_feat = impute_missing(df_feat, feature_cols)

    # Normalizacion
    df_feat, scaler = normalize_features(df_feat, feature_cols)

    # Generacion de secuencias
    X, y = create_sequences(df_feat, feature_cols, window_size=window_size, step=step)

    # Codificacion de etiquetas
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_encoded,
    )

    return DatasetSplits(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        label_encoder=label_encoder,
        feature_names=feature_cols,
        scaler=scaler,
    )


def build_lstm_model(input_shape: Tuple[int, int], num_classes: int, params: Dict) -> models.Model:
    reg = regularizers.l2(params.get("l2", 0.0))
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(params["lstm_units"], return_sequences=True, kernel_regularizer=reg),
            layers.Dropout(params["dropout"]),
            layers.LSTM(params["lstm_units"] // 2, kernel_regularizer=reg),
            layers.Dropout(params["dropout"]),
            layers.Dense(params["dense_units"], activation="relu", kernel_regularizer=reg),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    opt = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "recall": metrics.recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": metrics.f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def cross_validate_lstm(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    param_grid: Dict,
    n_splits: int,
    max_combos: Optional[int],
    epochs: int,
) -> CVResult:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    combos = list(ParameterGrid(param_grid))
    if max_combos:
        combos = combos[:max_combos]

    best_result: Optional[CVResult] = None
    input_shape = (X.shape[1], X.shape[2])
    y_cat = utils.to_categorical(y, num_classes=num_classes)

    for combo_idx, params in enumerate(combos, start=1):
        print(f"\n[Grid {combo_idx}/{len(combos)}] Probando hiperparámetros: {params}")
        fold_metrics: List[Dict[str, float]] = []
        for train_idx, val_idx in skf.split(X, y):
            model = build_lstm_model(input_shape, num_classes, params)
            es = callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss")
            model.fit(
                X[train_idx],
                y_cat[train_idx],
                validation_data=(X[val_idx], y_cat[val_idx]),
                epochs=epochs,
                batch_size=params["batch_size"],
                verbose=1,
                callbacks=[es],
            )
            preds = model.predict(X[val_idx], verbose=0)
            pred_labels = np.argmax(preds, axis=1)
            fold_scores = compute_metrics(y[val_idx], pred_labels)
            fold_metrics.append(fold_scores)
            print(
                "  Fold metrics -> "
                f"accuracy: {fold_scores['accuracy']:.4f}, "
                f"recall: {fold_scores['recall']:.4f}, "
                f"f1: {fold_scores['f1']:.4f}"
            )

        avg_acc = float(np.mean([m["accuracy"] for m in fold_metrics]))
        avg_rec = float(np.mean([m["recall"] for m in fold_metrics]))
        avg_f1 = float(np.mean([m["f1"] for m in fold_metrics]))
        print(
            f"[Grid {combo_idx}] Promedios -> accuracy: {avg_acc:.4f}, "
            f"recall: {avg_rec:.4f}, f1: {avg_f1:.4f}"
        )

        if (best_result is None) or (avg_f1 > best_result.f1):
            best_result = CVResult(
                params=params,
                accuracy=avg_acc,
                recall=avg_rec,
                f1=avg_f1,
                history={"folds": fold_metrics},
            )

    if best_result is None:
        raise RuntimeError("No se pudo evaluar ningun conjunto de hiperparametros.")
    return best_result


def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    params: Dict,
    epochs: int,
    verbose: int = 1,
    return_history: bool = False,
    class_weight: Optional[Dict[int, float]] = None,
) -> models.Model | Tuple[models.Model, callbacks.History]:
    """
    Entrena el modelo final LSTM.

    Args:
        X_train: Tensores de entrada (n_samples, timesteps, features).
        y_train: Etiquetas codificadas como enteros.
        num_classes: Número de clases de salida.
        params: Diccionario de hiperparámetros (lstm_units, dropout, etc.).
        epochs: Épocas máximas de entrenamiento.
        verbose: Nivel de verbosidad de Keras.
        return_history: Si True, devuelve también el History de Keras.
        class_weight: (Opcional) Pesos por clase para balancear el entrenamiento.
    """
    y_cat = utils.to_categorical(y_train, num_classes=num_classes)
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]), num_classes, params)
    # Usamos una paciencia algo mayor para permitir convergencia con class_weight
    es = callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_loss")
    history = model.fit(
        X_train,
        y_cat,
        validation_split=0.1,
        epochs=epochs,
        batch_size=params["batch_size"],
        callbacks=[es],
        verbose=verbose,
        class_weight=class_weight,
    )
    if return_history:
        return model, history
    return model


def run_pipeline(args: argparse.Namespace) -> None:
    data_path = resolve_project_path(args.data_path)
    df = load_raw_wisdm(data_path)
    print(f"Dataset cargado: {len(df):,} filas despues de limpieza inicial.")
    print(f"Actividades disponibles: {sorted(df['activity'].unique())}")

    splits = prepare_splits(df, window_size=args.window_size, step=args.step)
    print(
        f"Secuencias generadas -> train: {splits.X_train.shape}, test: {splits.X_test.shape}, "
        f"num_clases: {len(splits.label_encoder.classes_)}"
    )

    # Definir grid amplio de hiperparametros
    param_grid = {
        "lstm_units": [32, 64, 96],
        "dropout": [0.1, 0.25, 0.35],
        "dense_units": [32, 64, 96],
        "learning_rate": [1e-3, 5e-4, 1e-4],
        "batch_size": [64, 96, 128],
        "l2": [0.0, 1e-4, 1e-3],
    }

    best_cv = cross_validate_lstm(
        splits.X_train,
        splits.y_train,
        num_classes=len(splits.label_encoder.classes_),
        param_grid=param_grid,
        n_splits=args.folds,
        max_combos=args.max_grid,
        epochs=args.epochs,
    )
    print(f"Mejores params CV (F1={best_cv.f1:.4f}): {best_cv.params}")

    # Entrenamiento final con mejores parametros
    final_model = train_final_model(
        splits.X_train,
        splits.y_train,
        num_classes=len(splits.label_encoder.classes_),
        params=best_cv.params,
        epochs=args.epochs,
    )

    test_preds = final_model.predict(splits.X_test, verbose=0)
    test_labels = np.argmax(test_preds, axis=1)
    test_metrics = compute_metrics(splits.y_test, test_labels)
    print("Metricas en test:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    inv_labels = splits.label_encoder.inverse_transform(range(len(splits.label_encoder.classes_)))
    print("Reporte de clasificacion detallado:")
    print(
        metrics.classification_report(
            splits.y_test,
            test_labels,
            target_names=inv_labels,
            zero_division=0,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depuracion + LSTM para WISDM.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Ruta al fichero crudo de WISDM.",
    )
    parser.add_argument("--window-size", type=int, default=50, help="Tamaño de ventana temporal.")
    parser.add_argument("--step", type=int, default=25, help="Paso entre ventanas.")
    parser.add_argument("--folds", type=int, default=4, help="Numero de folds para CV.")
    parser.add_argument("--epochs", type=int, default=10, help="Epocas maximas por entrenamiento.")
    parser.add_argument(
        "--max-grid",
        type=int,
        default=12,
        help="Limite de combinaciones del grid para acotar el tiempo (None para usar todas).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
