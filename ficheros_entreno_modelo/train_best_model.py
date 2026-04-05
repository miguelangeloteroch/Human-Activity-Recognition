"""
Entrena el mejor modelo LSTM detectado previamente y lo guarda en el directorio raiz del proyecto.

Permite especificar los hiperparametros ganadores mediante JSON o fichero, de forma que el script
solo se enfoque en reentrenar con todo el conjunto de entrenamiento y exportar el modelo/metadata.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sklearn import metrics

from wisdm_lstm_pipeline import (
    compute_metrics,
    load_raw_wisdm,
    prepare_splits,
    train_final_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_DATA_PATH = DATA_DIR / "WISDM_ar_latest" / "WISDM_ar_v1.1" / "WISDM_ar_v1.1_raw.txt"
DEFAULT_MODEL_PATH = MODELS_DIR / "best_wisdm_lstm.h5"
DEFAULT_METADATA_PATH = MODELS_DIR / "best_wisdm_lstm_metadata.pkl"
DEFAULT_HISTORY_PATH = MODELS_DIR / "training_history.pkl"

DEFAULT_PARAMS = {
    "lstm_units": 64,
    "dropout": 0.25,
    "dense_units": 64,
    "learning_rate": 1e-3,
    "batch_size": 96,
    "l2": 1e-4,
}


def parse_param_source(params_json: Optional[str], params_file: Optional[str]) -> Dict:
    if params_json:
        return json.loads(params_json)
    if params_file:
        with open(params_file, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return DEFAULT_PARAMS


def resolve_project_path(path_str: str) -> Path:
    """Resuelve rutas relativas con base en el directorio raiz del proyecto."""
    raw_path = Path(path_str).expanduser()
    if raw_path.is_absolute():
        return raw_path
    return PROJECT_ROOT / raw_path


def run_training(args: argparse.Namespace) -> None:
    params_file = resolve_project_path(args.params_file) if args.params_file else None
    params = parse_param_source(args.params_json, str(params_file) if params_file else None)
    print(f"Hiperpar?metros utilizados: {params}")

    data_path = resolve_project_path(args.data_path)
    df = load_raw_wisdm(data_path)
    splits = prepare_splits(df, window_size=args.window_size, step=args.step)

    model, history = train_final_model(
        splits.X_train,
        splits.y_train,
        num_classes=len(splits.label_encoder.classes_),
        params=params,
        epochs=args.epochs,
        return_history=True,
    )

    test_preds = model.predict(splits.X_test, verbose=0)
    pred_labels = np.argmax(test_preds, axis=1)
    metrics_dict = compute_metrics(splits.y_test, pred_labels)
    print("Metricas finales en test:")
    for k, v in metrics_dict.items():
        print(f"  {k}: {v:.4f}")
    print("Reporte detallado por clase:")
    print(
        metrics.classification_report(
            splits.y_test,
            pred_labels,
            target_names=splits.label_encoder.classes_,
            zero_division=0,
        )
    )

    model_output = resolve_project_path(args.model_path)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_output)
    print(f"Modelo guardado en {model_output}")

    metadata = {
        "classes": splits.label_encoder.classes_.tolist(),
        "feature_names": splits.feature_names,
        "window_size": args.window_size,
        "step": args.step,
        "scaler": splits.scaler,
    }
    metadata_path = resolve_project_path(args.metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "wb") as fh:
        pickle.dump(metadata, fh)
    print(f"Metadata guardada en {metadata_path}")

    history_path = resolve_project_path(args.history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "wb") as fh:
        pickle.dump(history.history, fh)
    print(f"Historial de entrenamiento guardado en {history_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena y guarda el mejor modelo LSTM de WISDM.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Ruta al fichero crudo de WISDM.",
    )
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--step", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument(
        "--params-json",
        type=str,
        default=None,
        help='JSON con hiperparámetros ganadores, ejemplo: \'{"lstm_units":64,...}\'.',
    )
    parser.add_argument(
        "--params-file",
        type=str,
        default=None,
        help="Ruta a un fichero JSON con los hiperparámetros ganadores.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Ruta destino para guardar el modelo entrenado.",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=str(DEFAULT_METADATA_PATH),
        help="Ruta destino para guardar metadata asociada (clases, scaler, etc.).",
    )
    parser.add_argument(
        "--history-path",
        type=str,
        default=str(DEFAULT_HISTORY_PATH),
        help="Ruta destino para guardar history.history del entrenamiento final.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
