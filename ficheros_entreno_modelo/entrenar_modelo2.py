"""
Script para entrenar el Modelo 2 (WISDM_at, 3 clases, SIN class_weight).
Este script entrena el modelo sin balanceo de clases para comparar con el modelo 3.
"""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import ParameterGrid, StratifiedKFold
import tensorflow as tf
from tensorflow.keras import callbacks, utils

from src.wisdm_lstm_pipeline import (
    build_lstm_model,
    compute_metrics,
    load_raw_wisdm,
    prepare_splits,
    remap_activity_labels,
    train_final_model,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "modelo2"
RESULTS_DIR = PROJECT_ROOT / "results" / "modelo2" / "dataset_original"

DEFAULT_DATA_PATH = DATA_DIR / "WISDM_ar_latest" / "WISDM_at_v2.0_raw.txt"
DEFAULT_MODEL_PATH = MODELS_DIR / "modelo.h5"
DEFAULT_METADATA_PATH = MODELS_DIR / "metadata.pkl"
DEFAULT_HISTORY_PATH = MODELS_DIR / "training_history.pkl"
DEFAULT_LOSS_FIG = RESULTS_DIR / "loss_curve.png"
DEFAULT_ACC_FIG = RESULTS_DIR / "accuracy_curve.png"
DEFAULT_CONFUSION_FIG = RESULTS_DIR / "confusion_matrix.png"
DEFAULT_F1_FIG = RESULTS_DIR / "f1_by_class.png"
DEFAULT_BOXPLOT_FIG = RESULTS_DIR / "cv_boxplots.png"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

sns.set(style="whitegrid")


def resolve_project_path(path_str: str) -> Path:
    """Resuelve rutas relativas respecto al directorio del proyecto."""
    raw_path = Path(path_str).expanduser()
    if raw_path.is_absolute():
        return raw_path
    return PROJECT_ROOT / raw_path


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def format_params(params: Dict[str, Any]) -> str:
    ordered_keys = ["lstm_units", "dropout", "dense_units", "learning_rate", "batch_size", "l2"]
    return ", ".join(f"{key}={params[key]}" for key in ordered_keys if key in params)


def cross_validate_with_tracking(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    param_grid: Dict[str, List[Any]],
    folds: int,
    epochs: int,
    max_combos: Optional[int],
) -> Tuple[Dict[str, Any], Dict[str, float], pd.DataFrame]:
    """Ejecuta CV SIN class_weight guardando métricas por combinación para graficar boxplots."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    combos = list(ParameterGrid(param_grid))
    if max_combos is not None and max_combos > 0:
        combos = combos[:max_combos]

    input_shape = (X.shape[1], X.shape[2])
    y_cat = utils.to_categorical(y, num_classes=num_classes)

    fold_records: List[Dict[str, Any]] = []
    best_params: Optional[Dict[str, Any]] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_f1 = -np.inf

    for combo_idx, params in enumerate(combos, start=1):
        print(f"\n[Combo {combo_idx}/{len(combos)}] Probando hiperparámetros: {params}")
        fold_scores: List[Dict[str, float]] = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            model = build_lstm_model(input_shape, num_classes, params)
            # SIN class_weight - patience original (3)
            es = callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss")
            history = model.fit(
                X[train_idx],
                y_cat[train_idx],
                validation_data=(X[val_idx], y_cat[val_idx]),
                epochs=epochs,
                batch_size=params["batch_size"],
                verbose=2,
                callbacks=[es],
                # NO se pasa class_weight
            )
            preds = model.predict(X[val_idx], verbose=0)
            pred_labels = np.argmax(preds, axis=1)
            metrics_dict = compute_metrics(y[val_idx], pred_labels)
            fold_scores.append(metrics_dict)

            fold_records.append(
                {
                    "combo_index": combo_idx,
                    "combo_name": f"Combo {combo_idx}",
                    "fold": fold_idx,
                    "accuracy": metrics_dict["accuracy"],
                    "recall": metrics_dict["recall"],
                    "f1": metrics_dict["f1"],
                    "params_summary": format_params(params),
                    **{key: params[key] for key in params},
                    "epochs_trained": len(history.history.get("loss", [])),
                }
            )
            print(
                f"  Fold {fold_idx}: accuracy={metrics_dict['accuracy']:.4f}, "
                f"recall={metrics_dict['recall']:.4f}, f1={metrics_dict['f1']:.4f}"
            )

        avg_acc = float(np.mean([score["accuracy"] for score in fold_scores]))
        avg_rec = float(np.mean([score["recall"] for score in fold_scores]))
        avg_f1 = float(np.mean([score["f1"] for score in fold_scores]))
        print(
            f"[Combo {combo_idx}] Promedio folds -> accuracy={avg_acc:.4f}, "
            f"recall={avg_rec:.4f}, f1={avg_f1:.4f}"
        )

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_params = params
            best_metrics = {"accuracy": avg_acc, "recall": avg_rec, "f1": avg_f1}

    if not best_params or not best_metrics:
        raise RuntimeError("La búsqueda en grid no produjo resultados válidos.")

    records_df = pd.DataFrame(fold_records)
    if not records_df.empty:
        order = [f"Combo {idx}" for idx in sorted(records_df["combo_index"].unique())]
        records_df["combo_name"] = pd.Categorical(records_df["combo_name"], categories=order, ordered=True)
    return best_params, best_metrics, records_df


def plot_loss_curve(history: Dict[str, List[float]], output_path: Path) -> None:
    epochs = range(1, len(history.get("loss", [])) + 1)
    if not epochs:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Curva de pérdida - Modelo 2")
    plt.legend()
    plt.tight_layout()
    ensure_parent_dir(output_path)
    plt.savefig(output_path)
    plt.close()


def plot_accuracy_curve(history: Dict[str, List[float]], output_path: Path) -> None:
    epochs = range(1, len(history.get("accuracy", [])) + 1)
    if not epochs:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["accuracy"], label="Train Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Curva de accuracy - Modelo 2")
    plt.legend()
    plt.tight_layout()
    ensure_parent_dir(output_path)
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix_fig(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    cm = metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, square=True)
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta real")
    plt.title("Matriz de confusión - Modelo 2")
    plt.tight_layout()
    ensure_parent_dir(output_path)
    plt.savefig(output_path)
    plt.close()


def plot_f1_by_class(report: Dict[str, Dict[str, float]], class_names: List[str], output_path: Path) -> None:
    f1_scores = [report[name]["f1-score"] for name in class_names]
    plt.figure(figsize=(10, 5))
    sns.barplot(x=class_names, y=f1_scores, palette="viridis")
    plt.ylabel("F1-score")
    plt.xlabel("Clase")
    plt.title("F1-score por clase - Modelo 2")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    ensure_parent_dir(output_path)
    plt.savefig(output_path)
    plt.close()


def plot_cv_boxplots(records_df: pd.DataFrame, output_path: Path) -> None:
    if records_df.empty:
        print("No hay resultados de CV para graficar boxplots.")
        return

    metrics_to_plot = ["accuracy", "recall", "f1"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(6 * len(metrics_to_plot), 5), sharey=False)
    if len(metrics_to_plot) == 1:
        axes = [axes]

    for ax, metric_name in zip(axes, metrics_to_plot):
        sns.boxplot(data=records_df, x="combo_name", y=metric_name, ax=ax)
        ax.set_title(metric_name.title())
        ax.set_xlabel("Combinación de hiperparámetros")
        ax.set_ylabel(metric_name)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Distribución de métricas por combinación en CV - Modelo 2", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    ensure_parent_dir(output_path)
    fig.savefig(output_path)
    plt.close(fig)


def save_pickle(obj: Any, path: Path) -> None:
    ensure_parent_dir(path)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


def run(args: argparse.Namespace) -> None:
    data_path = resolve_project_path(args.data_path)
    df = load_raw_wisdm(data_path)
    print(f"Dataset cargado: {len(df):,} filas tras limpieza inicial.")

    # Remapear a 3 clases: walking, jogging, stationary
    df = remap_activity_labels(df)
    print(f"Actividades remapeadas a 3 clases. Total: {len(df):,} filas.")
    print(f"Distribución: {df['activity'].value_counts().to_dict()}")

    splits = prepare_splits(df, window_size=args.window_size, step=args.step)
    print(
        f"Secuencias -> train={splits.X_train.shape}, test={splits.X_test.shape}, "
        f"clases={len(splits.label_encoder.classes_)}"
    )

    param_grid = {
        "lstm_units": [32, 64, 96],
        "dropout": [0.1, 0.25, 0.35],
        "dense_units": [32, 64, 96],
        "learning_rate": [1e-3, 5e-4, 1e-4],
        "batch_size": [64, 96, 128],
        "l2": [0.0, 1e-4, 1e-3],
    }
    max_grid = args.max_grid if args.max_grid and args.max_grid > 0 else None

    best_params, best_metrics, cv_records = cross_validate_with_tracking(
        splits.X_train,
        splits.y_train,
        num_classes=len(splits.label_encoder.classes_),
        param_grid=param_grid,
        folds=args.folds,
        epochs=args.epochs,
        max_combos=max_grid,
    )
    print(
        f"\nMejor combinación CV (F1={best_metrics['f1']:.4f}): {best_params} "
        f"// accuracy={best_metrics['accuracy']:.4f}, recall={best_metrics['recall']:.4f}"
    )

    boxplot_fig = resolve_project_path(args.boxplot_path)
    plot_cv_boxplots(cv_records, boxplot_fig)

    # Entrenamiento final SIN class_weight
    final_model, history = train_final_model(
        splits.X_train,
        splits.y_train,
        num_classes=len(splits.label_encoder.classes_),
        params=best_params,
        epochs=args.epochs,
        verbose=1,
        return_history=True,
        class_weight=None,  # SIN class_weight
    )

    test_probs = final_model.predict(splits.X_test, verbose=0)
    test_pred = np.argmax(test_probs, axis=1)
    test_metrics = compute_metrics(splits.y_test, test_pred)
    print("\nMétricas finales en test:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")

    class_names = splits.label_encoder.classes_.tolist()
    report = classification_report(
        splits.y_test,
        test_pred,
        target_names=class_names,
        zero_division=0,
    )
    print("\nReporte por clase en test:")
    print(report)

    model_path = resolve_project_path(args.model_path)
    ensure_parent_dir(model_path)
    final_model.save(model_path)
    print(f"Modelo guardado en {model_path}")

    metadata = {
        "classes": class_names,
        "feature_names": splits.feature_names,
        "window_size": args.window_size,
        "step": args.step,
        "scaler": splits.scaler,
        "best_params": best_params,
    }
    metadata_path = resolve_project_path(args.metadata_path)
    save_pickle(metadata, metadata_path)
    print(f"Metadata guardada en {metadata_path}")

    history_path = resolve_project_path(args.history_path)
    save_pickle(history.history, history_path)
    print(f"Historial guardado en {history_path}")

    loss_fig = resolve_project_path(args.loss_fig_path)
    acc_fig = resolve_project_path(args.acc_fig_path)
    confusion_fig = resolve_project_path(args.confusion_fig_path)
    f1_fig = resolve_project_path(args.f1_fig_path)

    plot_loss_curve(history.history, loss_fig)
    plot_accuracy_curve(history.history, acc_fig)
    plot_confusion_matrix_fig(splits.y_test, test_pred, class_names, confusion_fig)
    report_dict = classification_report(
        splits.y_test,
        test_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    plot_f1_by_class(report_dict, class_names, f1_fig)

    print("\nGráficas generadas:")
    for path in [loss_fig, acc_fig, confusion_fig, f1_fig, boxplot_fig]:
        print(f" - {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el Modelo 2 (WISDM_at, SIN class_weight).")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Ruta al fichero crudo WISDM_at_v2.0.",
    )
    parser.add_argument("--window-size", type=int, default=50, help="Tamaño de ventana temporal.")
    parser.add_argument("--step", type=int, default=25, help="Paso entre ventanas.")
    parser.add_argument("--folds", type=int, default=4, help="Número de folds en la validación cruzada.")
    parser.add_argument("--epochs", type=int, default=10, help="Épocas máximas por entrenamiento.")
    parser.add_argument(
        "--max-grid",
        type=int,
        default=12,
        help="Límite de combinaciones del grid (usa 0 para recorrer todo).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Ruta destino para guardar el modelo final.",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=str(DEFAULT_METADATA_PATH),
        help="Ruta destino para guardar la metadata.",
    )
    parser.add_argument(
        "--history-path",
        type=str,
        default=str(DEFAULT_HISTORY_PATH),
        help="Ruta destino para guardar history.history.",
    )
    parser.add_argument(
        "--loss-fig-path",
        type=str,
        default=str(DEFAULT_LOSS_FIG),
        help="Ruta para la curva de pérdida.",
    )
    parser.add_argument(
        "--acc-fig-path",
        type=str,
        default=str(DEFAULT_ACC_FIG),
        help="Ruta para la curva de accuracy.",
    )
    parser.add_argument(
        "--confusion-fig-path",
        type=str,
        default=str(DEFAULT_CONFUSION_FIG),
        help="Ruta para la matriz de confusión.",
    )
    parser.add_argument(
        "--f1-fig-path",
        type=str,
        default=str(DEFAULT_F1_FIG),
        help="Ruta para el gráfico de F1 por clase.",
    )
    parser.add_argument(
        "--boxplot-path",
        type=str,
        default=str(DEFAULT_BOXPLOT_FIG),
        help="Ruta para los boxplots de la validación cruzada.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())



















