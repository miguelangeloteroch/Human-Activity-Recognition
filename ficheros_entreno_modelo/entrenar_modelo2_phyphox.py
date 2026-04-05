"""
Script para REENTRENAR (fine-tuning) el Modelo 2 usando DATOS REALES de PhyPhox.
División 60% train / 40% test.

Carga el modelo 2 original entrenado con WISDM y lo ajusta con datos de PhyPhox.
Guarda el modelo reentrenado en models/modelo2_phyphox/ para no sobrescribir el original.
"""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, utils

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "modelo2_phyphox"
ORIGINAL_MODEL_DIR = PROJECT_ROOT / "models" / "modelo2"
RESULTS_DIR = PROJECT_ROOT / "results" / "modelo2_phyphox"

DEFAULT_DATA_PATH = DATA_DIR / "phyphox_all_labeled.csv"
DEFAULT_ORIGINAL_MODEL_PATH = ORIGINAL_MODEL_DIR / "modelo.h5"
DEFAULT_ORIGINAL_METADATA_PATH = ORIGINAL_MODEL_DIR / "metadata.pkl"
DEFAULT_MODEL_PATH = MODELS_DIR / "modelo.h5"
DEFAULT_METADATA_PATH = MODELS_DIR / "metadata.pkl"
DEFAULT_HISTORY_PATH = MODELS_DIR / "training_history.pkl"
DEFAULT_LOSS_FIG = RESULTS_DIR / "loss_curve.png"
DEFAULT_ACC_FIG = RESULTS_DIR / "accuracy_curve.png"
DEFAULT_CONFUSION_FIG = RESULTS_DIR / "confusion_matrix.png"
DEFAULT_F1_FIG = RESULTS_DIR / "f1_by_class.png"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

sns.set(style="whitegrid")

BASE_FEATURES = ["x", "y", "z", "accel_mag"]


def ensure_parent_dir(path: Path) -> None:
    """Crea el directorio padre si no existe."""
    path.parent.mkdir(parents=True, exist_ok=True)


def load_phyphox_processed(path: Path) -> pd.DataFrame:
    """Carga el CSV procesado de PhyPhox."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en {path}")
    
    df = pd.read_csv(path)
    required_cols = ["time", "x", "y", "z", "accel_mag", "activity"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")
    
    print(f"Dataset cargado: {len(df):,} filas")
    print(f"Distribución de clases: {df['activity'].value_counts().to_dict()}")
    return df


def create_sequences_phyphox(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = 50,
    step: int = 25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera ventanas temporales para datos de PhyPhox.
    Agrupa por actividad para evitar mezclar actividades en una ventana.
    """
    sequences: List[np.ndarray] = []
    labels: List[str] = []

    for activity, group_df in df.groupby("activity"):
        group_df = group_df.sort_values("time").reset_index(drop=True)
        feats = group_df[feature_cols].values
        
        for start in range(0, len(group_df) - window_size + 1, step):
            end = start + window_size
            window_feats = feats[start:end]
            sequences.append(window_feats)
            labels.append(activity)

    X = np.stack(sequences)
    y = np.array(labels)
    print(f"Secuencias generadas: {X.shape[0]} ventanas de {window_size} timesteps x {len(feature_cols)} features")
    return X, y


def normalize_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    """Normaliza las features usando StandardScaler."""
    scaler = StandardScaler()
    df_out = df.copy()
    df_out[feature_cols] = scaler.fit_transform(df_out[feature_cols])
    return df_out, scaler


def prepare_splits_phyphox(
    df: pd.DataFrame,
    window_size: int,
    step: int,
    test_size: float = 0.4,
) -> Dict[str, Any]:
    """Prepara los splits de train/test para datos de PhyPhox."""
    feature_cols = BASE_FEATURES
    
    # Normalización
    df_norm, scaler = normalize_features(df, feature_cols)
    
    # Generación de secuencias
    X, y = create_sequences_phyphox(df_norm, feature_cols, window_size=window_size, step=step)
    
    # Codificación de etiquetas
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # División train/test (60/40)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y_encoded,
    )
    
    print(f"División: {int((1-test_size)*100)}% train / {int(test_size*100)}% test")
    print(f"  Train: {X_train.shape[0]} muestras")
    print(f"  Test: {X_test.shape[0]} muestras")
    print(f"  Clases: {label_encoder.classes_.tolist()}")
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "label_encoder": label_encoder,
        "feature_names": feature_cols,
        "scaler": scaler,
    }


def load_pretrained_model(model_path: Path, metadata_path: Path) -> Tuple[models.Model, Dict[str, Any]]:
    """Carga el modelo 2 preentrenado y su metadata."""
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo preentrenado en {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"No se encontró la metadata en {metadata_path}")
    
    print(f"Cargando modelo preentrenado desde {model_path}...")
    model = models.load_model(model_path)
    
    print(f"Cargando metadata desde {metadata_path}...")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    print(f"Modelo cargado: {len(metadata.get('classes', []))} clases")
    print(f"  Clases: {metadata.get('classes', [])}")
    print(f"  Window size: {metadata.get('window_size', 'N/A')}")
    print(f"  Step: {metadata.get('step', 'N/A')}")
    
    return model, metadata


def prepare_model_for_finetuning(
    model: models.Model,
    learning_rate: float = 1e-4,
    freeze_lstm_layers: bool = False,
) -> models.Model:
    """
    Prepara el modelo para fine-tuning ajustando el learning rate.
    
    Args:
        model: Modelo preentrenado
        learning_rate: Learning rate para fine-tuning (más bajo que el original)
        freeze_lstm_layers: Si True, congela las capas LSTM y solo entrena las Dense
    """
    # Compilar con nuevo learning rate
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    
    # Opcionalmente congelar capas LSTM
    if freeze_lstm_layers:
        print("Congelando capas LSTM (solo se entrenarán las capas Dense)...")
        for layer in model.layers:
            if isinstance(layer, layers.LSTM):
                layer.trainable = False
            elif isinstance(layer, layers.Dropout):
                # Los Dropout se mantienen entrenables
                pass
            else:
                layer.trainable = True
    else:
        print("Todas las capas son entrenables (fine-tuning completo)")
    
    # Mostrar resumen de capas entrenables
    trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    print(f"Parámetros entrenables: {trainable_count:,}")
    print(f"Parámetros congelados: {non_trainable_count:,}")
    
    return model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula métricas de evaluación."""
    return {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "recall": metrics.recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": metrics.f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def validate_model_compatibility(
    original_metadata: Dict[str, Any],
    phyphox_classes: List[str],
    window_size: int,
    step: int,
) -> None:
    """Verifica que el modelo original sea compatible con los nuevos datos."""
    original_classes = original_metadata.get("classes", [])
    original_window = original_metadata.get("window_size")
    original_step = original_metadata.get("step")
    
    # Verificar clases
    if set(original_classes) != set(phyphox_classes):
        print(f"ADVERTENCIA: Las clases no coinciden exactamente:")
        print(f"  Original: {original_classes}")
        print(f"  PhyPhox: {phyphox_classes}")
        print("  Continuando de todas formas...")
    
    # Verificar window_size y step
    if original_window != window_size or original_step != step:
        print(f"ADVERTENCIA: Parámetros de ventana diferentes:")
        print(f"  Original: window_size={original_window}, step={original_step}")
        print(f"  PhyPhox: window_size={window_size}, step={step}")
        print("  Continuando de todas formas...")
    
    print("[OK] Compatibilidad verificada")


def finetune_model(
    model: models.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
    epochs: int,
    learning_rate: float = 1e-4,
) -> Tuple[models.Model, callbacks.History]:
    """
    Realiza fine-tuning del modelo preentrenado con los nuevos datos.
    
    Args:
        model: Modelo preentrenado ya preparado para fine-tuning
        X_train: Datos de entrenamiento
        y_train: Etiquetas de entrenamiento
        batch_size: Tamaño de batch
        epochs: Número máximo de épocas
        learning_rate: Learning rate para fine-tuning
    """
    print("\nIniciando fine-tuning del modelo preentrenado...")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epocas maximas: {epochs}")
    
    # Obtener número de clases desde la forma de salida del modelo
    num_classes = model.output_shape[1]
    y_cat = utils.to_categorical(y_train, num_classes=num_classes)
    
    # Early stopping con paciencia mayor para fine-tuning
    es = callbacks.EarlyStopping(
        patience=7, 
        restore_best_weights=True, 
        monitor="val_loss",
        verbose=1
    )
    
    # Reduce learning rate on plateau
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_cat,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es, reduce_lr],
        verbose=1,
    )
    
    return model, history


def plot_loss_curve(history: Dict[str, List[float]], output_path: Path) -> None:
    """Genera gráfica de curva de pérdida."""
    epochs = range(1, len(history.get("loss", [])) + 1)
    if not epochs:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Curva de pérdida - Modelo 2 (PhyPhox)")
    plt.legend()
    plt.tight_layout()
    ensure_parent_dir(output_path)
    plt.savefig(output_path)
    plt.close()


def plot_accuracy_curve(history: Dict[str, List[float]], output_path: Path) -> None:
    """Genera gráfica de curva de accuracy."""
    epochs = range(1, len(history.get("accuracy", [])) + 1)
    if not epochs:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["accuracy"], label="Train Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Curva de accuracy - Modelo 2 (PhyPhox)")
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
    """Genera matriz de confusión."""
    cm = metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names, square=True)
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta real")
    plt.title("Matriz de confusión - Modelo 2 (PhyPhox)")
    plt.tight_layout()
    ensure_parent_dir(output_path)
    plt.savefig(output_path)
    plt.close()


def plot_f1_by_class(report: Dict[str, Dict[str, float]], class_names: List[str], output_path: Path) -> None:
    """Genera gráfico de F1 por clase."""
    f1_scores = [report[name]["f1-score"] for name in class_names]
    plt.figure(figsize=(10, 5))
    sns.barplot(x=class_names, y=f1_scores, palette="viridis")
    plt.ylabel("F1-score")
    plt.xlabel("Clase")
    plt.title("F1-score por clase - Modelo 2 (PhyPhox)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    ensure_parent_dir(output_path)
    plt.savefig(output_path)
    plt.close()


def save_pickle(obj: Any, path: Path) -> None:
    """Guarda un objeto como pickle."""
    ensure_parent_dir(path)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


def run(args: argparse.Namespace) -> None:
    """Función principal de reentrenamiento (fine-tuning)."""
    print("=" * 60)
    print("REENTRENAMIENTO (FINE-TUNING) MODELO 2 - DATOS PHYPHOX (60/40)")
    print("=" * 60)
    
    # Cargar modelo preentrenado
    original_model_path = Path(args.original_model_path)
    original_metadata_path = Path(args.original_metadata_path)
    pretrained_model, original_metadata = load_pretrained_model(
        original_model_path, 
        original_metadata_path
    )
    
    # Cargar datos de PhyPhox procesados
    df = load_phyphox_processed(Path(args.data_path))

    # Preparar splits (60/40)
    splits = prepare_splits_phyphox(
        df,
        window_size=args.window_size,
        step=args.step,
        test_size=0.4,  # 40% test
    )
    
    # Verificar compatibilidad
    validate_model_compatibility(
        original_metadata,
        splits["label_encoder"].classes_.tolist(),
        args.window_size,
        args.step,
    )
    
    # Preparar modelo para fine-tuning
    model_for_finetuning = prepare_model_for_finetuning(
        pretrained_model,
        learning_rate=args.learning_rate,
        freeze_lstm_layers=args.freeze_lstm,
    )

    # Fine-tuning
    final_model, history = finetune_model(
        model_for_finetuning,
        splits["X_train"],
        splits["y_train"],
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    # Evaluación en test
    print("\n" + "=" * 60)
    print("EVALUACIÓN EN TEST (40%)")
    print("=" * 60)
    test_pred = np.argmax(final_model.predict(splits["X_test"], verbose=0), axis=1)
    test_metrics = compute_metrics(splits["y_test"], test_pred)
    print("\nMétricas en TEST:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")

    class_names = splits["label_encoder"].classes_.tolist()
    print("\nReporte detallado por clase:")
    report = classification_report(splits["y_test"], test_pred, target_names=class_names, zero_division=0)
    print(report)

    # Guardar modelo y metadata
    model_path = Path(args.model_path)
    ensure_parent_dir(model_path)
    final_model.save(model_path)
    print(f"\nModelo guardado: {model_path}")

    metadata = {
        "classes": class_names,
        "feature_names": splits["feature_names"],
        "window_size": args.window_size,
        "step": args.step,
        "scaler": splits["scaler"],
        "test_size": 0.4,
        "data_source": "phyphox",
        "test_metrics": test_metrics,
        "original_model_path": str(original_model_path),
        "fine_tuning_params": {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "freeze_lstm": args.freeze_lstm,
            "epochs_trained": len(history.history.get("loss", [])),
        },
    }
    metadata_path = Path(args.metadata_path)
    save_pickle(metadata, metadata_path)
    print(f"Metadata guardada: {metadata_path}")

    history_path = Path(args.history_path)
    save_pickle(history.history, history_path)
    print(f"Historial guardado: {history_path}")

    # Generar gráficas
    loss_fig = Path(args.loss_fig_path)
    acc_fig = Path(args.acc_fig_path)
    confusion_fig = Path(args.confusion_fig_path)
    f1_fig = Path(args.f1_fig_path)

    plot_loss_curve(history.history, loss_fig)
    plot_accuracy_curve(history.history, acc_fig)
    plot_confusion_matrix_fig(splits["y_test"], test_pred, class_names, confusion_fig)
    
    report_dict = classification_report(
        splits["y_test"], test_pred, 
        target_names=class_names, 
        zero_division=0, 
        output_dict=True
    )
    plot_f1_by_class(report_dict, class_names, f1_fig)

    print("\nGráficas generadas:")
    for path in [loss_fig, acc_fig, confusion_fig, f1_fig]:
        print(f"  - {path}")

    print("\n" + "=" * 60)
    print("REENTRENAMIENTO (FINE-TUNING) COMPLETADO")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reentrena (fine-tuning) Modelo 2 con datos de PhyPhox (60% train / 40% test)"
    )
    parser.add_argument(
        "--data-path", 
        type=str, 
        default=str(DEFAULT_DATA_PATH),
        help="Ruta al CSV procesado de PhyPhox"
    )
    parser.add_argument(
        "--original-model-path",
        type=str,
        default=str(DEFAULT_ORIGINAL_MODEL_PATH),
        help="Ruta al modelo 2 original preentrenado"
    )
    parser.add_argument(
        "--original-metadata-path",
        type=str,
        default=str(DEFAULT_ORIGINAL_METADATA_PATH),
        help="Ruta a la metadata del modelo original"
    )
    parser.add_argument("--window-size", type=int, default=50, help="Tamaño de ventana temporal")
    parser.add_argument("--step", type=int, default=25, help="Paso entre ventanas")
    parser.add_argument("--epochs", type=int, default=20, help="Épocas máximas para fine-tuning")
    parser.add_argument("--batch-size", type=int, default=64, help="Tamaño de batch para fine-tuning")
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=1e-4, 
        help="Learning rate para fine-tuning (default: 1e-4, más bajo que entrenamiento inicial)"
    )
    parser.add_argument(
        "--freeze-lstm",
        action="store_true",
        help="Congela las capas LSTM y solo entrena las capas Dense"
    )
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--metadata-path", type=str, default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--history-path", type=str, default=str(DEFAULT_HISTORY_PATH))
    parser.add_argument("--loss-fig-path", type=str, default=str(DEFAULT_LOSS_FIG))
    parser.add_argument("--acc-fig-path", type=str, default=str(DEFAULT_ACC_FIG))
    parser.add_argument("--confusion-fig-path", type=str, default=str(DEFAULT_CONFUSION_FIG))
    parser.add_argument("--f1-fig-path", type=str, default=str(DEFAULT_F1_FIG))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

