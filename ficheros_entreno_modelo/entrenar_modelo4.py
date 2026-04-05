"""
Script para crear Modelo 4 mediante fine-tuning del Modelo 2 con dataset híbrido.
Dataset híbrido: 70% PhyPhox + 30% WISDM balanceado por clases.
División 80% train / 20% test.

Carga el modelo 2 original entrenado con WISDM y lo ajusta con el dataset híbrido.
Guarda el modelo en models/modelo4/.
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
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, utils

from src.wisdm_lstm_pipeline import (
    create_sequences,
    engineer_features,
    impute_missing,
    load_raw_wisdm,
    remap_activity_labels,
    winsorize_features,
    RANDOM_SEED,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "modelo4"
ORIGINAL_MODEL_DIR = PROJECT_ROOT / "models" / "modelo2"
RESULTS_DIR = PROJECT_ROOT / "results" / "modelo4"

DEFAULT_PHYPHOX_DATA_PATH = PROCESSED_DATA_DIR / "phyphox_all_labeled.csv"
DEFAULT_WISDM_DATA_PATH = DATA_DIR / "WISDM_ar_latest" / "WISDM_at_v2.0_raw.txt"
DEFAULT_ORIGINAL_MODEL_PATH = ORIGINAL_MODEL_DIR / "modelo.h5"
DEFAULT_ORIGINAL_METADATA_PATH = ORIGINAL_MODEL_DIR / "metadata.pkl"
DEFAULT_MODEL_PATH = MODELS_DIR / "modelo.h5"
DEFAULT_METADATA_PATH = MODELS_DIR / "metadata.pkl"
DEFAULT_HISTORY_PATH = MODELS_DIR / "training_history.pkl"
DEFAULT_LOSS_FIG = RESULTS_DIR / "loss_curve.png"
DEFAULT_ACC_FIG = RESULTS_DIR / "accuracy_curve.png"
DEFAULT_CONFUSION_FIG = RESULTS_DIR / "confusion_matrix.png"
DEFAULT_F1_FIG = RESULTS_DIR / "f1_by_class.png"

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
    
    print(f"Dataset PhyPhox cargado: {len(df):,} filas")
    print(f"Distribución de clases: {df['activity'].value_counts().to_dict()}")
    return df


def load_wisdm_balanced_subset(
    wisdm_path: Path,
    scaler: Any,
    feature_names: List[str],
    window_size: int,
    step: int,
    phyphox_ratio: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga WISDM, lo procesa y extrae un subconjunto balanceado por clases.
    
    Args:
        wisdm_path: Ruta al archivo WISDM raw
        scaler: Scaler del modelo original para normalizar
        feature_names: Nombres de features esperadas
        window_size: Tamaño de ventana temporal
        step: Paso entre ventanas
        phyphox_ratio: Ratio de PhyPhox (para calcular cuánto WISDM necesitamos)
    
    Returns:
        Tuple (X_wisdm, y_wisdm) con ventanas balanceadas de WISDM
    """
    print("\n" + "=" * 60)
    print("CARGANDO Y PROCESANDO WISDM PARA DATASET HÍBRIDO")
    print("=" * 60)
    
    # Cargar WISDM raw
    print(f"Cargando WISDM desde {wisdm_path}...")
    df = load_raw_wisdm(wisdm_path)
    print(f"Dataset WISDM cargado: {len(df):,} filas tras limpieza inicial.")
    
    # Remapear a 3 clases
    df = remap_activity_labels(df)
    print(f"Actividades remapeadas a 3 clases. Total: {len(df):,} filas.")
    print(f"Distribución: {df['activity'].value_counts().to_dict()}")
    
    # Engineer features
    df_feat, feature_cols = engineer_features(df)
    
    # Verificar que las features coincidan
    if set(feature_cols) != set(feature_names):
        print(f"ADVERTENCIA: Features diferentes:")
        print(f"  Dataset: {feature_cols}")
        print(f"  Modelo: {feature_names}")
        feature_cols = [f for f in feature_cols if f in feature_names]
        if len(feature_cols) != len(feature_names):
            raise ValueError(f"No se pueden alinear las features: {feature_cols} vs {feature_names}")
    
    # Ordenar features según el orden del modelo
    feature_cols = [f for f in feature_names if f in feature_cols]
    
    # Winsorizar outliers
    df_feat = winsorize_features(df_feat, feature_cols, limits=(0.01, 0.01))
    
    # Imputar missing
    df_feat = impute_missing(df_feat, feature_cols)
    
    # Normalizar usando el scaler del modelo original
    df_feat[feature_cols] = scaler.transform(df_feat[feature_cols])
    
    # Generar secuencias
    X_all, y_all = create_sequences(df_feat, feature_cols, window_size=window_size, step=step)
    print(f"\nSecuencias WISDM generadas: {X_all.shape[0]} ventanas")
    
    # Calcular cuántas ventanas necesitamos por clase para balancear
    # Si PhyPhox tiene N ventanas y queremos 70/30, entonces:
    # N_phyphox = 0.7 * total
    # N_wisdm = 0.3 * total
    # N_wisdm = N_phyphox * (0.3 / 0.7) = N_phyphox * (1 - phyphox_ratio) / phyphox_ratio
    # Pero como aún no sabemos cuántas ventanas tiene PhyPhox, vamos a generar todas
    # y luego balancear después de saber cuántas tiene PhyPhox
    
    # Por ahora, extraer un subconjunto balanceado
    # Dividir por clase
    unique_classes = np.unique(y_all)
    print(f"\nClases encontradas en WISDM: {unique_classes}")
    
    # Contar ventanas por clase
    class_counts = {cls: np.sum(y_all == cls) for cls in unique_classes}
    print(f"Ventanas por clase: {class_counts}")
    
    # Retornar todas las ventanas (el balanceo se hará después de saber cuántas tiene PhyPhox)
    return X_all, y_all


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
    print(f"Secuencias PhyPhox generadas: {X.shape[0]} ventanas de {window_size} timesteps x {len(feature_cols)} features")
    return X, y


def create_hybrid_dataset(
    phyphox_df: pd.DataFrame,
    X_wisdm: np.ndarray,
    y_wisdm: np.ndarray,
    feature_cols: List[str],
    window_size: int,
    step: int,
    phyphox_ratio: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Crea un dataset híbrido combinando PhyPhox (70%) y WISDM balanceado (30%).
    
    Args:
        phyphox_df: DataFrame de PhyPhox procesado
        X_wisdm: Ventanas de WISDM
        y_wisdm: Etiquetas de WISDM (strings)
        feature_cols: Nombres de features
        window_size: Tamaño de ventana
        step: Paso entre ventanas
        phyphox_ratio: Ratio de PhyPhox (default 0.7 = 70%)
    
    Returns:
        Tuple (X_hybrid, y_hybrid, label_encoder)
    """
    print("\n" + "=" * 60)
    print("CREANDO DATASET HÍBRIDO")
    print("=" * 60)
    
    # Generar ventanas PhyPhox
    X_phyphox, y_phyphox = create_sequences_phyphox(phyphox_df, feature_cols, window_size, step)
    n_phyphox = len(X_phyphox)
    print(f"\nVentanas PhyPhox: {n_phyphox}")
    
    # Calcular cuántas ventanas WISDM necesitamos
    # Si PhyPhox es 70% del total, entonces:
    # total = n_phyphox / phyphox_ratio
    # n_wisdm = total * (1 - phyphox_ratio) = n_phyphox * (1 - phyphox_ratio) / phyphox_ratio
    n_wisdm_target = int(n_phyphox * (1 - phyphox_ratio) / phyphox_ratio)
    print(f"Ventanas WISDM objetivo: {n_wisdm_target} (30% del total)")
    
    # Balancear WISDM: extraer igual número de ventanas por clase
    unique_classes = np.unique(y_wisdm)
    n_per_class = n_wisdm_target // len(unique_classes)
    print(f"Ventanas por clase WISDM: {n_per_class}")
    
    X_wisdm_balanced = []
    y_wisdm_balanced = []
    
    for cls in unique_classes:
        # Obtener índices de esta clase
        cls_indices = np.where(y_wisdm == cls)[0]
        
        # Si hay suficientes, tomar n_per_class aleatoriamente
        if len(cls_indices) >= n_per_class:
            selected_indices = np.random.choice(cls_indices, size=n_per_class, replace=False)
        else:
            # Si no hay suficientes, tomar todas y repetir si es necesario
            print(f"  ADVERTENCIA: Clase '{cls}' solo tiene {len(cls_indices)} ventanas, tomando todas")
            selected_indices = cls_indices
            if n_per_class > len(cls_indices):
                # Repetir algunas para llegar al objetivo
                extra_needed = n_per_class - len(cls_indices)
                extra_indices = np.random.choice(cls_indices, size=extra_needed, replace=True)
                selected_indices = np.concatenate([selected_indices, extra_indices])
        
        X_wisdm_balanced.append(X_wisdm[selected_indices])
        y_wisdm_balanced.extend(y_wisdm[selected_indices])
    
    X_wisdm_balanced = np.vstack(X_wisdm_balanced)
    y_wisdm_balanced = np.array(y_wisdm_balanced)
    
    print(f"\nVentanas WISDM seleccionadas: {len(X_wisdm_balanced)}")
    print(f"Distribución WISDM: {dict(zip(*np.unique(y_wisdm_balanced, return_counts=True)))}")
    
    # Combinar datasets
    X_hybrid = np.vstack([X_phyphox, X_wisdm_balanced])
    y_hybrid = np.concatenate([y_phyphox, y_wisdm_balanced])
    
    print(f"\nDataset híbrido creado:")
    print(f"  Total ventanas: {len(X_hybrid)}")
    print(f"  PhyPhox: {len(X_phyphox)} ({len(X_phyphox)/len(X_hybrid)*100:.1f}%)")
    print(f"  WISDM: {len(X_wisdm_balanced)} ({len(X_wisdm_balanced)/len(X_hybrid)*100:.1f}%)")
    print(f"  Distribución total: {dict(zip(*np.unique(y_hybrid, return_counts=True)))}")
    
    # Codificar etiquetas
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_hybrid)
    
    return X_hybrid, y_encoded, label_encoder


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
    hybrid_classes: List[str],
    window_size: int,
    step: int,
) -> None:
    """Verifica que el modelo original sea compatible con los nuevos datos."""
    original_classes = original_metadata.get("classes", [])
    original_window = original_metadata.get("window_size")
    original_step = original_metadata.get("step")
    
    # Verificar clases
    if set(original_classes) != set(hybrid_classes):
        print(f"ADVERTENCIA: Las clases no coinciden exactamente:")
        print(f"  Original: {original_classes}")
        print(f"  Híbrido: {hybrid_classes}")
        print("  Continuando de todas formas...")
    
    # Verificar window_size y step
    if original_window != window_size or original_step != step:
        print(f"ADVERTENCIA: Parámetros de ventana diferentes:")
        print(f"  Original: window_size={original_window}, step={original_step}")
        print(f"  Híbrido: window_size={window_size}, step={step}")
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
    plt.title("Curva de pérdida - Modelo 4 (Híbrido)")
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
    plt.title("Curva de accuracy - Modelo 4 (Híbrido)")
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
    plt.title("Matriz de confusión - Modelo 4 (Híbrido)")
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
    plt.title("F1-score por clase - Modelo 4 (Híbrido)")
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
    """Función principal de entrenamiento del Modelo 4."""
    print("=" * 60)
    print("ENTRENAMIENTO MODELO 4 - FINE-TUNING CON DATASET HÍBRIDO")
    print("70% PhyPhox + 30% WISDM balanceado")
    print("=" * 60)
    
    # Cargar modelo preentrenado
    original_model_path = Path(args.original_model_path)
    original_metadata_path = Path(args.original_metadata_path)
    pretrained_model, original_metadata = load_pretrained_model(
        original_model_path, 
        original_metadata_path
    )
    
    # Obtener parámetros del modelo original
    window_size = original_metadata.get("window_size", args.window_size)
    step = original_metadata.get("step", args.step)
    feature_names = original_metadata.get("feature_names", BASE_FEATURES)
    original_scaler = original_metadata.get("scaler")
    
    if original_scaler is None:
        raise ValueError("El modelo original no tiene scaler guardado")
    
    print(f"\nParámetros del modelo original:")
    print(f"  Window size: {window_size}")
    print(f"  Step: {step}")
    print(f"  Features: {feature_names}")
    
    # Cargar datos de PhyPhox procesados
    print("\n" + "=" * 60)
    print("CARGANDO DATOS PHYPHOX")
    print("=" * 60)
    phyphox_df = load_phyphox_processed(Path(args.phyphox_data_path))
    
    # Normalizar PhyPhox con el scaler del modelo original
    print("\nNormalizando PhyPhox con scaler del modelo original...")
    phyphox_df[feature_names] = original_scaler.transform(phyphox_df[feature_names])
    
    # Cargar y procesar WISDM
    X_wisdm, y_wisdm = load_wisdm_balanced_subset(
        Path(args.wisdm_data_path),
        original_scaler,
        feature_names,
        window_size,
        step,
        phyphox_ratio=args.phyphox_ratio,
    )
    
    # Crear dataset híbrido
    X_hybrid, y_hybrid, label_encoder = create_hybrid_dataset(
        phyphox_df,
        X_wisdm,
        y_wisdm,
        feature_names,
        window_size,
        step,
        phyphox_ratio=args.phyphox_ratio,
    )
    
    # Verificar compatibilidad
    validate_model_compatibility(
        original_metadata,
        label_encoder.classes_.tolist(),
        window_size,
        step,
    )
    
    # División train/test (80/20)
    print("\n" + "=" * 60)
    print("DIVISIÓN TRAIN/TEST (80/20)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X_hybrid, y_hybrid,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_hybrid,
    )
    
    print(f"División: 80% train / 20% test")
    print(f"  Train: {X_train.shape[0]} muestras")
    print(f"  Test: {X_test.shape[0]} muestras")
    print(f"  Clases: {label_encoder.classes_.tolist()}")
    
    # Preparar modelo para fine-tuning
    model_for_finetuning = prepare_model_for_finetuning(
        pretrained_model,
        learning_rate=args.learning_rate,
        freeze_lstm_layers=args.freeze_lstm,
    )

    # Fine-tuning
    final_model, history = finetune_model(
        model_for_finetuning,
        X_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    # Evaluación en test
    print("\n" + "=" * 60)
    print("EVALUACIÓN EN TEST (20%)")
    print("=" * 60)
    test_pred = np.argmax(final_model.predict(X_test, verbose=0), axis=1)
    test_metrics = compute_metrics(y_test, test_pred)
    print("\nMétricas en TEST:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")

    class_names = label_encoder.classes_.tolist()
    print("\nReporte detallado por clase:")
    report = classification_report(y_test, test_pred, target_names=class_names, zero_division=0)
    print(report)

    # Guardar modelo y metadata
    model_path = Path(args.model_path)
    ensure_parent_dir(model_path)
    final_model.save(model_path)
    print(f"\nModelo guardado: {model_path}")

    metadata = {
        "classes": class_names,
        "feature_names": feature_names,
        "window_size": window_size,
        "step": step,
        "scaler": original_scaler,  # Usar el scaler original
        "test_size": 0.2,
        "data_source": "hybrid_70phyphox_30wisdm",
        "phyphox_ratio": args.phyphox_ratio,
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
    plot_confusion_matrix_fig(y_test, test_pred, class_names, confusion_fig)
    
    report_dict = classification_report(
        y_test, test_pred, 
        target_names=class_names, 
        zero_division=0, 
        output_dict=True
    )
    plot_f1_by_class(report_dict, class_names, f1_fig)

    print("\nGráficas generadas:")
    for path in [loss_fig, acc_fig, confusion_fig, f1_fig]:
        print(f"  - {path}")

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO MODELO 4 COMPLETADO")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena Modelo 4 con fine-tuning usando dataset híbrido (70% PhyPhox + 30% WISDM balanceado)"
    )
    parser.add_argument(
        "--phyphox-data-path", 
        type=str, 
        default=str(DEFAULT_PHYPHOX_DATA_PATH),
        help="Ruta al CSV procesado de PhyPhox"
    )
    parser.add_argument(
        "--wisdm-data-path",
        type=str,
        default=str(DEFAULT_WISDM_DATA_PATH),
        help="Ruta al archivo WISDM raw"
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
    parser.add_argument("--window-size", type=int, default=50, help="Tamaño de ventana temporal (se usa el del modelo original si está disponible)")
    parser.add_argument("--step", type=int, default=25, help="Paso entre ventanas (se usa el del modelo original si está disponible)")
    parser.add_argument("--epochs", type=int, default=20, help="Épocas máximas para fine-tuning")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamaño de batch para fine-tuning")
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=5e-5, 
        help="Learning rate para fine-tuning (default: 5e-5)"
    )
    parser.add_argument(
        "--phyphox-ratio",
        type=float,
        default=0.7,
        help="Ratio de PhyPhox en el dataset híbrido (default: 0.7 = 70%)"
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



















