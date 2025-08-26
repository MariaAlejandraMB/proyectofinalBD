import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    classification_report, confusion_matrix, ConfusionMatrixDisplay, 
    roc_auc_score, balanced_accuracy_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import time

# Configuración de la página
st.set_page_config(
    page_title="AutoML con Diagnóstico Completo",
    page_icon="🤖",
    layout="wide",
)

# =========================== CONFIGURACIÓN Y CONSTANTES ===========================
# XGBoost es opcional
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

METRIC_FUNCS = {
    'accuracy': accuracy_score,
    'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
    'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
    'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'),
    'balanced_accuracy': balanced_accuracy_score,
    'roc_auc': roc_auc_score,
}

ALGO_KEY = {
    'Random Forest': 'random_forest',
    'Logistic Regression': 'logistic_regression',
    'XGBoost': 'xgboost',
}

PAYMENT_KEYWORDS = ['payment', 'pay', 'method', 'type', 'credit', 'debit', 'boleto', 'voucher']
# PAYMENT_KEYWORDS = ['payment_method', 'payment_type']

# =========================== FUNCIONES DE UTILIDAD ===========================
def parse_list_from_text(text, cast_fn=float):
    """Convierte "1, 2, 3" -> [1,2,3] con el tipo indicado."""
    items = [t.strip() for t in text.split(',') if t.strip() != '']
    out = []
    for it in items:
        if it.lower() == 'none':
            out.append(None)
        else:
            out.append(cast_fn(it))
    return out

def compute_metric(y_true, y_pred, metric_name: str) -> float:
    return METRIC_FUNCS[metric_name](y_true, y_pred)

def split_train_test_oot(X, y, test_size, oot_size, random_state=42):
    """Divide en Test primero y luego separa OOT del remanente, con estratificación."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    adjusted_oot = oot_size / (1 - test_size)
    X_train, X_oot, y_train, y_oot = train_test_split(
        X_temp, y_temp, test_size=adjusted_oot, stratify=y_temp, random_state=random_state
    )
    return X_train, X_test, X_oot, y_train, y_test, y_oot

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Genera una matriz de confusión visual"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.title('Matriz de Confusión')
    plt.xticks(rotation=45)
    return fig

def verify_balancing(y_train, balance_method, model_name):
    """Verifica y muestra información sobre el balanceo aplicado"""
    # st.write(f"**Verificación de balanceo para {model_name}:**")
    # st.write(f"- Método de balanceo: {balance_method}")
    # st.write(f"- Distribución original: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    
    # if balance_method == 'class_weight':
    #     st.write("- Se aplicaron pesos automáticos a las clases")
    # elif balance_method == 'scale_pos_weight':
    #     if len(np.unique(y_train)) == 2:
    #         ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    #         st.write(f"- Se aplicó scale_pos_weight con ratio: {ratio:.2f}")
    #     else:
    #         st.warning("- scale_pos_weight solo funciona para problemas binarios")
    # elif balance_method in ['smote', 'oversample', 'undersample']:
    #     st.write("- Se aplicó remuestreo con la técnica seleccionada")
    # else:
    #     st.write("- No se aplicó técnica de balanceo")

def plot_class_distribution_comparison(y_original, y_resampled, title):
    """Compara visualmente la distribución de clases antes y después del balanceo"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Distribución original
    original_counts = y_original.value_counts()
    ax1.bar([str(x) for x in original_counts.index], original_counts.values)
    ax1.set_title('Distribución Original')
    ax1.set_xlabel('Clase')
    ax1.set_ylabel('Count')
    
    # Distribución después del balanceo
    if y_resampled is not None:
        resampled_counts = y_resampled.value_counts()
        ax2.bar([str(x) for x in resampled_counts.index], resampled_counts.values)
        ax2.set_title('Distribución después del Balanceo')
        ax2.set_xlabel('Clase')
        ax2.set_ylabel('Count')
    else:
        ax2.text(0.5, 0.5, 'No se aplicó balanceo', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Distribución después del Balanceo')
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

# =========================== FUNCIONES DE ENTRENAMIENTO ===========================
def gridsearch_train(X_train, y_train, algo_name, grid, cv_folds, scoring, balance_method=None):
    # Definir el modelo base
    if algo_name == 'random_forest':
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        if balance_method == 'class_weight':
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    elif algo_name == 'logistic_regression':
        base_model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1, solver='lbfgs')
        if balance_method == 'class_weight':
            base_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    elif algo_name == 'xgboost':
        if not XGB_AVAILABLE:
            raise RuntimeError('XGBoost no está disponible en este entorno')
        base_model = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
        # scale_pos_weight solo para problemas binarios
        if balance_method == 'scale_pos_weight' and len(np.unique(y_train)) == 2:
            ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
            base_model = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1, scale_pos_weight=ratio)
    else:
        raise ValueError(f"Algoritmo no soportado: {algo_name}")

    # Crear pipeline con técnica de balanceo si es necesario
    X_train_resampled = None
    y_train_resampled = None
    
    if balance_method in ['smote', 'oversample', 'undersample']:
        if balance_method == 'smote':
            sampler = SMOTE(random_state=42)
        elif balance_method == 'oversample':
            sampler = RandomOverSampler(random_state=42)
        elif balance_method == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
        
        # Para XGBoost, necesitamos manejar el caso especial donde no usamos class_weight
        if algo_name == 'xgboost':
            # Asegurarnos de que el modelo base no tenga scale_pos_weight cuando usamos sampling
            base_model = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
        
        pipeline = ImbPipeline([
            ('sampler', sampler),
            ('classifier', base_model)
        ])
        param_grid = {'classifier__' + key: value for key, value in grid.items()}
        model = pipeline
        
        # Aplicar el sampling para obtener los datos balanceados
        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
    else:
        model = base_model
        param_grid = grid
        X_train_resampled, y_train_resampled = X_train, y_train

    # Reducir el espacio de búsqueda si hay muchos parámetros
    for key in param_grid:
        if len(param_grid[key]) > 3:
            param_grid[key] = param_grid[key][:3]  # Limitar a 3 valores por parámetro

    gs = GridSearchCV(model, param_grid, cv=min(cv_folds, 3), scoring=scoring, n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)  # Nota: usamos X_train, y_train originales aquí
    
    # Devolvemos también los datos resampleados para visualización
    return gs.best_estimator_, gs.best_params_, float(gs.best_score_), X_train_resampled, y_train_resampled

def select_best_model(test_metrics, target_metric: str):
    """Selecciona el mejor modelo basado en la métrica objetivo."""
    scores = {}
    for name, metrics in test_metrics.items():
        scores[name] = metrics[target_metric]
    
    if not scores:
        return None, None
    best_name = max(scores, key=scores.get)
    return best_name, scores[best_name]

@st.cache_data(show_spinner=False)
def load_data(file, subsample_size=None):
    """Carga y opcionalmente submuestrea los datos"""
    df = pd.read_csv(file)
    if subsample_size and len(df) > subsample_size:
        df = df.sample(subsample_size, random_state=42)
    return df

# =========================== FUNCIONES DE DIAGNÓSTICO ===========================
def remove_problematic_variables(X, y, target):
    """Elimina variables problemáticas y retorna las características limpias"""
    variables_eliminadas = []
    
    # 1. Eliminar variables con correlación casi perfecta (solo si son numéricas)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        correlations = X[numeric_cols].corrwith(y).abs().sort_values(ascending=False)
        high_corr_vars = correlations[correlations > 0.7].index.tolist()
        
        for var in high_corr_vars:
            if var in X.columns:
                X = X.drop(columns=[var])
                variables_eliminadas.append(f"{var} (corr: {correlations[var]:.4f})")

    # 2. Eliminar variables relacionadas con payment methods (patrón común)
    for col in X.columns:
        if any(keyword in col.lower() for keyword in PAYMENT_KEYWORDS):
            if col in X.columns:
                X = X.drop(columns=[col])
                variables_eliminadas.append(f"{col} (related to payment)")

    # 3. Eliminar variables con nombres similares al target
    for col in X.columns:
        if target.lower() in col.lower() and col != target:
            if col in X.columns:
                X = X.drop(columns=[col])
                variables_eliminadas.append(f"{col} (similar to target)")

    # 4. Eliminar variables constantes o casi constantes
    constant_vars = X.columns[X.nunique() <= 1].tolist()
    for var in constant_vars:
        if var in X.columns:
            X = X.drop(columns=[var])
            variables_eliminadas.append(f"{var} (constant)")
    
    return X, variables_eliminadas

def check_data_leakage(X, y, target):
    """Verifica y reporta problemas de data leakage"""
    leakage_detected = False
    
    # Verificar si el target está en las features
    if target in X.columns:
        st.error(f"❌ DATA LEAKAGE: El target '{target}' está en las features!")
        X = X.drop(columns=[target])
        leakage_detected = True

    # Buscar correlaciones perfectas (solo variables numéricas)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        correlations = X[numeric_cols].corrwith(y).abs().sort_values(ascending=False)
        high_corr = correlations[correlations > 0.7]
        if len(high_corr) > 0:
            st.warning("⚠️ Variables con correlación alta (>0.7):")
            for var, corr in high_corr.items():
                st.write(f"- {var}: {corr:.4f}")
        else:
            st.success("✅ No hay variables con correlación peligrosa")

    if not leakage_detected:
        st.success("✅ No se detectó data leakage evidente después de la limpieza")
    
    return X, leakage_detected

def analyze_target_distribution(y):
    """Analiza y visualiza la distribución del target"""
    class_dist = y.value_counts()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].bar([str(x) for x in class_dist.index], class_dist.values)
    ax[0].set_title('Distribución de clases')
    ax[0].set_xlabel('Clase')
    ax[0].set_ylabel('Count')

    ax[1].pie(class_dist.values, labels=[str(x) for x in class_dist.index], autopct='%1.1f%%')
    ax[1].set_title('Proporción de clases')

    st.pyplot(fig)

    # Análisis de desbalanceo
    if len(class_dist) > 1:
        minority_ratio = min(class_dist) / max(class_dist)
        st.info(f"**Ratio of desbalanceo:** {minority_ratio:.3f}")

        if minority_ratio < 0.3:
            st.warning("⚠️ Desbalanceo severo detectado - Se recomienda usar técnicas de balanceo")
        elif minority_ratio < 0.5:
            st.warning("⚠️ Desbalanceo moderado detectado")
        else:
            st.success("✅ Distribución balanceada")
        
        return minority_ratio
    return 0

# =========================== INTERFAZ PRINCIPAL ===========================
def main():
    st.title("🤖 AutoML con Diagnóstico Completo")
    st.markdown("""
    Pipeline para entrenar y comparar algoritmos con diagnóstico avanzado de data leakage y problemas de datos.
    """)

    # ========== CONFIGURACIÓN EN LA BARRA LATERAL ==========
    st.sidebar.header("⚙️ Configuración")
    file = st.sidebar.file_uploader("Cargar dataset (CSV)", type=['csv'])

    if file is None:
        st.info("Cargue un CSV para comenzar.")
        return

    # Parámetros básicos
    metric_options = ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy', 'roc_auc']
    metric = st.sidebar.selectbox("Métrica objetivo", metric_options, index=1)
    cv_folds = st.sidebar.slider("Folds para validación cruzada", 3, 10, 3)

    # Filtrar algoritmos disponibles
    available_algos = ['Random Forest', 'Logistic Regression']
    if XGB_AVAILABLE:
        available_algos.append('XGBoost')
    
    algos = st.sidebar.multiselect(
        "Algoritmos a competir",
        available_algos,
        default=['Random Forest', 'Logistic Regression']
    )

    st.sidebar.subheader("División de datos")
    test_pct = st.sidebar.slider("% Test", 10, 30, 20, step=5) / 100.0
    oot_pct = st.sidebar.slider("% OOT", 10, 30, 20, step=5) / 100.0

    if test_pct + oot_pct >= 1.0:
        st.sidebar.error("La suma de Test + OOT no puede exceder 100%")
        train_pct = 0
    else:
        train_pct = 1.0 - test_pct - oot_pct

    st.sidebar.caption(f"Train: {train_pct*100:.0f}% | Test: {test_pct*100:.0f}% | OOT: {oot_pct*100:.0f}%")

    # Técnicas para manejar desbalanceo
    st.sidebar.subheader("Manejo de Desbalanceo")
    balance_method = st.sidebar.selectbox(
        "Técnica para clases desbalanceadas",
        ['ninguna', 'smote', 'oversample', 'undersample'],
        index=0,
        help="Seleccione técnica para manejar desbalanceo de clases"
    )

    # Grids por algoritmo
    st.sidebar.subheader("Grids de hiperparámetros")

    st.sidebar.markdown("**Random Forest**")
    rf_n_estimators = st.sidebar.text_input("n_estimators", "50, 100")
    rf_max_depth = st.sidebar.text_input("max_depth", "None, 10")
    rf_min_samples_split = st.sidebar.text_input("min_samples_split", "2, 5")
    rf_grid = {
        'n_estimators': parse_list_from_text(rf_n_estimators, int),
        'max_depth': parse_list_from_text(rf_max_depth, int),
        'min_samples_split': parse_list_from_text(rf_min_samples_split, int),
    }

    st.sidebar.markdown("**Logistic Regression**")
    lr_c_vals = st.sidebar.text_input("C", "0.1, 1.0")
    lr_grid = {
        'C': parse_list_from_text(lr_c_vals, float),
        'penalty': ['l2'],
    }

    xgb_grid = None
    if XGB_AVAILABLE and 'XGBoost' in algos:
        st.sidebar.markdown("**XGBoost**")
        xgb_n_estimators = st.sidebar.text_input("n_estimators (XGB)", "50, 100")
        xgb_max_depth = st.sidebar.text_input("max_depth (XGB)", "3, 6")
        xgb_learning_rate = st.sidebar.text_input("learning_rate (XGB)", "0.1")
        xgb_grid = {
            'n_estimators': parse_list_from_text(xgb_n_estimators, int),
            'max_depth': parse_list_from_text(xgb_max_depth, int),
            'learning_rate': parse_list_from_text(xgb_learning_rate, float),
        }

    # Añadir opción para usar submuestra
    use_subsample = st.sidebar.checkbox("Usar submuestra para diagnóstico rápido", value=True)
    subsample_size = st.sidebar.slider("Tamaño de submuestra", 1000, 10000, 5000) if use_subsample else None

    run = st.sidebar.button("🚀 Ejecutar pipeline", type='primary')

    # ========== CARGA DE DATOS ==========
    df = load_data(file, subsample_size if use_subsample else None)
    st.success(f"Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")

    # ========== DIAGNÓSTICO DE DATOS ==========
    st.header("🔍 Diagnóstico de Calidad de Datos")

    # Elegir target
    cols = df.columns.tolist()
    target = st.selectbox("Variable objetivo (clasificación)", cols, index=len(cols)-1)

    if target not in df.columns:
        st.error(f"La variable objetivo '{target}' no se encuentra en el dataset")
        return

    X = df.drop(columns=[target])
    y = df[target]

    # ========== LIMPIEZA AUTOMÁTICA DE VARIABLES ==========
    st.header("🔧 Corrección Automática de Data Leakage")
    
    X, variables_eliminadas = remove_problematic_variables(X, y, target)

    # Mostrar resultados de la limpieza
    if variables_eliminadas:
        st.warning("🚨 Variables eliminadas automáticamente:")
        for var in variables_eliminadas:
            st.write(f"- {var}")
        st.success(f"✅ Total de variables eliminadas: {len(variables_eliminadas)}")
        st.info(f"📊 Variables restantes: {X.shape[1]}")
    else:
        st.success("✅ No se encontraron variables problemáticas para eliminar")

    # ========== ANÁLISIS DE VARIABLES ==========
    st.subheader("1. Análisis de Variables (Después de limpieza)")

    if X.shape[1] > 20:
        st.write(f"**Total de variables:** {X.shape[1]}")
        st.write(f"**Tipos de datos:**")
        st.write(X.dtypes.value_counts())
        st.write("**Primeras 10 variables:**")
        st.write(X.iloc[:, :10].dtypes)
    else:
        col_stats = []
        for col in X.columns:
            col_stats.append({
                'Variable': col,
                'Tipo': str(X[col].dtype),
                'Valores Únicos': X[col].nunique(),
                'Valores Faltantes': X[col].isnull().sum(),
                'Ejemplo': X[col].iloc[0] if len(X) > 0 else 'N/A'
            })
        st.table(pd.DataFrame(col_stats).head(10))

    # ========== VERIFICACIÓN OF DATA LEAKAGE ==========
    st.subheader("2. Verificación de Data Leakage (Después de limpieza)")
    X, leakage_detected = check_data_leakage(X, y, target)

    # ========== ANÁLISIS DE DISTRIBUCIÓN DEL TARGET ==========
    st.subheader("3. Distribución de la Variable Objetivo")
    minority_ratio = analyze_target_distribution(y)

    if len(y.value_counts()) == 1:
        st.error("❌ ¡Solo hay una clase en el target! No es un problema de clasificación válido.")
        return

    # ========== MODELO DE LÍNEA BASE ==========
    st.subheader("4. Modelo de Línea Base (Después de limpieza)")
    try:
        dummy = DummyClassifier(strategy='most_frequent')
        sample_size = min(1000, len(X))
        X_sample = X.iloc[:sample_size]
        y_sample = y.iloc[:sample_size]
        dummy.fit(X_sample, y_sample)
        dummy_acc = dummy.score(X_sample, y_sample)
        
        st.info(f"Accuracy de línea base (siempre predecir mayoritaria): {dummy_acc:.4f}")
        
        if dummy_acc > 0.95:
            st.warning("⚠️ El problema puede ser trivial - accuracy base muy alta")
        elif dummy_acc > 0.8:
            st.info("ℹ️ Accuracy base moderada - Problema desbalanceado")
        else:
            st.success("✅ Accuracy base normal")
            
    except Exception as e:
        st.warning(f"No se pudo calcular modelo base: {e}")

    # Mostrar recomendación de técnica
    if minority_ratio < 0.3 and balance_method == 'ninguna':
        st.warning("💡 Recomendación: Considere usar SMOTE o class_weight para manejar el desbalanceo")

    # ========== EJECUCIÓN PRINCIPAL ==========
    if run:
        # Validar que queden variables después de la limpieza
        if X.shape[1] == 0:
            st.error("❌ No hay variables disponibles después de la limpieza. Revise el dataset.")
            return
            
        # Validar división de datos
        if test_pct + oot_pct >= 1.0:
            st.error("❌ La suma de Test + OOT no puede ser igual o mayor al 100%")
            return
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Ejecutando pipeline..."):
            # División de datos con OOT
            status_text.text("Dividiendo datos...")
            X_train, X_test, X_oot, y_train, y_test, y_oot = split_train_test_oot(
                X, y, test_size=test_pct, oot_size=oot_pct, random_state=42
            )
            progress_bar.progress(20)

            # Preparar grids elegidos
            selected = []
            for label in algos:
                key = ALGO_KEY.get(label)
                if key is None:
                    continue
                if key == 'random_forest':
                    selected.append(('random_forest', rf_grid))
                elif key == 'logistic_regression':
                    selected.append(('logistic_regression', lr_grid))
                elif key == 'xgboost' and xgb_grid is not None:
                    selected.append(('xgboost', xgb_grid))

            training_results = {}
            trained_models = {}
            train_metrics = {}
            resampled_data = {}  # Para almacenar datos después del balanceo

            total_models = len(selected)
            for i, (name, grid) in enumerate(selected):
                status_text.text(f"Entrenando {name} ({i+1}/{total_models})...")
                try:
                    model, best_params, best_cv, X_resampled, y_resampled = gridsearch_train(
                        X_train, y_train, name, grid, cv_folds=min(cv_folds, 3),
                        scoring=metric, balance_method=balance_method if balance_method != 'ninguna' else None
                    )
                    trained_models[name] = model
                    training_results[name] = {
                        'best_params': best_params,
                        'best_cv_score': best_cv,
                        'balance_method': balance_method if balance_method != 'ninguna' else 'ninguna'
                    }
                    resampled_data[name] = (X_resampled, y_resampled)
                    
                    # Verificar el balanceo aplicado
                    st.subheader(f"Balanceo aplicado para {name}")
                    fig = plot_class_distribution_comparison(y_train, y_resampled, f"Distribución de clases - {name}")
                    st.pyplot(fig)
                    
                    verify_balancing(y_train, balance_method if balance_method != 'ninguna' else 'ninguna', name)
                    
                    progress_bar.progress(20 + ((i+1) * 60 // total_models))
                except Exception as e:
                    st.error(f"Error entrenando {name}: {str(e)}")
                    continue

            if not trained_models:
                st.error("❌ No se pudo entrenar ningún modelo")
                return

            # ========== EVALUACIÓN ==========
            status_text.text("Evaluando modelos...")
            test_metrics = {}
            oot_metrics = {}
            train_metrics = {}
            oot_drops = {}

            for name, model in trained_models.items():
                try:
                    # Train performance
                    yp_train = model.predict(X_train)
                    
                    train_metrics[name] = {
                        'accuracy': accuracy_score(y_train, yp_train),
                        'f1': f1_score(y_train, yp_train, average='weighted'),
                        'precision': precision_score(y_train, yp_train, average='weighted'),
                        'recall': recall_score(y_train, yp_train, average='weighted'),
                        'balanced_accuracy': balanced_accuracy_score(y_train, yp_train),
                    }
                    
                    # Test performance
                    yp_test = model.predict(X_test)
                    
                    test_metrics[name] = {
                        'accuracy': accuracy_score(y_test, yp_test),
                        'f1': f1_score(y_test, yp_test, average='weighted'),
                        'precision': precision_score(y_test, yp_test, average='weighted'),
                        'recall': recall_score(y_test, yp_test, average='weighted'),
                        'balanced_accuracy': balanced_accuracy_score(y_test, yp_test),
                        'report': classification_report(y_test, yp_test, output_dict=True),
                    }
                    
                    # OOT performance
                    yp_oot = model.predict(X_oot)
                    
                    oot_metrics[name] = {
                        'accuracy': accuracy_score(y_oot, yp_oot),
                        'f1': f1_score(y_oot, yp_oot, average='weighted'),
                        'precision': precision_score(y_oot, yp_oot, average='weighted'),
                        'recall': recall_score(y_oot, yp_oot, average='weighted'),
                        'balanced_accuracy': balanced_accuracy_score(y_oot, yp_oot),
                    }

                    # Calcular caída en OOT
                    oot_drops[name] = max(0.0, test_metrics[name][metric] - oot_metrics[name][metric])

                    # Verificar si hay sobreajuste perfecto
                    if train_metrics[name]['accuracy'] == 1.0 and test_metrics[name]['accuracy'] == 1.0:
                        st.error(f"❌ POSIBLE PROBLEMA: {name} tiene accuracy perfecto en train y test")
                        
                except Exception as e:
                    st.error(f"Error evaluando {name}: {str(e)}")
                    continue

            best_name, best_score = select_best_model(test_metrics, target_metric=metric)
            progress_bar.progress(90)

        # ========== REPORTE FINAL ==========
        status_text.text("Generando reporte final...")
        st.header("📊 Resultados del Pipeline")
        st.info(f"**Técnica de balanceo utilizada:** {balance_method.upper() if balance_method != 'ninguna' else 'Ninguna'}")
        
        # col1, col2, col3 = st.columns(3)
        # col1.metric("Train", f"{len(X_train)} muestras")
        # col2.metric("Test", f"{len(X_test)} muestras")
        # col3.metric("OOT", f"{len(X_oot)} muestras")

        # # Distribución de clases
        # st.subheader("📈 Distribución de clases por conjunto")
        # dist_data = {
        #     'Conjunto': ['Train', 'Test', 'OOT'],
        #     'Clase 0': [sum(y_train == 0), sum(y_test == 0), sum(y_oot == 0)],
        #     'Clase 1': [sum(y_train == 1), sum(y_test == 1), sum(y_oot == 1)],
        #     'Total': [len(y_train), len(y_test), len(y_oot)]
        # }
        # st.table(pd.DataFrame(dist_data))

        # Métricas de evaluación
        st.subheader("🔍 Métricas de Evaluación")
        
        if best_name is not None:
            st.success(f"🏆 **Mejor modelo:** {best_name} - {metric.upper()}: {best_score:.4f}")
            
            metrics_data = []
            for name in trained_models.keys():
                metrics_data.append({
                    'Modelo': name,
                    'Técnica Balanceo': training_results[name]['balance_method'],
                    'Accuracy': f"{test_metrics[name]['accuracy']:.4f}",
                    'Precision': f"{test_metrics[name]['precision']:.4f}",
                    'Recall': f"{test_metrics[name]['recall']:.4f}",
                    'F1 Score': f"{test_metrics[name]['f1']:.4f}",
                    'Balanced Acc': f"{test_metrics[name]['balanced_accuracy']:.4f}",
                })
            
            st.table(pd.DataFrame(metrics_data))

        # Validación OOT
        st.subheader("✅ Validación en OOT (Out-of-Time)")
        oot_table = []
        for name in trained_models.keys():
            oot_table.append({
                'Modelo': name,
                f'{metric.upper()} Test': round(test_metrics[name][metric], 4),
                f'{metric.upper()} OOT': round(oot_metrics[name][metric], 4),
                'Recall Test': round(test_metrics[name]['recall'], 4),
                'Recall OOT': round(oot_metrics[name]['recall'], 4),
                'Caída OOT': round(oot_drops[name], 4),
            })
        st.table(pd.DataFrame(oot_table))

        # Matriz de confusión
        if best_name is not None:
            st.subheader("📊 Matriz de Confusión del Mejor Modelo (Test)")
            bm = trained_models[best_name]
            y_pred = bm.predict(X_test)
            cm_fig = plot_confusion_matrix(y_test, y_pred, sorted(y.unique()))
            st.pyplot(cm_fig)

        progress_bar.progress(100)
        status_text.text("¡Completado!")
        st.balloons()

        st.caption("Pipeline de AutoML con diagnóstico completo y limpieza automática de data leakage.")

if __name__ == "__main__":
    main()
