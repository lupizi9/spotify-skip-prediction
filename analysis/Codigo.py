# -------------------------------------------------
# 1. LIBRERÍAS
# -------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import argparse

# Sklearn
from sklearn.model_selection import train_test_split, GroupKFold, KFold, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

# XGBoost
from xgboost import XGBClassifier, plot_importance

# Plotting
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", 200)
sns.set(style="whitegrid", palette="muted")

# Path
COMPETITION_PATH = os.path.dirname(os.path.abspath(__file__))


# -------------------------------------------------
# 2. CARGA DE DATOS
# -------------------------------------------------
def load_data(data_dir, sample_frac=None, random_state=42):
    """
    Carga train/test y concatena para transformar de forma consistente.
    Parámetros:
        data_dir: carpeta con train_data.txt y test_data.txt
        sample_frac: fracción de train para pruebas rápidas
        random_state: semilla para el sample
    """
    train_file = os.path.join(data_dir, "train_data.txt")
    test_file = os.path.join(data_dir, "test_data.txt")

    train_df = pd.read_csv(train_file, sep="\t", low_memory=False)
    test_df = pd.read_csv(test_file, sep="\t", low_memory=False)

    if sample_frac:
        train_df = train_df.sample(frac=sample_frac, random_state=random_state)

    combined = pd.concat([train_df, test_df], ignore_index=True)

    print(f"Train: {train_df.shape}, Test: {test_df.shape}, Combined: {combined.shape}")
    return combined


# -------------------------------------------------
# 3. PREPROCESAMIENTO INICIAL
# -------------------------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformaciones iniciales:
    - Convierte timestamps a datetime
    - Crea target e is_test
    - Orden de reproducción por usuario
    - Variables de hora y día
    - Tipo de contenido (music/podcast/audiobook)
    """
    df = df.copy()

    # Conversión de timestamps
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["offline_timestamp"] = pd.to_datetime(df["offline_timestamp"], unit="s", errors="coerce", utc=True)

    # Target y máscara de test
    df["target"] = (df["reason_end"] == "fwdbtn").astype(int)
    df["is_test"] = df["reason_end"].isna()

    # Orden dentro del usuario
    df = df.sort_values(["username", "ts"])
    df["user_order"] = df.groupby("username", observed=True).cumcount() + 1

    # Variables de tiempo
    df["hour"] = df["ts"].dt.hour
    df["weekday"] = df["ts"].dt.weekday

    # Tipo de contenido simple
    df["content_type"] = np.where(
        df["spotify_track_uri"].notna(), "music",
        np.where(df["episode_name"].notna(), "podcast", "audiobook"),
    )

    print("Preprocesamiento inicial completado.")
    return df


# -------------------------------------------------
# 4. ANÁLISIS EXPLORATORIO DE DATOS
# -------------------------------------------------
# Estas funciones permiten explorar visualmente patrones relevantes:
# - Correlaciones con la variable target
# - Tasa de skips (target=1) por hora y tipo de contenido
# - Actividad diaria de los usuarios

def plot_full_correlation(df, sample_size=100_000, save_fig=True):
    """
    Genera matriz de correlación y muestra las variables más relacionadas con 'target'.
    """
    print("=== Generando matriz de correlación ===")

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    cols_to_keep = [
        "target", "hour", "weekday", "shuffle", "offline",
        "incognito_mode", "content_type"
    ]
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]

    df_encoded = pd.get_dummies(df[cols_to_keep], drop_first=False)
    corr = df_encoded.corr(numeric_only=True)

    # Mostrar correlaciones con el target
    if "target" in corr.columns:
        target_corr = corr["target"].sort_values(ascending=False)
        print("\nCorrelaciones más altas con target:")
        print(target_corr.head(10))

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, cbar_kws={'shrink': 0.6})
    plt.title("Matriz de correlación (subset de variables)")
    plt.tight_layout()
    if save_fig:
        plt.savefig("eda_correlation_matrix.png", dpi=300)
    plt.show()

def plot_skip_rate_patterns(df, save_fig=True):
    """
    Analiza patrones del target (skip):
    - Tasa de skips según tipo de contenido
    - Tasa de skips a lo largo del día
    """
    print("=== Analizando patrones de skip ===")

    if "target" not in df.columns:
        print("No se encontró la columna 'target', se omite análisis.")
        return

    # Skip rate por tipo de contenido
    if "content_type" in df.columns:
        content_skip = df.groupby("content_type")["target"].mean().sort_values(ascending=False)
        plt.figure(figsize=(6, 4))
        sns.barplot(x=content_skip.index, y=content_skip.values, palette="viridis")
        plt.ylabel("Tasa de skips (target=1)")
        plt.title("Tasa de skips según tipo de contenido")
        plt.tight_layout()
        if save_fig:
            plt.savefig("eda_skip_by_content.png", dpi=300)
        plt.show()

    # Skip rate por hora
    if "hour" in df.columns:
        hour_skip = df.groupby("hour")["target"].mean()
        plt.figure(figsize=(7, 4))
        sns.lineplot(x=hour_skip.index, y=hour_skip.values, marker="o")
        plt.ylabel("Tasa de skips (target=1)")
        plt.xlabel("Hora del día")
        plt.title("Tasa de skips a lo largo del día")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_fig:
            plt.savefig("eda_skip_by_hour.png", dpi=300)
        plt.show()

def plot_activity_over_time(df, save_fig=True):
    """
    Muestra el volumen diario de reproducciones (actividad total de usuarios).
    """
    if "ts" not in df.columns:
        print("No se encontró columna 'ts', se omite gráfico de actividad temporal.")
        return

    df_daily = df.groupby(df["ts"].dt.date).size()
    plt.figure(figsize=(10, 4))
    sns.lineplot(x=df_daily.index, y=df_daily.values)
    plt.title("Volumen diario de reproducciones")
    plt.ylabel("Cantidad de reproducciones")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_fig:
        plt.savefig("eda_activity_over_time.png", dpi=300)
    plt.show()

def run_analisis(df):
    """
    Ejecuta el análisis exploratorio completo sobre el dataset de entrenamiento.
    """
    print("\n=== INICIO DEL ANÁLISIS EXPLORATORIO ===")
    print("Filas totales:", len(df))
    print("Columnas:", len(df.columns))
    print("Tipos de datos:\n", df.dtypes.value_counts())
    print("\nPorcentaje de nulos (Top 10):")
    print(df.isna().mean().sort_values(ascending=False).head(10))

    if "content_type" in df.columns:
        print("\nDistribución de tipos de contenido:")
        print(df["content_type"].value_counts())

    plot_activity_over_time(df)
    plot_skip_rate_patterns(df)
    plot_full_correlation(df)

    print("=== Análisis completo ===\n")

# Para ver sólo el análisis, descomentar este main y comentar el otro
# -------------------------------------------------
# MAIN PARA EJECUTAR SOLO EL ANÁLISIS EXPLORATORIO
# -------------------------------------------------
# Ejecuta las funciones de carga, preprocesamiento y análisis completo,
# generando los gráficos que se guardan automáticamente:
#   - eda_activity_over_time.png
#   - eda_skip_by_content.png
#   - eda_correlation_matrix.png

# if __name__ == "__main__":
#     print("=== Análisis - Inicio ===")

#     # Carga + preprocesamiento básico
#     df = load_data(COMPETITION_PATH, sample_frac=0.2, random_state=1234)
#     df = preprocess(df)

#     # Ejecutar análisis completo con gráficos
#     run_analisis(df)

#     print("=== Análisis - Finalizado ===")
#     print("Gráficos guardados en el directorio actual:")
#     print(" - eda_activity_over_time.png")
#     print(" - eda_skip_by_content.png")
#     print(" - eda_correlation_matrix.png")


# -------------------------------------------------
# 5. INGENIERÍA DE ATRIBUTOS
# -------------------------------------------------
# Esta sección crea todas las variables utilizadas por los modelos:
# - Comportamiento temporal y reciente por usuario
# - Popularidad y contexto de contenido
# - Variables geográficas y de plataforma
# - Codificación KFold sin leakage (target encoding)
# - One-hot final y limpieza de nombres

# Target encoding (bin counting) KFold sin leakage
def bin_counting_kfold_df(df, cat_cols, target_col = "target", is_test_col = "is_test", n_splits = 5, seed = 42):
    """
    Mean target encoding (bin counting) por KFold sin leakage:
    - En TRAIN: promedia el target por categoría, sin usar los datos del fold.
    - En TEST: mapea con las medias del TRAIN completo.
    Crea nuevas columnas <col>_bin para cada variable categórica.
    """
    df = df.copy()
    train_mask = ~df[is_test_col]
    tr = df.loc[train_mask].copy()
    te = df.loc[~train_mask].copy()

    y = tr[target_col].astype(int)
    global_mean = float(y.mean())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for col in cat_cols:
        if col not in df.columns:
            continue
        new_col = f"{col}_bin"
        tr[new_col] = np.nan

        # KFold dentro del train
        for tr_idx, va_idx in kf.split(tr):
            tr_part = tr.iloc[tr_idx]
            means = tr_part.groupby(col)[target_col].mean()
            tr.iloc[va_idx, tr.columns.get_loc(new_col)] = (
                tr.iloc[va_idx][col].map(means).values
            )

        tr.fillna({new_col: global_mean}, inplace=True)

        # Mapeo a test con medias del train completo
        full_means = tr.groupby(col)[new_col].mean()
        te[new_col] = te[col].map(full_means).fillna(global_mean)

    df_encoded = pd.concat([tr, te], axis=0).sort_index()
    return df_encoded

# Feature engineering principal
def create_user_content_features(df):
    """
    Genera todas las variables predictoras del modelo final.
    Crea variables de usuario, contenido, región, modos de reproducción
    y aplica bin counting (target encoding sin leakage)
    """
    df = df.copy()

    # ORDEN E HISTORIAL
    df = df.sort_values(["username", "ts"]).copy()

    # Orden promedio por usuario
    user_order_mean = df.groupby("username")["user_order"].mean().rename("user_order_mean")
    df = df.merge(user_order_mean, on="username", how="left")

    # Skip previo y promedio de skips recientes
    df["prev_target"] = df.groupby("username")["target"].shift(1).fillna(0).astype(float)
    df["mean_last3"] = (
        df.groupby("username")["target"]
        .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        .reset_index(level=0, drop=True)
        .fillna(0)
        .astype(float)
    )

    # Diferencia de tiempo con la reproducción anterior (minutos)
    df["dt_prev_min"] = (df.groupby("username")["ts"].diff() / pd.Timedelta(minutes=1)).fillna(9999).clip(0, 180)

    # ACTIVIDAD DE USUARIO
    user_freq = df.groupby("username").size().rename("user_activity")
    df = df.merge(user_freq, on="username", how="left")
    df["user_activity_log"] = np.log1p(df["user_activity"]).astype(float)

    # ARTISTA, ÁLBUM Y CONTENIDO
    if "master_metadata_album_artist_name" in df.columns:
        artist_count = df.groupby("master_metadata_album_artist_name").size().rename("artist_count")
        df = df.merge(artist_count, on="master_metadata_album_artist_name", how="left")

    if "master_metadata_album_album_name" in df.columns:
        album_count = df.groupby("master_metadata_album_album_name").size().rename("album_count")
        df = df.merge(album_count, on="master_metadata_album_album_name", how="left")

    df["content_id"] = df["spotify_track_uri"].fillna(
        df["spotify_episode_uri"].fillna(df["audiobook_uri"].fillna("unknown"))
    )
    popularity = df.groupby("content_id").size().rename("content_popularity")
    df = df.merge(popularity, on="content_id", how="left")

    # VARIABLES TEMPORALES Y DE CONTEXTO
    bins = [0, 6, 12, 18, 24]
    labels = ["madrugada", "mañana", "tarde", "noche"]
    df["daytime"] = pd.cut(df["hour"], bins=bins, labels=labels, right=False, include_lowest=True)
    df["month"] = df["ts"].dt.month
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    # Tipo de contenido unificado
    df["content_type"] = "other"
    df.loc[df["spotify_track_uri"].notna(), "content_type"] = "track"
    df.loc[df["spotify_episode_uri"].notna(), "content_type"] = "podcast"
    df.loc[df["audiobook_uri"].notna(), "content_type"] = "audiobook"

    # MODO DE REPRODUCCIÓN
    df["playback_mode"] = (
        df["shuffle"].astype(int).astype(str)
        + "_"
        + df["offline"].astype(int).astype(str)
        + "_"
        + df["incognito_mode"].astype(int).astype(str)
    )

    # PAÍS --> REGIÓN
    country_to_region = {
        "FR": "Europe", "DE": "Europe", "IT": "Europe", "ES": "Europe", "UK": "Europe",
        "PT": "Europe", "NL": "Europe", "BE": "Europe", "CH": "Europe", "AT": "Europe",
        "SE": "Europe", "NO": "Europe", "DK": "Europe", "FI": "Europe", "PL": "Europe",
        "US": "North America", "CA": "North America", "MX": "North America",
        "AR": "South America", "BR": "South America", "CL": "South America", "UY": "South America",
        "CN": "Asia", "JP": "Asia", "KR": "Asia", "IN": "Asia", "ID": "Asia",
        "AU": "Oceania", "NZ": "Oceania",
        "ZA": "Africa", "NG": "Africa", "EG": "Africa", "MA": "Africa",
    }
    df["region"] = df["conn_country"].map(country_to_region).fillna("Other")

    # MÁS VARIABLES TEMPORALES
    df = df.sort_values(["username", "ts"]).reset_index(drop=True)
    prev_artist = df.groupby("username")["master_metadata_album_artist_name"].shift(1)
    prev_track = df.groupby("username")["spotify_track_uri"].shift(1)
    df["same_artist_prev"] = (prev_artist == df["master_metadata_album_artist_name"]).astype(int)
    df["same_track_prev"] = (prev_track == df["spotify_track_uri"]).astype(int)

    last_time_artist = df.groupby(["username", "master_metadata_album_artist_name"])["ts"].shift(1)
    df["mins_since_artist"] = (
        (df["ts"] - last_time_artist) / pd.Timedelta(minutes=1)
    ).clip(0, 1440).fillna(1440)

    # Hora en componentes cíclicos
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # HISTORIAL Y STEAKS
    df["target_masked"] = df["target"].where(~df["is_test"], np.nan)

    def _streak_from_prev(s: pd.Series, value: int) -> pd.Series:
        prev = s.shift(1)
        mask = (prev == value)
        grp = (~mask).cumsum()
        out = mask.groupby(grp).cumsum().fillna(0)
        return out.clip(0, 50)

    g = df.groupby("username", sort=False)
    df["streak_skip_prev"] = g["target_masked"].apply(lambda s: _streak_from_prev(s, 1)).reset_index(level=0, drop=True)
    df["streak_play_prev"] = g["target_masked"].apply(lambda s: _streak_from_prev(s, 0)).reset_index(level=0, drop=True)
    df["streak_balance"] = (df["streak_skip_prev"] - df["streak_play_prev"]).astype(int)
    df["streak_skip_prev_log"] = np.log1p(df["streak_skip_prev"])

    # TARGET ENCODING
    te_cols = [
        c for c in [
            "platform", "conn_country", "content_type",
            "master_metadata_album_artist_name", "master_metadata_album_album_name",
            "episode_show_name", "audiobook_title",
            "playback_mode", "region",
            "platform_daytime"
        ] if c in df.columns
    ]

    df["platform_daytime"] = df["platform"].astype(str) + "_" + df["daytime"].astype(str)

    if len(te_cols) > 0:
        df = bin_counting_kfold_df(
            df, cat_cols=te_cols, target_col="target",
            is_test_col="is_test", n_splits=5, seed=123
        )

    # SELECCIÓN FINAL Y ONE-HOT
    keep_cols = [
        "obs_id", "username", "user_order", "hour", "weekday", "user_order_mean",
        "artist_count", "album_count", "target", "is_test", "daytime",
        "content_type", "month", "is_weekend", "shuffle", "offline",
        "incognito_mode", "playback_mode", "region", "conn_country",
        "content_popularity", "prev_target", "mean_last3", "user_activity_log",
        "dt_prev_min", "same_artist_prev", "same_track_prev", "mins_since_artist",
        "hour_sin", "hour_cos", "streak_skip_prev_log", "streak_balance", "platform"
    ] + [f"{c}_bin" for c in te_cols]

    df_model = df[keep_cols].copy()

    cat_cols = ["daytime", "content_type", "playback_mode", "region", "conn_country", "platform"]
    df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

    df_model.columns = (
        df_model.columns.astype(str)
        .str.replace("[", "(", regex=False)
        .str.replace("]", ")", regex=False)
        .str.replace("<", " menor ", regex=False)
        .str.replace(">", " mayor ", regex=False)
    )

    df_model.fillna(0, inplace=True)
    print("Feature engineering finalizado.")
    return df_model


# -------------------------------------------------
# 6. SPLIT DE ENTRENAMIENTO Y VALIDACIÓN
# -------------------------------------------------
# Dividimos los datos en train, test y validation asegurando que ningún usuario aparezca 
# en ambos conjuntos, evitando fuga de información (data leakage)

def split_train_val_by_user(df, val_size = 0.2, random_state = 42):
    """
    Split train/val asegurando que un usuario no esté en ambos sets.
    Devuelve: X_train, y_train, X_val, y_val, X_test, test_obs_ids
    """
    # Separar train (con target) y test (sin target)
    train_df = df[~df["is_test"]].copy()
    test_df = df[df["is_test"]].copy()

    # Usuarios únicos
    unique_users = train_df["username"].unique()
    train_users, val_users = train_test_split(unique_users, test_size=val_size, random_state=random_state)

    # Train
    X_train = train_df[train_df["username"].isin(train_users)].drop(columns=["target", "is_test", "username", "obs_id"], errors="ignore")
    y_train = train_df[train_df["username"].isin(train_users)]["target"]

    # Validation
    X_val = train_df[train_df["username"].isin(val_users)].drop(columns=["target", "is_test", "username", "obs_id"], errors="ignore")
    y_val = train_df[train_df["username"].isin(val_users)]["target"]

    # Test (solo features)
    X_test = test_df.drop(columns=["target", "is_test", "username", "obs_id"], errors="ignore")
    test_obs_ids = test_df["obs_id"].values

    print(f"Split realizado | Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Usuarios en train: {len(train_users)} | Usuarios en val: {len(val_users)}")
    return X_train, y_train, X_val, y_val, X_test, test_obs_ids


# -------------------------------------------------
# 7. ENTRENAMIENTO DE MODELOS
# -------------------------------------------------
# Árbol de decisión
def train_decision_tree(X_train, y_train, max_depth = None, min_samples_split = 2, min_samples_leaf = 1, random_state = 42):
    """
    Entrena un árbol de decisión simple.
    """
    print("Entrenando Decision Tree...")
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=None,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    print("Decision Tree entrenado.")
    return model

# Random Forest
def train_random_forest(X_train, y_train, n_estimators = 200, max_depth = None, min_samples_split = 5, min_samples_leaf = 2, max_features="sqrt", random_state = 42):
    """
    Entrena un modelo Random Forest robusto y escalable.
    """
    print("Entrenando Random Forest...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
        max_features=max_features,
        bootstrap=True,
    )
    model.fit(X_train, y_train)
    print("Random Forest entrenado.")
    return model

# XGBoost
def train_xgboost(X_train, y_train, n_estimators = 500, learning_rate = 0.05, max_depth = 6, subsample = 0.8, colsample_bytree = 0.8, random_state = 42):
    """
    Entrena un modelo XGBoost con hiperparámetros razonables por defecto.
    """
    print("Entrenando XGBoost...")
    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric="auc",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    print("XGBoost entrenado.")
    return model


# -------------------------------------------------
# 8. EVALUACIÓN DE MODELOS
# -------------------------------------------------
def evaluate_model(model, X_val, y_val, name="Modelo"):
    """
    Calcula el ROC-AUC en el set de validación y muestra el resultado.
    Ideal para chequeos rápidos o durante grid/random search.
    """
    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    print(f"{name} - ROC-AUC: {auc:.4f}")
    return auc, y_pred

def predict_test(model, X_test, test_obs_ids, filename = "predicciones.csv"):
    """
    Genera predicciones sobre el set de test y guarda un CSV con: obs_id | pred_proba
    """
    preds_proba = model.predict_proba(X_test)[:, 1]
    preds_df = pd.DataFrame({"obs_id": test_obs_ids, "pred_proba": preds_proba})
    preds_df.to_csv(filename, index=False)
    print(f"Predicciones guardadas en {filename}")

def show_xgb_feature_importance(model, max_num_features = 20, save_fig = False, filename = "xgb_feature_importance.png"):
    """
    Muestra las features más importantes de un modelo XGBoost según 'gain'.
    """
    importance = model.get_booster().get_score(importance_type="gain")
    importance_df = (
        pd.DataFrame({"feature": list(importance.keys()), "gain": list(importance.values())})
        .sort_values(by="gain", ascending=False)
    )
    print("Top variables por Gain (XGBoost):")
    print(importance_df.head(max_num_features))

    plot_importance(model, importance_type="gain", max_num_features=max_num_features)
    plt.tight_layout()
    if save_fig:
        plt.savefig(filename)
    plt.show()
    
    return importance_df


# -------------------------------------------------
# 9. TUNING DE HIPERPARÁMETROS
# -------------------------------------------------
# Árbol de decisión
def tune_decision_tree(X_train, y_train, X_val, y_val, param_grid):
    """
    Búsqueda exhaustiva (grid search) para Decision Tree.
    """
    best_auc, best_model, best_params = 0, None, None
    for combination in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), combination))
        print(f"Probando params Decision Tree: {params}")
        model = train_decision_tree(X_train, y_train, **params)
        auc, _ = evaluate_model(model, X_val, y_val)
        if auc > best_auc:
            best_auc, best_model, best_params = auc, model, params
    print(f"Mejor ROC-AUC Decision Tree: {best_auc:.4f} con params: {best_params}")
    return best_model, best_params

# Random Forest
def tune_random_forest(X_train, y_train, X_val, y_val, param_grid):
    """
    Búsqueda exhaustiva (grid search) para Random Forest.
    """
    best_auc, best_model, best_params = 0, None, None
    for combination in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), combination))
        print(f"Probando params RF: {params}")
        model = train_random_forest(X_train, y_train, **params)
        auc, _ = evaluate_model(model, X_val, y_val)
        if auc > best_auc:
            best_auc, best_model, best_params = auc, model, params
    print(f"Mejor ROC-AUC RF: {best_auc:.4f} con params: {best_params}")
    return best_model, best_params

def randomized_search_random_forest(X_train, y_train, n_iter: int = 30, cv: int = 3, random_state: int = 42):
    """
    Randomized Search para Random Forest.
    """
    param_dist = {
        "n_estimators": [200, 400, 600, 1000],
        "max_depth": [None, 15, 30, 50],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2", 0.5],
        "bootstrap": [True, False],
    }
    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        verbose=2,
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print("Mejores hiperparámetros RF encontrados:", search.best_params_)
    print("Mejor score (AUC):", search.best_score_)
    return search.best_estimator_, search.best_params_

# XGBoost
def tune_min_child_weight(X_train, y_train, X_val, y_val, values=[1, 3, 5, 7, 10, 15]):
    '''
    Tuning de un único hiperparámetro de XGBoost, manteniendo los demás parámetros
    en valores predeterminados: `min_child_weight`.
    '''
    best_auc = -1
    best_model = None
    best_value = None

    for val in values:
        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=val,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        auc, _ = evaluate_model(model, X_val, y_val, name=f"min_child_weight={val}")

        if auc > best_auc:
            best_auc = auc
            best_value = val
            best_model = model

    print(f"Mejor min_child_weight={best_value} | AUC={best_auc:.4f}")
    return best_model, best_value

def tune_xgboost(X_train, y_train, X_val, y_val, param_grid):
    """
    Búsqueda exhaustiva (grid search) para XGBoost.
    """
    best_auc = 0
    best_model = None
    best_params = None
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    for combination in product(*values):
        params = dict(zip(keys, combination))
        print(f"Probando params XGBoost: {params}")

        model = XGBClassifier(
            **params,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        auc, _ = evaluate_model(model, X_val, y_val)

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_params = params

    print(f"Mejor ROC-AUC XGBoost: {best_auc:.4f} con params: {best_params}")
    return best_model, best_params

def randomized_search_xgboost(X_train, y_train, X_val, y_val, n_iter = 20, random_state = 42):
    """
    Randomized Search para XGBoost.
    """
    param_dist = {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.3, 0.5],
    }

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=random_state,
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        n_jobs=-1,
        cv=3,
        verbose=2,
        random_state=random_state,
    )
    search.fit(X_train, y_train)
    print("Mejores hiperparámetros XGB:", search.best_params_)
    print("Mejor AUC:", search.best_score_) 

    best_model = search.best_estimator_
    best_model.fit(X_train, y_train)

    return best_model, search.best_params_


# -------------------------------------------------
# 10. BLEND ENTRE RF Y XGB
# -------------------------------------------------
# Se combinan ambos modelos para mejorar el AUC.
# Estrategia:
#   1) Convertir probas a ranks para mayor robustez
#   2) Blend lineal: blend = (1 - w) * RF + w * XGB
#   3) Buscar el mejor w en validación

def blend_probas_ranked(proba_rf, proba_xgb, w):
    """
    Devuelve blend lineal rankeado: (1-w)*rank(RF) + w*rank(XGB).
    """
    n = len(proba_rf)
    r_rf = pd.Series(proba_rf).rank(method="average") / n
    r_xgb = pd.Series(proba_xgb).rank(method="average") / n
    return (1 - w) * r_rf + w * r_xgb

def tune_blend_weight(y_val, proba_rf_val, proba_xgb_val, grid=None):
    """
    Búsqueda del mejor 'w' entre RF y XGB optimizando AUC.
    """
    if grid is None:
        grid = np.linspace(0.0, 1.0, 21)  # 0.00, 0.05, ..., 1.00

    n = len(y_val)
    r_rf  = pd.Series(proba_rf_val).rank(method="average") / n
    r_xgb = pd.Series(proba_xgb_val).rank(method="average") / n

    best_w, best_auc = None, -1.0
    for w in grid:
        blend_val = (1 - w) * r_rf + w * r_xgb
        auc = roc_auc_score(y_val, blend_val)
        if auc > best_auc:
            best_w, best_auc = float(w), float(auc)
    print(f"Mejor w en VALID: {best_w:.2f} | AUC={best_auc:.5f}")
    return best_w, best_auc

def save_blend_predictions(proba_rf_test, proba_xgb_test, w, test_obs_ids, filename="blend.csv"):
    """
    Genera el CSV final de blend usando el mismo 'w' encontrado en validación.
    """
    blend_test = blend_probas_ranked(proba_rf_test, proba_xgb_test, w)
    preds = pd.DataFrame({"obs_id": test_obs_ids, "target": blend_test})
    preds.to_csv(filename, index=False)
    print(f"Blend guardado en {filename}")
    return preds

'''
# -------------------------------------------------
# 11. MAIN - EJECUCIÓN DE MODELOS
# -------------------------------------------------
def main():
    # Se cargan los datos para realizar un preprocesamiento y feature engineering
    print("=== Cargando datos y preprocesamiento ===")
    df = load_data(COMPETITION_PATH, sample_frac=0.2, random_state=1234)
    df = preprocess(df)
    df_model = create_user_content_features(df)

    # SE FILTRAN SOLO NUMERICAS??????

    # Split por usuario
    X_train, y_train, X_val, y_val, X_test, test_obs_ids = split_train_val_by_user(df_model)

    resultados = []  # Para guardar métricas de cada modelo

    # ------------------------------
    # Árbol de decisión
    # ------------------------------
    print("\n================ Árbol de decisión ================")

    # Entrenamiento básico
    dt_model = train_decision_tree(X_train, y_train)
    metrics_dt = evaluate_model(dt_model, X_val, y_val)
    resultados.append({"modelo": "Decision Tree (base)", **metrics_dt})
    predict_test(dt_model, X_test, test_obs_ids, filename="pred_decision_tree.csv")

    # Tuning
    #print("\n--- Tuning Decision Tree ---")
    dt_param_grid = {
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5]
    }
    best_dt_model, best_dt_params = tune_decision_tree(X_train, y_train, X_val, y_val, dt_param_grid)
    metrics_dt_tuned = evaluate_model(best_dt_model, X_val, y_val)
    resultados.append({"modelo": f"Decision Tree (tuned) {best_dt_params}", **metrics_dt_tuned})
    predict_test(best_dt_model, X_test, test_obs_ids, filename="pred_dt_tuned.csv")

if __name__ == "__main__":
    main()
'''


# -------------------------------------------------
# 11. MAIN - EJECUCIÓN DE MODELOS
# -------------------------------------------------

def main(run_dt=True, run_rf=True, run_xgb=True, sample_frac=None, tuning_mode="grid"):
    # Carga de datos y preprocesamiento
    print("=== Cargando datos y preprocesamiento ===")
    df = load_data(COMPETITION_PATH, sample_frac=sample_frac, random_state=1234)
    df = preprocess(df)
    
    # Feature engineering 
    df_model = create_user_content_features(df) 

    # Split
    X_train, y_train, X_val, y_val, X_test, test_obs_ids = split_train_val_by_user(df_model)

    # Para re-entrenar final
    X_full = df_model[~df_model["is_test"]].drop(columns=["target", "is_test", "username", "obs_id"])
    y_full = df_model[~df_model["is_test"]]["target"]

    proba_rf_val = proba_xgb_val = None
    proba_rf_test = proba_xgb_test = None

    # ÁRBOL DE DECISIÓN
    if run_dt:
        print("\n================ Decision Tree ================")
        dt_model = train_decision_tree(X_train, y_train)
        evaluate_model(dt_model, X_val, y_val)
        predict_test(dt_model, X_test, test_obs_ids, filename="dt_base.csv")

        # Tuning
        if tuning_mode != "none":
            grid = {
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            }
            best_dt, _ = tune_decision_tree(X_train, y_train, X_val, y_val, grid)
        else:
            best_dt = dt_model

        # Entrenamiento final
        dt_final = train_decision_tree(X_full, y_full)
        predict_test(dt_final, X_test, test_obs_ids, filename="dt_final.csv")

    # RANDOM FOREST
    if run_rf:
        print("\n================ Random Forest ================")
        rf_base = train_random_forest(X_train, y_train)
        auc_rf, proba_rf_val = evaluate_model(rf_base, X_val, y_val)
        proba_rf_test = rf_base.predict_proba(X_test)[:, 1]
        predict_test(rf_base, X_test, test_obs_ids, filename="rf_base.csv")

        # Tuning
        if tuning_mode == "grid":
            rf_grid = {
                "n_estimators": [200, 400],
                "max_depth": [None, 15],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt"]
            }
            best_rf, _ = tune_random_forest(X_train, y_train, X_val, y_val, rf_grid)

        elif tuning_mode == "random":
            best_rf, _ = randomized_search_random_forest(X_train, y_train)

        else:
            best_rf = rf_base

        # Entrenamiento final
        rf_final = train_random_forest(X_full, y_full)
        proba_rf_test = rf_final.predict_proba(X_test)[:, 1]
        predict_test(rf_final, X_test, test_obs_ids, filename="rf_final.csv")

    # XGBOOST
    if run_xgb:
        print("\n================ XGBoost ================")
        xgb_base = train_xgboost(X_train, y_train)
        auc_xgb, proba_xgb_val = evaluate_model(xgb_base, X_val, y_val)
        proba_xgb_test = xgb_base.predict_proba(X_test)[:, 1]
        predict_test(xgb_base, X_test, test_obs_ids, filename="xgb_base.csv")

        # Tuning
        if tuning_mode == "grid":
            xgb_grid = {
                "n_estimators": [200, 400],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }
            best_xgb, _ = tune_xgboost(X_train, y_train, X_val, y_val, xgb_grid)

        elif tuning_mode == "random":
            best_xgb, _ = randomized_search_xgboost(X_train, y_train, X_val, y_val)

        else:
            best_xgb = xgb_base

        # Entrenamiento final
        xgb_final = XGBClassifier(**best_xgb.get_params())
        xgb_final.fit(X_full, y_full)
        proba_xgb_test = xgb_final.predict_proba(X_test)[:, 1]
        predict_test(xgb_final, X_test, test_obs_ids, filename="xgb_final.csv")
        show_xgb_feature_importance(xgb_final)

    # BLEND 
    print("\n================ Blending RF + XGB ================")
    if proba_rf_val is not None and proba_xgb_val is not None:
        w_values = np.linspace(0, 1, 21)
        best_w, best_auc = tune_blend_weight(y_val, proba_rf_val, proba_xgb_val, w_values)
        print(f"Mejor w={best_w} | AUC Valid={best_auc:.5f}")

        blend_test = blend_probas_ranked(proba_rf_test, proba_xgb_test, best_w)
        pd.DataFrame({"obs_id": test_obs_ids, "target": blend_test}).to_csv("blend.csv", index=False)
        print("Blend final guardado en blend.csv")
    else:
        print("No se pudo hacer blend (faltan RF o XGB)")

    print("\n=== Pipeline completo ===")


# MODO DE USO
# Parámetros disponibles:
# --dt        Ejecuta Decision Tree
# --rf        Ejecuta Random Forest
# --xgb       Ejecuta XGBoost
# --tuning    Define el método de tuning: grid | random | none
# --sample    Usa solo una fracción del dataset (ej: 0.1 = 10%)
# (sin flags) Si no se especifica ningún modelo, corre TODOS.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf", action="store_true")
    parser.add_argument("--xgb", action="store_true")
    parser.add_argument("--dt", action="store_true")
    parser.add_argument("--sample", type=float, default=None)
    parser.add_argument("--tuning", type=str, default="grid", choices=["grid", "random", "none"])
    args = parser.parse_args()

    if not args.rf and not args.xgb and not args.dt:
        run_dt = True
        run_rf = True
        run_xgb = True
    else:
        run_dt = args.dt
        run_rf = args.rf
        run_xgb = args.xgb

    main(run_dt=run_dt, run_rf=run_rf, run_xgb=run_xgb, sample_frac=args.sample, tuning_mode=args.tuning)