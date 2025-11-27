#############
# libraries #
#############

from pathlib import Path
import re
import sqlite3
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from category_encoders import CatBoostEncoder
from quiniela.io import load_laliga_data, prepare_df_rank, parse_training_seasons, map_code_to_quiniela
from quiniela.settings import LOGS_PATH, DATABASE_PATH

###################
# Quiniela Model #
##################

class QuinielaModel:
    """
    Modelo Quiniela para la predicción de resultados de partidos de fútbol.

    Flujo general:
      1) train(training_seasons): entrena el modelo usando las seasons indicadas.
      2) predict_matchday(season, division, matchday): predice una jornada concreta.
      3) predict_season(season, division): predice todas las jornadas de una temporada.
      4) save / load: guarda y carga el modelo completo (modelo, encoder, scaler).
    """

    def __init__(self):
        self.best_model = None
        self.encoder = None
        self.scaler = None
        self.feature_cols = None  # columnas utilizadas como features tras la codificación

    # -----------------------------------------------------
    # 1. Entrenamiento por seasons
    # -----------------------------------------------------
    def train(
        self,
        training_seasons: str = "2010:2020",
        test_size: float = 0.2,
        random_state: int = 42,
        verbose: bool = True,
    ):
        """
        Entrena el modelo utilizando únicamente las seasons indicadas.
        """

        # 1) Carga de datos desde SQLite
        if verbose:
            print("1) Carga de datos desde SQLite...")
        df = load_laliga_data(export_csv=True)

        # 2) Preparación y enriquecimiento de variables (df_rank)
        if verbose:
            print("\n2) Generación de df_rank con variables derivadas...")
        df_rank = prepare_df_rank(df, export_csv=True)

        # 3) Filtrado por seasons de entrenamiento
        seasons = parse_training_seasons(training_seasons)
        if verbose:
            print(f"\n3) Filtrado de seasons para entrenamiento: {seasons}")

        df_trainable = df_rank[df_rank["season"].isin(seasons)].dropna(
            subset=["result"]
        )

        # 4) Construcción de X e y
        drop_cols = ["score", "result", "home_goals", "away_goals", "date"]
        X = df_trainable.drop(columns=drop_cols, errors="ignore")
        y = df_trainable["result"].astype(int)

        # 5) División train / test
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )

        # 6) Codificación de variables categóricas
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        if verbose:
            print(f"\n4) Columnas categóricas para CatBoostEncoder: {cat_cols}")

        self.encoder = CatBoostEncoder(cols=cat_cols, random_state=random_state)
        self.encoder.fit(X_train, y_train)

        X_train_enc = self.encoder.transform(X_train).fillna(0)
        X_test_enc = self.encoder.transform(X_test).fillna(0)

        # 7) Escalado de variables numéricas
        self.feature_cols = X_train_enc.columns  # columnas tras la codificación
        self.scaler = MinMaxScaler()
        self.scaler.fit(X_train_enc)

        X_train_scaled = pd.DataFrame(
            self.scaler.transform(X_train_enc),
            columns=self.feature_cols,
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test_enc),
            columns=self.feature_cols,
        )

        if verbose:
            print("\n5) Tamaños de los conjuntos (tras codificación y escalado):")
            print(f"   Train: {X_train_scaled.shape}")
            print(f"   Test : {X_test_scaled.shape}")

        # 8) Definición del modelo y búsqueda de hiperparámetros
        if verbose:
            print("\n6) Entrenamiento de RandomForest con GridSearchCV...")

        rf = RandomForestClassifier(random_state=random_state, class_weight="balanced")
        param_grid_rf = {
            "n_estimators": [200],
            "criterion": ["gini"],
            "max_depth": [None, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt"],
            "bootstrap": [True],
        }

        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid_rf,
            cv=5,
            n_jobs=-1,
            scoring="accuracy",
        )
        grid_search.fit(X_train_scaled, y_train)

        self.best_model = grid_search.best_estimator_

        # 9) Evaluación en el conjunto de prueba
        y_pred = self.best_model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        if verbose:
            print("\n7) Resultados del entrenamiento:")
            print(f"\n  Parámetros del mejor modelo: {grid_search.best_params_}")
            print(f"\n  Accuracy: {acc:.4f}")
            print("\n   Matriz de confusión:\n", cm)
            print("\n   Reporte de clasificación:\n", classification_report(y_test, y_pred))

        # 10) Almacenamiento de métricas en Excel, incluyendo información de seasons
        safe_seasons_name = training_seasons.replace(":", "_").replace(" ", "")
        metrics_filename = f"metrics_seasons_{safe_seasons_name}.xlsx"
        metrics_path = LOGS_PATH / metrics_filename

        with pd.ExcelWriter(metrics_path, engine="xlsxwriter") as writer:
            # Hoja con accuracy e información de seasons
            df_acc = pd.DataFrame(
                {
                    "Accuracy": [acc],
                    "training_seasons_param": [training_seasons],
                    "training_seasons_list": [", ".join(seasons)],
                }
            )
            df_acc.to_excel(writer, sheet_name="Accuracy", index=False)

            # Matriz de confusión
            pd.DataFrame(cm).to_excel(
                writer, sheet_name="Confusion_Matrix", index=False
            )

            # Reporte de clasificación detallado
            pd.DataFrame(report).transpose().to_excel(
                writer, sheet_name="Classification_Report"
            )

        if verbose:
            print(f"\nMétricas guardadas en: {metrics_path}")

        return self.best_model

    # -----------------------------------------------------
    # 2. Predicción por season / división / jornada
    # -----------------------------------------------------
    def predict_matchday(
        self,
        season: str,
        division: int,
        matchday: int,
        save_csv: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Predice los partidos de una jornada concreta.

        Devuelve df_result con:
        - 'prediction' (0/1/2)
        - 'quiniela' (1/2/X)
        - 'confidence' (probabilidad de la clase predicha)
        """

        if self.best_model is None or self.encoder is None or self.scaler is None:
            raise ValueError(
                "El modelo no ha sido entrenado o cargado correctamente. "
                "Ejecutar previamente train() o load()."
            )

        # 1) Carga de datos completos
        if verbose:
            print("\n1) Carga de datos completos desde SQLite...")
        df = load_laliga_data(export_csv=False)

        # 2) Generación de df_rank
        if verbose:
            print("2) Generación de df_rank para predicción...")
        df_rank = prepare_df_rank(df, export_csv=False)

        # 3) Filtrado de la jornada
        if verbose:
            print(
                f"3) Filtrado de season={season}, division={division}, matchday={matchday}..."
            )

        df_day = df_rank[
            (df_rank["season"] == season)
            & (df_rank["division"] == division)
            & (df_rank["matchday"] == matchday)
        ].copy()

        if df_day.empty:
            raise ValueError(
                f"No se encontraron partidos para "
                f"season={season}, division={division}, matchday={matchday}."
            )

        # 4) Construcción de X para predicción
        drop_cols = ["score", "result", "home_goals", "away_goals", "date"]
        X_new = df_day.drop(columns=drop_cols, errors="ignore")

        # Codificación categórica
        X_new_enc = self.encoder.transform(X_new).fillna(0)

        # Asegurar mismas columnas que en entrenamiento
        for col in self.feature_cols:
            if col not in X_new_enc.columns:
                X_new_enc[col] = 0
        X_new_enc = X_new_enc[self.feature_cols]

        # Escalado
        X_new_scaled = pd.DataFrame(
            self.scaler.transform(X_new_enc),
            columns=self.feature_cols,
        )

        if verbose:
            print("4) Generación de predicciones para la jornada seleccionada...")

        # 5) Predicción + probabilidad
        y_pred = self.best_model.predict(X_new_scaled)
        y_proba = self.best_model.predict_proba(X_new_scaled)
        confidence = y_proba.max(axis=1)  # probabilidad de la clase elegida

        df_result = df_day.copy()
        df_result["prediction"] = y_pred
        df_result["confidence"] = confidence
        df_result["quiniela"] = df_result["prediction"].apply(map_code_to_quiniela)

        # 6) Guardar en CSV
        if save_csv:
            filename = f"predictions_{season}_div{division}_md{matchday}.csv"
            result_path = LOGS_PATH / filename
            df_result.to_csv(result_path, index=False)
            if verbose:
                print(f"\nPredicciones guardadas en: {result_path}")

        # 7) Mostrar resumen
        if verbose:
            print("\nResultados de la jornada:\n")
            cols_to_show = []
            for c in ["home_team", "away_team", "prediction", "quiniela", "confidence"]:
                if c in df_result.columns:
                    cols_to_show.append(c)
            if cols_to_show:
                print(df_result[cols_to_show])
            else:
                print(df_result.head())

        # 8) Guardar también en la base de datos (tabla Predictions)
        self._save_predictions_to_sqlite(df_result, verbose=verbose)

        return df_result

    # -----------------------------------------------------
    # 2.b Predicción de toda una temporada
    # -----------------------------------------------------
    def predict_season(
        self,
        season: str,
        division: int,
        save_csv: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Genera predicciones para todas las jornadas de una temporada+división.

        Devuelve df_result con:
        - 'prediction' (0/1/2)
        - 'quiniela' (1/2/X)
        - 'confidence' (probabilidad de la clase predicha)
        """

        if self.best_model is None or self.encoder is None or self.scaler is None:
            raise ValueError(
                "El modelo no ha sido entrenado o cargado correctamente. "
                "Ejecutar previamente train() o load()."
            )

        # 1) Carga de datos completos y df_rank
        if verbose:
            print("\n[Season] 1) Carga de datos completos desde SQLite...")
        df = load_laliga_data(export_csv=False)

        if verbose:
            print("[Season] 2) Generación de df_rank para predicción...")
        df_rank = prepare_df_rank(df, export_csv=False)

        # 2) Filtrar season/division
        if verbose:
            print(f"[Season] 3) Filtrado de season={season}, division={division}...")

        df_season = df_rank[
            (df_rank["season"] == season) & (df_rank["division"] == division)
        ].copy()

        if df_season.empty:
            raise ValueError(
                f"No se encontraron partidos para season={season}, division={division}."
            )

        # Jornadas disponibles
        matchdays = (
            df_season["matchday"]
            .dropna()
            .astype(int)
            .sort_values()
            .unique()
            .tolist()
        )

        if verbose:
            print(f"[Season] Jornadas detectadas: {matchdays}")

        all_results = []

        # 3) Predicción jornada a jornada
        for md in matchdays:
            df_day = df_season[df_season["matchday"] == md].copy()
            if df_day.empty:
                continue

            # X de esta jornada
            drop_cols = ["score", "result", "home_goals", "away_goals", "date"]
            X_new = df_day.drop(columns=drop_cols, errors="ignore")

            # Codificación
            X_new_enc = self.encoder.transform(X_new).fillna(0)

            # Asegurar mismas columnas
            for col in self.feature_cols:
                if col not in X_new_enc.columns:
                    X_new_enc[col] = 0
            X_new_enc = X_new_enc[self.feature_cols]

            # Escalado
            X_new_scaled = pd.DataFrame(
                self.scaler.transform(X_new_enc),
                columns=self.feature_cols,
            )

            # Predicción + probas
            y_pred = self.best_model.predict(X_new_scaled)
            y_proba = self.best_model.predict_proba(X_new_scaled)
            confidence = y_proba.max(axis=1)

            df_day_result = df_day.copy()
            df_day_result["prediction"] = y_pred
            df_day_result["confidence"] = confidence
            df_day_result["quiniela"] = df_day_result["prediction"].apply(
                map_code_to_quiniela
            )

            all_results.append(df_day_result)

        # 4) Unir todas las jornadas
        df_result = pd.concat(all_results, ignore_index=True)

        # 5) Guardar CSV
        if save_csv:
            filename = f"predictions_{season}_div{division}_all_matchdays.csv"
            result_path = LOGS_PATH / filename
            df_result.to_csv(result_path, index=False)
            if verbose:
                print(
                    f"\n[Season] Predicciones de toda la temporada guardadas en: {result_path}"
                )

        if verbose:
            print("\n[Season] Predicción de temporada completada.")
            cols_to_show = []
            for c in ["matchday", "home_team", "away_team", "quiniela", "confidence"]:
                if c in df_result.columns:
                    cols_to_show.append(c)
            if cols_to_show:
                print(df_result[cols_to_show].head())

        # 6) Guardar en SQLite
        self._save_predictions_to_sqlite(df_result, verbose=verbose)

        return df_result

    # -----------------------------------------------------
    # 2.c Helper: guardar predicciones en SQLite
    # -----------------------------------------------------
    
    def _save_predictions_to_sqlite(
        self,
        df_result: pd.DataFrame,
        verbose: bool = True,
    ) -> None:
        """
        Guarda las predicciones en la tabla 'Predictions' de laliga.sqlite.

        - Si la columna 'confidence' no existe en la tabla, la crea.
        - Almacena 'quiniela' como 'pred' y la probabilidad en 'confidence'.
        """

        # Copia para no modificar df_result original
        df_result = df_result.copy()

        # Si la columna 'time' no existe, la creamos
        if "time" not in df_result.columns:
            df_result["time"] = None

        # Columnas requeridas en df_result
        required_cols = [
            "season",
            "division",
            "matchday",
            "date",
            "time",
            "home_team",
            "away_team",
            "score",
            "quiniela",
            "confidence",  # ahora siempre esperamos esta columna
        ]
        missing = [c for c in required_cols if c not in df_result.columns]
        if missing:
            raise ValueError(
                f"Faltan columnas necesarias para guardar en SQLite: {missing}"
            )

        # Selección de columnas necesarias
        df_db = df_result[required_cols].copy()

        # ----------------------------
        # Conversión de tipos a SQLite
        # ----------------------------

        # Convertir date → formato MM-DD-YYYY
        df_db["date"] = (
            pd.to_datetime(df_db["date"], errors="coerce")
            .dt.strftime("%m-%d-%Y")
        )

        # Convertir time a string
        df_db["time"] = df_db["time"].astype(str)

        # Asegurar tipo float de confidence
        df_db["confidence"] = df_db["confidence"].astype(float)

        # Renombrar quiniela → pred
        df_db = df_db.rename(columns={"quiniela": "pred"})

        # CONEXIÓN A SQLITE
        with sqlite3.connect(DATABASE_PATH) as conn:
            cur = conn.cursor()

            # 1) Comprobar si la columna 'confidence' existe en la tabla
            cur.execute("PRAGMA table_info(Predictions);")
            existing_cols = [row[1] for row in cur.fetchall()]

            if "confidence" not in existing_cols:
                # Añadimos la columna si no existe
                if verbose:
                    print("Añadiendo columna 'confidence' a la tabla Predictions...")
                cur.execute(
                    "ALTER TABLE Predictions ADD COLUMN confidence REAL;"
                )
                conn.commit()

            # 2) Insertar filas (borrando antes si ya existían)
            for row in df_db.itertuples(index=False):

                # Eliminamos cualquier fila previa del mismo partido
                cur.execute(
                    """
                    DELETE FROM Predictions
                    WHERE season=? AND division=? AND matchday=?
                          AND home_team=? AND away_team=?;
                    """,
                    (
                        row.season,
                        int(row.division),
                        int(row.matchday),
                        row.home_team,
                        row.away_team,
                    ),
                )

                # Insertamos nueva fila con confidence
                cur.execute(
                    """
                    INSERT INTO Predictions(
                        season, division, matchday, date, time,
                        home_team, away_team, score, pred, confidence
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        row.season,
                        int(row.division),
                        int(row.matchday),
                        row.date,       # formato MM-DD-YYYY
                        row.time,       # string
                        row.home_team,
                        row.away_team,
                        row.score,
                        row.pred,
                        float(row.confidence),
                    ),
                )

            conn.commit()

        if verbose:
            print(
                f"{len(df_db)} filas guardadas/actualizadas en la tabla 'Predictions' ({DATABASE_PATH})."
            )

    # -----------------------------------------------------
    # 3. Guardado y carga de modelos
    # -----------------------------------------------------
    def save(self, filename: str | Path):
        """
        Guarda en disco el objeto QuinielaModel completo, incluyendo el
        modelo entrenado, el codificador categórico y el escalador.
        """
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"Modelo guardado en: {filename}")

    @classmethod
    def load(cls, filename: str | Path):
        """
        Carga desde disco un objeto QuinielaModel previamente guardado.
        """
        filename = Path(filename)
        with open(filename, "rb") as f:
            model = pickle.load(f)
            assert type(model) is cls
        return model

# class QuinielaModel:
#     """
#     Modelo Quiniela para la predicción de resultados de partidos de fútbol.

#     Flujo general:
#       1) train(training_seasons): entrena el modelo usando las seasons indicadas.
#       2) predict_matchday(season, division, matchday): predice una jornada concreta.
#       3) save / load: guarda y carga el modelo completo (modelo, encoder, scaler).
#     """

#     def __init__(self):
#         self.best_model = None
#         self.encoder = None
#         self.scaler = None
#         self.feature_cols = None  # columnas utilizadas como features tras la codificación

#     # -----------------------------------------------------
#     # 1. Entrenamiento por seasons
#     # -----------------------------------------------------
#     def train(
#         self,
#         training_seasons: str = "2010:2020",
#         test_size: float = 0.2,
#         random_state: int = 42,
#         verbose: bool = True,
#     ):
#         """
#         Entrena el modelo utilizando únicamente las seasons indicadas.

#         Parámetros
#         ----------
#         training_seasons : str, opcional
#             Cadena que define el conjunto de seasons de entrenamiento.
#             Ejemplos: '2010:2020' o '2019-2020'.
#         test_size : float, opcional
#             Proporción de datos reservada para el conjunto de prueba (por defecto 0.2).
#         random_state : int, opcional
#             Semilla para reproducibilidad (por defecto 42).
#         verbose : bool, opcional
#             Si es True, muestra información del proceso por pantalla.
#         """

#         # 1) Carga de datos desde SQLite
#         if verbose:
#             print("1) Carga de datos desde SQLite...")
#         df = load_laliga_data(export_csv=True)

#         # 2) Preparación y enriquecimiento de variables (df_rank)
#         if verbose:
#             print("\n2) Generación de df_rank con variables derivadas...")
#         df_rank = prepare_df_rank(df, export_csv=True)

#         # 3) Filtrado por seasons de entrenamiento
#         seasons = parse_training_seasons(training_seasons)
#         if verbose:
#             print(f"\n3) Filtrado de seasons para entrenamiento: {seasons}")

#         df_trainable = df_rank[df_rank["season"].isin(seasons)].dropna(subset=["result"])

#         # 4) Construcción de X e y
#         drop_cols = ["score", "result", "home_goals", "away_goals", "date"]
#         X = df_trainable.drop(columns=drop_cols, errors="ignore")
#         y = df_trainable["result"].astype(int)

#         # 5) División train / test
#         X_train, X_test, y_train, y_test = train_test_split(
#             X,
#             y,
#             test_size=test_size,
#             stratify=y,
#             random_state=random_state,
#         )

#         # 6) Codificación de variables categóricas
#         cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
#         if verbose:
#             print(f"\n4) Columnas categóricas para CatBoostEncoder: {cat_cols}")

#         self.encoder = CatBoostEncoder(cols=cat_cols, random_state=random_state)
#         self.encoder.fit(X_train, y_train)

#         X_train_enc = self.encoder.transform(X_train).fillna(0)
#         X_test_enc = self.encoder.transform(X_test).fillna(0)

#         # 7) Escalado de variables numéricas
#         self.feature_cols = X_train_enc.columns  # columnas tras la codificación
#         self.scaler = MinMaxScaler()
#         self.scaler.fit(X_train_enc)

#         X_train_scaled = pd.DataFrame(
#             self.scaler.transform(X_train_enc),
#             columns=self.feature_cols,
#         )
#         X_test_scaled = pd.DataFrame(
#             self.scaler.transform(X_test_enc),
#             columns=self.feature_cols,
#         )

#         if verbose:
#             print("\n5) Tamaños de los conjuntos (tras codificación y escalado):")
#             print(f"   Train: {X_train_scaled.shape}")
#             print(f"   Test : {X_test_scaled.shape}")

#         # 8) Definición del modelo y búsqueda de hiperparámetros
#         if verbose:
#             print("\n6) Entrenamiento de RandomForest con GridSearchCV...")

#         rf = RandomForestClassifier(random_state=random_state, class_weight="balanced")
#         param_grid_rf = {
#             "n_estimators": [200],
#             "criterion": ["gini"],
#             "max_depth": [None, 10],
#             "min_samples_split": [2, 5],
#             "min_samples_leaf": [1, 2, 4],
#             "max_features": ["sqrt"],
#             "bootstrap": [True],
#         }

#         grid_search = GridSearchCV(
#             estimator=rf,
#             param_grid=param_grid_rf,
#             cv=5,
#             n_jobs=-1,
#             scoring="accuracy",
#         )
#         grid_search.fit(X_train_scaled, y_train)

#         self.best_model = grid_search.best_estimator_

#         # 9) Evaluación en el conjunto de prueba
#         y_pred = self.best_model.predict(X_test_scaled)

#         acc = accuracy_score(y_test, y_pred)
#         cm = confusion_matrix(y_test, y_pred)
#         report = classification_report(y_test, y_pred, output_dict=True)

#         if verbose:
#             print("\n7) Resultados del entrenamiento:")
#             print(f"\n  Parámetros del mejor modelo: {grid_search.best_params_}")
#             print(f"\n  Accuracy: {acc:.4f}")
#             print("\n   Matriz de confusión:\n", cm)
#             print("\n   Reporte de clasificación:\n", classification_report(y_test, y_pred))

#         # 10) Almacenamiento de métricas en Excel, incluyendo información de seasons
#         safe_seasons_name = training_seasons.replace(":", "_").replace(" ", "")
#         metrics_filename = f"metrics_seasons_{safe_seasons_name}.xlsx"
#         metrics_path = LOGS_PATH / metrics_filename

#         with pd.ExcelWriter(metrics_path, engine="xlsxwriter") as writer:
#             # Hoja con accuracy e información de seasons
#             df_acc = pd.DataFrame(
#                 {
#                     "Accuracy": [acc],
#                     "training_seasons_param": [training_seasons],
#                     "training_seasons_list": [", ".join(seasons)],
#                 }
#             )
#             df_acc.to_excel(writer, sheet_name="Accuracy", index=False)

#             # Matriz de confusión
#             pd.DataFrame(cm).to_excel(
#                 writer, sheet_name="Confusion_Matrix", index=False
#             )

#             # Reporte de clasificación detallado
#             pd.DataFrame(report).transpose().to_excel(
#                 writer, sheet_name="Classification_Report"
#             )

#         if verbose:
#             print(f"\nMétricas guardadas en: {metrics_path}")

#         return self.best_model

#     # -----------------------------------------------------
#     # 2. Predicción por season / división / jornada
#     # -----------------------------------------------------
#     def predict_matchday(
#         self,
#         season: str,
#         division: int,
#         matchday: int,
#         save_csv: bool = True,
#         verbose: bool = True,
#     ) -> pd.DataFrame:
#         """
#         Predice los partidos correspondientes a una jornada concreta y devuelve
#         un DataFrame con la información del partido y las predicciones.

#         Parámetros
#         ----------
#         season : str
#             Identificador de la temporada, por ejemplo '2021-2022'.
#         division : int
#             División de la liga, por ejemplo 1 para Primera División.
#         matchday : int
#             Número de jornada.
#         save_csv : bool, opcional
#             Si es True, guarda las predicciones en un archivo CSV en LOGS_PATH.
#         verbose : bool, opcional
#             Si es True, muestra información del proceso por pantalla.

#         Retorna
#         -------
#         pd.DataFrame
#             DataFrame con la información de los partidos de la jornada y las
#             columnas añadidas 'prediction' (0/1/2) y 'quiniela' (1/2/X).
#         """

#         if self.best_model is None or self.encoder is None or self.scaler is None:
#             raise ValueError(
#                 "El modelo no ha sido entrenado o cargado correctamente. "
#                 "Ejecutar previamente train() o load()."
#             )

#         # 1) Carga de datos completos
#         if verbose:
#             print("\n1) Carga de datos completos desde SQLite...")
#         df = load_laliga_data(export_csv=False)

#         # 2) Generación de df_rank con las mismas transformaciones que en entrenamiento
#         if verbose:
#             print("2) Generación de df_rank para predicción...")
#         df_rank = prepare_df_rank(df, export_csv=False)

#         # 3) Filtrado de la jornada solicitada
#         if verbose:
#             print(
#                 f"3) Filtrado de season={season}, division={division}, matchday={matchday}..."
#             )

#         df_day = df_rank[
#             (df_rank["season"] == season)
#             & (df_rank["division"] == division)
#             & (df_rank["matchday"] == matchday)
#         ].copy()

#         if df_day.empty:
#             raise ValueError(
#                 f"No se encontraron partidos para "
#                 f"season={season}, division={division}, matchday={matchday}."
#             )

#         # 4) Construcción de la matriz de características para predicción
#         drop_cols = ["score", "result", "home_goals", "away_goals", "date"]
#         X_new = df_day.drop(columns=drop_cols, errors="ignore")

#         # Codificación categórica coherente con el entrenamiento
#         X_new_enc = self.encoder.transform(X_new).fillna(0)

#         # Asegurar que las columnas coinciden con las utilizadas en entrenamiento
#         for col in self.feature_cols:
#             if col not in X_new_enc.columns:
#                 X_new_enc[col] = 0
#         X_new_enc = X_new_enc[self.feature_cols]

#         # Escalado
#         X_new_scaled = pd.DataFrame(
#             self.scaler.transform(X_new_enc),
#             columns=self.feature_cols,
#         )

#         if verbose:
#             print("4) Generación de predicciones para la jornada seleccionada...")

#         # 5) Predicción
#         y_pred = self.best_model.predict(X_new_scaled)

#         df_result = df_day.copy()
#         df_result["prediction"] = y_pred
#         df_result["quiniela"] = df_result["prediction"].apply(map_code_to_quiniela)

#         # 6) Almacenamiento opcional en CSV
#         if save_csv:
#             filename = f"predictions_{season}_div{division}_md{matchday}.csv"
#             result_path = LOGS_PATH / filename
#             df_result.to_csv(result_path, index=False)
#             if verbose:
#                 print(f"\nPredicciones guardadas en: {result_path}")

#         # 7) Visualización resumida (opcional)
#         if verbose:
#             print("\nResultados de la jornada:\n")
#             cols_to_show = []
#             for c in ["home_team", "away_team", "prediction", "quiniela"]:
#                 if c in df_result.columns:
#                     cols_to_show.append(c)
#             if cols_to_show:
#                 print(df_result[cols_to_show])
#             else:
#                 print(df_result.head())

#         return df_result
    
#     # -----------------------------------------------------
#     # 2.b Predicción de todas las jornadas de una temporada
#     # -----------------------------------------------------
#     def predict_season(
#         self,
#         season: str,
#         division: int,
#         save_csv: bool = True,
#         verbose: bool = True,
#     ) -> pd.DataFrame:
#         """
#         Genera predicciones para todas las jornadas disponibles de una
#         temporada y división dadas. Devuelve un DataFrame con todos los
#         partidos y las columnas 'prediction' (0/1/2) y 'quiniela' (1/2/X).

#         Parámetros
#         ----------
#         season : str
#             Temporada en formato 'YYYY-YYYY', por ejemplo '2021-2022'.
#         division : int
#             División de la competición (1 para Primera División, etc.).
#         save_csv : bool, opcional
#             Si es True, guarda las predicciones de toda la temporada en un CSV.
#         verbose : bool, opcional
#             Si es True, muestra información del proceso por pantalla.

#         Retorna
#         -------
#         pd.DataFrame
#             DataFrame con las predicciones de todos los partidos de la temporada.
#         """

#         if self.best_model is None or self.encoder is None or self.scaler is None:
#             raise ValueError(
#                 "El modelo no ha sido entrenado o cargado correctamente. "
#                 "Ejecutar previamente train() o load()."
#             )

#         # 1) Carga de datos completos y generación de df_rank
#         if verbose:
#             print("\n[Season] 1) Carga de datos completos desde SQLite...")
#         df = load_laliga_data(export_csv=False)

#         if verbose:
#             print("[Season] 2) Generación de df_rank para predicción...")
#         df_rank = prepare_df_rank(df, export_csv=False)

#         # 2) Filtrar toda la temporada para la división solicitada
#         if verbose:
#             print(f"[Season] 3) Filtrado de season={season}, division={division}...")

#         df_season = df_rank[
#             (df_rank["season"] == season) & (df_rank["division"] == division)
#         ].copy()

#         if df_season.empty:
#             raise ValueError(
#                 f"No se encontraron partidos para season={season}, division={division}."
#             )

#         # Jornadas disponibles (ordenadas)
#         matchdays = (
#             df_season["matchday"]
#             .dropna()
#             .astype(int)
#             .sort_values()
#             .unique()
#             .tolist()
#         )

#         if verbose:
#             print(f"[Season] Jornadas detectadas: {matchdays}")

#         all_results = []

#         # 3) Predicción jornada a jornada
#         for md in matchdays:
#             df_day = df_season[df_season["matchday"] == md].copy()
#             if df_day.empty:
#                 continue

#             # Construcción de X para esta jornada
#             drop_cols = ["score", "result", "home_goals", "away_goals", "date"]
#             X_new = df_day.drop(columns=drop_cols, errors="ignore")

#             # Codificación categórica coherente con el entrenamiento
#             X_new_enc = self.encoder.transform(X_new).fillna(0)

#             # Asegurar mismas columnas que en entrenamiento
#             for col in self.feature_cols:
#                 if col not in X_new_enc.columns:
#                     X_new_enc[col] = 0
#             X_new_enc = X_new_enc[self.feature_cols]

#             # Escalado
#             X_new_scaled = pd.DataFrame(
#                 self.scaler.transform(X_new_enc),
#                 columns=self.feature_cols,
#             )

#             # Predicción
#             y_pred = self.best_model.predict(X_new_scaled)

#             df_day_result = df_day.copy()
#             df_day_result["prediction"] = y_pred
#             df_day_result["quiniela"] = df_day_result["prediction"].apply(
#                 map_code_to_quiniela
#             )

#             all_results.append(df_day_result)

#         # 4) Concatenar resultados de todas las jornadas
#         df_result = pd.concat(all_results, ignore_index=True)

#         # 5) Guardar en CSV (toda la temporada) si procede
#         if save_csv:
#             filename = f"predictions_{season}_div{division}_all_matchdays.csv"
#             result_path = LOGS_PATH / filename
#             df_result.to_csv(result_path, index=False)
#             if verbose:
#                 print(f"\n[Season] Predicciones de toda la temporada guardadas en: {result_path}")

#         if verbose:
#             print("\n[Season] Predicción de temporada completada.")
#             cols_to_show = []
#             for c in ["matchday", "home_team", "away_team", "quiniela"]:
#                 if c in df_result.columns:
#                     cols_to_show.append(c)
#             if cols_to_show:
#                 print(df_result[cols_to_show].head())

#         return df_result

#     # -----------------------------------------------------
#     # 3. Guardado y carga de modelos
#     # -----------------------------------------------------
#     def save(self, filename: str | Path):
#         """
#         Guarda en disco el objeto QuinielaModel completo, incluyendo el
#         modelo entrenado, el codificador categórico y el escalador.

#         Parámetros
#         ----------
#         filename : str | Path
#             Ruta del archivo de salida.
#         """
#         filename = Path(filename)
#         filename.parent.mkdir(parents=True, exist_ok=True)
#         with open(filename, "wb") as f:
#             pickle.dump(self, f)
#         print(f"Modelo guardado en: {filename}")

#     @classmethod
#     def load(cls, filename: str | Path):
#         """
#         Carga desde disco un objeto QuinielaModel previamente guardado.

#         Parámetros
#         ----------
#         filename : str | Path
#             Ruta del archivo .pkl que contiene el modelo.

#         Retorna
#         -------
#         QuinielaModel
#             Instancia de QuinielaModel cargada desde el archivo.
#         """
#         filename = Path(filename)
#         with open(filename, "rb") as f:
#             model = pickle.load(f)
#             assert type(model) is cls
#         return model

# class QuinielaModel:
#     """
#     Modelo Quiniela para la predicción de resultados de partidos de fútbol.
#     Flujo general:
#         1) split_data: genera conjuntos de entrenamiento, prueba y predicción.
#         2) train_model: entrena el modelo, evalúa resultados y guarda métricas.
#         3) predict: genera y guarda las predicciones finales.
#     """

#     def __init__(self):
#         self.best_model = None
#         self.encoder = None
#         self.scaler = None
#         self.feature_cols = None

#     # -----------------------------------------------------
#     # 1. Split Data
#     # -----------------------------------------------------
#     def split_data(self, df: pd.DataFrame | None = None, verbose: bool = True):
#         if verbose:
#             print("1) Lectura data desde SQLite:")

#         if df is None:
#             df = load_laliga_data(export_csv=True)

#         if verbose:
#             print("\n2) Creación de nuevas variables:")

#         df_rank = prepare_df_rank(df, export_csv=True)

#         fecha_ini = pd.Timestamp("2021-08-30")
#         fecha_fin = pd.Timestamp("2022-05-29")

#         df_trainable = df_rank[df_rank["date"] < fecha_ini].dropna(subset=["result"])
#         df_to_predict = df_rank[
#             (df_rank["date"] >= fecha_ini)
#             & (df_rank["date"] <= fecha_fin)
#             & (df_rank["result"].isna())
#         ].drop(columns=["score", "result", "home_goals", "away_goals", "date"])

#         X = df_trainable.drop(columns=["score", "result", "home_goals", "away_goals", "date"])
#         y = df_trainable["result"].astype(int)

#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, stratify=y, random_state=42
#         )

#         cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
#         self.encoder = CatBoostEncoder(cols=cat_cols, random_state=42)
#         self.encoder.fit(X_train, y_train)

#         X_train = self.encoder.transform(X_train).fillna(0)
#         X_test = self.encoder.transform(X_test).fillna(0)
#         X_to_predict = self.encoder.transform(df_to_predict).fillna(0)

#         self.scaler = MinMaxScaler()
#         self.scaler.fit(X_train)

#         X_train = pd.DataFrame(self.scaler.transform(X_train), columns=X_train.columns)
#         X_test = pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns)
#         X_to_predict = pd.DataFrame(self.scaler.transform(X_to_predict), columns=X_to_predict.columns)

#         if verbose:
#             print("\n3) Spli Data:")
#             print("Tamaño train (scaled):", X_train.shape)
#             print("Tamaño test (scaled):", X_test.shape)
#             print("Tamaño predict (scaled):", X_to_predict.shape)

#         self.feature_cols = X_train.columns
#         return X_train, X_test, y_train, y_test, X_to_predict, df_to_predict


#     # -----------------------------------------------------
#     # 2. Train Model
#     # -----------------------------------------------------
#     def train_model(self, X_train, X_test, y_train, y_test):
#         """
#         Entrena RandomForest con GridSearchCV, imprime métricas y
#         guarda logs/metrics.xlsx.
#         """

#         print("\n")
#         print("4) Entrenando modelo RandomForest:")

#         rf = RandomForestClassifier(random_state=42, class_weight="balanced")
#         param_grid_rf = {
#             "n_estimators": [200],
#             "criterion": ["gini"],
#             "max_depth": [None, 10],
#             "min_samples_split": [2, 5],
#             "min_samples_leaf": [1, 2, 4],
#             "max_features": ["sqrt"],
#             "bootstrap": [True],
#         }

#         grid_search = GridSearchCV(
#             estimator=rf,
#             param_grid=param_grid_rf,
#             cv=5,
#             n_jobs=-1,
#             scoring="accuracy",
#         )
#         grid_search.fit(X_train, y_train)

#         self.best_model = grid_search.best_estimator_
#         y_pred = self.best_model.predict(X_test)

#         acc = accuracy_score(y_test, y_pred)
#         cm = confusion_matrix(y_test, y_pred)
#         report = classification_report(y_test, y_pred, output_dict=True)

#         print("Entrenamiento completado.")
#         print("Parámetros del mejor modelo:", grid_search.best_params_)
#         print(f"Accuracy: {acc:.4f}")
#         print("\nMatriz de confusión:\n", cm)
#         print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

#         metrics_path = LOGS_PATH / "metrics.xlsx"
#         with pd.ExcelWriter(metrics_path, engine="xlsxwriter") as writer:
#             pd.DataFrame({"Accuracy": [acc]}).to_excel(writer, sheet_name="Accuracy", index=False)
#             pd.DataFrame(cm).to_excel(writer, sheet_name="Confusion_Matrix", index=False)
#             pd.DataFrame(report).transpose().to_excel(writer, sheet_name="Classification_Report")
#         print(f"Métricas guardadas en: {metrics_path}")

#         return self.best_model

#     # -----------------------------------------------------
#     # 3. Predict
#     # -----------------------------------------------------
#     def predict(self, X_to_predict, df_to_predict):
#         """
#         Predice con el mejor modelo entrenado, imprime df_result y
#         guarda logs/predictions.csv.
#         """
#         if self.best_model is None:
#             raise ValueError("El modelo no ha sido entrenado. Ejecute train_model() primero.")

#         print("1) Generando predicciones:")

#         y_pred_new = self.best_model.predict(X_to_predict)
#         df_pred = pd.DataFrame(y_pred_new, columns=["prediction"])
#         df_pred.index = df_to_predict.index
#         df_result = pd.concat([df_to_predict, df_pred], axis=1)

#         result_path = LOGS_PATH / "predictions.csv"
#         df_result.to_csv(result_path, index=False)

#         # print("\n")
#         print(f"Predicciones guardadas en: {result_path}")

#         # print("\n")
#         print("\nResultados:\n")
#         print(df_result)  # imprime el df_result completo

#         return df_result

#     # -----------------------------------------------------
#     # 4. Save / Load
#     # -----------------------------------------------------
#     def save(self, filename):
#         """Guarda el modelo, codificador y escalador."""
#         with open(filename, "wb") as f:
#             pickle.dump(self, f)
#         print(f"Modelo guardado en: {filename}")

#     @classmethod
#     def load(cls, filename):
#         """Carga un modelo previamente guardado."""
#         with open(filename, "rb") as f:
#             model = pickle.load(f)
#             assert type(model) is cls
#         # print(f"Modelo cargado desde: {filename}")
#         return model

