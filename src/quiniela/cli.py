#############
# libraries #
#############

import argparse
import logging
from datetime import datetime
from quiniela.models import QuinielaModel
from quiniela.settings import LOGS_PATH

#############
# Parameter #
#############

DEFAULT_MODEL_PATH = LOGS_PATH / "quiniela.pkl"

###########
# main()  #
###########

def main():
    """
    Interfaz de línea de comandos (CLI) para el proyecto Quiniela.

    Subcomandos principales:
      - quiniela train   : entrena el modelo y guarda métricas.
      - quiniela predict : genera predicciones para una jornada concreta.
    """
    parser = argparse.ArgumentParser(
        prog="quiniela",
        description="CLI para entrenamiento y predicción del modelo Quiniela.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ----------------------------
    # Subcomando: train
    # ----------------------------
    train_parser = subparsers.add_parser(
        "train",
        help="Entrena el modelo Quiniela y guarda las métricas en logs/.",
    )
    train_parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Ruta donde se guardará el modelo entrenado (por defecto logs/quiniela.pkl).",
    )
    train_parser.add_argument(
        "--training-seasons",
        dest="training_seasons",
        default="2010:2020",
        help=(
            "Rango de seasons utilizadas para el entrenamiento. "
            "Ejemplos: '2010:2020' o '2019-2020'."
        ),
    )

    # ----------------------------
    # Subcomando: predict
    # ----------------------------
    predict_parser = subparsers.add_parser(
        "predict",
        help="Carga el modelo Quiniela y genera predicciones para una jornada.",
    )
    predict_parser.add_argument(
        "season",
        type=str,
        help="Temporada a predecir en formato 'YYYY-YYYY', por ejemplo '2021-2022'.",
    )
    predict_parser.add_argument(
        "division",
        type=int,
        help="División de la competición, por ejemplo 1 para Primera División.",
    )
    predict_parser.add_argument(
        "matchday",
        type=int,
        help="Número de jornada a predecir.",
    )
    predict_parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Ruta desde donde cargar el modelo entrenado (por defecto logs/quiniela.pkl).",
    )

    # ----------------------------
    # Subcomando: predict-season
    # ----------------------------
    predict_season_parser = subparsers.add_parser(
        "predict-season",
        help="Genera predicciones para todas las jornadas de una temporada.",
    )
    predict_season_parser.add_argument(
        "season",
        type=str,
        help="Temporada a predecir en formato 'YYYY-YYYY', por ejemplo '2021-2022'.",
    )
    predict_season_parser.add_argument(
        "division",
        type=int,
        help="División de la competición, por ejemplo 1 para Primera División.",
    )
    predict_season_parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Ruta desde donde cargar el modelo entrenado (por defecto logs/quiniela.pkl).",
    )

    args = parser.parse_args()

    # Configuración del log
    log_file = LOGS_PATH / f"{args.command}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logging.basicConfig(
        filename=log_file,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        level=logging.INFO,
    )

    # ----------------------------
    # Acción: TRAIN
    # ----------------------------
    if args.command == "train":
        logging.info(
            "Inicio de entrenamiento del modelo Quiniela. "
            f"training_seasons={args.training_seasons}"
        )
        print("\n==========  ENTRENAMIENTO DEL MODELO QUINIELA ========== \n")

        model = QuinielaModel()
        model.train(training_seasons=args.training_seasons)
        model.save(args.model_path)

        print(f"\nModelo entrenado y guardado en: {args.model_path}")
        print(f"Registro del entrenamiento: {log_file}")
        logging.info("Entrenamiento completado y modelo guardado.")

    # ----------------------------
    # Acción: PREDICT
    # ----------------------------
    elif args.command == "predict":
        logging.info(
            "Inicio de predicción con el modelo Quiniela. "
            f"season={args.season}, division={args.division}, matchday={args.matchday}"
        )
        print("\n========== PREDICCIÓN CON EL MODELO QUINIELA ========== \n")

        # Carga del modelo
        model = QuinielaModel.load(args.model_path)

        # Predicción de la jornada solicitada
        df_result = model.predict_matchday(
            season=args.season,
            division=args.division,
            matchday=args.matchday,
            save_csv=True,
            verbose=False,  # la salida "bonita" se gestiona desde la CLI
        )

        # Formato de salida tipo enunciado
        header = (
            f"Matchday {args.matchday} - LaLiga - Division {args.division} - Season {args.season}"
        )

        print("-" * 80) 
        print(header)
        print("-" * 80)

        # Impresión fila a fila
        for _, row in df_result.iterrows():
            home = str(row.get("home_team", "")).strip()
            away = str(row.get("away_team", "")).strip()
            quiniela = str(row.get("quiniela", "")).strip()

            print(f"{home:>20}          vs            {away:<20} --> {quiniela}")

        print("-" * 80)
        print(f"\nRegistro de la ejecución: {log_file}")
        logging.info("Predicciones completadas y archivo CSV de resultados guardado.")
    
    # ----------------------------
    # Acción: PREDICT-SEASON
    # ----------------------------
    elif args.command == "predict-season":
        logging.info(
            "Inicio de predicción de temporada completa. "
            f"season={args.season}, division={args.division}"
        )
        print("\n========== PREDICCIÓN COMPLETA DE TEMPORADA ==========\n")

        model = QuinielaModel.load(args.model_path)

        df_result = model.predict_season(
            season=args.season,
            division=args.division,
            save_csv=True,
            verbose=False,
        )

        # Impresión agrupada por jornada
        for md, group in df_result.sort_values("matchday").groupby("matchday"):
            header = (
                f"Matchday {int(md)} - LaLiga - Division {args.division} - Season {args.season}"
            )
            print("-" * 80)
            print(header)
            print("-" * 80)
            for _, row in group.iterrows():
                home = str(row.get("home_team", "")).strip()
                away = str(row.get("away_team", "")).strip()
                quiniela = str(row.get("quiniela", "")).strip()
                print(f"{home:>20}          vs            {away:<20} --> {quiniela}")
            print()  # línea en blanco entre jornadas

        print("-" * 80)
        print(f"\nRegistro de la ejecución: {log_file}")
        logging.info("Predicción de temporada completada y archivo CSV de resultados guardado.")

# def main():
#     """
#     Interfaz de línea de comandos (CLI) para el proyecto Quiniela.
#     Permite ejecutar:
#       - quiniela train  : Entrena el modelo y guarda métricas.
#       - quiniela predict: Genera predicciones con el modelo entrenado.
#     """
#     parser = argparse.ArgumentParser(
#         prog="quiniela",
#         description="CLI para entrenamiento y predicción del modelo Quiniela."
#     )

#     subparsers = parser.add_subparsers(dest="command", required=True)

#     # ----------------------------
#     # Subcomando: train
#     # ----------------------------
#     train_parser = subparsers.add_parser(
#         "train",
#         help="Entrenar el modelo Quiniela y guardar las métricas en logs/."
#     )
#     train_parser.add_argument(
#         "--model-path",
#         default=str(DEFAULT_MODEL_PATH),
#         help="Ruta donde se guardará el modelo entrenado (por defecto logs/quiniela.pkl)."
#     )

#     # ----------------------------
#     # Subcomando: predict
#     # ----------------------------
#     predict_parser = subparsers.add_parser(
#         "predict",
#         help="Cargar el modelo Quiniela y generar predicciones."
#     )
#     predict_parser.add_argument(
#         "--model-path",
#         default=str(DEFAULT_MODEL_PATH),
#         help="Ruta desde donde cargar el modelo entrenado (por defecto logs/quiniela.pkl)."
#     )

#     args = parser.parse_args()

#     # Configuración del log
#     log_file = LOGS_PATH / f"{args.command}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
#     logging.basicConfig(
#         filename=log_file,
#         format="%(asctime)s - [%(levelname)s] - %(message)s",
#         level=logging.INFO,
#     )

#     # ----------------------------
#     # Acción: TRAIN
#     # ----------------------------
#     if args.command == "train":
#         logging.info("Inicio de entrenamiento del modelo Quiniela.")
#         print("\n=== ENTRENAMIENTO DEL MODELO QUINIELA ===\n")

#         model = QuinielaModel()
#         X_train, X_test, y_train, y_test, X_to_predict, df_to_predict = model.split_data()
#         model.train_model(X_train, X_test, y_train, y_test)
#         model.save(args.model_path)

#         print(f"\nModelo entrenado y guardado en: {args.model_path}")
#         print(f"Registro del entrenamiento: {log_file}")
#         logging.info("Entrenamiento completado y modelo guardado.")

#     # ----------------------------
#     # Acción: PREDICT
#     # ----------------------------
#     elif args.command == "predict":
#         logging.info("Inicio de predicción con el modelo Quiniela.")
#         print("\n=== PREDICCIÓN CON EL MODELO QUINIELA ===\n")

#         model = QuinielaModel.load(args.model_path)
#         X_train, X_test, y_train, y_test, X_to_predict, df_to_predict =  model.split_data(verbose=False)
#         df_result = model.predict(X_to_predict, df_to_predict)

#         print("\nPredicciones generadas correctamente y guardadas en logs/predictions.csv")
#         print(f"Registro de la ejecución: {log_file}")
#         logging.info("Predicciones completadas y archivo guardado.")

