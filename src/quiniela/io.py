#############
# libraries #
#############

import re
import sqlite3
import pandas as pd
import numpy as np
from quiniela.settings import DATABASE_PATH, LOGS_PATH

#############
# Functions #
#############

# ------------------------------------
# Function 1: Load data from SQLite 
# ------------------------------------

def load_laliga_data(export_csv: bool = True, csv_path: str = None) -> pd.DataFrame:
    """
    Carga los datos de la tabla 'Matches' desde la base SQLite especificada en settings.py.
    Opcionalmente exporta el DataFrame resultante a un archivo CSV en el directorio de logs.

    Parámetros
    ----------
    export_csv : bool, opcional
        Si es True, exporta el DataFrame a un archivo CSV (por defecto True).
    csv_path : str, opcional
        Ruta de exportación personalizada para el CSV. Si es None, se utiliza logs/matches.csv.

    Retorna
    -------
    pd.DataFrame
        DataFrame con los registros de la tabla 'Matches'.
    """
    # Conexión a la base de datos
    conn = sqlite3.connect(DATABASE_PATH)

    # Lectura de la tabla 'Matches'
    df = pd.read_sql_query("SELECT * FROM Matches;", conn)

    # Cierre de la conexión
    conn.close()

    # Exportación opcional
    if export_csv:
        export_path = LOGS_PATH / "matches.csv" if csv_path is None else csv_path
        df.to_csv(export_path, index=False)
        # print(f"Archivo de SQLite exportado correctamente en: {export_path}")

    return df

# -----------------------------------------
# Function 2: Treatment and Preparation   
# -----------------------------------------
def prepare_df_rank(df: pd.DataFrame, export_csv: bool = True, csv_path: str | None = None) -> pd.DataFrame:
    """
    Prepara el DataFrame de partidos para modelado:
      - Normaliza fechas y corrige años fuera de rango.
      - Extrae goles locales/visitantes desde 'score'.
      - Construye 'result' con codificación 0 (local), 1 (visitante), 2 (empate).
      - Genera rankings acumulados por jornada sin fuga temporal.
      - Calcula forma reciente (últimos 5) por goles y puntos.
      - Calcula ratios H2H históricos en ventana de 5 temporadas.
      - Imputa valores neutrales para faltantes iniciales.
      - Ordena y exporta opcionalmente a CSV.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame original de partidos.
    export_csv : bool, opcional
        Si es True, exporta el df_rank a CSV (por defecto True).
    csv_path : str | None, opcional
        Ruta de exportación del CSV. Si es None, se usa logs/df_rank.csv.

    Retorna
    -------
    pd.DataFrame
        DataFrame procesado df_rank.
    """
    # Copia defensiva para no mutar el df original.
    df = df.copy()

    # ==============================
    # 1) Fechas y columnas básicas
    # ==============================
    # - Estandariza 'date' a tipo datetime (formato fuente mm/dd/yy).
    # - Corrige fechas futuras erróneas (p. ej., 2121 → 2021) restando 100 años.
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce', format='%m/%d/%y')
        mask_future = df['date'] > pd.Timestamp('2025-01-01')
        df.loc[mask_future, 'date'] = df.loc[mask_future, 'date'] - pd.DateOffset(years=100)

    # 'time' no se utiliza en el pipeline de features → se elimina si existe.
    if 'time' in df.columns:
        df = df.drop(columns=['time'])

    # =============================================
    # 2) Parsing de goles a partir de la columna score
    # =============================================
    # - 'score' viene como "goles_local:goles_visitante".
    # - Se separa en dos columnas numéricas: home_goals y away_goals.
    if 'score' in df.columns:
        goals = df['score'].str.split(':', expand=True)
        if goals is not None and goals.shape[1] == 2:
            df[['home_goals', 'away_goals']] = goals
    df['home_goals'] = pd.to_numeric(df.get('home_goals'), errors='coerce')
    df['away_goals'] = pd.to_numeric(df.get('away_goals'), errors='coerce')

    # ==================================================
    # 3) Etiqueta 'result' (0=home, 1=away, 2=draw)
    # ==================================================
    # - Si faltan goles, el resultado queda como NaN.
    def _result_from_goals(r):
        if pd.isna(r['home_goals']) or pd.isna(r['away_goals']):
            return np.nan
        if r['home_goals'] > r['away_goals']:
            return 0
        if r['home_goals'] < r['away_goals']:
            return 1
        return 2

    if 'result' not in df.columns:
        df['result'] = df.apply(_result_from_goals, axis=1)

    # Copia de trabajo para enriquecer sin tocar 'df'.
    df_rank = df.copy()

    # ==========================================================
    # 4) season_start numérico (año inicial de la temporada)
    # ==========================================================
    # - Convierte '2019-2020' → 2019, robusto a formatos '2019/2020' o '2019'.
    def season_to_int(season):
        s = str(season)
        try:
            return int(re.split(r'[-/]', s)[0])
        except Exception:
            try:
                return int(s)
            except Exception:
                return np.nan

    df_rank['season_start'] = df_rank['season'].apply(season_to_int).astype('Int64')

    # ===================================================
    # 5) Orden temporal base para evitar fuga de info
    # ===================================================
    # - Ordena por season_start/division/matchday para que las acumulaciones y shifts
    #   se hagan cronológicamente y no usen datos "del futuro".
    sort_cols_base = [c for c in ['season_start', 'division', 'matchday'] if c in df_rank.columns]
    df_rank = df_rank.sort_values(sort_cols_base).reset_index(drop=True)

    # ===========================================================
    # 6) Rankings acumulados por equipo (sin fuga temporal)
    # ===========================================================
    # - Se apilan partidos locales y visitantes en una tabla "matches" por equipo.
    # - Se calculan acumulados (GF, GA, W, L, T) y a partir de ellos:
    #   GD = GF - GA, Pts = 3*W + T, y un ranking denso por jornada.
    home = df_rank[['season_start','season','division','matchday','home_team','home_goals','away_goals']].copy()
    home = home.rename(columns={'home_team':'team','home_goals':'GF','away_goals':'GA'})
    home['result_WLT'] = np.where(home['GF'] > home['GA'], 'W', np.where(home['GF'] < home['GA'], 'L', 'T'))

    away = df_rank[['season_start','season','division','matchday','away_team','away_goals','home_goals']].copy()
    away = away.rename(columns={'away_team':'team','away_goals':'GF','home_goals':'GA'})
    away['result_WLT'] = np.where(away['GF'] > away['GA'], 'W', np.where(away['GF'] < away['GA'], 'L', 'T'))

    matches = pd.concat([home, away], ignore_index=True)
    matches = matches.sort_values(['season_start','season','division','matchday']).reset_index(drop=True)

    # Flags binarios de resultado para acumular por equipo.
    matches['W'] = (matches['result_WLT'] == 'W').astype(int)
    matches['L'] = (matches['result_WLT'] == 'L').astype(int)
    matches['T'] = (matches['result_WLT'] == 'T').astype(int)

    # Acumulados por equipo dentro de season/division.
    agg_cols = ['GF','GA','W','L','T']
    matches[agg_cols] = matches.groupby(['season_start','division','team'], dropna=False)[agg_cols].cumsum()

    # Diferencial de goles y puntos acumulados.
    matches['GD']  = matches['GF'] - matches['GA']
    matches['Pts'] = matches['W']*3 + matches['T']

    # Ranking por jornada: prioridad Pts >> GD >> GF (rank denso descendente).
    matches['rank_key'] = matches['Pts']*1000 + matches['GD']*10 + matches['GF']
    matches['ranking'] = matches.groupby(['season_start','division','matchday'], dropna=False)['rank_key'] \
                                .rank(method='dense', ascending=False).astype('Int64')
    matches = matches.drop(columns='rank_key')

    # Para evitar fuga: usamos métricas "previas" (shift) al partido actual.
    matches = matches.sort_values(['season_start','division','team','matchday'])
    matches['prev_ranking'] = matches.groupby(['season_start','division','team'], dropna=False)['ranking'].shift(1)
    matches['prev_pts']     = matches.groupby(['season_start','division','team'], dropna=False)['Pts'].shift(1)
    matches['prev_gd']      = matches.groupby(['season_start','division','team'], dropna=False)['GD'].shift(1)
    matches['prev_pts'] = matches['prev_pts'].fillna(0)  # inicio de temporada
    matches['prev_gd']  = matches['prev_gd'].fillna(0)

    # Proyección de métricas previas a las filas home/away de df_rank.
    home_rank = matches.rename(columns={
        'team':'home_team','prev_ranking':'home_ranking','prev_pts':'home_pts','prev_gd':'home_gd'
    })[['season','division','matchday','home_team','home_ranking','home_pts','home_gd']].drop_duplicates(
        subset=['season','division','matchday','home_team']
    )

    away_rank = matches.rename(columns={
        'team':'away_team','prev_ranking':'away_ranking','prev_pts':'away_pts','prev_gd':'away_gd'
    })[['season','division','matchday','away_team','away_ranking','away_pts','away_gd']].drop_duplicates(
        subset=['season','division','matchday','away_team']
    )

    df_rank = df_rank.merge(home_rank, on=['season','division','matchday','home_team'], how='left')
    df_rank = df_rank.merge(away_rank, on=['season','division','matchday','away_team'], how='left')

    # ==============================================================
    # 7) Forma reciente por goles (rolling 5, sin incluir el actual)
    # ==============================================================
    # - Calcula el goal difference por partido y luego la media móvil de los
    #   últimos 5 partidos por equipo antes del encuentro actual.
    df_rank['goal_diff_home'] = df_rank['home_goals'] - df_rank['away_goals']
    df_rank['goal_diff_away'] = df_rank['away_goals'] - df_rank['home_goals']

    home_stats = df_rank[['season_start','season','division','matchday','home_team','goal_diff_home']].copy()
    home_stats = home_stats.rename(columns={'home_team':'team','goal_diff_home':'goal_diff'})
    away_stats = df_rank[['season_start','season','division','matchday','away_team','goal_diff_away']].copy()
    away_stats = away_stats.rename(columns={'away_team':'team','goal_diff_away':'goal_diff'})

    team_stats = pd.concat([home_stats, away_stats], ignore_index=True)
    team_stats = team_stats.sort_values(['season_start','division','team','matchday'])

    team_stats['last5_goal_diff'] = (
        team_stats.groupby(['season_start','division','team'], dropna=False)['goal_diff']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    home_last5 = team_stats[['season','division','matchday','team','last5_goal_diff']].copy()
    home_last5 = home_last5.rename(columns={'team':'home_team','last5_goal_diff':'HGD_last5'})
    away_last5 = team_stats[['season','division','matchday','team','last5_goal_diff']].copy()
    away_last5 = away_last5.rename(columns={'team':'away_team','last5_goal_diff':'AGD_last5'})

    home_last5 = home_last5.drop_duplicates(subset=['season','division','matchday','home_team'])
    away_last5 = away_last5.drop_duplicates(subset=['season','division','matchday','away_team'])

    df_rank = df_rank.merge(home_last5, on=['season','division','matchday','home_team'], how='left')
    df_rank = df_rank.merge(away_last5, on=['season','division','matchday','away_team'], how='left')
    df_rank = df_rank.drop(columns=['goal_diff_home','goal_diff_away'])

    # =========================================================================
    # 8) Forma reciente por puntos (rolling 5, sin incluir el actual)
    # =========================================================================
    # - Traducimos 'result' a puntos por local/visita y promediamos últimos 5.
    HOME_POINTS_MAP = {0:3, 1:0, 2:1, '0':3, '1':0, '2':1}
    AWAY_POINTS_MAP = {0:0, 1:3, 2:1, '0':0, '1':3, '2':1}

    df_rank['points_home'] = df_rank['result'].map(HOME_POINTS_MAP)
    df_rank['points_away'] = df_rank['result'].map(AWAY_POINTS_MAP)

    home_stats = df_rank[['season_start','season','division','matchday','home_team','points_home']].copy()
    home_stats = home_stats.rename(columns={'home_team':'team','points_home':'points'})
    away_stats = df_rank[['season_start','season','division','matchday','away_team','points_away']].copy()
    away_stats = away_stats.rename(columns={'away_team':'team','points_away':'points'})

    team_stats = pd.concat([home_stats, away_stats], ignore_index=True)
    team_stats = team_stats.sort_values(['season_start','division','team','matchday'])

    team_stats['last5_points_avg'] = (
        team_stats.groupby(['season_start','division','team'], dropna=False)['points']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    home_last5 = team_stats[['season','division','matchday','team','last5_points_avg']].copy()
    home_last5 = home_last5.rename(columns={'team':'home_team','last5_points_avg':'Hpts_last5'})
    away_last5 = team_stats[['season','division','matchday','team','last5_points_avg']].copy()
    away_last5 = away_last5.rename(columns={'team':'away_team','last5_points_avg':'Apts_last5'})

    home_last5 = home_last5.drop_duplicates(subset=['season','division','matchday','home_team'])
    away_last5 = away_last5.drop_duplicates(subset=['season','division','matchday','away_team'])

    df_rank = df_rank.merge(home_last5, on=['season','division','matchday','home_team'], how='left')
    df_rank = df_rank.merge(away_last5, on=['season','division','matchday','away_team'], how='left')
    df_rank = df_rank.drop(columns=['points_home','points_away'])

    # ================================================================
    # 9) H2H histórico (ventana de 5 temporadas, sin fuga temporal)
    # ================================================================
    # - Para cada par de equipos (orden indiferente), se calcula la razón de
    #   victorias históricas del local y del visitante, considerando solo los
    #   enfrentamientos de las últimas 5 temporadas respecto de la actual.
    df_rank = df_rank.sort_values(["season_start","matchday"]).reset_index(drop=True)

    df_rank["pair"] = df_rank.apply(
        lambda r: "_".join(sorted([str(r["home_team"]), str(r["away_team"])])),
        axis=1
    )

    RESULT_NUM_MAP = {0: 1, 1: -1, 2: 0, '0': 1, '1': -1, '2': 0}
    df_rank["result_num"] = df_rank["result"].map(RESULT_NUM_MAP)

    df_rank["hist_home_win_ratio"] = np.nan
    df_rank["hist_away_win_ratio"] = np.nan

    for pair, group in df_rank.groupby("pair", sort=False):
        group = group.sort_values(["season_start","matchday"]).copy()
        hist_home, hist_away, prev_matches = [], [], []

        for _, row in group.iterrows():
            current_season = int(row["season_start"]) if not pd.isna(row["season_start"]) else None
            # Filtra a los partidos de las últimas 5 temporadas respecto a la actual.
            recent = [m for m in prev_matches if (m.get("season_start") is not None and current_season is not None and int(m["season_start"]) >= current_season - 5)]

            if len(recent) > 0:
                # Cuenta victorias del equipo local del registro actual (sin importar si jugó de local o visita).
                wins_home = sum(
                    (m["home_team"] == row["home_team"] and m["result_num"] == 1) or
                    (m["away_team"] == row["home_team"] and m["result_num"] == -1)
                    for m in recent
                )
                # Cuenta victorias del equipo visitante del registro actual.
                wins_away = sum(
                    (m["home_team"] == row["away_team"] and m["result_num"] == 1) or
                    (m["away_team"] == row["away_team"] and m["result_num"] == -1)
                    for m in recent
                )
                total = len(recent)
                hist_home.append(wins_home / total if total > 0 else np.nan)
                hist_away.append(wins_away / total if total > 0 else np.nan)
            else:
                hist_home.append(np.nan)
                hist_away.append(np.nan)

            # Agrega el partido actual al buffer de históricos para el siguiente loop.
            prev_matches.append(row)

        # Vuelca los vectores calculados en las filas correspondientes del grupo.
        df_rank.loc[group.index, "hist_home_win_ratio"] = hist_home
        df_rank.loc[group.index, "hist_away_win_ratio"] = hist_away

    # Limpieza de columnas auxiliares.
    df_rank.drop(columns=["pair","result_num"], inplace=True)

    # ==========================================
    # 10) Imputaciones (valores neutrales)
    # ==========================================
    # - H2H sin historial suficiente → 0.5 (neutral).
    # - Forma reciente al inicio → 0.0 (neutral).
    # - Ranking faltante → n_teams (peor ranking posible).
    df_rank['hist_home_win_ratio'] = df_rank['hist_home_win_ratio'].fillna(0.5)
    df_rank['hist_away_win_ratio'] = df_rank['hist_away_win_ratio'].fillna(0.5)

    for col in ['HGD_last5', 'AGD_last5', 'Hpts_last5', 'Apts_last5']:
        if col in df_rank.columns:
            df_rank[col] = df_rank[col].fillna(0.0)

    teams_per_sd = (
        matches.groupby(['season','division'])['team']
        .nunique()
        .rename('n_teams')
        .reset_index()
    )
    df_rank = df_rank.merge(teams_per_sd, on=['season','division'], how='left')
    df_rank['home_ranking'] = df_rank['home_ranking'].fillna(df_rank['n_teams'])
    df_rank['away_ranking'] = df_rank['away_ranking'].fillna(df_rank['n_teams'])
    df_rank = df_rank.drop(columns=['n_teams'])

    # ==============================================
    # 11) Orden final y exportación opcional a CSV
    # ==============================================
    df_rank = df_rank.sort_values(['season_start','division','matchday']).reset_index(drop=True)

    if export_csv:
        export_path = LOGS_PATH / "df_rank.csv" if csv_path is None else csv_path
        df_rank.to_csv(export_path, index=False)
        # print(f"Archivo exportado: {export_path}")

    return df_rank

# -----------------------------------------
# Function 3: Parse Traininig Seasons  
# -----------------------------------------

def parse_training_seasons(training_seasons: str) -> list[str]:
    """
    Convierte una cadena que define el conjunto de seasons de entrenamiento
    en una lista estándar de identificadores de season.

    - Si el parámetro es de la forma '2010:2020', se genera:
      ['2010-2011', '2011-2012', ..., '2020-2021'].

    - Si el parámetro es un identificador de season concreto,
      por ejemplo '2019-2020', se devuelve ['2019-2020'].

    Parámetros
    ----------
    training_seasons : str
        Cadena que define el rango o la season específica de entrenamiento.

    Retorna
    -------
    list[str]
        Lista de identificadores de season en formato 'YYYY-YYYY'.
    """
    training_seasons = training_seasons.strip()
    if ":" in training_seasons:
        start, end = training_seasons.split(":")
        start = int(start)
        end = int(end)
        return [f"{year}-{year+1}" for year in range(start, end + 1)]
    else:
        return [training_seasons]

# -----------------------------------------
# Function 4: Map Coding   
# -----------------------------------------

def map_code_to_quiniela(code: int) -> str:
    """
    Convierte la codificación interna de clases en el formato tradicional de quiniela.

    Codificación utilizada:
        0 → '1'  (victoria del equipo local)
        1 → '2'  (victoria del equipo visitante)
        2 → 'X'  (empate)

    Parámetros
    ----------
    code : int
        Código de clase predicho por el modelo.

    Retorna
    -------
    str
        Símbolo de quiniela asociado ('1', '2' o 'X').
    """
    if code == 0:
        return "1"
    elif code == 1:
        return "2"
    else:
        return "X"











# def load_matchday(season, division, matchday):
#     with sqlite3.connect(settings.DATABASE_PATH) as conn:
#         data = pd.read_sql(
#             f"""
#             SELECT * FROM Matches
#                 WHERE season = '{season}'
#                   AND division = {division}
#                   AND matchday = {matchday}
#         """,
#             conn,
#         )
#     if data.empty:
#         raise ValueError("There is no matchday data for the values given")
#     return data


# def load_historical_data(seasons):
#     with sqlite3.connect(settings.DATABASE_PATH) as conn:
#         if seasons == "all":
#             data = pd.read_sql("SELECT * FROM Matches", conn)
#         else:
#             data = pd.read_sql(
#                 f"""
#                 SELECT * FROM Matches
#                     WHERE season IN {tuple(seasons)}
#             """,
#                 conn,
#             )
#     if data.empty:
#         raise ValueError(f"No data for seasons {seasons}")
#     return data


# def save_predictions(predictions):
#     with sqlite3.connect(settings.DATABASE_PATH) as conn:
#         predictions.to_sql(
#             name="Predictions", con=conn, if_exists="append", index=False
#         )
