import pandas as pd

from sklearn.ensemble import RandomForestRegressor

#___ UPDATED: refreshed 2026-27 CSV is loaded here when the script is rerun.
# 1. LOAD ALL THREE SEASONS

files = {
    "2024-25": "data/players_data-2024_2025.csv",
    "2025-26": "data/players_data-2025_2026.csv",
    "2026-27": "data/players_data-2026_2027.csv"
}

dfs = {
    season: pd.read_csv(file)
    for season, file in files.items()
}

# 2. KEEP COLUMNS COMMON TO ALL SEASONS

common_columns = (
    set(dfs["2024-25"].columns)
    & set(dfs["2025-26"].columns)
    & set(dfs["2026-27"].columns)
)

common_dfs = {
    season: dfs[season][list(common_columns)].copy()
    for season in dfs
}

# 3. STANDARDIZE PLAYER POSITIONS

def standardize_position(positions):

    positions = positions.split(",")

    position_order = {
        "DF": 0,
        "MF": 1,
        "FW": 2
    }

    positions = sorted(
        positions,
        key=position_order.get
    )

    return ",".join(positions)


for season in common_dfs:

    common_dfs[season]["Pos"] = (
        common_dfs[season]["Pos"]
        .apply(standardize_position)
    )

# 4. FIX KNOWN AGE VALUES

age_updates = {
    "Olabade Aluko": 17,
    "Hannes Behrens": 17,
    "Pape Daouda Diongue": 18,
    "Jake Evans": 17,
    "Fer López": 20,
    "Mateus Mane": 16,
    "Max Moerstedt": 18,
    "Jeremy Monga": 15
}

for player, age in age_updates.items():

    mask = common_dfs["2024-25"]["Player"] == player

    common_dfs["2024-25"].loc[mask, "Age"] = age

# 5. FIX KNOWN NATION VALUES

nation_updates = {
    "Olabade Aluko": "ENG",
    "Jake Evans": "ENG",
    "Atakan Karazor": "GER",
    "Fer López": "ESP",
    "Mateus Mane": "POR",
    "Jeremy Monga": "ENG",
    "Plamedi Nsingi": "FRA"
}

for player, nation in nation_updates.items():

    mask = common_dfs["2024-25"]["Player"] == player

    common_dfs["2024-25"].loc[mask, "Nation"] = nation

# 6. STANDARDIZE NATION VALUES

for season in common_dfs:

    common_dfs[season]["Nation"] = (
        common_dfs[season]["Nation"]
        .str.split()
        .str[-1]
        .str.upper()
    )

# 7. REMOVE GOALKEEPERS

for season in common_dfs:

    common_dfs[season] = (
        common_dfs[season][
            common_dfs[season]["Pos"] != "GK"
        ].copy()
    )

# 8. BUILD ONE ROW PER PLAYER PER SEASON

def build_season_table(df):

    # Find total minutes for each player at each squad
    squad_minutes = (
        df.groupby(
            ["Player", "Nation", "Squad"]
        )["Min"]
        .sum()
        .reset_index()
    )

    # Find the squad where each player played the most minutes
    highest_idx = (
        squad_minutes
        .groupby(["Player", "Nation"])["Min"]
        .idxmax()
    )

    highest_squads = squad_minutes.loc[highest_idx]

    # Keep player information from their canonical squad
    player_info = df.merge(
        highest_squads[
            ["Player", "Nation", "Squad"]
        ],
        on=["Player", "Nation", "Squad"],
        how="inner"
    )[[
        "Player",
        "Nation",
        "Age",
        "Pos"
    ]]

    player_info = player_info.drop_duplicates(
        ["Player", "Nation"]
    )

    # Calculate season totals at player level
    totals = (
        df.groupby(["Player", "Nation"])[
            ["Min", "90s", "Gls", "Ast", "G+A"]
        ]
        .sum()
        .reset_index()
    )

    # Combine player information with season totals
    season_table = totals.merge(
        player_info,
        on=["Player", "Nation"],
        how="left"
    )

    # Calculate per-90 statistics
    season_table["Gls_per90"] = (
        season_table["Gls"] /
        season_table["90s"]
    )

    season_table["Ast_per90"] = (
        season_table["Ast"] /
        season_table["90s"]
    )

    season_table["GA_per90"] = (
        season_table["G+A"] /
        season_table["90s"]
    )

    return season_table


# Create player-level tables for all three seasons
season_tables = {
    season: build_season_table(common_dfs[season])
    for season in common_dfs
}

# 9. CREATE ORIGINAL 2024-25 FEATURES

features_24 = season_tables["2024-25"][[
    "Player",
    "Nation",
    "Age",
    "Pos",
    "Gls_per90",
    "Ast_per90",
    "GA_per90",
    "Min"
]].copy()

features_24 = features_24.rename(columns={
    "Age": "Age_24",
    "Pos": "Pos_24",
    "Gls_per90": "Gls_per90_24",
    "Ast_per90": "Ast_per90_24",
    "GA_per90": "GA_per90_24",
    "Min": "Min_24"
})

# 10. CREATE ORIGINAL 2025-26 TARGET

label_25 = season_tables["2025-26"][[
    "Player",
    "Nation",
    "GA_per90",
    "Min"
]].copy()

label_25 = label_25.rename(columns={
    "GA_per90": "GA_per90_25",
    "Min": "Min_25"
})

# 11. BUILD ORIGINAL 1231-PLAYER DATASET

prediction_df = features_24.merge(
    label_25,
    on=["Player", "Nation"],
    how="inner"
)

# 12. APPLY ORIGINAL 450-MINUTE FILTER

prediction_df = prediction_df[
    (prediction_df["Min_24"] >= 450) &
    (prediction_df["Min_25"] >= 450)
].copy()

# 13. DEFINE THE ML FEATURES

numeric_features = [
    "Gls_per90_24",
    "Ast_per90_24",
    "GA_per90_24",
    "Age_24",
    "Min_24"
]

position_columns = [
    "DF",
    "MF",
    "FW"
]

# 14. ENCODE POSITIONS

all_pos_encoded = pd.DataFrame({
    pos: prediction_df["Pos_24"]
        .str.split(",")
        .apply(lambda x: pos in x)
        .astype(int)
    for pos in position_columns
}, index=prediction_df.index)

# 15. CREATE FULL TRAINING FEATURES AND TARGET

X_all = pd.concat(
    [
        prediction_df[numeric_features],
        all_pos_encoded
    ],
    axis=1
)

y_all = prediction_df["GA_per90_25"]

# 16. FINAL RANDOM FOREST CONFIGURATION

final_rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

#============
#chunk --1 WORST 2 LR CASE
#============

players_to_check = [
    "Héctor Fort",
    "Christoph Baumgartner"
]

worst_cases = prediction_df[
    prediction_df["Player"].isin(players_to_check)
][[
    "Player",
    "Nation",
    "Age_24",
    "Min_24"
]]

#print(worst_cases)

#============
#chunk --2 REAL APPLICATION DATASET
#============

features_25 = season_tables["2025-26"][[
    "Player",
    "Nation",
    "Age",
    "Pos",
    "Gls_per90",
    "Ast_per90",
    "GA_per90",
    "Min"
]].copy()

features_25 = features_25.rename(columns={#features
    "Age": "Age_25",
    "Pos": "Pos_25",
    "Gls_per90": "Gls_per90_25",
    "Ast_per90": "Ast_per90_25",
    "GA_per90": "GA_per90_25",
    "Min": "Min_25"
})

#creating the actual 26/27 data

actual_26 = season_tables["2026-27"][[
    "Player",
    "Nation",
    "GA_per90",
    "Min"
]].copy()

actual_26 = actual_26.rename(columns={#target
    "GA_per90": "GA_per90_26",
    "Min": "Min_26"
})

#merging them 

application_df = features_25.merge(
    actual_26,
    on=["Player", "Nation"],
    how="inner"
)

#450 min threshold

application_df = application_df[
    application_df["Min_25"] >= 450
].copy()

#encoding the position

position_columns_25 = [
    "DF",
    "MF",
    "FW"
]

pos_encoded_25 = pd.DataFrame({
    pos: application_df["Pos_25"]
        .str.split(",")
        .apply(lambda x: pos in x)
        .astype(int)
    for pos in position_columns_25
}, index=application_df.index)

#putting them into exact feature structure the Random Forest expects.

X_application = pd.concat(
    [
        application_df[
            [
                "Gls_per90_25",
                "Ast_per90_25",
                "GA_per90_25",
                "Age_25",
                "Min_25"
            ]
        ],
        pos_encoded_25
    ],
    axis=1
)

#creating a refernce of 26/27 for later comparison

application_results = application_df[
    [
        "Player",
        "Nation",
        "GA_per90_26",
        "Min_26"
    ]
].copy()

#============
#chunk --3 RETRAIN THE FINAL RANDOM FOREST
#============

#fitting rf to x_all/y_all(entire 1231 players)

final_rf.fit(X_all, y_all)

#___ UPDATED: align only the prediction copy with the feature names used during model fitting.
X_application_model = X_application[
    [
        "Gls_per90_25",
        "Ast_per90_25",
        "GA_per90_25",
        "Age_25",
        "Min_25",
        "DF",
        "MF",
        "FW"
    ]
]

X_application_for_model = X_application_model.copy()

X_application_for_model.columns = X_all.columns

#making the predictions

predicted_26 = final_rf.predict(X_application_for_model)

#bringing pred and actual vallues togather

application_results["Predicted_GA_per90_26"] = predicted_26

#comaprision dataset

comparison_df = application_results.copy()

#threshold for 26/27

comparison_df = comparison_df[
    comparison_df["Min_26"] >= 450
].copy()

#calc error = pred-actual

comparison_df["Error"] = (
    comparison_df["Predicted_GA_per90_26"]
    - comparison_df["GA_per90_26"]
)

comparison_df["Absolute_Error"] = (
    comparison_df["Error"].abs()
)

print(
    "Players in 2026-27 comparison:",
    len(comparison_df)
)

#============
#chunk --4 ANALYZE PREDICTIONS
#============

closest_predictions = comparison_df.sort_values(
    "Absolute_Error"
).head(5)

print(
    closest_predictions[[
        "Player",
        "GA_per90_26",
        "Predicted_GA_per90_26",
        "Absolute_Error",
        "Min_26"
    ]].to_string(index=False)
)
#==================================================
# Closest prediction results:
# Oihan Sancet       -> Actual: 0.175439 | Predicted: 0.177492 | Error: 0.002053 | Minutes: 514
# Arsène Kouassi     -> Actual: 0.200000 | Predicted: 0.197873 | Error: 0.002127 | Minutes: 450
# Marcos Llorente    -> Actual: 0.192308 | Predicted: 0.186298 | Error: 0.006009 | Minutes: 464
# Jan Paul van Hecke -> Actual: 0.200000 | Predicted: 0.190886 | Error: 0.009114 | Minutes: 450
# Pedri              -> Actual: 0.363636 | Predicted: 0.352340 | Error: 0.011296 | Minutes: 496
#
# These were the 5 predictions with the smallest absolute errors.
# The model was very close to the observed 2026-27 GA_per90 for all five players.
#==================================================

#biggest errors
biggest_misses = comparison_df.sort_values(
    "Absolute_Error",
    ascending=False
).head(5)

print(
    biggest_misses[[
        "Player",
        "GA_per90_26",
        "Predicted_GA_per90_26",
        "Absolute_Error",
        "Min_26"
    ]].to_string(index=False)
)

#==========================================================
# Biggest miss results:
# Raphinha         -> Actual: 2.459016 | Predicted: 0.599475 | Error: 1.859541 | Minutes: 550
# Sergio Camello   -> Actual: 1.538462 | Predicted: 0.528802 | Error: 1.009660 | Minutes: 466
# Pascal Groß      -> Actual: 1.200000 | Predicted: 0.205876 | Error: 0.994124 | Minutes: 450
# Lamine Yamal     -> Actual: 1.718750 | Predicted: 0.785639 | Error: 0.933111 | Minutes: 579
# Kylian Mbappé    -> Actual: 1.285714 | Predicted: 0.690452 | Error: 0.595263 | Minutes: 630
#
# All five biggest misses were underpredictions.
# These errors are based on the partial 2026-27 season, so they are not
# evidence of final-season model performance.
#==========================================================


#medium size error
medium_errors = comparison_df[
    (comparison_df["Absolute_Error"] >= 0.10) &
    (comparison_df["Absolute_Error"] <= 0.30)
].sort_values("Absolute_Error")

print(
    medium_errors[[
        "Player",
        "GA_per90_26",
        "Predicted_GA_per90_26",
        "Absolute_Error",
        "Min_26"
    ]].head(10).to_string(index=False)
)
#===========================================================
# Interesting comparison results:
# Olasagasti       -> Actual: 0.357143 | Predicted: 0.257085 | Error: 0.100058
# Malick Thiaw     -> Actual: 0.000000 | Predicted: 0.101541 | Error: 0.101541
# Bruno Fernandes  -> Actual: 0.800000 | Predicted: 0.686022 | Error: 0.113978
# Eric García      -> Actual: 0.000000 | Predicted: 0.109521 | Error: 0.109521
# Catena           -> Actual: 0.000000 | Predicted: 0.115607 | Error: 0.115607
#
# These examples show that the model can both underpredict and overpredict.
# It tends to predict a non-zero value for some players whose current
# observed GA_per90 is 0, while it can also underestimate higher observed values.
# The 2026-27 values are based on a partial season.
#===========================================================

#============
#chunk --5 MODEL LIMITATIONs
#============

#players with no row in 25/26
new_26_players = actual_26.merge(
    features_25[["Player", "Nation"]],
    on=["Player", "Nation"],
    how="left",
    indicator=True
)

new_26_players = new_26_players[
    new_26_players["_merge"] == "left_only"
].drop(columns="_merge")

# print(
#     new_26_players[[
#         "Player",
#         "Nation",
#         "Min_26"
#     ]].to_string(index=False)
# )

print(
    "\nPlayers with no 2025-26 row:",
    len(new_26_players)
)

#players exclude in 25/26 due to <450

low_min_25 = features_25[features_25["Min_25"] < 450].copy()

print(
    "Players with 2025-26 row but below 450 minutes:",
    len(low_min_25)
)



#FINAL MODEL SUMMARY

# Final model:
# RandomForestRegressor
# n_estimators = 200
# max_depth = 6
# random_state = 42

# Final test MAE from the original 247-player test set:
# 0.113956

# Linear Regression test MAE:
# 0.116310

# Decision Tree (max_depth=4) test MAE:
# 0.119854