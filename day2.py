import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score

# LOAD DATA

files = {
    "2024-25": "data/players_data-2024_2025.csv",
    "2025-26": "data/players_data-2025_2026.csv"
}

dfs = {
    season: pd.read_csv(file)
    for season, file in files.items()
}

# COMMON COLUMNS

common_columns = (
    set(dfs["2024-25"].columns)
    & set(dfs["2025-26"].columns)
)

common_dfs = {
    season: dfs[season][list(common_columns)].copy()
    for season in dfs
}

# STANDARDIZE POSITIONS
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
    common_dfs[season]["Pos"] = common_dfs[season]["Pos"].apply(
        standardize_position
    )

# AGE FIXES

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

# NATION FIXES

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


# STANDARDIZE NATION
for season in common_dfs:

    common_dfs[season]["Nation"] = (
        common_dfs[season]["Nation"]
        .str.split()
        .str[-1]
        .str.upper()
    )

# REMOVE GOALKEEPERS
for season in common_dfs:

    common_dfs[season] = common_dfs[season][
        common_dfs[season]["Pos"] != "GK"
    ].copy()

# BUILD PLAYER-LEVEL SEASON TABLE
def build_season_table(df):

    squad_minutes = df.groupby(
        ["Player", "Nation", "Squad"]
    )["Min"].sum().reset_index()

    highest_idx = squad_minutes.groupby(
        ["Player", "Nation"]
    )["Min"].idxmax()

    highest_squads = squad_minutes.loc[highest_idx]

    player_info = df.merge(
        highest_squads[
            ["Player", "Nation", "Squad"]
        ],
        on=["Player", "Nation", "Squad"],
        how="inner"
    )[["Player", "Nation", "Age", "Pos"]]

    player_info = player_info.drop_duplicates(
        ["Player", "Nation"]
    )

    totals = df.groupby(
        ["Player", "Nation"]
    )[["Min", "90s", "Gls", "Ast", "G+A"]].sum().reset_index()

    season_table = totals.merge(
        player_info,
        on=["Player", "Nation"],
        how="left"
    )

    season_table["Gls_per90"] = (
        season_table["Gls"] / season_table["90s"]
    )

    season_table["Ast_per90"] = (
        season_table["Ast"] / season_table["90s"]
    )

    season_table["GA_per90"] = (
        season_table["G+A"] / season_table["90s"]
    )

    return season_table

# SEASON TABLES
season_tables = {
    season: build_season_table(common_dfs[season])
    for season in common_dfs
}

# 2024-25 FEATURES
features = season_tables["2024-25"][
    [
        "Player",
        "Nation",
        "Age",
        "Pos",
        "Gls_per90",
        "Ast_per90",
        "GA_per90",
        "Min"
    ]
].copy()

features = features.rename(
    columns={
        "Age": "Age_24",
        "Pos": "Pos_24",
        "Gls_per90": "Gls_per90_24",
        "Ast_per90": "Ast_per90_24",
        "GA_per90": "GA_per90_24",
        "Min": "Min_24"
    }
)

# 2025-26 LABEL
label = season_tables["2025-26"][
    [
        "Player",
        "Nation",
        "GA_per90",
        "Min"
    ]
].copy()

label = label.rename(
    columns={
        "GA_per90": "GA_per90_25",
        "Min": "Min_25"
    }
)

# SUPERVISED DATASET
prediction_df = features.merge(
    label,
    on=["Player", "Nation"],
    how="inner"
)


# MINUTES FILTER

prediction_df = prediction_df[
    (prediction_df["Min_24"] >= 450)
    &
    (prediction_df["Min_25"] >= 450)
].copy()

# TRAIN / TEST SPLIT
train_df, test_df = train_test_split(
    prediction_df,
    test_size=0.20,
    random_state=42
)

#==============================================================================
#chunk --1  categorically arranging posn
#==============================================================================


# Create separate binary features for each position

position_columns = ["DF", "MF", "FW"]

train_pos_encoded = pd.DataFrame(
    {
        pos: train_df["Pos_24"].str.split(",").apply(lambda x: pos in x).astype(int)#astype converts bool into 0 and 1
        for pos in position_columns
    },
    index=train_df.index
)

test_pos_encoded = pd.DataFrame(
    {
        pos: test_df["Pos_24"].str.split(",").apply(lambda x: pos in x).astype(int)#lambda checking if that plaer posn lies there or not
        for pos in position_columns
    },
    index=test_df.index
)

print(position_columns)

print("\nEncoded training positions:")
print(train_pos_encoded.head())

#assemble encoded posn with numerical feature x and y
numeric_features = [
    "Gls_per90_24",
    "Ast_per90_24",
    "GA_per90_24",
    "Age_24",
    "Min_24"
]

X_train = pd.concat(
    [
        train_df[numeric_features],
        train_pos_encoded
    ],
    axis=1
)

X_test = pd.concat(
    [
        test_df[numeric_features],
        test_pos_encoded
    ],
    axis=1
)

y_train = train_df["GA_per90_25"]
y_test = test_df["GA_per90_25"]
print(X_train.shape)
print(X_test.shape)
print(X_train.columns)

#==============================================================================
#chunk --2  Building baseline
#==============================================================================

#calc baseline prediction
baseline_prediction = y_train.mean()

print("Baseline prediction:", baseline_prediction)#setting a reference point

#creating prediction for every test player
baseline_pred = [baseline_prediction] * len(y_test)

print("Number of baseline predictions:", len(baseline_pred))

#calculating baseline MAE
#org-reference = error

baseline_mae = mean_absolute_error(
    y_test,
    baseline_pred
)

#print("Baseline MAE:", baseline_mae)

#==============================================================================
#chunk --3 TRAINING LINEAR REGRESSION
#==============================================================================

#creating model
model = LinearRegression()
#training model based on training data
model.fit(X_train,y_train)
#testing the model based on its prediction
model_pred = model.predict(X_test)
print("Number of model prediction:",len(model_pred))

#==============================================================================
#chunk --4 MODEL EVALUATION
#==============================================================================

#calculting models MAE
#MAE tells us how far pred are from actual values
model_mae = mean_absolute_error(
    y_test,
    model_pred
)
#print("Model MAE :",model_mae)

#calculation R2
#tells us how much variation model explained
model_r2=r2_score(
    y_test,
    model_pred
)
#print("Model R2:", model_r2)

#compare the model with the baseline
mae_improvement = baseline_mae - model_mae

print("Baseline MAE:", baseline_mae)
print("Model MAE:", model_mae)
print("MAE improvement:", mae_improvement)
print("Model R2:", model_r2)

#==============================================================================
#chunk --5 ACTUAL vs PREDICT
#==============================================================================

#comparing 10 random players
comparison = test_df[["Player", "GA_per90_25"]].copy()

comparison["Predicted_GA_per90_25"] = model_pred

comparison = comparison.sample(
    10,
    random_state=42
)

print(comparison)
