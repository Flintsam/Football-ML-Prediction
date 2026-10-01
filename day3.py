import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# 1. Load datasets

files = {
    "2024-25": "data/players_data-2024_2025.csv",
    "2025-26": "data/players_data-2025_2026.csv"
}

dfs = {
    season: pd.read_csv(file)
    for season, file in files.items()
}

# 2. Keep common columns

common_columns = (
    set(dfs["2024-25"].columns)
    & set(dfs["2025-26"].columns)
)

common_dfs = {
    season: dfs[season][list(common_columns)].copy()
    for season in dfs
}

# 3. Standardize positions

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

# 4. Fix known ages

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

# 5. Fix known nations

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

# 6. Standardize nations

for season in common_dfs:
    common_dfs[season]["Nation"] = (
        common_dfs[season]["Nation"]
        .str.split()
        .str[-1]
        .str.upper()
    )

# 7. Remove goalkeepers

for season in common_dfs:
    common_dfs[season] = common_dfs[season][
        common_dfs[season]["Pos"] != "GK"
    ].copy()

# 8. Build one row per player

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
    )[[
        "Player",
        "Nation",
        "Age",
        "Pos"
    ]]

    player_info = player_info.drop_duplicates(
        ["Player", "Nation"]
    )

    totals = df.groupby(
        ["Player", "Nation"]
    )[[
        "Min",
        "90s",
        "Gls",
        "Ast",
        "G+A"
    ]].sum().reset_index()

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


season_tables = {
    season: build_season_table(common_dfs[season])
    for season in common_dfs
}

# 9. Build features

features = season_tables["2024-25"][[
    "Player",
    "Nation",
    "Age",
    "Pos",
    "Gls_per90",
    "Ast_per90",
    "GA_per90",
    "Min"
]].copy()

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

# 10. Build target

label = season_tables["2025-26"][[
    "Player",
    "Nation",
    "GA_per90",
    "Min"
]].copy()

label = label.rename(
    columns={
        "GA_per90": "GA_per90_25",
        "Min": "Min_25"
    }
)

# 11. Merge features + target

prediction_df = features.merge(
    label,
    on=["Player", "Nation"],
    how="inner"
)

# 12. Apply minutes threshold

prediction_df = prediction_df[
    (prediction_df["Min_24"] >= 450)
    &
    (prediction_df["Min_25"] >= 450)
].copy()

# 13. Train/test split

train_df, test_df = train_test_split(
    prediction_df,
    test_size=0.20,
    random_state=42
)

# 14. Define ML features

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

# 15. Encode positions

train_pos_encoded = pd.DataFrame(
    {
        pos: train_df["Pos_24"]
        .str.split(",")
        .apply(lambda x: pos in x)
        .astype(int)
        for pos in position_columns
    },
    index=train_df.index
)

test_pos_encoded = pd.DataFrame(
    {
        pos: test_df["Pos_24"]
        .str.split(",")
        .apply(lambda x: pos in x)
        .astype(int)
        for pos in position_columns
    },
    index=test_df.index
)

# 16. Build X and y

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

# Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
lr_train_pred = lr_model.predict(X_train)
lr_test_pred = lr_model.predict(X_test)

# MAE
lr_train_mae = mean_absolute_error(y_train, lr_train_pred)
lr_test_mae = mean_absolute_error(y_test, lr_test_pred)

#=================================================================
#CHUNK --1 LR BASELINE
#=================================================================

train_pred = lr_model.predict(X_train)
train_mae = mean_absolute_error(
    y_train,
    train_pred
)
test_mae = mean_absolute_error(
    y_test,
    lr_test_pred
)

print ("LR Train MAE:",lr_train_mae)
print("LR Test MAE:", lr_test_mae)

#=================================================================
#CHUNK --2 UNRESTRICTED DECISION TREE
#=================================================================

#what is unrestriced decision tree 
# ml model tht predicts a numerical value by making a sequence of decisions
#GA_per90_25 is a regressor bcoz output is a number

#tree is much more complex than linear regression model


#creating the model
tree_model = DecisionTreeRegressor(random_state=42)
#training the model
tree_model.fit(X_train,y_train)

#asking tree pred for both train and test data
tree_train_pred = tree_model.predict(X_train)
tree_test_pred = tree_model.predict(X_test)

#calc MAE for train and test using tree_pred
tree_train_mae = mean_absolute_error(
    y_train,
    tree_train_pred
)

tree_test_mae = mean_absolute_error(
    y_test,
    tree_test_pred
)

print("Tree Train MAE:", tree_train_mae)
print("Tree Test MAE:", tree_test_mae)


#=================================================================
#CHUNK --3 CONTROL THE TREE DEPTH
#=================================================================

#it is bcoz the the tree's pred became very accurate for training_data
#so it had a huge MAE for test data 

#fixing depth = 3 , 5 , 10 , 7
tree_depth3_model = DecisionTreeRegressor(
    max_depth=7,
    random_state=42
)

tree_depth3_model.fit(X_train, y_train)

tree_depth3_train_pred = tree_depth3_model.predict(X_train)
tree_depth3_test_pred = tree_depth3_model.predict(X_test)

tree_depth3_train_mae = mean_absolute_error(
    y_train,
    tree_depth3_train_pred
)

tree_depth3_test_mae = mean_absolute_error(
    y_test,
    tree_depth3_test_pred
)

print("Depth 7 Train MAE:", tree_depth3_train_mae)
print("Depth 7 Test MAE:", tree_depth3_test_mae)

#with depth 3 the prediction became less flexible 
#Depth 3 Train MAE: 0.10962872944698711
#Depth 3 Test MAE: 0.11722586948497851

#with depth 5 we made the tree more complex
#but the MAE came< then depth3
#Depth 5 Train MAE: 0.10036743659775267
#Depth 5 Test MAE: 0.11372132006081798


#with depth 10 tree became more complex test mae inc and train mae dec
#Depth 10 Train MAE: 0.04820542211839741
#Depth 10 Test MAE: 0.14650006986130384

#with depth = 7
#Depth 7 Train MAE: 0.08192446494100111
#Depth 7 Test MAE: 0.1267376952877732