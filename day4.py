import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

# 1. LOAD DATASETS

files = {
    "2024-25": "data/players_data-2024_2025.csv",
    "2025-26": "data/players_data-2025_2026.csv"
}

dfs = {
    season: pd.read_csv(file)
    for season, file in files.items()
}

# 2. KEEP COMMON COLUMNS

common_columns = (
    set(dfs["2024-25"].columns)
    & set(dfs["2025-26"].columns)
)

common_dfs = {
    season: dfs[season][list(common_columns)].copy()
    for season in dfs
}

# 3. STANDARDIZE POSITIONS

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

# 4. FIX KNOWN AGES

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

# 5. FIX KNOWN NATIONS
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

# 6. STANDARDIZE NATIONS

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

    squad_minutes = (
        df.groupby(
            ["Player", "Nation", "Squad"]
        )["Min"]
        .sum()
        .reset_index()
    )

    highest_idx = (
        squad_minutes
        .groupby(["Player", "Nation"])["Min"]
        .idxmax()
    )

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

    totals = (
        df.groupby(["Player", "Nation"])[
            ["Min", "90s", "Gls", "Ast", "G+A"]
        ]
        .sum()
        .reset_index()
    )

    season_table = totals.merge(
        player_info,
        on=["Player", "Nation"],
        how="left"
    )

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


season_tables = {
    season: build_season_table(common_dfs[season])
    for season in common_dfs
}

# 9. BUILD FEATURES

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

features = features.rename(columns={
    "Age": "Age_24",
    "Pos": "Pos_24",
    "Gls_per90": "Gls_per90_24",
    "Ast_per90": "Ast_per90_24",
    "GA_per90": "GA_per90_24",
    "Min": "Min_24"
})

# 10. BUILD TARGET

label = season_tables["2025-26"][[
    "Player",
    "Nation",
    "GA_per90",
    "Min"
]].copy()

label = label.rename(columns={
    "GA_per90": "GA_per90_25",
    "Min": "Min_25"
})

# 11. MERGE FEATURES + TARGET

prediction_df = features.merge(
    label,
    on=["Player", "Nation"],
    how="inner"
)

# 12. MINUTES THRESHOLD

prediction_df = prediction_df[
    (prediction_df["Min_24"] >= 450) &
    (prediction_df["Min_25"] >= 450)
].copy()

# 13. TRAIN / TEST SPLIT

train_df, test_df = train_test_split(
    prediction_df,
    test_size=0.20,
    random_state=42
)

# 14. ML FEATURES

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

# 15. ENCODE POSITIONS
train_pos_encoded = pd.DataFrame({
    pos: train_df["Pos_24"]
        .str.split(",")
        .apply(lambda x: pos in x)
        .astype(int)
    for pos in position_columns
}, index=train_df.index)

test_pos_encoded = pd.DataFrame({
    pos: test_df["Pos_24"]
        .str.split(",")
        .apply(lambda x: pos in x)
        .astype(int)
    for pos in position_columns
}, index=test_df.index)

# 16. X / Y

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

# 17. DAY 2 LINEAR REGRESSION
lr_model = LinearRegression()

lr_model.fit(
    X_train,
    y_train
)

# 18. LINEAR REGRESSION PREDICTIONS

lr_train_pred = lr_model.predict(X_train)
lr_test_pred = lr_model.predict(X_test)

# 19. LINEAR REGRESSION MAE

lr_train_mae = mean_absolute_error(
    y_train,
    lr_train_pred
)

lr_test_mae = mean_absolute_error(
    y_test,
    lr_test_pred
)

# 20. DAY 3 UNRESTRICTED TREE

tree_model = DecisionTreeRegressor(
    random_state=42
)

tree_model.fit(
    X_train,
    y_train
)

tree_train_pred = tree_model.predict(X_train)
tree_test_pred = tree_model.predict(X_test)

tree_train_mae = mean_absolute_error(
    y_train,
    tree_train_pred
)

tree_test_mae = mean_absolute_error(
    y_test,
    tree_test_pred
)

#======
#CHUNK --1 VALIDATION TEST/HONEST TREE SELECTION
#======

#spliting the training data ninto 70-30 fir training and validation
X_train_small, X_val, y_train_small, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.30,
    random_state=42
)
# print(X_train_small.shape)
# print(X_val.shape)
# print(X_test.shape)

# finding the best depth 
# fitting train_small
#validating train_val
depths = [3,4, 5, 7, 10, None]

validation_results = {}

for depth in depths:

    model = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )#trying different depth of tree

    model.fit(
        X_train_small,
        y_train_small
    )#fitting model on train_small

    val_pred = model.predict(X_val)#validating the model on train_val

    val_mae = mean_absolute_error(
        y_val,
        val_pred
    )#calc MAE for diff depth on train_small-train_val

    validation_results[depth] = val_mae
    #print(f"MAE for {depth} : {val_mae}")

#print(validation_results)

#======================================================
#doing some crazy shitt very much as we saw 3 is best for 70-30 and 4 is for 80-20

#======
#CHUNK --2 k-fold cross validation
#======

#so dividing train in 5 equal parts 
#first we train on 4 parts and val on the last part  
#and calc mae and in the ennd avg out all 5 mae for diff depth 
#see which depth is best for the whole training data


depths = [3, 4, 5, 7, None]

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)#creating 5 folds for cross validation

cv_results = {}#stroing result

for depth in depths:

    fold_maes = []#storing MAEs for each depth

    for train_index, val_index in kf.split(X_train):

        X_cv_train = X_train.iloc[train_index]#creates training data of 80%
        X_cv_val = X_train.iloc[val_index]

        y_cv_train = y_train.iloc[train_index]#created val data for 20%
        y_cv_val = y_train.iloc[val_index]

        model = DecisionTreeRegressor(#the tree
            max_depth=depth,
            random_state=42
        )

        model.fit(#training on 80%
            X_cv_train,
            y_cv_train
        )

        cv_pred = model.predict(X_cv_val)#predicting on 805

        fold_mae = mean_absolute_error(#calc mae
            y_cv_val,
            cv_pred
        )

        fold_maes.append(fold_mae)

    cv_results[depth] = fold_maes

# cv_table = pd.DataFrame({
#     "Max Depth": list(cv_results.keys()),
#     "Fold 1 MAE": [maes[0] for maes in cv_results.values()],
#     "Fold 2 MAE": [maes[1] for maes in cv_results.values()],
#     "Fold 3 MAE": [maes[2] for maes in cv_results.values()],
#     "Fold 4 MAE": [maes[3] for maes in cv_results.values()],
#     "Fold 5 MAE": [maes[4] for maes in cv_results.values()],
#     "Average MAE": [
#         sum(maes) / len(maes)
#         for maes in cv_results.values()
#     ]
# })

# print(cv_table.to_string(index=False))

#=========================================================================================================================
#  Max Depth  Fold 1 MAE  Fold 2 MAE  Fold 3 MAE  Fold 4 MAE  Fold 5 MAE  Average MAE
#        3.0    0.127088    0.125372    0.117926    0.115551    0.109525     0.119092
#        4.0    0.124950    0.117580    0.114616    0.120327    0.108768     0.117248
#        5.0    0.128599    0.124386    0.119092    0.122401    0.108066     0.120509
#        7.0    0.144045    0.134337    0.132174    0.131589    0.120539     0.132537
#        NaN    0.179047    0.161445    0.151738    0.158307    0.153278     0.160763
#
# Best max_depth = 4
# Lowest Average MAE = 0.117248
#=========================================================================================================================

#======
#CHUNK --3 FINAL TREE
#======
final_tree = DecisionTreeRegressor(
    max_depth=4,
    random_state=42
)

final_tree.fit(
    X_train,
    y_train
)
#prediction and cal mae
final_tree_train_pred = final_tree.predict(X_train)
final_tree_test_pred = final_tree.predict(X_test)

final_tree_train_mae = mean_absolute_error(
    y_train,
    final_tree_train_pred
)

final_tree_test_mae = mean_absolute_error(
    y_test,
    final_tree_test_pred
)

print("Final Tree Train MAE:", final_tree_train_mae)
print("Final Tree Test MAE:", final_tree_test_mae)

#FINAL TREE TRAIN-TEST GAP

final_tree_gap = (
    final_tree_test_mae -
    final_tree_train_mae
)

print("Final Tree Train-Test MAE Gap:", final_tree_gap)

#======
#CHUNK --4 MODEL COMPARISON
#======

prediction_results = test_df[
    ["Player", "Nation", "Age_24", "Pos_24",
     "Gls_per90_24", "Ast_per90_24",
     "GA_per90_24", "Min_24"]
].copy()

prediction_results["Actual"] = y_test
prediction_results["Predicted"] = lr_test_pred

prediction_results["Error"] = (
    prediction_results["Actual"] -
    prediction_results["Predicted"]
)

prediction_results["Absolute_Error"] = (
    prediction_results["Error"].abs()
)

#worst 10 players based on LR
worst_predictions = prediction_results.sort_values(
    "Absolute_Error",
    ascending=False
)

# print(
#     worst_predictions[
#         ["Player", "Nation", "Age_24", "Pos_24",
#          "GA_per90_24", "Min_24",
#          "Actual", "Predicted",
#          "Error", "Absolute_Error"]
#     ].head(10).to_string(index=False)
# )

#LR_coeff
coefficient_table = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": lr_model.coef_
})

# print(
#     coefficient_table
#     .sort_values("Coefficient", ascending=False)
#     .to_string(index=False)
# )

#DECISION TREE FEATURE IMPORTANCES

tree_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": final_tree.feature_importances_
})

# print(
#     tree_importance
#     .sort_values("Importance", ascending=False)
#     .to_string(index=False)
# )


#======
#CHUNK --6 RANDOM FOREST
#======

#what is random forest 
#A Random forest builds many Decision tree and combines thier prediction

#creating the base line for random forest

n_estimators_list = [50, 100, 200, 300]#testing number of trees

rf_validation_results = {}

for n in n_estimators_list:

    rf_model = RandomForestRegressor(
        n_estimators=n,
        random_state=42
    )

    rf_model.fit(#training based on train_small
        X_train_small,
        y_train_small
    )

    rf_val_pred = rf_model.predict(X_val)#predicting train_val

    rf_val_mae = mean_absolute_error(#calc mae
        y_val,
        rf_val_pred
    )

    rf_validation_results[n] = rf_val_mae

#for n, mae in rf_validation_results.items():
#    print(f"n_estimators = {n}: Validation MAE = {mae:.6f}")

#==================================================
# RANDOM FOREST - n_estimators VALIDATION RESULTS
#
# n_estimators = 50:  Validation MAE = 0.119998
# n_estimators = 100: Validation MAE = 0.119893
# n_estimators = 200: Validation MAE = 0.119456
# n_estimators = 300: Validation MAE = 0.119892
#
# Best n_estimators on validation set = 200
#===================================================

#no checking diff depth for n=200
rf_depths = [3, 4, 5, 6, 7, 8, 9, None]#checking diff depth

rf_depth_results = {}

for depth in rf_depths:

    rf_model = RandomForestRegressor(#creating rf
        n_estimators=200,
        max_depth=depth,
        random_state=42
    )

    rf_model.fit(#training rf
        X_train_small,
        y_train_small
    )

    rf_val_pred = rf_model.predict(X_val)#rf prediciton

    rf_val_mae = mean_absolute_error(#mae for rf pred
        y_val,
        rf_val_pred
    )

    rf_depth_results[depth] = rf_val_mae

# for depth, mae in rf_depth_results.items():
#     print(f"max_depth = {depth}: Validation MAE = {mae:.6f}")
#=============================================================
# RANDOM FOREST - max_depth VALIDATION RESULTS
#
# max_depth = 3:    Validation MAE = 0.115780
# max_depth = 4:    Validation MAE = 0.116045
# max_depth = 5:    Validation MAE = 0.115770
# max_depth = 6:    Validation MAE = 0.115478
# max_depth = 7:    Validation MAE = 0.115741
# max_depth = 8:    Validation MAE = 0.116183
# max_depth = 9:    Validation MAE = 0.116607
# max_depth = None: Validation MAE = 0.119456
#
# Best max_depth on validation set = 6
# Best n_estimators = 200
#=============================================================


#FINAL RANDOM FOREST MODEL for training o

final_rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

final_rf.fit(
    X_train_small,
    y_train_small
)

rf_small_train_pred = final_rf.predict(X_train_small)
rf_val_pred = final_rf.predict(X_val)

rf_small_train_mae = mean_absolute_error(
    y_train_small,
    rf_small_train_pred
)

rf_val_mae = mean_absolute_error(
    y_val,
    rf_val_pred
)

print("Random Forest Train-Small MAE:", rf_small_train_mae)
print("Random Forest Validation MAE:", rf_val_mae)
#=================================
# Random Forest Train-Small MAE = 0.085508
# Random Forest Validation MAE = 0.115478
#========================-========

#final rf based predicting test data
final_rf.fit(
    X_train,
    y_train
)

rf_final_test_pred = final_rf.predict(X_test)

rf_final_test_mae = mean_absolute_error(
    y_test,
    rf_final_test_pred
)

print("Final Random Forest Test MAE:", rf_final_test_mae)

#======
#CHUNK --7 FINAL MODEL COMPARISON
#======

baseline_mae = 0.165279
lr_test_mae = 0.116310
tree_test_mae = 0.119854
rf_test_mae = 0.113956

models = {
    "Baseline": baseline_mae,
    "Linear Regression": lr_test_mae,
    "Decision Tree": tree_test_mae,
    "Random Forest": rf_test_mae
}

for model, mae in models.items():

    improvement = baseline_mae - mae

    print(
        f"{model}: "
        f"Test MAE = {mae:.6f}, "
        f"Improvement over baseline = {improvement:.6f}"
    )

#cal reduction inmae in %
for model, mae in models.items():

    percentage_improvement = (
        (baseline_mae - mae) / baseline_mae
    ) * 100

    print(
        f"{model}: "
        f"{percentage_improvement:.2f}% reduction in MAE"
    )
