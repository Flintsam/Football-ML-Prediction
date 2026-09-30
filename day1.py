import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
files = {
    "2024-25": "data/players_data-2024_2025.csv",
    "2025-26": "data/players_data-2025_2026.csv"
}

#==============================================================================
#chunk --1  
#==============================================================================

#creating dfs
dfs = {}

for season, file in files.items():
    dfs[season] = pd.read_csv(file)

print(dfs.keys())

#creating dfs with common col

common_dfs = {}
common_columns = set(dfs["2024-25"].columns) & set(dfs["2025-26"].columns)
for season in dfs:
    common_dfs[season]= dfs[season][list(common_columns)]
print(common_dfs["2024-25"].shape)
print(common_dfs["2025-26"].shape)

#manually adding missing nation and age 
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

nation_updates = {
    "Olabade Aluko": "ENG",
    "Jake Evans": "ENG",
    "Atakan Karazor": "GER",
    "Fer López": "ESP",
    "Mateus Mane": "POR",
    "Jeremy Monga": "ENG",
    "Plamedi Nsingi": "FRA"
}

for player, age in age_updates.items():
    mask = common_dfs["2024-25"]["Player"] == player
    common_dfs["2024-25"].loc[mask, "Age"] = age

for player, nation in nation_updates.items():
    mask = common_dfs["2024-25"]["Player"] == player
    common_dfs["2024-25"].loc[mask, "Nation"] = nation

#standarizing posn 
def standardize_position(positions):
    positions = positions.split(",")

    position_order = {
    "DF": 0,
    "MF": 1,
    "FW": 2
}
    positions = sorted(positions, key=position_order.get)
    return ",".join(positions)

for season in common_dfs:
    common_dfs[season]["Pos"] = common_dfs[season]["Pos"].apply(
        standardize_position
    )

#removing gk
for season in common_dfs:
    #print(season, "before:", len(common_dfs[season]))
    common_dfs[season] = common_dfs[season][
        common_dfs[season]["Pos"] != "GK"
    ]
for season in common_dfs:
    pass
    #rint("\n", season)
    #print(common_dfs[season]["Pos"].value_counts())

# Keep only the 3-letter nation code
for season in common_dfs:
    common_dfs[season]["Nation"] = (
        common_dfs[season]["Nation"]
        .str.split()
        .str[-1]
        .str.upper()
    )


#==============================================================================
#chunk --2
#==============================================================================

df = common_dfs["2024-25"]
def build_season_table(df):

    #Find squad with most min for each player
    squad_minutes = df.groupby(
        ["Player", "Nation","Squad"]
    )["Min"].sum().reset_index()

    #print(squad_minutes.head())

    highest_idx = squad_minutes.groupby(
        ["Player","Nation"]
    )["Min"].idxmax()

    #print(highest_idx.head())

    highest_squads = squad_minutes.loc[highest_idx]

    #print(highest_squads.head(10))
    #print(squad_minutes.columns)

    #Keep Age and Pos from the highest-minute squad
    player_info = df.merge(
        highest_squads[["Player", "Nation", "Squad"]],
        on=["Player", "Nation", "Squad"],
        how="inner"
    )[["Player", "Nation", "Age", "Pos"]]

    # One row per Player + Nation
    player_info = player_info.drop_duplicates(
        ["Player", "Nation"]
    )

    # Sum statistics across ALL squads
    totals = df.groupby(
        ["Player", "Nation"]
    )[["Min", "90s", "Gls", "Ast", "G+A"]].sum().reset_index()

    # Combine totals with Age and Position
    season_table = totals.merge(
        player_info,
        on=["Player", "Nation"],
        how="left"
    )

    # Calculate per-90 metrics from the summed totals
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

season_tables = {}

#checking total rows and outfield player rows
for season in common_dfs:
    season_tables[season] = build_season_table(common_dfs[season])

    print(
        season,
        "raw rows:", len(common_dfs[season]),
        "player rows:", len(season_tables[season])
    )

#checking for duplicates
for season, df in season_tables.items():
    print(
        season,
        "duplicate Player+Nation:",
        df.duplicated(["Player", "Nation"]).sum()
    )

#==============================================================================
#chunk --3 Building the prediction dataset
#==============================================================================

#things we wanna predict for next season
features = season_tables["2024-25"][
    ["Player",
    "Nation",
    "Age",
    "Pos",
    "Gls_per90",
    "Ast_per90",
    "GA_per90",
    "Min"]
].copy()
# print(features.head())
# print(features.columns)

#labels
label = season_tables["2025-26"][
    ["Player",
    "Nation",
    "GA_per90",
    "Min"]
].copy()

# print(label.head())
# print(label.columns)

#Renaming based on year
features = features.rename(columns={
    "Age": "Age_24",
    "Pos": "Pos_24",
    "Gls_per90": "Gls_per90_24",
    "Ast_per90": "Ast_per90_24",
    "GA_per90": "GA_per90_24",
    "Min": "Min_24"
})
label = label.rename(columns={
    "GA_per90": "GA_per90_25",
    "Min": "Min_25"
})
# print(features.columns)
# print(label.columns)

#merging based on player + nation
prediction_df = features.merge(
    label,
    on=["Player", "Nation"],
    how="inner"
)
# print("2024-25 players:", len(features))
# print("2025-26 players:", len(label))
# print("Players in both:", len(prediction_df))

# print(prediction_df.head())

#verifying duplicates
print(
    "Duplicate keys:",
    prediction_df.duplicated(
        ["Player", "Nation"]
    ).sum()
)

#applying 450min filter
#print(prediction_df["Min_24"].describe())
#print(prediction_df["Min_25"].describe())

prediction_df = prediction_df[
    (prediction_df["Min_24"] >= 450) &
    (prediction_df["Min_25"] >= 450)
].copy()
#print("Players after 450-minute filter:", len(prediction_df))
#print("Minimum 2024-25 minutes:", prediction_df["Min_24"].min())
#print("Minimum 2025-26 minutes:", prediction_df["Min_25"].min())


#==============================================================================
# Chunk 4 -- Train/Test Split
#==============================================================================

#dividing data 
train_df, test_df = train_test_split(
    prediction_df,
    test_size=0.20,
    random_state=42
)
print("Total:", len(prediction_df))
print("Train:", len(train_df))
print("Test:", len(test_df))

train_keys = set(
    zip(train_df["Player"], train_df["Nation"])
)

test_keys = set(
    zip(test_df["Player"], test_df["Nation"])
)
print("Overlap:", len(train_keys & test_keys))


#visualization
# plt.figure(figsize=(8, 6))

# plt.scatter(
#     train_df["GA_per90_24"],
#     train_df["GA_per90_25"]
# )
# plt.xlabel("2024-25 G+A per 90")
# plt.ylabel("2025-26 G+A per 90")
# plt.title("Training Set: G+A per 90 Across Seasons")
# plt.show()