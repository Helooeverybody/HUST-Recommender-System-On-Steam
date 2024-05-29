import numpy as np
from sklearn.linear_model import Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import math
def build_data(user_id,games,rec,pca_matrix):
    data=rec[rec["user_id"]==user_id]
    data=games.merge(data,on="app_id")
    Y=data["rating"].to_numpy()
    rated_game_list=data["app_id"].to_list()
    games_rated_index=games[games["app_id"].isin(rated_game_list)].index.tolist()
    X=pca_matrix[games_rated_index]
    return (X,Y)
def rated_game_index_list(user_id,games,rec):
    rated_game_list=rec[rec["user_id"]==user_id]["app_id"].to_list()
    games_rated_index=games[games["app_id"].isin(rated_game_list)].index.to_numpy()
    return games_rated_index
def rec_by_regression(user_id,model_id,games,rec,pca_matrix):
    models=[Ridge(alpha=1),Lasso(alpha=0.01),RandomForestRegressor(n_estimators=100,max_features=200,min_samples_leaf=5,random_state=5)]
    model=models[model_id]
    X,Y=build_data(user_id,games,rec,pca_matrix)
    model.fit(X,Y)
    pred_ratings=model.predict(pca_matrix)
    candidate_list=pred_ratings.argsort()[::-1]
    rated_list=rated_game_index_list(user_id,games,rec)
    candidate_list =np.array([i for i in candidate_list if i not in rated_list])
    top_5=candidate_list[:5]
    return games.loc[top_5]["app_id"].to_list()