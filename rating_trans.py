import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
rec_data = pd.read_csv("data/recommendations.csv")
hours_intervals = [(0.0, 2.0),(2.0,6.0) ,(6.0, 14.1), (14.1, 39.7),(39.7,float('inf'))]  
ratings = [2,2.5,3,3.5,4]  
def assign_rating(hours, is_recommended):
    for i, (start, end) in enumerate(hours_intervals):
        if start <= hours < end:
            return ratings[i] + 2 * is_recommended - 1
    return ratings[-1] + 2* is_recommended - 1  

rec_data['explicit_rating'] = rec_data.apply(lambda row: assign_rating(row['hours'], row['is_recommended']), axis=1)
renamed_data = rec_data.rename(columns = {'is_recommended':'implicit_rating'})
ratings_data = renamed_data[['user_id', 'app_id','explicit_rating','implicit_rating']].copy()
ratings_data.to_csv("ratings.csv",index = False)