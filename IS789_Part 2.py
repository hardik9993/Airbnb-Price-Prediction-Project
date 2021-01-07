import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files
uploaded = files.upload()

data2 = pd.read_csv("calendar.csv") 
data2.head()

data2.listing_id

data2.describe()

data2['occupied']=np.where(data2['available']=='t', 0, 1)

data2

num_cols = data2.select_dtypes(exclude='object').columns
cat_cols = data2.select_dtypes(include='object').columns
data2[cat_cols] = data2[cat_cols].apply(
    lambda col: col.fillna(col.mode()[0]))
data2[num_cols] = data2[num_cols].apply(
    lambda col: col.fillna(col.median()))

corr_matrix = data2.corr()
corr_matrix = abs(corr_matrix)
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corr_matrix, vmax=0.8, vmin=0.05, annot=True)

#start  of knn model
minimum_nights_input = 3
reviews = data2.loc[1,'minimum_nights']
initial_review = np.abs(reviews - minimum_nights_input)
print(initial_review)

#There are 27 listings with distance 0, meaning that have same number of minimum nights as the input
data2['distance'] = np.abs(data2.occupied - minimum_nights_input)
data2.distance.value_counts().sort_index()

data2=data2.sample(frac=1,random_state=0)
data2 = data2.sort_values('distance')
data2.price.head()

mean_price = data2.price.iloc[:5].mean()
mean_price

#evaluating knn
data2.drop('distance',axis=1)
train_df = data2.copy().iloc[:36671]
test_df = data2.copy().iloc[36671:]

def predict_price(new_listing_value,feature_column):
    temp_df = train_df
    temp_df['distance'] = np.abs(data2[feature_column] - new_listing_value)
    temp_df = temp_df.sort_values('distance')
    knn_5 = temp_df.price.iloc[:5]
    predicted_price = knn_5.mean()
    return(predicted_price)

test_df['predicted_price'] = test_df.number_of_reviews.apply(predict_price,feature_column='number_of_reviews')



