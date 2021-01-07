import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from google.colab import files
uploaded = files.upload()

data = pd.read_csv("AB_NYC_2019.csv") 
data.head()

data.name

data.describe()

data['host_experience']=np.where(data['availability_365']<=122, 'part_time',
                                 (np.where(data['availability_365']>244, 'professional', 'full_time')))

data

correlation = data.corr()
correlation.head()

plt.figure(figsize=(10,10)) 
# play with the figsize until the plot is big enough to plot all the columns
# of your dataset, or the way you desire it to look like otherwise

sns.heatmap(data.corr())

X = data.loc[:, 'minimum_nights':'reviews_per_month']
y = data[['price']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

plt.figure(figsize=(12, 12))
sns.heatmap(data.groupby(['room_type', 'neighbourhood_group']).price.mean().unstack(), cmap='coolwarm', annot=True, fmt='.0f')

plt.figure(figsize=(12, 12))
sns.heatmap(data.groupby(['host_experience', 'neighbourhood_group']).price.mean().unstack(), cmap='coolwarm', annot=True, fmt='.0f')

#adjusting the data: most commmon value for empty categorical values , median values for empty numerical
num_cols = data.select_dtypes(exclude='object').columns
cat_cols = data.select_dtypes(include='object').columns
data[cat_cols] = data[cat_cols].apply(
    lambda col: col.fillna(col.mode()[0]))
data[num_cols] = data[num_cols].apply(
    lambda col: col.fillna(col.median()))

corr_matrix = data.corr()
corr_matrix = abs(corr_matrix)
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corr_matrix, vmax=0.8, vmin=0.05, annot=True)

#start  of knn model
input_review_value = 5
reviews = data.loc[0,'number_of_reviews']
initial_review = np.abs(reviews - input_review_value)
print(initial_review)

#checking the distnaces in the observations. There are 1618 listings with distance 0, meaning that have same number of reviews as the input(choosing values with distance of 0 prediction would be biased)
data['distance'] = np.abs(data.number_of_reviews - input_review_value)
data.distance.value_counts().sort_index()

#using a random state to shuffle the data, 
data=data.sample(frac=1,random_state=0)
data = data.sort_values('distance')
data.price.head()

#prediction of a price for a listing with # reviews
mean_price = data.price.iloc[:9].mean()
mean_price

#splitting dataset for training and testing
data.drop('distance',axis=1)
train_df = data.copy().iloc[:36671]
test_df = data.copy().iloc[36671:]

def predict_price(new_listing_value,feature_column):
    temp_df = train_df
    temp_df['distance'] = np.abs(data[feature_column] - new_listing_value)
    temp_df = temp_df.sort_values('distance')
    knn_5 = temp_df.price.iloc[:9]
    predicted_price = knn_5.mean()
    return(predicted_price)

test_df['predicted_price'] = test_df.number_of_reviews.apply(predict_price,feature_column='number_of_reviews')

print(test_df['predicted_price'])

#prediction error of the predictions
test_df['squared_error'] = (test_df['predicted_price'] - test_df['price'])**(2)
mse = test_df['squared_error'].mean()
rmse = mse ** (1/2)
rmse

for feature in ['number_of_reviews','reviews_per_month']:
    test_df['predicted_price'] = test_df.number_of_reviews.apply(predict_price,feature_column=feature)
    test_df['squared_error'] = (test_df['predicted_price'] - test_df['price'])**(2)
    mse = test_df['squared_error'].mean()
    rmse = mse ** (1/2)
    print("RMSE for the {} column: {}".format(feature,rmse))

from pandas import DataFrame
data2 = {'number_of_reviews': [5,10,15,20,25],
         'suggested_price':[115,141,89.67,205.58,103.50],
         'prediction_error':[120.52,121.74,187.75,222.59,272.13]
         }

df = DataFrame(data2, columns=['number_of_reviews','suggested_price','prediction_error'])
print (df)

df.plot(x ='number_of_reviews',y=['suggested_price','prediction_error'], kind='line')
