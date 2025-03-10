import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams as rcP



df = pd.read_csv('pune.csv')
df.head()

df.shape

df.groupby('area_type')['area_type'].agg('count')
df.groupby('availability')['availability'].agg('count')
df.groupby('size')['size'].agg('count')
df.groupby('site_location')['site_location'].agg('count')
df = df.drop('society', axis='columns')
df.head()
df.isnull().sum()

from math import floor

balcony_median = float(floor(df.balcony.median()))
bath_median = float(floor(df.bath.median()))

df.balcony = df.balcony.fillna(balcony_median)
df.bath = df.bath.fillna(bath_median)


df.isnull().sum()


df = df.dropna()
df.isnull().sum()


df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))
df = df.drop('size', axis='columns')
df.groupby('bhk')['bhk'].agg('count')


df.total_sqft.unique()


def isFloat(x):
    try:
        float(x)
    except:
        return False
    return True


df[~df['total_sqft'].apply(isFloat)]


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
df['new_total_sqft'] = df.total_sqft.apply(convert_sqft_to_num)
df = df.drop('total_sqft', axis='columns')
df.head()


df.isna().sum()


df = df.dropna()
df.isna().sum()


df1 = df.copy()

df1['price_per_sqft'] = (df1['price']*100000)/df1['new_total_sqft']
df1.head()

locations = list(df['site_location'].unique())
print(len(locations))


df1.site_location = df1.site_location.apply(lambda x: x.strip())


location_stats = df1.groupby('site_location')['site_location'].agg('count').sort_values(ascending=False)
location_stats


print(len(location_stats[location_stats<=10]), len(df1.site_location.unique()))
df1.head()


locations_less_than_10 = location_stats[location_stats<=10]

df1.site_location = df1.site_location.apply(lambda x: 'other' if x in locations_less_than_10 else x)
len(df1.site_location.unique())


df1.groupby('availability')['availability'].agg('count').sort_values(ascending=False)

dates = df1.groupby('availability')['availability'].agg('count').sort_values(ascending=False)

dates_not_ready = dates[dates<10000]
df1.availability = df1.availability.apply(lambda x: 'Not Ready' if x in dates_not_ready else x)

len(df1.availability.unique())
df1.head()

df1.groupby('area_type')['area_type'].agg('count').sort_values(ascending=False)


df2 = df1[~(df1.new_total_sqft/df1.bhk<300)]
print(len(df2), len(df1))
df2.price_per_sqft.describe()

def remove_pps_outliers(df):

    df_out = pd.DataFrame()

    for key, sub_df in df.groupby('site_location'):
        m = np.mean(sub_df.price_per_sqft)
        sd = np.std(sub_df.price_per_sqft)
        reduce_df = sub_df[(sub_df.price_per_sqft>(m-sd)) & (sub_df.price_per_sqft<(m+sd))]
        df_out = pd.concat([df_out, reduce_df], ignore_index=True)

    return df_out

df3 = remove_pps_outliers(df2)
print(len(df2), len(df3))

def plot_scatter_chart(df, site_location):
    bhk2 = df[(df.site_location == site_location) & (df.bhk == 2)]
    bhk3 = df[(df.site_location == site_location) & (df.bhk == 3)]
    rcP['figure.figsize'] = (15,10)
    plt.scatter(bhk2.new_total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.new_total_sqft, bhk3.price, color='green', marker='+', label='3 BHK', s=50)
    plt.xlabel('Total Square Feet Area')
    plt.ylabel('Price (in Lakhs)')
    plt.title(site_location)
    plt.legend()

plot_scatter_chart(df3, 'Hadapsar')

def remove_bhk_outliers(df):
    exclude_indices = np.array([])

    for site_location, site_location_df in df.groupby('site_location'):
        bhk_stats = {}

        for bhk, bhk_df in site_location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }

        for bhk, bhk_df in site_location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)

    return df.drop(exclude_indices, axis='index')

df4 = remove_bhk_outliers(df3)
print(len(df3), len(df4))

plot_scatter_chart(df4, 'Wagholi')
plt.hist(df4.price_per_sqft, rwidth=0.5)
plt.xlabel('Price Per Square Feet')
plt.ylabel('Count')

plt.hist(df4.bath, rwidth=0.5)
plt.xlabel('Number of Bathrooms')
plt.ylabel('Count')

df5 = df4[df4.bath<(df4.bhk+2)]
print(len(df4), len(df5))

df5.tail()

df6 = df5.copy()
df6 = df6.drop('price_per_sqft', axis='columns')

df6.head()

dummy_cols = pd.get_dummies(df6.site_location)
df6 = pd.concat([df6,dummy_cols], axis='columns')


dummy_cols = pd.get_dummies(df6.availability).drop('Not Ready', axis='columns')
df6 = pd.concat([df6,dummy_cols], axis='columns')


dummy_cols = pd.get_dummies(df6.area_type).drop('Super built-up  Area', axis='columns')
df6 = pd.concat([df6,dummy_cols], axis='columns')

df6.drop(['area_type','availability','site_location'], axis='columns', inplace=True)
df6.head(10)

df6.shape

X = df6.drop('price', axis='columns')
y = df6['price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)

X.columns
locations=df.site_location.unique()

np.where(X.columns=='Balaji Nagar')[0][0]

np.where(X.columns=='Built-up  Area')[0][0]

def prediction(location, bhk, bath, balcony, sqft, area_type, availability):

    loc_index, area_index, avail_index = -1,-1,-1

    if location!='other':
        loc_index = int(np.where(X.columns==location)[0][0])

    if area_type!='Super built-up  Area':
        area_index = np.where(X.columns==area_type)[0][0]

    if availability!='Not Ready':
        avail_index = np.where(X.columns==availability)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = bath
    x[1] = balcony
    x[2] = bhk
    x[3] = sqft

    if loc_index >= 0:
        x[loc_index] = 1
    if area_index >= 0:
        x[area_index] = 1
    if avail_index >= 0:
        x[avail_index] = 1

    return model.predict([x])[0]

prediction('Balaji Nagar', 2, 2, 2, 1000, 'Built-up  Area', 'Ready To Move')

prediction('Wagholi', 1, 1, 1, 800, 'Built-up  Area', 'Ready To Move')
