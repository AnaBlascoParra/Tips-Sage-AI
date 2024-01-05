import pandas as pd

df = pd.read_csv("tips.csv")    
df = df.dropna()  

# nominal/categorical columns to ordinal
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in ["smoker", "day", "time", "sex"]:
    col_tmp = le.fit_transform(df[col])
    df[col] = col_tmp

from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.3,random_state=42)


# dataset split
import numpy as np
train_cols = df_train.columns.drop("tip")

X_train = df_train.loc[:, train_cols]
y_train = df_train[['tip']].to_numpy() 
X_test = df_test.loc[:, train_cols]
y_test = df_test[['tip']].to_numpy() 

#imported models
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

model = SVR(gamma='auto')
#model = RandomForestRegressor(random_state=42)
#model = LinearRegression()
#model = MLPRegressor(random_state=1, max_iter=500)
#model = make_pipeline(StandardScaler(), model) 



model.fit(X_train, y_train) 
y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

print ("MSE: ", mean_squared_error(y_test, y_pred))
print ("MAE: ", mean_absolute_error(y_test, y_pred))
print ("R2: ", r2_score(y_test, y_pred))
print ("MAPE: ", mean_absolute_percentage_error(y_test, y_pred))

