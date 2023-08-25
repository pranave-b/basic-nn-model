# Developing a Neural Network Regression Model

## Aim:

To develop a neural network regression model for the given dataset.

## Theory:

Neural networks consist of simple input/output units called neurons. These units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

Build your training and test set from the dataset, here we are making the neural network 3 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model:
![nnnn](https://github.com/Aashima02/basic-nn-model/assets/93427086/21a17037-96ee-420a-801d-556a4d54c073)




## Design Steps:

1. Loading the dataset

2. Split the dataset into training and testing

3. Create MinMaxScalar objects ,fit the model and transform the data.

4. Build the Neural Network Model and compile the model.

5. Train the model with the training data.

6. Plot the performance plot

7. Evaluate the model with the testing data.

## Program:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from google.colab import auth
import gspread
from google.auth import default

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('DLExp1').sheet1
data = worksheet.get_all_values()
df = pd.DataFrame(data[1:], columns=data[0])
df = df.astype({'Input':'float'})
df = df.astype({'Output':'float'})
df.head()

X = df[['Input']].values
y = df[['Output']].values
X

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
X_train1

ai=Sequential([
    Dense(7,activation='relu'),
    Dense(14,activation='relu'),
    Dense(1)
])
ai.compile(optimizer='rmsprop',loss='mse')
ai.fit(X_train1,y_train,epochs=2000)
ai.fit(X_train1,y_train,epochs=2000)

## Plot the loss
loss_df = pd.DataFrame(ai.history.history)
loss_df.plot()

## Evaluate the model
X_test1 = Scaler.transform(X_test)
ai.evaluate(X_test1,y_test)

# Prediction
X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai.predict(X_n1_1)
```

## Dataset Information:

![image](https://github.com/Aashima02/basic-nn-model/assets/93427086/c6f062bc-91f8-486d-852b-3c9a8578937e)


## Output:

### Training Loss Vs Iteration Plot:

![image](https://github.com/Aashima02/basic-nn-model/assets/93427086/c831c691-939b-4021-b655-f9f038588079)


### Test Data Root Mean Squared Error:

![image](https://github.com/Aashima02/basic-nn-model/assets/93427086/29dec5d3-187f-491f-b5f4-b90294b4b85e)


### New Sample Data Prediction:

![image](https://github.com/Aashima02/basic-nn-model/assets/93427086/1b94db2d-14de-4ef8-b62b-ae7c78db8d57)


## Result:
Thus a neural network regression model for the given dataset is written and executed successfully.
