"""
Clase 1: Workflow modelo con scikit-learn
"""
# %% Import
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR

# %% Dataset
np.random.seed(1234)
X = np.random.uniform(size=(1000, 5)) #Features
y = 2. + X @ np.array([1., 3., -2., 7., 5.]) + np.random.normal(size=1000) #Target


# %% Train & Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1,
                                                    random_state=1234)

# %% Models
lr_model = LinearRegression()  
svr_model = SVR(kernel='rbf', gamma='scale') 

# %% Cross-Validation
print('Linear Regression')
lr_cv = cross_val_score(lr_model, X_train, y_train, cv=5)
print(lr_cv)
print(f'Average 5-Fold CV Score = {lr_cv.mean():.4%}')
print('\n')
print('Support Vector Regression')
svr_cv = cross_val_score(svr_model, X_train, y_train, cv=5)
print(svr_cv)
print(f'Average 5-Fold CV Score = {svr_cv.mean():.4%}')

# %% Methods Fit & predict to the training data
#LinearRegression
lr_model.fit(X_train, y_train)
print(f'Linear Model Score : {lr_model.score(X, y):.2%}')
print(f'Train Set Score : {lr_model.score(X_train, y_train):.4%}')
print(f'Test Set Score: {lr_model.score(X_test, y_test):.4%}')

# Predict on the test data: y_pred
y_pred_lr = lr_model.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print(f'Root Mean Squared Error LRM: {rmse_lr:.4}')

#SVR
svr_model.fit(X_train, y_train)
print(f'SVR Model Score : {svr_model.score(X, y):.2%}')
print(f'Train Set Score : {svr_model.score(X_train, y_train):.4%}')
print(f'Test Set Score: {svr_model.score(X_test, y_test):.4%}')

# Predict on the test data: y_pred
y_pred_sv = svr_model.predict(X_test)
rmse_sv = np.sqrt(mean_squared_error(y_test, y_pred_sv))
print(f'Root Mean Squared Error SVR: {rmse_sv:.4}')

# %% Attributes
#LinearRegression
print(f'Coeficients = {lr_model.coef_}')
print(f'Intercept = {lr_model.intercept_}')

#SVR
print(f'Coeficients = {svr_model.coef_}')
print(f'Intercept = {svr_model.intercept_}') #This is only available in the case of a 'linear' kernel.
