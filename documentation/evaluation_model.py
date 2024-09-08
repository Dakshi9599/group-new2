from sklearn.linear_model import LinearRegression
regressor_mlr = LinearRegression()
regressor_mlr.fit(x_train, y_train)

y_pred = regressor_mlr.predict(x_test)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
