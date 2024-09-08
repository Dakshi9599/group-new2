from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor()
regressor_rf.fit(x_train, y_train)

y_pred = regressor_rf.predict(x_test)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


