x = dataset.drop(columns='Selling_Price')
y = dataset['Selling_Price']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

x_train.shape

y_train.shape
x_test.shape
y_test.shape

