import numpy
import pandas
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


dataset_df = pandas.read_csv("vente_maillots_de_bain.csv")
dataset_df["Years"] = pandas.to_datetime(dataset_df["Years"])


def show_dataset(df, x_column, y_column):
	plt.figure(figsize=(15, 6))
	plt.plot(df[x_column], df[y_column], "b-")
	plt.show()

def prepare_data(df, y_column):
	df["time_index"] = numpy.arange(0, len(df), 1)
	split_index = int(len(df)*0.8)
	train_df = df.iloc[split_index : ]
	test_df = df.iloc[ : split_index]

	x_train = train_df[["time_index"]]
	y_train = train_df[[y_column]]

	x_test = test_df[["time_index"]]
	y_test = test_df[[y_column]]

	return x_train, y_train, x_test, y_test



def genrate_fitted_model(x_train, y_train):
	model = LinearRegression()
	model.fit(x_train, y_train)

	return model


def test_model(model, x_test, y_test):
	y_test_predicted = model.predict(x_test)
	plt.figure(figsize=(15, 6))
	plt.plot(x_test, y_test, "b-")
	plt.plot(x_test, y_test_predicted, "r-")
	plt.show()


# show_dataset(dataset_df, "Years", "Sales")

x_train, y_train, x_test, y_test = prepare_data(df=dataset_df, y_column="Sales")

model = genrate_fitted_model(x_train, y_train)

test_model(model, x_test, y_test)