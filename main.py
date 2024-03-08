import numpy
import pandas
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf

dataset_df = pandas.read_csv("vente_maillots_de_bain.csv")
dataset_df["Years"] = pandas.to_datetime(dataset_df["Years"])


def show_dataset(df, x_column, y_column):
	plt.figure(figsize=(15, 6))
	plt.plot(df[x_column], df[y_column], "b-")
	plt.show()

def prepare_data(df, y_column):
	df["time_index"] = numpy.arange(1, len(df)+1, 1)
	split_index = int(len(df)*0.7)
	df_mean, df_std = df["Sales"].mean(), df["Sales"].std()

	df["Sales"] = (df["Sales"] - df_mean) / df_std

	train_df = df.iloc[ : split_index]
	test_df = df.iloc[split_index : ]

	x_train = train_df[["time_index"]]
	y_train = train_df[[y_column]]

	x_test = test_df[["time_index"]]
	y_test = test_df[[y_column]]

	return x_train, y_train, x_test, y_test, df_mean, df_std



def genrate_fitted_model(x_train, y_train):
	model = LinearRegression()
	model.fit(x_train, y_train)

	return model


def test_model(model, x_test, y_test, df_mean, df_std):
	y_test_predicted = model.predict(x_test).squeeze()
	y_test = y_test["Sales"].values
	rmse = math.sqrt(sum((y_test- y_test_predicted)**2/len(x_test))) # donnée normalisée
	print(rmse)
	
	y_test_predicted = y_test_predicted*df_std + df_mean
	y_test = y_test*df_std+df_mean	
	

	plt.figure(figsize=(15, 6))
	
	plt.plot(x_test, y_test, "b-") # donnée originale
	plt.plot(x_test, y_test_predicted, "r-")
	
	plt.show()


def show_correlogram(df, y_column):
	y_values = df[y_column].values	
	y_stationarized = y_values[1:]- y_values[:-1]
	plot_acf(y_stationarized)
	plt.show()



# show_dataset(dataset_df, "Years", "Sales")

# x_train, y_train, x_test, y_test, df_mean, df_std = prepare_data(df=dataset_df, y_column="Sales")

# model = genrate_fitted_model(x_train, y_train)

# test_model(model, x_test, y_test, df_mean, df_std)
show_correlogram(dataset_df, "Sales")