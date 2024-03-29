import numpy
import pandas
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf


pandas.options.mode.copy_on_write = True
dataset_df = pandas.read_csv("vente_maillots_de_bain.csv")
dataset_df["Years"] = pandas.to_datetime(dataset_df["Years"])


def show_dataset(df, x_column, y_column):
	plt.figure(figsize=(15, 6))
	plt.plot(df[x_column], df[y_column], "b-")
	plt.show()

def prepare_data_regression(df, y_column):
	df["time_index"] = numpy.arange(1, len(df)+1, 1)
	df["Month"] = df["Years"].dt.month_name()
	split_index = int(len(df)*0.7)
	df_mean, df_std = df["Sales"].mean(), df["Sales"].std()

	df["Sales"] = (df["Sales"] - df_mean) / df_std	

	train_df = df.iloc[ : split_index]
	test_df = df.iloc[split_index : ]

	x_train = train_df[["time_index"]]
	y_train = train_df[[y_column]]

	x_test = test_df[["time_index"]]
	y_test = test_df[[y_column]]

	return train_df, test_df, x_train, y_train, x_test, y_test, df_mean, df_std




def prepare_data_additif(df, y_column):
	df["Years"] = pandas.to_datetime(df["Years"])
	df["time_index"] = numpy.arange(1, len(df)+1, 1)

	df["Month"] = df["Years"].dt.month_name()
	df = pandas.get_dummies(df, columns=["Month"], prefix="", prefix_sep="", dtype=int)

	split_index = int(len(df)*0.7)
	df_mean, df_std = df["Sales"].mean(), df["Sales"].std()

	df["Sales"] = (df["Sales"] - df_mean) / df_std

	train_df = df.iloc[ : split_index]
	test_df = df.iloc[split_index : ]

	x_train = train_df[["time_index"] + list(df["Years"].dt.month_name().unique())]
	y_train = train_df[[y_column]]

	x_test = test_df[["time_index"] + list(df["Years"].dt.month_name().unique())]
	y_test = test_df[[y_column]]

	return x_train, y_train, x_test, y_test, df_mean, df_std





def genrate_fitted_model(x_train, y_train):
	model = LinearRegression()
	model.fit(x_train, y_train)

	return model


def test_model(model, x_test, y_test, df_mean, df_std):
	y_test_predicted = model.predict(x_test).squeeze()  # return a * x + b
	y_test = y_test["Sales"].values
	rmse = math.sqrt(sum((y_test- y_test_predicted)**2/len(x_test))) # donnée normalisée
	print(rmse)
	
	y_test_predicted = y_test_predicted*df_std + df_mean
	y_test = y_test*df_std+df_mean	
	

	plt.figure(figsize=(15, 6))
	
	plt.plot(x_test["time_index"], y_test, "b-") # donnée originale
	plt.plot(x_test["time_index"], y_test_predicted, "r-")
	
	plt.show()


def show_correlogram(df, y_column):
	y_values = df[y_column].values	
	y_stationarized = y_values[1:]- y_values[:-1]
	plot_acf(y_stationarized)
	plt.show()



def get_mean_seasonal_deviation(model, train_df, x_train, df_mean, df_std):

	train_df["regression_prediction"] = model.predict(x_train).squeeze() * df_std + df_mean
	train_df["seasonal deviation"] = (train_df["Sales"] * df_std + df_mean) / train_df["regression_prediction"] 
	mean_seasonal_deviation = train_df.groupby("Month").mean()["seasonal deviation"]

	return mean_seasonal_deviation




def test_multiplicatif_model(model, test_df, x_test, y_test, df_mean, df_std, mean_seasonal_deviation):
	test_df["regression_prediction"] = model.predict(x_test).squeeze()
	test_df = test_df.merge(mean_seasonal_deviation, on="Month")
	test_df.sort_values(by=["Years"], inplace=True)


	y_test = y_test["Sales"].values
	y_test_predicted = test_df["regression_prediction"] * test_df["seasonal deviation"]
	
	rmse = math.sqrt(sum((y_test- y_test_predicted)**2/len(x_test))) # donnée normalisée
	print(rmse)

	y_test = y_test * df_std + df_mean
	y_test_predicted = y_test_predicted * df_std + df_mean
	plt.figure(figsize=(15,6))
	plt.plot(x_test["time_index"], y_test, "b-") # donnée originale
	plt.plot(x_test["time_index"], y_test_predicted, "r-")
	
	plt.show()



# show_dataset(dataset_df, "Years", "Sales")


# model = genrate_fitted_model(x_train, y_train)

# test_model(model, x_test, y_test, df_mean, df_std)
# show_correlogram(dataset_df, "Sales")


# x_train, y_train, x_test, y_test, df_mean, df_std = prepare_data_additif(df=dataset_df, y_column="Sales")



print("regression simple")
train_df, test_df, x_train, y_train, x_test, y_test,\
									 df_mean, df_std = prepare_data_regression(df=dataset_df, y_column="Sales")
model = genrate_fitted_model(x_train, y_train)
test_model(model, x_test, y_test, df_mean, df_std)


print("additif")
x_train, y_train, x_test, y_test, df_mean, df_std = prepare_data_additif(df=dataset_df, y_column="Sales")
model = genrate_fitted_model(x_train, y_train)
test_model(model, x_test, y_test, df_mean, df_std)

print("multiplicatif")
dataset_df = pandas.read_csv("vente_maillots_de_bain.csv")
dataset_df["Years"] = pandas.to_datetime(dataset_df["Years"])
train_df, test_df, x_train, y_train, x_test, y_test,\
									 df_mean, df_std = prepare_data_regression(df=dataset_df, y_column="Sales")
model = genrate_fitted_model(x_train, y_train)
mean_seasonal_deviation = get_mean_seasonal_deviation(model, train_df, x_train, df_mean, df_std)
test_multiplicatif_model(model, test_df, x_test, y_test, df_mean, df_std, mean_seasonal_deviation)




