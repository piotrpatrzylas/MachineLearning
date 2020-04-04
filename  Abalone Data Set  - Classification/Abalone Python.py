import pandas as pd
import seaborn as sns
from sklearn import model_selection
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

# Data pre-processing and EDA

df = pd.read_csv("abalone.data")
df.columns = ["Sex", "Length", "Diameter", "Height", "W_Weight", "S_Weight", "V_Weight", "Sh_Weight", "Rings"]
df.head(10)
# 1 - Changing the response from Rings to Age
df["Age"] = df["Rings"] + 1.5
del df["Rings"]
# 2 - One hot encoding for "Sex" predictor
df.Sex.unique()
dummies = pd.get_dummies(df.Sex, drop_first=True)  # drop_first to avoid dummy variable trap
df = df.join(dummies)
del df["Sex"]
# 3 - Re-ordering columns
df = df[["I", "M", "Length", "Diameter", "Height", "W_Weight", "S_Weight", "V_Weight", "Sh_Weight", "Age"]]
# 4 - Pre-processing end results
df.info()
# 5 - Checking correlation
corr = df.corr()
sns.heatmap(corr, cmap="RdBu_r")  # Multicolinearity candidates: Length/Diameter and weights.

# Varience inflation factor - cutoff > 5
X1 = sm.tools.add_constant(df)
S1 = pd.Series([variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])], index=X1.columns)
df_dropped = df.drop(["Length", "W_Weight", "S_Weight", "V_Weight", "Sh_Weight"], axis=1)
X2 = sm.tools.add_constant(df_dropped)
S2 = pd.Series([variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])], index=X2.columns)
pd.plotting.scatter_matrix(df_dropped[["Diameter", "Height", "Age"]])

# 6. EDA
desc = df.describe()
pd.set_option('display.expand_frame_repr', False)
iqr = desc.loc["75%"] - desc.loc["25%"]
desc.loc["+Outlier"] = desc.loc["75%"] + (1.5 * iqr)
desc.loc["-Outlier"] = desc.loc["25%"] - (1.5 * iqr)
desc.loc["+4std"] = desc.loc["mean"] + (4 * desc.loc["std"])
desc.loc["-4std"] = desc.loc["mean"] - (4 * desc.loc["std"])
# Height seems to have some outliers that need to be examined closer.
desc_drop = df[(np.abs(stats.zscore(df)) < 5).all(axis=1)]
df.index.difference(desc_drop.index)
df_dropped = df_dropped.drop([162, 293, 479, 1208, 1416, 1762, 2050, 2107, 2208])


# Linear Regression

X = df_dropped.drop("Age", axis=1)
Y = df_dropped[["Age"]]
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=1)

X_train = sm.add_constant(X_train)
model = sm.OLS(Y_train, X_train).fit()
model.summary()
plot_size = plt.figure(figsize=(20, 12))
plot = sm.graphics.plot_partregress_grid(model, fig=plot_size)
# All predictors p-values are statistically significant. Omnibus > 0.5, data is probably normal.

X_test = sm.add_constant(X_test)
y_pred = model.predict(X_test)
print(y_pred[:10])
print(Y_test[:10])

# To check model accuracy we would need to define the threshold when our predictions are too far.
# Age values for test (Y_test) range from 4.5 to 24.5.
# Let's check how good this model is to guess age +-5, +-1 years and +- 1 month.
y_pred2 = pd.core.frame.DataFrame(y_pred)
y1 = np.isclose(y_pred2, Y_test, rtol=1, atol=1)
m1 = np.isclose(y_pred2, Y_test, rtol=0.083, atol=0.083)
d1 = np.isclose(y_pred2, Y_test, rtol=0.0027, atol=0.083)
acc = [y1, m1, d1]
acc_t = ["1 year", "1 month", "1 day"]
for i in range(3):
    print("Model accuracy for " + acc_t[i] + " is " + str(acc[i].sum()/1251*100) + "%")

# For one year age prediction test accuracy is 100%, but it falls quickly for 1 month (37%) and further for 1 day (5%).
# Still, not that bad.
