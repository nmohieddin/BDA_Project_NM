import numpy as np
import pandas
import statsmodels.api
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tabulate import tabulate

df = pandas.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
).dropna().reset_index()

print(df)
print(df.dtypes)
print(df.columns)
predictors = ["sex", "age", "sibsp", "embarked", "class"]
response = ["alive"]
x = df[predictors]
y = df[response]


# for categorical predictors
y_categorical = y.select_dtypes(include=['bool', 'object'])
# fo continuous predictors
y_continuous = y.select_dtypes(include=['int', 'float'])

# for categorical predictors
x_categorical = x.select_dtypes(include=['bool', 'object'])
# fo continuous predictors
x_continuous = x.select_dtypes(include=['int', 'float'])


# predictor - cont: Age
# predictor - cat: sex, alone
# response - cat: survived, alive

# build code for all 4 senarios

# 1_cat response - cat predictor
def cat_response_cat_predictor(var_1, var_2):
    conf_matrix = confusion_matrix([var_1], [var_2])

    fig_no_relationship = go.Figure(
        data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
    )

    fig_no_relationship.update_layout(
        title="Categorical Predictor by Categorical Response (with relationship)",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    fig_no_relationship.show()

    conf_matrix = confusion_matrix([var_1], [var_2])

    fig_no_relationship = go.Figure(
        data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
    )
    fig_no_relationship.update_layout(
        title="Categorical Predictor by Categorical Response (with relationship)",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    fig_no_relationship.show()
    return

# 2_cat response - cont predictor
def cat_resp_cont_predictor(var_3):
    n = 200
    # test matrix
    # x = titanic_df['age'].to_numpy()
    # x = x[~numpy.isnan(x)]
    # y = titanic_df['fare'].to_numpy()
    # y = y[~numpy.isnan(y)]
    # z = [x, y]

    group_labels = []
    for (columnName, columnData) in x.iteritems():
        group_labels.append("group")

    # Create distribution plot with custom bin_size
    fig_1 = ff.create_distplot([var_3], group_labels, bin_size=0.2)
    fig_1.update_layout(
        title="Continuous Predictor by Categorical Response",
        xaxis_title="Predictor",
        yaxis_title="Distribution",
    )
    fig_1.show()

    fig_2 = go.Figure()
    for y, curr_group in zip([var_3], group_labels):
        fig_2.add_trace(
            go.Violin(
                x=np.repeat(curr_group, n),
                y=y,
                name=curr_group,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig_2.update_layout(
        title="Continuous Predictor by Categorical Response",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    fig_2.show()
    return

# 3_con response - cat predictor
def cont_resp_cat_predictor(var_4):
    n = 200
    # Add histogram data
    # a = titanic_df['sex']
    # b = titanic_df['pclass']
    # c = titanic_df['deck']
    # z = [a, b, c]

    group_labels = []
    for (columnName, columnData) in x.iteritems():
        group_labels.append("group")

    # Create distribution plot with custom bin_size
    fig_1 = ff.create_distplot(var_4, group_labels, bin_size=0.2)
    fig_1.update_layout(
        title="Continuous Response by Categorical Predictor",
        xaxis_title="Response",
        yaxis_title="Distribution",
    )
    fig_1.show()

    fig_2 = go.Figure()
    for y, curr_group in zip(var_4, group_labels):
        fig_2.add_trace(
            go.Violin(
                x=np.repeat(curr_group, n),
                y=y,
                name=curr_group,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig_2.update_layout(
        title="Continuous Response by Categorical Predictor",
        xaxis_title="Groupings",
        yaxis_title="Response",
    )
    fig_2.show()
    return

# 4_con response - con predictor
def cont_response_cont_predictor(var_5, var_6):
    fig = px.scatter(x=var_5, y=var_6, trendline="ols")
    fig.update_layout(
        title="Continuous Response by Continuous Predictor",
        xaxis_title="Predictor",
        yaxis_title="Response",
    )
    fig.show()
    return

def main():
    for i in x:
        cat_response_cat_predictor(x_categorical[i], y_categorical[i])
        cont_response_cont_predictor(x_continuous[i], y_continuous[i])
        cat_resp_cont_predictor(x_continuous[i])
        cont_resp_cat_predictor(y_continuous[i])
    return

# p-value & tscore & Regression (Continuous)
resp_test = None
pred_test = None

# for response
for val in response:
    if y[val].dtype == "bool":
        resp_test = "boolean"
    else:
        resp_test = "continuous"

# for predictors
for val in predictors:  # x.columns
    if (
        x[val].dtype == "string"
        or x[val].dtype == "object"
        or x[val].dtype == "bool"
        or x[val].dtype == "category"
    ):
        pred_test = "categorical"
    else:
        pred_test = "continuous"
def main():
    if pred_test == "continuous":
        feature_name = val
        predictor = statsmodels.api.add_constant(x[val].values)
        linear_regression_model = statsmodels.api.OLS(
            y, predictor
        )  # Logit -logistic regression
        linear_regression_model_fitted = linear_regression_model.fit()
        print(f"Variable: {feature_name}")
        print(linear_regression_model_fitted.summary())

        # Get the stats
        t_value = round(linear_regression_model_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

        # Plot the figure
        fig = px.scatter(x=x[val].values, y=y, trendline="ols")
        fig.update_layout(
            title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {feature_name}",
            yaxis_title="y",
        )
        fig.show()
        return


# Logistic Regression
def main():
    if x[val].dtype == "boolean":
        feature_name = val
        predictor = statsmodels.api.add_constant(x[val].values)
        log_regression_model = statsmodels.api.Logit(y, predictor)
        log_regression_model_fitted = log_regression_model.fit()
        print(f"Variable: {feature_name}")
        print(log_regression_model_fitted.summary())

        # Get the stats
        t_value = round(log_regression_model_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(log_regression_model_fitted.pvalues[1])

        # Plot the figure
        fig = px.scatter(x=x[val].values, y=y, trendline="ols")
        fig.update_layout(
            title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {feature_name}",
            yaxis_title="y",
        )
        fig.show()
        return


# RandomForest Regression:
def main():
    X = pandas.DataFrame(y.data, columns=y.feature_names)
    y = y.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=12
    )

    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)

    px.create_distplot(x.feature_names, rf.feature_importances_)
    return


# mean calculation
def main():
    mae = mean_absolute_error(x_continuous, y_continuous)


    ## plot the data & predictions with the mae #
    plt.plot(x_continuous,y_continuous)
    plt.errorbar(x_continuous,y_train,mae)
    plt.title('Sinusoidal Data with Noise + Predictions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['y_true','y_pred'])
    plt.show()
    return

#Create Table
data = [["t_value", t_value],
        ["p_value", p_value],
        ["Linear Regression", linear_regression_model_fitted],
        ["Logistic Regression", log_regression_model_fitted],
        ["RandomForestRegressor", rf],
        ["MeanSquaredDiffWeighted", mae]]
print(tabulate(data, tablefmt="fancy_grid"))