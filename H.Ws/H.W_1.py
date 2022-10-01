import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# import sklearn


# Load the data into dataframe
Iris_df = pd.read_csv(
    "/Users/nassermohieddin/Desktop/BDA_602/iris.csv",
    names=["sepal length", "sepal width", "petal length", "petal width", "class"],
)

# select columns with numbers to calculate mean,min,max, and quartile
Num_Iris = Iris_df.iloc[:, :4]

# print to check the data
# print(Num_Iris)

# Statistical Analysis:

# mean
print("Mean values are:")
print(Num_Iris.mean(numeric_only=True), "\n")

# min
print("Min values are:")
print(Num_Iris.min(numeric_only=True), "\n")

# max
print("Max values are:")
print(Num_Iris.max(numeric_only=True), "\n")

# quantile
print("Quantiles are:")
print(Num_Iris.quantile(numeric_only=True), "\n")

# Plotting
plot_Iris = Iris_df.iloc[:, 4:5]

# Figure 1 - Histogram
fig = px.histogram(plot_Iris, x="class", title="Histogram of Different Classes")
fig.show()

# Figure 2 - Pie Chart
fig = px.pie(plot_Iris, names="class", title="Pie Chart of Classes")
fig.show()

# Figure 3 - Scatter Plot
fig = px.scatter(
    Iris_df,
    x="sepal width",
    y="sepal length",
    color="class",
    title="Scatter Plot of Classes Length & Width",
)
fig.show()

# Figure 4 - Violin
fig = px.violin(
    Iris_df,
    y="sepal length",
    x="sepal width",
    color="class",
    box=True,
    points="all",
    hover_data=Iris_df.columns,
    title="Violin of Classes",
)
fig.show()

# Figure 5 -line graph
fig = px.line(Iris_df, y="sepal length", color="class")
fig.show()

# scikit-learn

# Data Setup
X_orig = Iris_df.data
y = Iris_df.target


def main():
    # StandardScaler
    one_hot_encoder = OneHotEncoder()
    print(one_hot_encoder.fit(X_orig))
    X = one_hot_encoder.transform(X_orig)

    random_forest = RandomForestClassifier(random_state=1234)
    random_forest.fit(X, y)

    # pipeline
    pipeline = Pipeline(
        [
            ("OneHotEncode", OneHotEncoder()),
            ("RandomForest", RandomForestClassifier(random_state=1234)),
        ]
    )
    pipeline.fit(X_orig, y)

    probability = pipeline.predict_proba(X)
    prediction = pipeline.predict(X)
    print(f"Probability: {probability}")
    print(f"Predictions: {prediction}")
    return
