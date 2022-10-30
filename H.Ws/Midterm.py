import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import seaborn as sns
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OrdinalEncoder

df = (
    pandas.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    )
    .dropna()
    .reset_index()
)

print(df)
print(df.dtypes)
print(df.columns)
predictors = ["sex", "age", "sibsp", "embarked", "class"]
response = ["alive"]
x = df[predictors]
y = df[response]

if y.nunique()[0] == 2:
    response_type = "bool"
else:
    response_type = "continuous"

x_categorical = []
x_continuous = []

for val in x:
    if x[val].dtype in ["string", "object", "bool", "category"]:
        pred_test = "categorical"
        x_categorical.append(x[val])
    else:
        pred_test = "continuous"
        x_continuous.append(x[val])

# onehot encoder
enc = OrdinalEncoder()
x_2 = enc.fit_transform(x_categorical).astype(int)
enc2 = OrdinalEncoder()
y_2 = enc2.fit_transform(y).astype(int)

# build code for all 4 senarios
# 1_cat response - cat predictor


def cat_response_cat_predictor(var_1, var_2):

    conf_matrix = confusion_matrix(var_1, var_2)
    fig_no_relationship = go.Figure(
        data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
    )

    fig_no_relationship.update_layout(
        title="Categorical Predictor by Categorical Response (with relationship)",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )

    fig_no_relationship.show()
    fig_no_relationship.write_html(
        file=f'{"cat_cat"}',
        include_plotlyjs="cdn",
    )
    return


# 2_cat response - cont predictor
def cat_resp_cont_predictor(var_3):
    n = 200
    hist_data = [var_3]
    group_labels = ["Response"]
    # Create distribution plot with custom bin_size
    fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
    fig_1.update_layout(
        title="Continuous Predictor by Categorical Response",
        xaxis_title="Predictor",
        yaxis_title="Distribution",
    )
    fig_1.show()
    fig_1.write_html(
        file=f'{"cat_con_1"}',
        include_plotlyjs="cdn",
    )

    fig_2 = go.Figure()
    for curr_hist, curr_group in zip(hist_data, group_labels):
        fig_2.add_trace(
            go.Violin(
                x=np.repeat(curr_group, n),
                y=curr_hist,
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
    fig_2.write_html(
        file=f'{"cat_con_2"}',
        include_plotlyjs="cdn",
    )
    return


# 3_con response - cat predictor
def cont_resp_cat_predictor(var_4):
    group_labels = [x_categorical]
    # for i in x_2:
    # group_labels.append("group")
    n = len(group_labels)

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
    for i in x_2:
        cat_response_cat_predictor(i, y_2)
    for i in x_continuous:
        cat_resp_cont_predictor(i)

    # correlation matrices
    def Pearson_Correlation():
        cont_table = df.corr()
        print(cont_table)
        cont_matrix = np.asmatrix(cont_table)
        print(cont_matrix)
        print(sns.heatmap(cont_table, cmap="coolwarm"))
        plt.title("Pearson Correlation Plot")
        plt.show()
        plt.savefig("Pearson.png")
        return

    Pearson_Correlation()

    def cat_corr():
        a = pandas.DataFrame(x_2)
        cat_table = a.corr()
        print(cat_table)
        cat_matrix = np.asmatrix(cat_table)
        print(cat_matrix)
        print(sns.heatmap(cat_table, cmap="coolwarm"))
        plt.title("Cat-Cat Correlation Plot")
        plt.show()
        plt.savefig("cat_corr.png")
        return

    cat_corr()

    def continuous_corr():
        cat = pandas.DataFrame(x_2)
        con = pandas.DataFrame(x_continuous)
        new = [con, cat]
        concat = pandas.concat(new)
        concat_table = concat.corr()
        print(concat_table)
        concat_matrix = np.asmatrix(concat_table)
        print(concat_matrix)
        pivot = pd.pivot_table(data=df, index="alive")
        print(sns.heatmap(pivot))
        plt.title("Con-Cat Correlation Plot")
        plt.show()
        plt.savefig("con_corr.png")
        return

    continuous_corr()

    # Brute Force
    brute_force_cat = [
        list(x) for x in np.array(np.meshgrid(*x_2)).T.reshape(-1, len(x_2))
    ]
    print(sns.heatmap(brute_force_cat))
    plt.show()
    plt.savefig("Heatmap.png")

    # creating the bar plot
    x = pandas.DataFrame(x_categorical)
    for (columnName, columnData) in x.iteritems():
        plt.bar(
            columnData.astype(str), columnData.astype(str), color="maroon", width=0.4
        )
    plt.show()

    def Brute_Froce(sample1, sample2):
        # n = len(sample1)
        # m = len(sample2)
        # std1 = np.std(sample1, ddof=1)
        # std2 = np.std(sample2, ddof=1)
        # ddof = (std1**2 / n + std2**2 / m) ** 2 / (
        # (std1**2 / n) ** 2 / (n - 1) + (std2**2 / m) ** 2 / (m - 1)
        # )
        diff = np.mean(sample1) - np.mean(sample2)
        return diff

    print(Brute_Froce(df["age"], df["sibsp"]))

    for i in x_continuous:
        print(Brute_Froce(df["i"], df["i" + 1]))


if __name__ == "__main__":
    main()

# Html Report
page_title_text = "My report"
title_text = "Daily S&P 500 prices report"
text = "Hello, welcome to your report!"
prices_text = "Historical prices of S&P 500"
stats_text = "Historical prices summary statistics"

html = f"""
    <html>
        <head>
            <title>{"Midterm_Report"}</title>
        </head>
        <body>
            <h1>{"H.W_Figures"}</h1>
            <img src='Pearson.png' width="700">
            <img src='cat_con_1' width="700">
            <img src='cat_con_2' width="700">
            <h1>{"Correlation_Plots"}</h1>
            <img src='cat_cat' width="700">
            <img src='cat_corr.png' width="700">
            <img src='con_corr' width="700">
            <h1>"Brute_Force"</h1>
            <img src='Heatmap.png' width="700">
        </body>
    </html>
"""
with open("html_report.html", "w") as f:
    f.write(html)
