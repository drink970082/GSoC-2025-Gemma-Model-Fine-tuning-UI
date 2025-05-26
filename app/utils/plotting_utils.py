def plot_line_chart(data, x_col, y_col, title="Line Chart"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x=x_col, y=y_col)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid()
    plt.show()

def plot_bar_chart(data, x_col, y_col, title="Bar Chart"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x=x_col, y=y_col)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid()
    plt.show()

def plot_histogram(data, column, title="Histogram"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], bins=30, kde=True)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()