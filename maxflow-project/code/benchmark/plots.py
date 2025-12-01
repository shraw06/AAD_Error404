import pandas as pd
import matplotlib.pyplot as plt


def plot_runtime_bar(csv_file, title="Runtime Comparison"):
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(10, 5))
    plt.bar(df["Algorithm"], df["MeanTime"])
    plt.xlabel("Algorithm")
    plt.ylabel("Mean Runtime (s)")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_runtime_vs_n(csv_file, x_name="Nodes"):
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(10, 5))
    for algo in df["Algorithm"].unique():
        sub = df[df["Algorithm"] == algo]
        plt.plot(sub[x_name], sub["MeanTime"], marker="o", label=algo)

    plt.xlabel(x_name)
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs Graph Size")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_runtime_vs_density(csv_path):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)

    # Group by algorithm
    algorithms = df["Algorithm"].unique()

    plt.figure(figsize=(10, 6))

    for algo in algorithms:
        sub = df[df["Algorithm"] == algo]
        plt.plot(sub["Density"], sub["MeanTime"],
                 marker='o', label=algo)

    plt.xlabel("Graph Density")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime vs Graph Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

