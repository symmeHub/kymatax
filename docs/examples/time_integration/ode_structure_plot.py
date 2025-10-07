import marimo

__generated_with = "0.14.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import plotly.express as px
    import polars as pl

    return pl, px


@app.cell
def _(pl):
    sdf = pl.read_csv("data/solutions.csv")
    sdf
    return (sdf,)


@app.cell
def _(pl):
    pdf = pl.read_csv("data/problems.csv")
    pdf
    return (pdf,)


@app.cell
def _(pl):
    cdf = pl.read_csv("data/finder_configs.csv")
    cdf
    return (cdf,)


@app.cell
def _(cdf, pdf, pl, sdf):
    df = pl.concat([sdf, pdf, cdf], how="horizontal").sort("subharmonics")
    df
    return (df,)


@app.cell
def _(df, px):
    fig = px.scatter_3d(
        df,
        x="Xa_0",
        z="Xa_1",
        y="fd",
        color=[str(v) for v in df["subharmonics"]],
        size=(df["subharmonics"] + 1.0) ** -1 * 1.0 + 0.25,
        labels={
            "Xa_0": "Position, x",
            "Xa_1": r"Speed, dot x",
            "fd": "Driving frequency, f_d [Hz]",
        },
    )
    # fig.update_traces(marker=dict(size=4))

    fig.show()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
