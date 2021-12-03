import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.figure_factory as ff

from tqdm.auto import tqdm
from glob import glob


ohlc_cols = ["Open", "High", "Low", "Close"]


def rsi(ohlc: pd.DataFrame, period=14):
    delta = ohlc["Close"].diff()

    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0

    RolUp = dUp.rolling(period).mean()
    RolDown = dDown.rolling(period).mean().abs()

    RS = RolUp / RolDown
    rsi = 100.0 - (100.0 / (1.0 + RS))
    return rsi


def df_logdiff(df1, df2):
    return np.log(df1) - np.log(df2)


def read_joined_dataframes(reader, resampler=id, **tickers_paths):
    result = pd.DataFrame()
    with tqdm(total=len(tickers_paths)) as pbar:
        for ticker, path in tickers_paths.items():
            pbar.set_description(f"doing {ticker} at path {path}")
            pbar.update()

            dataframe = pd.rename(
                resampler(reader(path))[ohlc_cols],
                columns=lambda c: f"{c}_{ticker}",
            )
            result = result.join(dataframe, how="outer")
    return result


def plot_column_histogram(dataframe, column):
    import plotly.express as px

    fig = px.histogram(dataframe, x=column)
    fig.write_html("/tmp/distplot.html", auto_open=True)
    return fig


def confusion_matrix_fig(confusion_matrix, labels):
    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in confusion_matrix]

    # set up figure
    fig = ff.create_annotated_heatmap(
        confusion_matrix,
        x=[f"{l}_pred" for l in labels],
        y=[f"{l}_real" for l in labels],
        annotation_text=z_text,
        colorscale="Viridis",
    )

    # add title
    fig.update_layout(
        title_text="<i><b>Confusion matrix</b></i>",
        # xaxis = dict(title='x'),
        # yaxis = dict(title='x')
    )

    # add custom xaxis title
    fig.add_annotation(
        dict(
            font=dict(color="black", size=14),
            x=0.5,
            y=-0.15,
            showarrow=False,
            text="Predicted value",
            xref="paper",
            yref="paper",
        )
    )

    # add custom yaxis title
    fig.add_annotation(
        dict(
            font=dict(color="black", size=14),
            x=-0.35,
            y=0.5,
            showarrow=False,
            text="Real value",
            textangle=-90,
            xref="paper",
            yref="paper",
        )
    )

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig["data"][0]["showscale"] = True
    return fig
