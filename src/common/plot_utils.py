from plotly.subplots import make_subplots
import wandb
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.figure_factory as ff

ohlc_cols = ["Open", "High", "Low", "Close"]


def plot_multi_lines(title=None, **name_data):
    traces = [
        go.Scatter(y=line_data, name=name, opacity=0.5)
        for name, line_data in name_data.items()
    ]
    return traces


def candlestick(data):
    if isinstance(data, pd.DataFrame):
        o, h, l, c = data[ohlc_cols].T.values
        candles = go.Candlestick(
            x=data.index,
            open=o,
            high=h,
            low=l,
            close=c,
            showlegend=False,
        )
    else:
        candles = go.Candlestick(
            x=np.arange(len(data)),
            open=data[:, 0],
            high=data[:, 1],
            low=data[:, 2],
            close=data[:, 3],
        )
    return candles


def plot_ohlcv(df):
    candles = candlestick(df)
    figure = go.Figure(data=[candles])
    figure.update_layout(xaxis_rangeslider_visible=False)
    return figure


def plot_multi_ohlcv(**kwargs):
    fig = make_subplots(
        rows=len(kwargs),
        cols=1,
        subplot_titles=[name for name in kwargs],
        vertical_spacing=0.4,
        shared_yaxes="all",
        shared_xaxes="all",
    )
    for i, df in enumerate(kwargs.values()):
        fig.add_trace(candlestick(df), row=i + 1, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_layout(xaxis2_rangeslider_visible=False)
    return fig


def confusion_matrix_table(confusion_matrix, labels: list) -> wandb.Table:
    # change each element of z to type string for annotations
    z_text = [
        [f"{l}_truth"] + [str(y) for y in x] for x, l in zip(confusion_matrix, labels)
    ]
    # on x there are predictions, on y there are ground truths
    columns = [""] + [f"{l}_pred" for l in labels]
    table = wandb.Table(data=z_text, columns=columns)
    return table


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


def plot_categorical_target(
    df, target_col="TargetCategorical", cmap={0: "red", 1: "white", 2: "green"}
):
    ax = df.Open.plot()
    for val, start, end in zip(df[target_col], df.index, df.index[1:]):
        ax.axvspan(start, end, facecolor=cmap[val], alpha=0.4)
    return ax.figure
