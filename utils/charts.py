import plotly.express as px


def class_distribution(df):
    fig = px.histogram(
        df,
        x="koi_disposition",
        color="koi_disposition",
        title="Planet Class Distribution"
    )

    fig.update_layout(
        template="plotly_dark",
        height=500
    )

    return fig