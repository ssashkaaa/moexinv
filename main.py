import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
import pandas as pd
import apimoex
import requests
import plotly.graph_objs as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("Инвестиционный дашборд по акциям Мосбиржи")

@st.cache_data(ttl=3600)
def get_securities():
    with requests.Session() as session:
        data = apimoex.get_board_securities(session, board='TQBR')
        df = pd.DataFrame(data)
        needed_cols = ['SECID', 'SHORTNAME', 'PREVPRICE', 'LOTSIZE', 'FACEVALUE', 'CURRENCYID', 'ISIN']
        existing_cols = [col for col in needed_cols if col in df.columns]
        df = df[existing_cols]
        return df

securities_df = get_securities()

tickers = st.multiselect(
    "Выберите акции для портфеля (до 5)",
    securities_df['SECID'],
    default=securities_df['SECID'][:2]
)

weights = []
st.write("Задайте доли для каждой акции (в сумме = 1.0):")
for ticker in tickers:
    weight = st.number_input(
        f"{ticker}:",
        min_value=0.0,
        max_value=1.0,
        value=1.0/len(tickers) if tickers else 0.0,
        step=0.01
    )
    weights.append(weight)
if sum(weights) != 1.0:
    st.warning("Сумма долей должна быть равна 1.0")

@st.cache_data(ttl=3600)
def get_history(ticker, start, end):
    with requests.Session() as session:
        data = apimoex.get_market_candles(
            session, security=ticker, interval=24, start=start, end=end
        )
    df = pd.DataFrame(data)
    if not df.empty:
        df['begin'] = pd.to_datetime(df['begin'])
        df.set_index('begin', inplace=True)
    return df

start_date = st.date_input("Дата начала", datetime.now() - timedelta(days=365))
end_date = st.date_input("Дата конца", datetime.now())
portfolio_df = pd.DataFrame()

for ticker in tickers:
    df = get_history(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    if not df.empty:
        portfolio_df[ticker] = df['close']

imoex_df = get_history('IMOEX', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

if not portfolio_df.empty:
    st.subheader("Динамика цен выбранных акций и индекса IMOEX")
    fig = go.Figure()
    for ticker in tickers:
        fig.add_trace(go.Scatter(x=portfolio_df.index, y=portfolio_df[ticker], mode='lines', name=ticker))
    if not imoex_df.empty:
        fig.add_trace(go.Scatter(x=imoex_df.index, y=imoex_df['close'], mode='lines', name='IMOEX', line=dict(dash='dash')))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Доходность и риск портфеля")
    returns = portfolio_df.pct_change().dropna()
    weighted_returns = returns.dot(weights)
    cumulative = (1 + weighted_returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    risk = weighted_returns.std() * (252 ** 0.5)

    st.write(f"**Доходность портфеля:** {total_return:.2%}")
    st.write(f"**Годовой риск (стандартное отклонение):** {risk:.2%}")

    if not imoex_df.empty:
        imoex_returns = imoex_df['close'].pct_change().dropna()
        imoex_cum = (1 + imoex_returns).cumprod()
        st.write(f"**Доходность IMOEX:** {imoex_cum.iloc[-1] - 1:.2%}")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=cumulative.index, y=cumulative, name="Портфель"))
        fig2.add_trace(go.Scatter(x=imoex_cum.index, y=imoex_cum, name="IMOEX"))
        st.plotly_chart(fig2, use_container_width=True)

    # --- Корреляции (тепловая карта) ---
    st.subheader("Корреляция доходностей акций в портфеле")
    if returns.shape[1] > 1:
        corr = returns.corr()
        st.dataframe(corr)
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        fig4, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, vmin=-1, vmax=1)
        st.pyplot(fig4)

