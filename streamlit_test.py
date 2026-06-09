import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

portfolio_excel_path = Path(r"D:\Investing\XTB\Repos\daily_portfolio_metrics.xlsx")
assets_excel_path = Path(r"D:\Investing\XTB\Repos\daily_asset_metrics.xlsx")

assets = pd.read_excel(assets_excel_path, sheet_name='Sheet1', index_col=0)

assets[['direction',
        'amount',
        'outstanding_position',
        'cost_cumsum',
        'price',
        'fx',
        'mv',
        'other_pnl',
        'closed_positions',
        'pnl_ltd',
        'pnl_dtd',
        'pnl_tot_dtd',
        'pnl_tot_ltd',
       # 'pnl_rel_ltd',
        'pnl_rel_tot_ltd',
        #'pnl_rel_dtd'
        ]] = (assets[['direction',
        'amount',
        'outstanding_position',
        'cost_cumsum',
        'price',
        'fx',
        'mv',
        'other_pnl',
        'closed_positions',
        'pnl_ltd',
        'pnl_dtd',
        'pnl_tot_dtd',
        'pnl_tot_ltd',
        #'pnl_rel_ltd',
        'pnl_rel_tot_ltd',
        #'pnl_rel_dtd'
                                    ]].fillna(0).astype('float64'))

assets.date = pd.to_datetime(assets.date, format='%Y-%m-%d')

tickers = assets.symbol.unique()

dates_to_filter = st.slider(
    'Timeline',
    min_value= datetime(2023, 1, 1),
    max_value= datetime(2025, 1, 1),
    value = (datetime(2023, 1, 1), datetime(2026, 5, 14))
    )

with st.container(border=True):
    tickers = st.multiselect("Tickers", tickers, default=tickers)

assets = assets.loc[(assets['date'] >= dates_to_filter[0]) & (assets['date'] <= dates_to_filter[1] ), :]

assets_pivot = assets.pivot(index='date', columns='symbol', values='pnl_ltd')


assets_pivot_selected = assets_pivot[tickers]


st.line_chart(assets_pivot_selected)