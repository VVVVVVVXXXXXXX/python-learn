# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 20:16:27 2021

@author: hp
"""

from jqdatasdk.technical_analysis import RSI
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import jqdatasdk as jq
import pandas as pd
from jqdatasdk import get_factor_values
jq.auth('18696160122', '9629189abC')

#%%多因子打分模型
def function(stocks,begin,GN):
    #按市值排序
    q = query(
        valuation.code,
        valuation.market_cap.label("MC"),        #市值
        indicator.roe,                           #roe
        income.net_profit.label("NP"),             #净利润
        indicator.gross_profit_margin.label("GP"), #毛利
        ( balance.total_liability/balance.total_assets*100).label("FZ") , #负债
        
    ).filter(
        valuation.code.in_(stocks)
    )
    
    fdf = get_fundamentals(q,begin)
    #################################################################
    factors = ['MC','roe','NP','GP','FZ']
    fdf.index = fdf['code']
    fdf.columns = ['code'] + factors
    #MarketCap越小序号越大，roe越大序号越大,
    effective_factors = {'MC':False,'roe':True,'NP':True,'GP':True,'FZ':False,}
    
    score = {}
    # 每个因子的序号
    for fac,value in effective_factors.items():
        score[fac] = fdf[fac].rank(ascending = value,method = 'first')
    DF=  pd.DataFrame(score).dropna()
   
   # log.info('order:'+str(DF))
    
    # 每支股票的打分=在每个因子数组中排列的序号*因子系数的总和，总和高的入选
    # --------------------------------MC-roe
    scores=(DF*np.array([0,0,0.5,0.4,0.2])).T.sum()
    
  #  log.info('scores:'+str(scores))
    buy_stock_set = list(scores.order(ascending = False).head(GN).index)
    
    log.info('buy_stock_set:'+str(buy_stock_set));
    return buy_stock_set
#%%
def get_stocks(date):       #获取主板股票
    from datetime import timedelta
    stocks_list = get_all_securities(['stock'], date=date)
    stocks_list = stocks_list[~stocks_list.index.str.startswith('688')]
    stocks_list = stocks_list[~stocks_list.index.str.startswith('300')]
    stocks_list = stocks_list[~stocks_list.display_name.str.contains('\*|ST|退', regex=True)]
    return list(stocks_list.index)
