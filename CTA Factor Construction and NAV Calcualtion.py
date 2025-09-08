# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


# %%SubFunction of Warehouse Calculation(仓单计算子函数)

def calculate_rolling_mean(df, start, end):
    window_size = start
    if end == 0:
        return df.rolling(window=window_size, min_periods=window_size).apply(lambda x: x[-start:].mean(), raw=True)
    else:
        return df.rolling(window=window_size, min_periods=window_size).apply(lambda x: x[-start:-end].mean(), raw=True)


# %%Time Series Momentum Factor(动量因子-时序)
def factor_calculate_momentumTS(MarketChange, FactorData, UniverseData
                                , Period, Leverage, HoldingPeriod):
    '''
    时间序列动量

    Args
    -------

    MarketChange: Return
    FactorData: Price
    UniverseData
    Period


    Returns
    -------
    两种因子净值序列

    '''

    print('Time Series Momentum Factor(时间序列动量)')

    FactorData = FactorData / FactorData.shift(Period) - 1

    FactorData = FactorData.shift(1)

    # FactorData[FactorData > 1] = 1
    # FactorData[FactorData < 1] = -1
    # FactorData[FactorData == 0] = 0

    PositionData = FactorData.copy()
    valid_row_sum = PositionData.where(UniverseData).abs().sum(axis=1)
    PositionData = PositionData.div(valid_row_sum, axis=0)
    PositionData = PositionData.where(UniverseData, 0).fillna(0) * Leverage

    # Update every HoldingPeriod
    # # Maintain Weight
    # Find the row index of the first non-zero value
    first_non_zero_idx = (PositionData != 0).any(axis=1).idxmax()
    # Convert to integer index position
    first_non_zero_idx_pos = PositionData.index.get_loc(first_non_zero_idx)
    # Starting from the first non-zero value row, perform grouping operations
    PositionData1 = PositionData.iloc[first_non_zero_idx_pos:, :]
    group_idx = np.arange(len(PositionData1)) // HoldingPeriod
    PositionData.iloc[first_non_zero_idx_pos:, :] = PositionData1.groupby(group_idx).transform('first')

    # # Maintain Position
    # MarketChange_w = MarketChange.copy()
    #
    # # Calculates the cumulative product within the group and lags one row, filling the first row with 1
    # def process_market_change(df):
    #     # Add 1 to each group and calculate the cumulative product
    #     df = (df + 1).cumprod()
    #     # Lag one row, fill the first row with 1
    #     df = df.shift(1).fillna(1)
    #     return df
    #
    # MarketChange_w.iloc[first_non_zero_idx_pos:, :] = (
    #     MarketChange.iloc[first_non_zero_idx_pos:, :].groupby(group_idx).apply(process_market_change))
    # PositionData = PositionData * MarketChange_w

    ###Calculate Return###
    DailyContribution = PositionData * MarketChange
    DailyReturn = DailyContribution.sum(axis=1)

    NAV = pd.DataFrame(index=DailyReturn.index)
    # Calculate the cumulative return without compounding
    NAV['NonCompounded'] = DailyReturn.cumsum() + 1
    # Calculate the cumulative return of compound interest
    NAV['Compounded'] = (1 + DailyReturn).cumprod()

    return NAV, DailyContribution, DailyReturn


# %%Cross Sectional Momentum Factor(动量因子-截面(资产的相对排名))
def factor_calculate_momentumPanel(MarketChange, FactorData, UniverseData
                                   , Period, Leverage, HoldingPeriod):
    '''
    横截面动量

    Args
    -------

    MarketChange
    FactorData
    UniverseData
    Period

    Returns
    -------
    两种因子净值序列

    '''

    # reciprocal_long_short_ratio=2# 2代表做多一半做空一半,4说明做多1/4

    print('横截面动量')
    FactorData = FactorData / FactorData.shift(Period) - 1
    FactorData = FactorData.shift(1)

    ###Convert factors to cross-sectional###
    Factor_filtered = FactorData.where(UniverseData)
    # Calculate quantile
    quantiles = Factor_filtered.quantile([0.5], axis=1)
    FactorData_adjusted = pd.DataFrame(0, index=FactorData.index, columns=FactorData.columns)
    # Set the first 20% of elements to 1 (long)
    FactorData_adjusted[Factor_filtered > quantiles.loc[0.5, :].values[:, None]] = 1
    # Set the bottom 20% of elements to -1 (empty)
    FactorData_adjusted[Factor_filtered <= quantiles.loc[0.5, :].values[:, None]] = -1

    PositionData = FactorData_adjusted.copy()
    valid_row_sum = PositionData.where(UniverseData).abs().sum(axis=1)
    PositionData = PositionData.div(valid_row_sum, axis=0)
    PositionData = PositionData.where(UniverseData, 0).fillna(0) * Leverage
    # check：PositionData.sum(axis=1) PositionData.abs().sum(axis=1)

    # Update every HoldingPeriod
    # # Maintain Weight
    # Find the row index of the first non-zero value
    first_non_zero_idx = (PositionData != 0).any(axis=1).idxmax()
    # Convert to integer index position
    first_non_zero_idx_pos = PositionData.index.get_loc(first_non_zero_idx)
    # Starting from the first non-zero value row, perform grouping operations
    PositionData1 = PositionData.iloc[first_non_zero_idx_pos:, :]
    group_idx = np.arange(len(PositionData1)) // HoldingPeriod
    PositionData.iloc[first_non_zero_idx_pos:, :] = PositionData1.groupby(group_idx).transform('first')

    # # Maintain Position
    # MarketChange_w = MarketChange.copy()
    #
    # # Calculates the cumulative product within the group and lags one row, filling the first row with 1
    # def process_market_change(df):
    #     # Add 1 to each group and calculate the cumulative product
    #     df = (df + 1).cumprod()
    #     # Lag one row, fill the first row with 1
    #     df = df.shift(1).fillna(1)
    #     return df
    #
    # MarketChange_w.iloc[first_non_zero_idx_pos:, :] = (
    #     MarketChange.iloc[first_non_zero_idx_pos:, :].groupby(group_idx).apply(process_market_change))
    # PositionData = PositionData * MarketChange_w

    ###Calculate Return###
    DailyContribution = PositionData * MarketChange
    DailyReturn = DailyContribution.sum(axis=1)

    NAV = pd.DataFrame(index=DailyReturn.index)
    # Calculate the cumulative return without compounding
    NAV['NonCompounded'] = DailyReturn.cumsum() + 1
    # Calculate the cumulative return of compound interest
    NAV['Compounded'] = (1 + DailyReturn).cumprod()

    return NAV, DailyContribution, DailyReturn


# %%Time Series Skew Factor(偏度因子-时序)
def factor_calculate_skewTS(MarketChange, FactorData, UniverseData
                            , Period, Leverage, HoldingPeriod):
    '''
    时间序列动量

    Args
    -------

    MarketChange
    FactorData
    UniverseData
    Period


    Returns
    -------
    两种因子净值序列

    '''

    print('时间序列偏度')

    FactorData = -(FactorData / FactorData.shift() - 1).rolling(Period).skew()
    FactorData = FactorData.shift(1)

    PositionData = FactorData.copy()
    valid_row_sum = PositionData.where(UniverseData).abs().sum(axis=1)
    PositionData = PositionData.div(valid_row_sum, axis=0)
    PositionData = PositionData.where(UniverseData, 0).fillna(0) * Leverage

    # Update every HoldingPeriod
    # # Maintain Weight
    # Find the row index of the first non-zero value
    first_non_zero_idx = (PositionData != 0).any(axis=1).idxmax()
    # Convert to integer index position
    first_non_zero_idx_pos = PositionData.index.get_loc(first_non_zero_idx)
    # Starting from the first non-zero value row, perform grouping operations
    PositionData1 = PositionData.iloc[first_non_zero_idx_pos:, :]
    group_idx = np.arange(len(PositionData1)) // HoldingPeriod
    PositionData.iloc[first_non_zero_idx_pos:, :] = PositionData1.groupby(group_idx).transform('first')

    # # Maintain Position
    # MarketChange_w = MarketChange.copy()
    #
    # # Calculates the cumulative product within the group and lags one row, filling the first row with 1
    # def process_market_change(df):
    #     # Add 1 to each group and calculate the cumulative product
    #     df = (df + 1).cumprod()
    #     # Lag one row, fill the first row with 1
    #     df = df.shift(1).fillna(1)
    #     return df
    #
    # MarketChange_w.iloc[first_non_zero_idx_pos:, :] = (
    #     MarketChange.iloc[first_non_zero_idx_pos:, :].groupby(group_idx).apply(process_market_change))
    # PositionData = PositionData * MarketChange_w

    ###Calculate Return###
    DailyContribution = PositionData * MarketChange
    DailyReturn = DailyContribution.sum(axis=1)

    NAV = pd.DataFrame(index=DailyReturn.index)
    # Calculate the cumulative return without compounding
    NAV['NonCompounded'] = DailyReturn.cumsum() + 1
    # Calculate the cumulative return of compound interest
    NAV['Compounded'] = (1 + DailyReturn).cumprod()

    return NAV, DailyContribution, DailyReturn


# %%Time Series Breakout Factor(突破因子-时序)
def factor_calculate_breakoutTS(MarketChange, FactorData, UniverseData
                                , Period, Leverage, HoldingPeriod):
    print('时间序列均价突破')

    moving_average = FactorData.rolling(window=Period).mean()

    FactorData = (FactorData - moving_average) / FactorData.shift(Period - 1)
    FactorData = FactorData.shift(1)

    PositionData = FactorData.copy()
    valid_row_sum = PositionData.where(UniverseData).abs().sum(axis=1)
    PositionData = PositionData.div(valid_row_sum, axis=0)
    PositionData = PositionData.where(UniverseData, 0).fillna(0) * Leverage

    # Update every HoldingPeriod
    # # Maintain Weight
    # Find the row index of the first non-zero value
    first_non_zero_idx = (PositionData != 0).any(axis=1).idxmax()
    # Convert to integer index position
    first_non_zero_idx_pos = PositionData.index.get_loc(first_non_zero_idx)
    # Starting from the first non-zero value row, perform grouping operations
    PositionData1 = PositionData.iloc[first_non_zero_idx_pos:, :]
    group_idx = np.arange(len(PositionData1)) // HoldingPeriod
    PositionData.iloc[first_non_zero_idx_pos:, :] = PositionData1.groupby(group_idx).transform('first')

    # # Maintain Position
    # MarketChange_w = MarketChange.copy()
    #
    # # Calculates the cumulative product within the group and lags one row, filling the first row with 1
    # def process_market_change(df):
    #     # Add 1 to each group and calculate the cumulative product
    #     df = (df + 1).cumprod()
    #     # Lag one row, fill the first row with 1
    #     df = df.shift(1).fillna(1)
    #     return df
    #
    # MarketChange_w.iloc[first_non_zero_idx_pos:, :] = (
    #     MarketChange.iloc[first_non_zero_idx_pos:, :].groupby(group_idx).apply(process_market_change))
    # PositionData = PositionData * MarketChange_w

    ###Calculate Return###
    DailyContribution = PositionData * MarketChange
    DailyReturn = DailyContribution.sum(axis=1)

    NAV = pd.DataFrame(index=DailyReturn.index)
    # Calculate the cumulative return without compounding
    NAV['NonCompounded'] = DailyReturn.cumsum() + 1
    # Calculate the cumulative return of compound interest
    NAV['Compounded'] = (1 + DailyReturn).cumprod()

    return NAV, DailyContribution, DailyReturn


# %%Time Series Breakout Factor_Huo Fu Niu(突破因子-时序-火富牛原版)
def factor_calculate_breakoutTS_HFN(MarketChange, FactorData, UniverseData
                                    , Period, Leverage, HoldingPeriod):
    print('时间序列均价突破火富牛')

    moving_average = FactorData.rolling(window=Period).mean()
    moving_std = FactorData.rolling(window=Period).std()

    FactorData = (FactorData - moving_average) / (FactorData.shift(Period - 1) * moving_std)
    FactorData = FactorData.shift(1)

    PositionData = FactorData.copy()
    valid_row_sum = PositionData.where(UniverseData).abs().sum(axis=1)
    PositionData = PositionData.div(valid_row_sum, axis=0)
    PositionData = PositionData.where(UniverseData, 0).fillna(0) * Leverage

    # Update every HoldingPeriod
    # # Maintain Weight
    # Find the row index of the first non-zero value
    first_non_zero_idx = (PositionData != 0).any(axis=1).idxmax()
    # Convert to integer index position
    first_non_zero_idx_pos = PositionData.index.get_loc(first_non_zero_idx)
    # Starting from the first non-zero value row, perform grouping operations
    PositionData1 = PositionData.iloc[first_non_zero_idx_pos:, :]
    group_idx = np.arange(len(PositionData1)) // HoldingPeriod
    PositionData.iloc[first_non_zero_idx_pos:, :] = PositionData1.groupby(group_idx).transform('first')

    # # Maintain Position
    # MarketChange_w = MarketChange.copy()
    #
    # # Calculates the cumulative product within the group and lags one row, filling the first row with 1
    # def process_market_change(df):
    #     # Add 1 to each group and calculate the cumulative product
    #     df = (df + 1).cumprod()
    #     # Lag one row, fill the first row with 1
    #     df = df.shift(1).fillna(1)
    #     return df
    #
    # MarketChange_w.iloc[first_non_zero_idx_pos:, :] = (
    #     MarketChange.iloc[first_non_zero_idx_pos:, :].groupby(group_idx).apply(process_market_change))
    # PositionData = PositionData * MarketChange_w

    ###Calculate Return###
    DailyContribution = PositionData * MarketChange
    DailyReturn = DailyContribution.sum(axis=1)

    NAV = pd.DataFrame(index=DailyReturn.index)
    # Calculate the cumulative return without compounding
    NAV['NonCompounded'] = DailyReturn.cumsum() + 1
    # Calculate the cumulative return of compound interest
    NAV['Compounded'] = (1 + DailyReturn).cumprod()

    return NAV, DailyContribution, DailyReturn


# %%Time Series Roll Factor(展期收益-时序)
def factor_calculate_rollTS(MarketChange, FactorData, UniverseData
                            , DC_Close, SDC_Close, DC_DaysUntil, SDC_DaysUntil
                            , Period, Leverage, HoldingPeriod):
    print('时间序列展期收益')

    FactorData1 = (np.log(DC_Close) - np.log(SDC_Close)) * 365 / (SDC_DaysUntil - DC_DaysUntil)

    FactorData1 = FactorData1.shift(1)

    PositionData = FactorData1.copy()
    valid_row_sum = PositionData.where(UniverseData).abs().sum(axis=1)
    PositionData = PositionData.div(valid_row_sum, axis=0)
    PositionData = PositionData.where(UniverseData, 0).fillna(0) * Leverage

    # Update every HoldingPeriod
    # # Maintain Weight
    # Find the row index of the first non-zero value
    first_non_zero_idx = (PositionData != 0).any(axis=1).idxmax()
    # Convert to integer index position
    first_non_zero_idx_pos = PositionData.index.get_loc(first_non_zero_idx)
    # Starting from the first non-zero value row, perform grouping operations
    PositionData1 = PositionData.iloc[first_non_zero_idx_pos:, :]
    group_idx = np.arange(len(PositionData1)) // HoldingPeriod
    PositionData.iloc[first_non_zero_idx_pos:, :] = PositionData1.groupby(group_idx).transform('first')

    # # Maintain Position
    # MarketChange_w = MarketChange.copy()
    #
    # # Calculates the cumulative product within the group and lags one row, filling the first row with 1
    # def process_market_change(df):
    #     # Add 1 to each group and calculate the cumulative product
    #     df = (df + 1).cumprod()
    #     # Lag one row, fill the first row with 1
    #     df = df.shift(1).fillna(1)
    #     return df
    #
    # MarketChange_w.iloc[first_non_zero_idx_pos:, :] = (
    #     MarketChange.iloc[first_non_zero_idx_pos:, :].groupby(group_idx).apply(process_market_change))
    # PositionData = PositionData * MarketChange_w

    ###Calculate Return###
    DailyContribution = PositionData * MarketChange
    DailyReturn = DailyContribution.sum(axis=1)

    NAV = pd.DataFrame(index=DailyReturn.index)
    # Calculate the cumulative return without compounding
    NAV['NonCompounded'] = DailyReturn.cumsum() + 1
    # Calculate the cumulative return of compound interest
    NAV['Compounded'] = (1 + DailyReturn).cumprod()

    return NAV, DailyContribution, DailyReturn

# %%Cross Sectional Roll Factor(展期收益-面板)
def factor_calculate_rollPanel(MarketChange, FactorData, UniverseData
                               , DC_Close, SDC_Close, DC_DaysUntil, SDC_DaysUntil
                               , Period, Leverage, HoldingPeriod):
    print('横截面展期收益')
    rollP = (np.log(DC_Close) - np.log(SDC_Close)) * 365 / (SDC_DaysUntil - DC_DaysUntil)

    ###Convert factors to cross-sectional###
    rollP = rollP.shift(1)
    Factor_filtered = rollP.where(UniverseData)
    # Calculate quantile
    quantiles = Factor_filtered.quantile([0.5], axis=1)
    FactorData_adjusted = pd.DataFrame(0, index=rollP.index, columns=rollP.columns)
    # Set the first 20% of elements to 1 (long)
    FactorData_adjusted[Factor_filtered > quantiles.loc[0.5, :].values[:, None]] = 1
    # Set the bottom 20% of elements to -1 (empty)
    FactorData_adjusted[Factor_filtered <= quantiles.loc[0.5, :].values[:, None]] = -1

    PositionData = FactorData_adjusted.copy()
    valid_row_sum = PositionData.where(UniverseData).abs().sum(axis=1)
    PositionData = PositionData.div(valid_row_sum, axis=0)
    PositionData = PositionData.where(UniverseData, 0).fillna(0) * Leverage
    # check：PositionData.sum(axis=1) PositionData.abs().sum(axis=1)

    # Update every HoldingPeriod
    # # Maintain Weight
    # Find the row index of the first non-zero value
    first_non_zero_idx = (PositionData != 0).any(axis=1).idxmax()
    # Convert to integer index position
    first_non_zero_idx_pos = PositionData.index.get_loc(first_non_zero_idx)
    # Starting from the first non-zero value row, perform grouping operations
    PositionData1 = PositionData.iloc[first_non_zero_idx_pos:, :]
    group_idx = np.arange(len(PositionData1)) // HoldingPeriod
    PositionData.iloc[first_non_zero_idx_pos:, :] = PositionData1.groupby(group_idx).transform('first')

    # # Maintain Position
    # MarketChange_w = MarketChange.copy()
    #
    # # Calculates the cumulative product within the group and lags one row, filling the first row with 1
    # def process_market_change(df):
    #     # Add 1 to each group and calculate the cumulative product
    #     df = (df + 1).cumprod()
    #     # Lag one row, fill the first row with 1
    #     df = df.shift(1).fillna(1)
    #     return df
    #
    # MarketChange_w.iloc[first_non_zero_idx_pos:, :] = (
    #     MarketChange.iloc[first_non_zero_idx_pos:, :].groupby(group_idx).apply(process_market_change))
    # PositionData = PositionData * MarketChange_w

    ###Calculate Return###
    DailyContribution = PositionData * MarketChange
    DailyReturn = DailyContribution.sum(axis=1)

    NAV = pd.DataFrame(index=DailyReturn.index)
    # Calculate the cumulative return without compounding
    NAV['NonCompounded'] = DailyReturn.cumsum() + 1
    # Calculate the cumulative return of compound interest
    NAV['Compounded'] = (1 + DailyReturn).cumprod()

    return NAV, DailyContribution, DailyReturn

# %%Time Series BasisMomentum Factor(基差动量-时序)
def factor_calculate_basisMomentumTS(MarketChange, FactorData, UniverseData
                                     , DC_Close, SDC_Close, DC_DaysUntil, SDC_DaysUntil
                                     , Period, Leverage, HoldingPeriod):
    print('时间序列基差动量')

    DC_return = DC_Close / DC_Close.shift(Period) - 1
    SDC_return = SDC_Close / SDC_Close.shift(Period) - 1
    FactorData1 = DC_return - SDC_return

    FactorData1 = FactorData1.shift(1)

    PositionData = FactorData1.copy()
    valid_row_sum = PositionData.where(UniverseData).abs().sum(axis=1)
    PositionData = PositionData.div(valid_row_sum, axis=0)
    PositionData = PositionData.where(UniverseData, 0).fillna(0) * Leverage

    # Update every HoldingPeriod
    # # Maintain Weight
    # Find the row index of the first non-zero value
    first_non_zero_idx = (PositionData != 0).any(axis=1).idxmax()
    # Convert to integer index position
    first_non_zero_idx_pos = PositionData.index.get_loc(first_non_zero_idx)
    # Starting from the first non-zero value row, perform grouping operations
    PositionData1 = PositionData.iloc[first_non_zero_idx_pos:, :]
    group_idx = np.arange(len(PositionData1)) // HoldingPeriod
    PositionData.iloc[first_non_zero_idx_pos:, :] = PositionData1.groupby(group_idx).transform('first')

    # # Maintain Position
    # MarketChange_w = MarketChange.copy()
    #
    # # Calculates the cumulative product within the group and lags one row, filling the first row with 1
    # def process_market_change(df):
    #     # Add 1 to each group and calculate the cumulative product
    #     df = (df + 1).cumprod()
    #     # Lag one row, fill the first row with 1
    #     df = df.shift(1).fillna(1)
    #     return df
    #
    # MarketChange_w.iloc[first_non_zero_idx_pos:, :] = (
    #     MarketChange.iloc[first_non_zero_idx_pos:, :].groupby(group_idx).apply(process_market_change))
    # PositionData = PositionData * MarketChange_w

    ###Calculate Return###
    DailyContribution = PositionData * MarketChange
    DailyReturn = DailyContribution.sum(axis=1)

    NAV = pd.DataFrame(index=DailyReturn.index)
    # Calculate the cumulative return without compounding
    NAV['NonCompounded'] = DailyReturn.cumsum() + 1
    # Calculate the cumulative return of compound interest
    NAV['Compounded'] = (1 + DailyReturn).cumprod()

    return NAV, DailyContribution, DailyReturn

# %%Warehouse Factor(仓单因子)
def factor_calculate_warehouse(MarketChange, FactorData, UniverseData
                               , Warehouse
                               , Period, Leverage, HoldingPeriod):
    print('仓单因子')

    # 找到A和B中都存在的列名
    common_columns = UniverseData.columns.intersection(Warehouse.columns)

    MarketChange_filtered = MarketChange[common_columns]
    FactorData_filtered = FactorData[common_columns]
    UniverseData_filtered = UniverseData[common_columns]

    FactorData1 = 1 - calculate_rolling_mean(Warehouse, Period, 0) / calculate_rolling_mean(Warehouse, 306, 180)
    FactorData1 = FactorData1.shift(1)
    # 将 DataFrame 中的 Inf 值替换为 0
    FactorData1.replace([np.inf, -np.inf], 0, inplace=True)

    PositionData = FactorData1.copy()
    PositionData = -PositionData

    valid_row_sum = PositionData.where(UniverseData_filtered).abs().sum(axis=1)
    PositionData = PositionData.div(valid_row_sum, axis=0)
    PositionData = PositionData.where(UniverseData_filtered, 0).fillna(0) * Leverage

    # Update every HoldingPeriod
    # # Maintain Weight
    # Find the row index of the first non-zero value
    first_non_zero_idx = (PositionData != 0).any(axis=1).idxmax()
    # Convert to integer index position
    first_non_zero_idx_pos = PositionData.index.get_loc(first_non_zero_idx)
    # Starting from the first non-zero value row, perform grouping operations
    PositionData1 = PositionData.iloc[first_non_zero_idx_pos:, :]
    group_idx = np.arange(len(PositionData1)) // HoldingPeriod
    PositionData.iloc[first_non_zero_idx_pos:, :] = PositionData1.groupby(group_idx).transform('first')

    # # Maintain Position
    # MarketChange_w = MarketChange.copy()
    #
    # # Calculates the cumulative product within the group and lags one row, filling the first row with 1
    # def process_market_change(df):
    #     # Add 1 to each group and calculate the cumulative product
    #     df = (df + 1).cumprod()
    #     # Lag one row, fill the first row with 1
    #     df = df.shift(1).fillna(1)
    #     return df
    #
    # MarketChange_w.iloc[first_non_zero_idx_pos:, :] = (
    #     MarketChange.iloc[first_non_zero_idx_pos:, :].groupby(group_idx).apply(process_market_change))
    # PositionData = PositionData * MarketChange_w

    DailyContribution = PositionData * MarketChange_filtered
    DailyReturn = DailyContribution.sum(axis=1)

    NAV = pd.DataFrame(index=DailyReturn.index)
    # Calculate the cumulative return without compounding
    NAV['NonCompounded'] = DailyReturn.cumsum() + 1
    # Calculate the cumulative return of compound interest
    NAV['Compounded'] = (1 + DailyReturn).cumprod()
    # return 0
    return NAV, DailyContribution, DailyReturn


# %%Cross Sectional Volatility Factor(波动率因子-截面)
def factor_calculate_volatilityPanel(MarketChange, FactorData, UniverseData
                                     , Period, Leverage, HoldingPeriod=1):
    print('横截面波动率因子')

    returns = FactorData / FactorData.shift() - 1
    vol = returns.rolling(Period).std()
    vol = vol.shift(1)

    ###Convert factors to cross-sectional###
    Factor_filtered = vol.where(UniverseData)
    # Calculate quantile
    quantiles = Factor_filtered.quantile([0.2, 0.8], axis=1)
    FactorData_adjusted = pd.DataFrame(0, index=vol.index, columns=vol.columns)
    # Set the first 20% of elements to 1 (long)
    FactorData_adjusted[Factor_filtered > quantiles.loc[0.8, :].values[:, None]] = 1
    # Set the bottom 20% of elements to -1 (empty)
    FactorData_adjusted[Factor_filtered <= quantiles.loc[0.2, :].values[:, None]] = -1

    PositionData = FactorData_adjusted.copy()
    valid_row_sum = PositionData.where(UniverseData).abs().sum(axis=1)
    PositionData = PositionData.div(valid_row_sum, axis=0)
    PositionData = PositionData.where(UniverseData, 0).fillna(0) * Leverage
    # check：PositionData.sum(axis=1) PositionData.abs().sum(axis=1)

    # Update every HoldingPeriod
    # # Maintain Weight
    # Find the row index of the first non-zero value
    first_non_zero_idx = (PositionData != 0).any(axis=1).idxmax()
    # Convert to integer index position
    first_non_zero_idx_pos = PositionData.index.get_loc(first_non_zero_idx)
    # Starting from the first non-zero value row, perform grouping operations
    PositionData1 = PositionData.iloc[first_non_zero_idx_pos:, :]
    group_idx = np.arange(len(PositionData1)) // HoldingPeriod
    PositionData.iloc[first_non_zero_idx_pos:, :] = PositionData1.groupby(group_idx).transform('first')

    # # Maintain Position
    # MarketChange_w = MarketChange.copy()
    # # Calculates the cumulative product within the group and lags one row, filling the first row with 1
    # def process_market_change(df):
    #     # Add 1 to each group and calculate the cumulative product
    #     df = (df + 1).cumprod()
    #     # Lag one row, fill the first row with 1
    #     df = df.shift(1).fillna(1)
    #     return df
    # MarketChange_w.iloc[first_non_zero_idx_pos:, :] = (MarketChange.iloc[first_non_zero_idx_pos:, :].groupby(group_idx).apply(process_market_change))
    # PositionData = PositionData * MarketChange_w

    ###Calculate Return###
    DailyContribution = PositionData * MarketChange
    DailyReturn = DailyContribution.sum(axis=1)

    NAV = pd.DataFrame(index=DailyReturn.index)
    # Calculate the cumulative return without compounding
    NAV['NonCompounded'] = DailyReturn.cumsum() + 1
    # Calculate the cumulative return of compound interest
    NAV['Compounded'] = (1 + DailyReturn).cumprod()

    return NAV, DailyContribution, DailyReturn


# %%Cross Sectional CV Factor(收益率变异系数因子-截面)
def factor_calculate_CVPanel(MarketChange, FactorData, UniverseData
                             , Period, Leverage, HoldingPeriod=1):
    print('横截面收益率变异系数')

    returns = pd.DataFrame(FactorData / FactorData.shift(1) - 1)
    CV = returns.rolling(Period).var() / returns.rolling(Period).mean().abs()
    CV = CV.shift(1)

    ###Convert factors to cross-sectional###
    Factor_filtered = CV.where(UniverseData)
    # Calculate quantile
    quantiles = Factor_filtered.quantile([0.2, 0.8], axis=1)
    FactorData_adjusted = pd.DataFrame(0, index=CV.index, columns=CV.columns)
    # Set the first 20% of elements to 1 (long)
    FactorData_adjusted[Factor_filtered > quantiles.loc[0.8, :].values[:, None]] = 1
    # Set the bottom 20% of elements to -1 (empty)
    FactorData_adjusted[Factor_filtered <= quantiles.loc[0.2, :].values[:, None]] = -1

    PositionData = FactorData_adjusted.copy()
    valid_row_sum = PositionData.where(UniverseData).abs().sum(axis=1)
    PositionData = PositionData.div(valid_row_sum, axis=0)
    PositionData = PositionData.where(UniverseData, 0).fillna(0) * Leverage
    # check：PositionData.sum(axis=1) PositionData.abs().sum(axis=1)

    # Update every HoldingPeriod
    # # Maintain Weight
    # Find the row index of the first non-zero value
    first_non_zero_idx = (PositionData != 0).any(axis=1).idxmax()
    # Convert to integer index position
    first_non_zero_idx_pos = PositionData.index.get_loc(first_non_zero_idx)
    # Starting from the first non-zero value row, perform grouping operations
    PositionData1 = PositionData.iloc[first_non_zero_idx_pos:, :]
    group_idx = np.arange(len(PositionData1)) // HoldingPeriod
    PositionData.iloc[first_non_zero_idx_pos:, :] = PositionData1.groupby(group_idx).transform('first')

    # # Maintain Position
    # MarketChange_w = MarketChange.copy()
    #
    # # Calculates the cumulative product within the group and lags one row, filling the first row with 1
    # def process_market_change(df):
    #     # Add 1 to each group and calculate the cumulative product
    #     df = (df + 1).cumprod()
    #     # Lag one row, fill the first row with 1
    #     df = df.shift(1).fillna(1)
    #     return df
    #
    # MarketChange_w.iloc[first_non_zero_idx_pos:, :] = (
    #     MarketChange.iloc[first_non_zero_idx_pos:, :].groupby(group_idx).apply(process_market_change))
    # PositionData = PositionData * MarketChange_w

    ###Calculate Return###
    DailyContribution = PositionData * MarketChange
    DailyReturn = DailyContribution.sum(axis=1)

    NAV = pd.DataFrame(index=DailyReturn.index)
    # Calculate the cumulative return without compounding
    NAV['NonCompounded'] = DailyReturn.cumsum() + 1
    # Calculate the cumulative return of compound interest
    NAV['Compounded'] = (1 + DailyReturn).cumprod()

    return NAV, DailyContribution, DailyReturn


# %%Cross Sectional Skew Factor(偏度因子-截面)
def factor_calculate_skewPanel(MarketChange, FactorData, UniverseData
                               , Period, Leverage, HoldingPeriod=1):
    print('横截面偏度因子')

    # skew = (FactorData / FactorData.shift() - 1).rolling(Period).apply(lambda x: stats.skew(x, bias=False), raw=True)# 太慢
    ret = FactorData / FactorData.shift(1) - 1
    mean_ret = ret.rolling(Period).mean()
    std_ret = ret.rolling(Period).std()
    skew = ((ret - mean_ret) / std_ret) ** 3
    skew = skew.rolling(Period).mean()
    skew = skew.shift(1)

    ###Convert factors to cross-sectional###
    Factor_filtered = skew.where(UniverseData)
    # Calculate quantile
    quantiles = Factor_filtered.quantile([0.2, 0.8], axis=1)
    FactorData_adjusted = pd.DataFrame(0, index=skew.index, columns=skew.columns)
    # Set the first 20% of elements to -1 (empty)
    FactorData_adjusted[Factor_filtered > quantiles.loc[0.8, :].values[:, None]] = -1
    # Set the bottom 20% of elements to 1 (long)
    FactorData_adjusted[Factor_filtered <= quantiles.loc[0.2, :].values[:, None]] = 1

    PositionData = FactorData_adjusted.copy()
    valid_row_sum = PositionData.where(UniverseData).abs().sum(axis=1)
    PositionData = PositionData.div(valid_row_sum, axis=0)
    PositionData = PositionData.where(UniverseData, 0).fillna(0) * Leverage
    # check：PositionData.sum(axis=1) PositionData.abs().sum(axis=1)

    # Update every HoldingPeriod
    # # Maintain Weight
    # Find the row index of the first non-zero value
    first_non_zero_idx = (PositionData != 0).any(axis=1).idxmax()
    # Convert to integer index position
    first_non_zero_idx_pos = PositionData.index.get_loc(first_non_zero_idx)
    # Starting from the first non-zero value row, perform grouping operations
    PositionData1 = PositionData.iloc[first_non_zero_idx_pos:, :]
    group_idx = np.arange(len(PositionData1)) // HoldingPeriod
    PositionData.iloc[first_non_zero_idx_pos:, :] = PositionData1.groupby(group_idx).transform('first')

    # # Maintain Position
    # MarketChange_w = MarketChange.copy()
    #
    # # Calculates the cumulative product within the group and lags one row, filling the first row with 1
    # def process_market_change(df):
    #     # Add 1 to each group and calculate the cumulative product
    #     df = (df + 1).cumprod()
    #     # Lag one row, fill the first row with 1
    #     df = df.shift(1).fillna(1)
    #     return df
    #
    # MarketChange_w.iloc[first_non_zero_idx_pos:, :] = (
    #     MarketChange.iloc[first_non_zero_idx_pos:, :].groupby(group_idx).apply(process_market_change))
    # PositionData = PositionData * MarketChange_w

    ###Calculate Return###
    DailyContribution = PositionData * MarketChange
    DailyReturn = DailyContribution.sum(axis=1)

    NAV = pd.DataFrame(index=DailyReturn.index)
    # Calculate the cumulative return without compounding
    NAV['NonCompounded'] = DailyReturn.cumsum() + 1
    # Calculate the cumulative return of compound interest
    NAV['Compounded'] = (1 + DailyReturn).cumprod()

    return NAV, DailyContribution, DailyReturn


# %%Cross Sectional Kurtosis Factor(峰度因子-截面)
def factor_calculate_KurtosisPanel(MarketChange, FactorData, UniverseData
                                   , Period, Leverage, HoldingPeriod=1):
    print('横截面峰度因子')

    # kurtosis = (FactorData / FactorData.shift() - 1).rolling(Period).apply(lambda x:  stats.kurtosis(x, bias=False), raw=True)# 太慢
    ret = FactorData / FactorData.shift(1) - 1
    mean_ret = ret.rolling(Period).mean()
    std_ret = ret.rolling(Period).std()
    kurtosis = ((ret - mean_ret) / std_ret) ** 4
    kurtosis = kurtosis.rolling(Period).mean() - 3
    kurtosis = kurtosis.shift(1)

    ###Convert factors to cross-sectional###
    Factor_filtered = kurtosis.where(UniverseData)
    # Calculate quantile
    quantiles = Factor_filtered.quantile([0.2, 0.8], axis=1)
    FactorData_adjusted = pd.DataFrame(0, index=kurtosis.index, columns=kurtosis.columns)
    # Set the first 20% of elements to -1 (empty)
    FactorData_adjusted[Factor_filtered > quantiles.loc[0.8, :].values[:, None]] = -1
    # Set the bottom 20% of elements to 1 (long)
    FactorData_adjusted[Factor_filtered <= quantiles.loc[0.2, :].values[:, None]] = 1

    PositionData = FactorData_adjusted.copy()
    valid_row_sum = PositionData.where(UniverseData).abs().sum(axis=1)
    PositionData = PositionData.div(valid_row_sum, axis=0)
    PositionData = PositionData.where(UniverseData, 0).fillna(0) * Leverage
    # check：PositionData.sum(axis=1) PositionData.abs().sum(axis=1)

    # Update every HoldingPeriod
    # # Maintain Weight
    # Find the row index of the first non-zero value
    first_non_zero_idx = (PositionData != 0).any(axis=1).idxmax()
    # Convert to integer index position
    first_non_zero_idx_pos = PositionData.index.get_loc(first_non_zero_idx)
    # Starting from the first non-zero value row, perform grouping operations
    PositionData1 = PositionData.iloc[first_non_zero_idx_pos:, :]
    group_idx = np.arange(len(PositionData1)) // HoldingPeriod
    PositionData.iloc[first_non_zero_idx_pos:, :] = PositionData1.groupby(group_idx).transform('first')

    # # Maintain Position
    # MarketChange_w = MarketChange.copy()
    #
    # # Calculates the cumulative product within the group and lags one row, filling the first row with 1
    # def process_market_change(df):
    #     # Add 1 to each group and calculate the cumulative product
    #     df = (df + 1).cumprod()
    #     # Lag one row, fill the first row with 1
    #     df = df.shift(1).fillna(1)
    #     return df
    #
    # MarketChange_w.iloc[first_non_zero_idx_pos:, :] = (
    #     MarketChange.iloc[first_non_zero_idx_pos:, :].groupby(group_idx).apply(process_market_change))
    # PositionData = PositionData * MarketChange_w

    ###Calculate Return###
    DailyContribution = PositionData * MarketChange
    DailyReturn = DailyContribution.sum(axis=1)

    NAV = pd.DataFrame(index=DailyReturn.index)
    # Calculate the cumulative return without compounding
    NAV['NonCompounded'] = DailyReturn.cumsum() + 1
    # Calculate the cumulative return of compound interest
    NAV['Compounded'] = (1 + DailyReturn).cumprod()

    return NAV, DailyContribution, DailyReturn


# %%Cross Sectional Amplitude Factor(振幅因子-截面)
def factor_calculate_AmplitudePanel(MarketChange, FactorData, UniverseData,
                                    Period, Highp, Leverage, HoldingPeriod=1):
    print('横截面振幅因子')

    Amplitude = ((Highp - FactorData) / FactorData).rolling(Period).mean()

    H_minus_L = Amplitude.copy()

    H_minus_L = H_minus_L.shift(1)

    ###Convert factors to cross-sectional###
    Factor_filtered = H_minus_L.where(UniverseData)
    # Calculate quantile
    quantiles = Factor_filtered.quantile([0.2, 0.8], axis=1)
    FactorData_adjusted = pd.DataFrame(0, index=H_minus_L.index, columns=H_minus_L.columns)
    # Set the first 20% of elements to 1 (long)
    FactorData_adjusted[Factor_filtered > quantiles.loc[0.8, :].values[:, None]] = 1
    # Set the bottom 20% of elements to -1 (empty)
    FactorData_adjusted[Factor_filtered <= quantiles.loc[0.2, :].values[:, None]] = -1

    PositionData = FactorData_adjusted.copy()
    valid_row_sum = PositionData.where(UniverseData).abs().sum(axis=1)
    PositionData = PositionData.div(valid_row_sum, axis=0)
    PositionData = PositionData.where(UniverseData, 0).fillna(0) * Leverage
    # check：PositionData.sum(axis=1) PositionData.abs().sum(axis=1)

    # Update every HoldingPeriod
    # # Maintain Weight
    # Find the row index of the first non-zero value
    first_non_zero_idx = (PositionData != 0).any(axis=1).idxmax()
    # Convert to integer index position
    first_non_zero_idx_pos = PositionData.index.get_loc(first_non_zero_idx)
    # Starting from the first non-zero value row, perform grouping operations
    PositionData1 = PositionData.iloc[first_non_zero_idx_pos:, :]
    group_idx = np.arange(len(PositionData1)) // HoldingPeriod
    PositionData.iloc[first_non_zero_idx_pos:, :] = PositionData1.groupby(group_idx).transform('first')

    # # Maintain Position
    # MarketChange_w = MarketChange.copy()
    #
    # # Calculates the cumulative product within the group and lags one row, filling the first row with 1
    # def process_market_change(df):
    #     # Add 1 to each group and calculate the cumulative product
    #     df = (df + 1).cumprod()
    #     # Lag one row, fill the first row with 1
    #     df = df.shift(1).fillna(1)
    #     return df
    #
    # MarketChange_w.iloc[first_non_zero_idx_pos:, :] = (
    #     MarketChange.iloc[first_non_zero_idx_pos:, :].groupby(group_idx).apply(process_market_change))
    # PositionData = PositionData * MarketChange_w

    ###Calculate Return###
    DailyContribution = PositionData * MarketChange
    DailyReturn = DailyContribution.sum(axis=1)

    NAV = pd.DataFrame(index=DailyReturn.index)
    # Calculate the cumulative return without compounding
    NAV['NonCompounded'] = DailyReturn.cumsum() + 1
    # Calculate the cumulative return of compound interest
    NAV['Compounded'] = (1 + DailyReturn).cumprod()

    return NAV, DailyContribution, DailyReturn


# %%Cross Sectional Liquidity Factor(流动性因子-截面)
def factor_calculate_liquidityPanel(MarketChange, FactorData, UniverseData
                                    , Period, Volume, Leverage, HoldingPeriod=1):
    print('横截面流动性因子')

    Liq = np.abs((Volume / FactorData.pct_change())).rolling(Period).mean()
    Liq = Liq.shift(1)
    ###Convert factors to cross-sectional###
    Factor_filtered = Liq.where(UniverseData)
    # Calculate quantile
    quantiles = Factor_filtered.quantile([0.5], axis=1)
    FactorData_adjusted = pd.DataFrame(0, index=Liq.index, columns=Liq.columns)
    # Set the first 20% of elements to -1 (empty)
    FactorData_adjusted[Factor_filtered > quantiles.loc[0.5, :].values[:, None]] = -1
    # Set the bottom 20% of elements to 1 (long)
    FactorData_adjusted[Factor_filtered <= quantiles.loc[0.5, :].values[:, None]] = 1

    PositionData = FactorData_adjusted.copy()
    valid_row_sum = PositionData.where(UniverseData).abs().sum(axis=1)
    PositionData = PositionData.div(valid_row_sum, axis=0)
    PositionData = PositionData.where(UniverseData, 0).fillna(0) * Leverage
    # check：PositionData.sum(axis=1) PositionData.abs().sum(axis=1)

    # Update every HoldingPeriod
    # # Maintain Weight
    # Find the row index of the first non-zero value
    first_non_zero_idx = (PositionData != 0).any(axis=1).idxmax()
    # Convert to integer index position
    first_non_zero_idx_pos = PositionData.index.get_loc(first_non_zero_idx)
    # Starting from the first non-zero value row, perform grouping operations
    PositionData1 = PositionData.iloc[first_non_zero_idx_pos:, :]
    group_idx = np.arange(len(PositionData1)) // HoldingPeriod
    PositionData.iloc[first_non_zero_idx_pos:, :] = PositionData1.groupby(group_idx).transform('first')

    # # Maintain Position
    # MarketChange_w = MarketChange.copy()
    #
    # # Calculates the cumulative product within the group and lags one row, filling the first row with 1
    # def process_market_change(df):
    #     # Add 1 to each group and calculate the cumulative product
    #     df = (df + 1).cumprod()
    #     # Lag one row, fill the first row with 1
    #     df = df.shift(1).fillna(1)
    #     return df
    #
    # MarketChange_w.iloc[first_non_zero_idx_pos:, :] = (
    #     MarketChange.iloc[first_non_zero_idx_pos:, :].groupby(group_idx).apply(process_market_change))
    # PositionData = PositionData * MarketChange_w

    ###Calculate Return###
    DailyContribution = PositionData * MarketChange
    DailyReturn = DailyContribution.sum(axis=1)

    NAV = pd.DataFrame(index=DailyReturn.index)
    # Calculate the cumulative return without compounding
    NAV['NonCompounded'] = DailyReturn.cumsum() + 1
    # Calculate the cumulative return of compound interest
    NAV['Compounded'] = (1 + DailyReturn).cumprod()

    return NAV, DailyContribution, DailyReturn


# %%MutiMomentum Factor(Absolute Ranking of Assets)(复合动量因子(资产的绝对排名))
def factor_calculate_Mutimom(MarketChange, FactorData, UniverseData
                                    , Period, Leverage, HoldingPeriod=1):
    print('复合动量因子')

    FactorData = FactorData / FactorData.shift(Period) - 1
    FactorData = FactorData.shift(1)

    Factor_filtered = FactorData.where(UniverseData)
    FactorData_adjusted = pd.DataFrame(0, index=FactorData.index, columns=FactorData.columns)

    # Get the conditions for positive values and assign them
    positive_condition = Factor_filtered > 0
    positive_quantile = Factor_filtered[positive_condition].quantile(0.5, axis=1)
    for date, threshold in positive_quantile.items():
        # The first 40% is set to 1
        FactorData_adjusted.loc[date] += np.where(
            (positive_condition.loc[date]) & (Factor_filtered.loc[date] > threshold),1,0)

    # Get the conditions for negative values and assign them
    negative_condition = Factor_filtered < 0
    negative_quantile = Factor_filtered[negative_condition].quantile(0.5, axis=1)
    for date, threshold in negative_quantile.items():
        # The last 40% is set to -1
        FactorData_adjusted.loc[date] += np.where(
            (negative_condition.loc[date]) & (Factor_filtered.loc[date] < threshold),-1, 0)

    PositionData = FactorData_adjusted.copy()
    valid_row_sum = PositionData.where(UniverseData).abs().sum(axis=1)
    PositionData = PositionData.div(valid_row_sum, axis=0)
    PositionData = PositionData.where(UniverseData, 0).fillna(0) * Leverage
    # check：PositionData.sum(axis=1) PositionData.abs().sum(axis=1)

    # Update every HoldingPeriod
    # # Maintain Weight
    # Find the row index of the first non-zero value
    first_non_zero_idx = (PositionData != 0).any(axis=1).idxmax()
    # Convert to integer index position
    first_non_zero_idx_pos = PositionData.index.get_loc(first_non_zero_idx)
    # Starting from the first non-zero value row, perform grouping operations
    PositionData1 = PositionData.iloc[first_non_zero_idx_pos:, :]
    group_idx = np.arange(len(PositionData1)) // HoldingPeriod
    PositionData.iloc[first_non_zero_idx_pos:, :] = PositionData1.groupby(group_idx).transform('first')

    # # Maintain Position
    # MarketChange_w = MarketChange.copy()
    #
    # # Calculates the cumulative product within the group and lags one row, filling the first row with 1
    # def process_market_change(df):
    #     # Add 1 to each group and calculate the cumulative product
    #     df = (df + 1).cumprod()
    #     # Lag one row, fill the first row with 1
    #     df = df.shift(1).fillna(1)
    #     return df
    #
    # MarketChange_w.iloc[first_non_zero_idx_pos:, :] = (
    #     MarketChange.iloc[first_non_zero_idx_pos:, :].groupby(group_idx).apply(process_market_change))
    # PositionData = PositionData * MarketChange_w

    ###Calculate Return###
    DailyContribution = PositionData * MarketChange
    DailyReturn = DailyContribution.sum(axis=1)

    NAV = pd.DataFrame(index=DailyReturn.index)
    # Calculate the cumulative return without compounding
    NAV['NonCompounded'] = DailyReturn.cumsum() + 1
    # Calculate the cumulative return of compound interest
    NAV['Compounded'] = (1 + DailyReturn).cumprod()

    return NAV, DailyContribution, DailyReturn


# %%Bias Factor(乖离率因子)
def factor_calculate_Bias(MarketChange, FactorData, UniverseData
                             , Period, Leverage, HoldingPeriod=1):
    print('乖离率因子')

    ret = FactorData.pct_change()
    Bias = (FactorData - FactorData.rolling(Period).mean()) / (ret.rolling(Period).std() * FactorData.shift(Period))
    Bias = Bias.shift(1)

    ###Convert factors to cross-sectional###
    Factor_filtered = Bias.where(UniverseData)
    # Calculate quantile
    quantiles = Factor_filtered.quantile([0.2,0.8], axis=1)
    FactorData_adjusted = pd.DataFrame(0, index=Bias.index, columns=Bias.columns)
    # Set the first 20% of elements to 1 (long)
    FactorData_adjusted[Factor_filtered > quantiles.loc[0.8, :].values[:, None]] = 1
    # Set the bottom 20% of elements to -1 (empty)
    FactorData_adjusted[Factor_filtered <= quantiles.loc[0.2, :].values[:, None]] = -1

    PositionData = FactorData_adjusted.copy()
    valid_row_sum = PositionData.where(UniverseData).abs().sum(axis=1)
    PositionData = PositionData.div(valid_row_sum, axis=0)
    PositionData = PositionData.where(UniverseData, 0).fillna(0) * Leverage
    # check：PositionData.sum(axis=1) PositionData.abs().sum(axis=1)

    # Update every HoldingPeriod
    # # Maintain Weight
    # Find the row index of the first non-zero value
    first_non_zero_idx = (PositionData != 0).any(axis=1).idxmax()
    # Convert to integer index position
    first_non_zero_idx_pos = PositionData.index.get_loc(first_non_zero_idx)
    # Starting from the first non-zero value row, perform grouping operations
    PositionData1 = PositionData.iloc[first_non_zero_idx_pos:, :]
    group_idx = np.arange(len(PositionData1)) // HoldingPeriod
    PositionData.iloc[first_non_zero_idx_pos:, :] = PositionData1.groupby(group_idx).transform('first')

    # # Maintain Position
    # MarketChange_w = MarketChange.copy()
    #
    # # Calculates the cumulative product within the group and lags one row, filling the first row with 1
    # def process_market_change(df):
    #     # Add 1 to each group and calculate the cumulative product
    #     df = (df + 1).cumprod()
    #     # Lag one row, fill the first row with 1
    #     df = df.shift(1).fillna(1)
    #     return df
    #
    # MarketChange_w.iloc[first_non_zero_idx_pos:, :] = (
    #     MarketChange.iloc[first_non_zero_idx_pos:, :].groupby(group_idx).apply(process_market_change))
    # PositionData = PositionData * MarketChange_w

    ###Calculate Return###
    DailyContribution = PositionData * MarketChange
    DailyReturn = DailyContribution.sum(axis=1)

    NAV = pd.DataFrame(index=DailyReturn.index)
    # Calculate the cumulative return without compounding
    NAV['NonCompounded'] = DailyReturn.cumsum() + 1
    # Calculate the cumulative return of compound interest
    NAV['Compounded'] = (1 + DailyReturn).cumprod()

    return NAV, DailyContribution, DailyReturn

# %%Trend Factor(趋势强度因子)
def factor_calculate_Trend(MarketChange, FactorData, UniverseData
                             , Period, Leverage, HoldingPeriod=1):
    print('趋势强度因子')

    Displace = np.abs(FactorData - FactorData.shift(1))
    Trend = (FactorData - FactorData.shift(Period)) / Displace.rolling(Period).sum()

    Trend = Trend.shift(1)

    ###Convert factors to cross-sectional###
    Factor_filtered = Trend.where(UniverseData)
    # Calculate quantile
    quantiles = Factor_filtered.quantile([0.2,0.8], axis=1)
    FactorData_adjusted = pd.DataFrame(0, index=Trend.index, columns=Trend.columns)
    # Set the first 20% of elements to 1 (long)
    FactorData_adjusted[Factor_filtered > quantiles.loc[0.8, :].values[:, None]] = 1
    # Set the bottom 20% of elements to -1 (empty)
    FactorData_adjusted[Factor_filtered <= quantiles.loc[0.2, :].values[:, None]] = -1

    PositionData = FactorData_adjusted.copy()
    valid_row_sum = PositionData.where(UniverseData).abs().sum(axis=1)
    PositionData = PositionData.div(valid_row_sum, axis=0)
    PositionData = PositionData.where(UniverseData, 0).fillna(0) * Leverage
    # check：PositionData.sum(axis=1) PositionData.abs().sum(axis=1)

    # Update every HoldingPeriod
    # # Maintain Weight
    # Find the row index of the first non-zero value
    first_non_zero_idx = (PositionData != 0).any(axis=1).idxmax()
    # Convert to integer index position
    first_non_zero_idx_pos = PositionData.index.get_loc(first_non_zero_idx)
    # Starting from the first non-zero value row, perform grouping operations
    PositionData1 = PositionData.iloc[first_non_zero_idx_pos:, :]
    group_idx = np.arange(len(PositionData1)) // HoldingPeriod
    PositionData.iloc[first_non_zero_idx_pos:, :] = PositionData1.groupby(group_idx).transform('first')

    # # Maintain Position
    # MarketChange_w = MarketChange.copy()
    #
    # # Calculates the cumulative product within the group and lags one row, filling the first row with 1
    # def process_market_change(df):
    #     # Add 1 to each group and calculate the cumulative product
    #     df = (df + 1).cumprod()
    #     # Lag one row, fill the first row with 1
    #     df = df.shift(1).fillna(1)
    #     return df
    #
    # MarketChange_w.iloc[first_non_zero_idx_pos:, :] = (
    #     MarketChange.iloc[first_non_zero_idx_pos:, :].groupby(group_idx).apply(process_market_change))
    # PositionData = PositionData * MarketChange_w

    ###Calculate Return###
    DailyContribution = PositionData * MarketChange
    DailyReturn = DailyContribution.sum(axis=1)

    NAV = pd.DataFrame(index=DailyReturn.index)
    # Calculate the cumulative return without compounding
    NAV['NonCompounded'] = DailyReturn.cumsum() + 1
    # Calculate the cumulative return of compound interest
    NAV['Compounded'] = (1 + DailyReturn).cumprod()

    return NAV, DailyContribution, DailyReturn