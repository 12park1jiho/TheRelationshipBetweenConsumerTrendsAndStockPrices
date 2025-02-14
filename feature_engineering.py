# feature_engineering.py
import pandas as pd

def create_lagged_feature(df, feature_name, lags):
    """Lagged Feature를 생성합니다."""
    for lag in lags:
        lagged_col_name = f'{feature_name}_lag{lag}'
        df[lagged_col_name] = df[feature_name].shift(lag)
    return df

def create_rolling_feature(df, feature_name, window, stat_funcs=['mean', 'std']):
    """Rolling Statistics Feature를 생성합니다."""
    for stat_func_name in stat_funcs:
        rolling_col_name = f'{feature_name}_rolling{window}_{stat_func_name}'
        if stat_func_name == 'mean':
            df[rolling_col_name] = df[feature_name].rolling(window=window).mean()
        elif stat_func_name == 'std':
            df[rolling_col_name] = df[feature_name].rolling(window=window).std()
    return df

def create_financial_ratio_feature(df):
    """재무 비율 Feature를 생성합니다."""
    if 'enpTdbtAmt' in df.columns and 'enpTcptAmt' in df.columns:
        df['debt_ratio'] = df['enpTdbtAmt'] / (df['enpTcptAmt'] + 1e-9)  # 분모에 아주 작은 값 더해서 0으로 나누는 것 방지
    if 'currentAsset' in df.columns and 'currentLiability' in df.columns:
        df['current_ratio'] = df['currentAsset'] / (df['currentLiability'] + 1e-9) # 분모에 아주 작은 값 더해서 0으로 나누는 것 방지
    if 'enpBzopPft' in df.columns and 'enpSaleAmt' in df.columns:
        df['profit_margin'] = df['enpBzopPft'] / (df['enpSaleAmt'] + 1e-9) # 분모에 아주 작은 값 더해서 0으로 나누는 것 방지
    return df

# (main.py에서 호출 예시)
# from feature_engineering import create_lagged_feature, create_rolling_feature, create_financial_ratio_feature
# df_stock_monthly = create_lagged_feature(df_stock_monthly, 'clpr', lags=[1, 3])
# df_stock_monthly = create_rolling_feature(df_stock_monthly, 'clpr', window=3)
# financial_data = create_financial_ratio_feature(financial_data)