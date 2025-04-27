#encoding:gbk

import pandas as pd
import talib
import numpy as np
import datetime


class StrategyState:
    pass


A = StrategyState()  # 创建空的类的实例 用来保存委托状态
A.last_buy_iteration = None  # 新增变量，用于记录最后一次买入信号的迭代次数
A.has_bought = False  # 新增变量，用于记录是否已经买入
A.first_buy = True  # 新增变量，用于标记是否为首次买入
A.exit_records = []  # 新增变量，用于记录每次卖出的原因和时间


def init(C):
    """初始化策略"""
    A.stock = C.stockcode + '.' + C.market
    A.acct = '******'  # 账户已脱敏
    A.acct_type = 'STOCK_OPTION'
    A.amount = 10000  # 单笔交易金额
    A.buy_code = 50
    A.sell_code = 51
    A.waiting_list = []
    A.buy_signal_count = 0   # 买入信号累计计数器
    A.iteration = 0          # 数据迭代计数器
    A.buy_signal_logs = []   # 记录每次买入信号的详细信息
    C.set_universe([A.stock])
    print(f'多指标实盘示例 {A.stock} {A.acct} {A.acct_type} 单笔买入金额 {A.amount}')
    A.last_buy_price = None  # 记录最近买入价格
    A.has_bought = False
    A.first_buy = True
    A.exit_records = []      # 重置卖出记录
    
    # 初始化QSDD指标值
    A.long_term_curve_value = None
    A.mid_term_curve_value = None
    A.short_term_curve_value = None


def handlebar(C):
    A.iteration += 1
    print(f"正在处理第 {A.iteration} 次数据迭代")

    account = get_trade_detail_data(A.acct, A.acct_type, 'account')
    if len(account) == 0:
        print(f'账号 {A.acct} 未登录 请检查')
        return
    account = account[0]
    available_cash = int(account.m_dAvailable)
    print(f"可用资金: {available_cash}")

    holdings = get_trade_detail_data(A.acct, A.acct_type, 'position')
    holdings = {i.m_strInstrumentID + '.' + i.m_strExchangeID: i.m_nCanUseVolume for i in holdings}
    print(f"持仓情况: {holdings}")

    required_count = 52
    request_count = required_count + 10

    df = C.get_market_data(
        fields=['open', 'high', 'low', 'close', 'volume'],
        count=request_count,
        skip_paused=True,
        period='1m'
    )

    if df is not None and len(df) >= required_count:
        df['datetime'] = pd.to_datetime(df.index)
        df = df.sort_values('datetime').reset_index(drop=True)
        print(f"成功获取 {len(df)} 条行情数据，数据示例：")
        print(df.head().to_csv(sep='\t', na_rep='nan'))

        df = calculate_technical_indicators(df)
        print("计算技术指标后的数据示例：")
        print(df[['close', 'MACD', 'K', 'RSI', 'OBV', 'MA_short', 'long_term_curve', 'mid_term_curve']].tail().to_csv(sep='\t', na_rep='nan'))

        if pd.notna(df['long_term_curve'].iloc[-1]) and pd.notna(df['mid_term_curve'].iloc[-1]):
            dt = df['datetime'].iloc[-1]
            # 保存当前QSDD指标值到A中，以便get_result函数使用
            A.long_term_curve_value = df['long_term_curve'].iloc[-1]
            A.mid_term_curve_value = df['mid_term_curve'].iloc[-1]
            A.short_term_curve_value = df['short_term_curve'].iloc[-1]
            
            print("最近的 QSDD 指标值:")
            print(f"  时间: {dt}")
            print(f"  Long Term: {df['long_term_curve'].iloc[-1]:.2f}")
            print(f"  Mid Term: {df['mid_term_curve'].iloc[-1]:.2f}")
            print(f"  Short Term: {df['short_term_curve'].iloc[-1]:.2f}")

            buy_signal, sell_signal, exit_reason = generate_signals(df)
            if buy_signal:
                A.buy_signal_count += 1
                signal_info = {
                    'iteration': A.iteration,
                    'datetime': dt,
                    'buy_signal_count': A.buy_signal_count
                }
                A.buy_signal_logs.append(signal_info)
                A.last_buy_iteration = A.iteration  # 记录最后一次买入信号的迭代次数
                A.has_bought = True
                if A.first_buy:
                    A.first_buy = False
                print(f"【第 {A.iteration} 次数据迭代】产生了买入信号: {signal_info}")
            else:
                print(f"【第 {A.iteration} 次数据迭代】本次没有买入信号")
            # 打印累计买入信号次数，无论是否产生新的信号
            print(f"累计买入信号数量: {A.buy_signal_count}")

            execute_trades(C, df, buy_signal, sell_signal, available_cash, holdings, exit_reason)
        else:
            print("警告：最新的技术指标值包含 NaN，无法生成有效信号。")
            print(f"  Long Term NaN: {pd.isna(df['long_term_curve'].iloc[-1])}")
            print(f"  Mid Term NaN: {pd.isna(df['mid_term_curve'].iloc[-1])}")
    elif df is not None:
        print(f"获取行情数据条数不足 ({len(df)} < {required_count})，跳过本次计算。")
    else:
        print("获取行情数据失败")

    if A.last_buy_iteration is not None:
        print(f"最后一次买入信号出现在第 {A.last_buy_iteration} 次数据迭代")
    
    # 如果是最后一次迭代，输出卖出记录统计
    if A.iteration >= 100000:  # 使用足够大的数字或其他条件来判断是否是最后一次迭代
        print_exit_records()


def calculate_technical_indicators(df):
    """计算各种技术指标
    
    注意：本代码中的指标参数已经过脱敏处理
    实际交易时请根据自己的策略调整参数
    """
    # 计算自定义趋势指标
    df = calculate_custom_trend_indicator(df)

    # MACD - 参数已脱敏
    df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = talib.MACD(
        df['close'], 
        fastperiod=12,  # 短期参数
        slowperiod=26,  # 长期参数
        signalperiod=9  # 信号参数
    )

    # KDJ - 参数已脱敏
    df['K'], df['D'] = talib.STOCH(
        df['high'], 
        df['low'], 
        df['close'], 
        fastk_period=9,   # K值周期
        slowk_period=3,   # K值平滑周期
        slowd_period=3    # D值平滑周期
    )
    df['J'] = 3 * df['K'] - 2 * df['D']

    # RSI - 参数已脱敏
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)

    # OBV及其均线
    df['OBV'] = talib.OBV(df['close'], df['volume'])
    df['OBV_MA'] = df['OBV'].rolling(5).mean()  # OBV均线周期已脱敏

    # 移动均线 (MA) - 参数已脱敏
    df['MA_short'] = df['close'].rolling(5).mean()  # 短期均线
    df['MA_long'] = df['close'].rolling(10).mean()   # 长期均线

    return df


def calculate_custom_trend_indicator(df):
    """计算自定义趋势指标
    
    注意：该指标的具体计算逻辑已脱敏
    实际交易时请使用自己的指标计算方法
    """
    # 处理缺失值
    df = df.dropna()

    # 计算自定义趋势指标的三条线
    # 实际计算逻辑已脱敏，这里仅展示框架
    df['long_term_curve'] = calculate_trend_line(df, 'long')   # 长期趋势线
    df['mid_term_curve'] = calculate_trend_line(df, 'mid')     # 中期趋势线
    df['short_term_curve'] = calculate_trend_line(df, 'short') # 短期趋势线

    # 计算各种形态指标
    df['顶部信号'] = calculate_top_signal(df)      # 顶部信号判断
    df['底部信号'] = calculate_bottom_signal(df)   # 底部信号判断
    df['金叉信号'] = calculate_golden_cross(df)    # 金叉信号判断

    return df


def calculate_trend_line(df, term_type):
    """计算趋势线
    
    参数:
        df (DataFrame): 数据框
        term_type (str): 趋势类型 ('long', 'mid', 'short')
    
    返回:
        Series: 趋势线数据
    """
    # 实际计算逻辑已脱敏
    # 这里返回一个示例计算
    if term_type == 'long':
        return df['close'].rolling(20).mean() + 100
    elif term_type == 'mid':
        return df['close'].rolling(10).mean() + 100
    else:  # short
        return df['close'].rolling(5).mean() + 100


def calculate_top_signal(df):
    """计算顶部信号
    实际判断逻辑已脱敏
    """
    return pd.Series(False, index=df.index)


def calculate_bottom_signal(df):
    """计算底部信号
    实际判断逻辑已脱敏
    """
    return pd.Series(False, index=df.index)


def calculate_golden_cross(df):
    """计算金叉信号
    实际判断逻辑已脱敏
    """
    return pd.Series(False, index=df.index)


def generate_signals(df):
    """生成买卖信号
    
    注意：实际信号生成逻辑已脱敏
    这里仅展示框架结构
    """
    last_close_price = df['close'].iloc[-1]

    # 买入信号条件
    buy_signal = check_buy_conditions(df)
    
    # 卖出信号条件
    sell_signal, exit_reason = check_sell_conditions(df)

    return buy_signal, sell_signal, exit_reason


def check_buy_conditions(df):
    """检查买入条件
    实际买入条件已脱敏
    """
    # 示例：检查趋势和技术指标的组合条件
    trend_condition = False  # 趋势条件
    technical_condition = False  # 技术指标条件
    
    if not A.has_bought and A.first_buy:
        return trend_condition and technical_condition
    elif not A.has_bought and not A.first_buy:
        return trend_condition or technical_condition
    return False


def check_sell_conditions(df):
    """检查卖出条件
    实际卖出条件已脱敏
    """
    # 示例：检查止盈止损和技术指标反转
    exit_reason = "未触发卖出"
    sell_signal = False
    
    # 实际判断逻辑已脱敏
    return sell_signal, exit_reason


def execute_trades(C, df, buy_signal, sell_signal, available_cash, holdings, exit_reason):
    stock_code = A.stock
    last_close_price = df['close'].iloc[-1]
    current_time = df['datetime'].iloc[-1]

    if buy_signal:
        # 在成功下单后记录买入价格
        A.last_buy_price = last_close_price  # 新增变量需要先在init中初始化
        # 调整买入数量计算方式，考虑期权合约单位
        # 假设期权合约单位为 10000，你需要根据实际情况修改
        contract_unit = 10000
        vol = int(A.amount / (last_close_price * contract_unit))
        if A.amount < available_cash and vol >= 1:
            msg = f"多指标实盘 {stock_code} 满足买入条件 买入 {vol} 张 @ {last_close_price:.2f}"
            passorder(
                A.buy_code, 1101, A.acct, stock_code, 14, -1, vol, '多指标实盘', 1, msg, C
            )
            print(msg)

        # 可以添加日志记录买入失败的原因
        elif vol < 1:
            print(f"计算买入数量不足1张 ({vol})，无法下单。金额: {A.amount}, 价格: {last_close_price:.2f}")
        elif A.amount >= available_cash:
            print(f"可用资金不足 ({available_cash})，无法下单。需要金额: {A.amount}")

    elif sell_signal:
        if stock_code in holdings and holdings[stock_code] > 0:
            sell_vol = holdings[stock_code]  # 获取可卖数量
            msg = f"多指标实盘 {stock_code} 满足卖出条件 卖出 {sell_vol} 张 @ {last_close_price:.2f}"
            passorder(
                A.sell_code, 1101, A.acct, stock_code, 14, -1, sell_vol, '多指标实盘', 1, msg, C
            )
            print(msg)
            
            # 记录卖出信息
            exit_record = {
                'datetime': current_time,
                'price': last_close_price,
                'buy_price': A.last_buy_price,
                'profit_pct': (last_close_price - A.last_buy_price) / A.last_buy_price * 100 if A.last_buy_price else 0,
                'reason': exit_reason,
                'iteration': A.iteration
            }
            A.exit_records.append(exit_record)
            
            A.has_bought = False  # 卖出后重置买入状态

        # 可以添加日志记录为何不卖出（例如无持仓）
        elif stock_code not in holdings or holdings.get(stock_code, 0) <= 0:
            print(f"满足卖出信号，但无可用持仓 {stock_code}。")


def print_exit_records():
    """在回测结束时输出所有卖出记录和统计信息"""
    if not A.exit_records:
        print("\n=== 回测结束：没有卖出记录 ===")
        return
    
    print("\n========== 卖出记录统计 ==========")
    print(f"总交易次数: {len(A.exit_records)}")
    
    # 按原因分类统计
    reason_stats = {}
    total_profit = 0
    win_count = 0
    loss_count = 0
    
    for record in A.exit_records:
        reason = record['reason']
        profit = record['profit_pct']
        
        if reason not in reason_stats:
            reason_stats[reason] = {'count': 0, 'total_profit': 0, 'wins': 0, 'losses': 0}
        
        reason_stats[reason]['count'] += 1
        reason_stats[reason]['total_profit'] += profit
        
        if profit > 0:
            reason_stats[reason]['wins'] += 1
            win_count += 1
        else:
            reason_stats[reason]['losses'] += 1
            loss_count += 1
            
        total_profit += profit
    
    # 输出总体统计
    print(f"总体盈利次数: {win_count}, 亏损次数: {loss_count}")
    print(f"总体平均收益率: {total_profit/len(A.exit_records):.2f}%")
    
    # 输出按原因分类的统计
    print("\n按卖出原因分类统计:")
    for reason, stats in reason_stats.items():
        avg_profit = stats['total_profit'] / stats['count'] if stats['count'] > 0 else 0
        win_rate = stats['wins'] / stats['count'] * 100 if stats['count'] > 0 else 0
        print(f"  {reason}: {stats['count']}次, 平均收益率: {avg_profit:.2f}%, 胜率: {win_rate:.1f}%")
    
    # 输出详细记录
    print("\n详细卖出记录:")
    for i, record in enumerate(A.exit_records):
        print(f"{i+1}. 时间: {record['datetime']}, 迭代: {record['iteration']}, "
              f"卖出价: {record['price']:.4f}, 买入价: {record['buy_price']:.4f}, "
              f"收益率: {record['profit_pct']:.2f}%, 原因: {record['reason']}")

def get_result():
    """获取交易结果并展示"""
    # 如果没有交易记录，直接返回
    if not A.exit_records:
        return {
            "总交易次数": 0,
            "总体收益率": 0,
            "盈利次数": 0,
            "亏损次数": 0,
            "胜率": 0,
            "平均收益率": 0,
            "最大收益": 0,
            "最大亏损": 0,
            "详细记录": []
        }
    
    # 计算统计数据
    win_count = sum(1 for record in A.exit_records if record['profit_pct'] > 0)
    loss_count = sum(1 for record in A.exit_records if record['profit_pct'] <= 0)
    total_profit = sum(record['profit_pct'] for record in A.exit_records)
    
    # 计算最大收益和最大亏损
    max_profit = max((record['profit_pct'] for record in A.exit_records), default=0)
    max_loss = min((record['profit_pct'] for record in A.exit_records), default=0)
    
    # 按卖出原因分类统计
    reason_stats = {}
    for record in A.exit_records:
        reason = record['reason']
        profit = record['profit_pct']
        
        if reason not in reason_stats:
            reason_stats[reason] = {'count': 0, 'total_profit': 0, 'wins': 0, 'losses': 0}
        
        reason_stats[reason]['count'] += 1
        reason_stats[reason]['total_profit'] += profit
        
        if profit > 0:
            reason_stats[reason]['wins'] += 1
        else:
            reason_stats[reason]['losses'] += 1
    
    # 计算每种原因的平均收益和胜率
    for reason, stats in reason_stats.items():
        stats['avg_profit'] = stats['total_profit'] / stats['count'] if stats['count'] > 0 else 0
        stats['win_rate'] = stats['wins'] / stats['count'] * 100 if stats['count'] > 0 else 0
    
    # 构建结果字典
    result = {
        "总交易次数": len(A.exit_records),
        "总体收益率": total_profit,
        "盈利次数": win_count,
        "亏损次数": loss_count,
        "胜率": win_count / len(A.exit_records) * 100 if A.exit_records else 0,
        "平均收益率": total_profit / len(A.exit_records) if A.exit_records else 0,
        "最大收益": max_profit,
        "最大亏损": max_loss,
        "按原因分类": reason_stats,
        "详细记录": A.exit_records,
        "多级指标": {
            "长期线": A.long_term_curve_value if hasattr(A, 'long_term_curve_value') else None,
            "中期线": A.mid_term_curve_value if hasattr(A, 'mid_term_curve_value') else None,
            "短期线": A.short_term_curve_value if hasattr(A, 'short_term_curve_value') else None
        }
    }
    
    # 打印结果
    print("=" * 50)
    print(f"交易结果统计:")
    print(f"总交易次数: {result['总交易次数']}")
    print(f"总体收益率: {result['总体收益率']:.2f}%")
    print(f"胜率: {result['胜率']:.2f}%")
    print(f"盈利次数: {result['盈利次数']}, 亏损次数: {result['亏损次数']}")
    print(f"平均收益率: {result['平均收益率']:.2f}%")
    print(f"最大收益: {result['最大收益']:.2f}%, 最大亏损: {result['最大亏损']:.2f}%")
    
    print("\n按卖出原因分类统计:")
    for reason, stats in reason_stats.items():
        print(f"  {reason}: {stats['count']}次, 平均收益率: {stats['avg_profit']:.2f}%, 胜率: {stats['win_rate']:.1f}%")
    
    return result