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
    A.acct = '100099994905'
    A.acct_type = 'STOCK_OPTION'
    A.amount = 10000
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
    """计算各种技术指标"""
    # QSDD 指标
    df = calculate_qsdd(df)

    # MACD
    df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = talib.MACD(df['close'], fastperiod=6, slowperiod=7, signalperiod=4)

    # KDJ
    df['K'], df['D'] = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=9, slowk_period=3, slowd_period=3)
    df['J'] = 3 * df['K'] - 2 * df['D']

    # RSI
    df['RSI'] = talib.RSI(df['close'], timeperiod=12)

    # OBV
    df['OBV'] = talib.OBV(df['close'], df['volume'])
    df['OBV_MA'] = df['OBV'].rolling(2).mean()

    # 移动均线 (MA)
    df['MA_short'] = df['close'].rolling(2).mean()
    df['MA_long'] = df['close'].rolling(4).mean()

    return df


def calculate_qsdd(df):
    """计算QSDD长短中线指标及相关判断，严格按照用户定义"""
    # 处理缺失值
    df = df.dropna()

    # 辅助函数: 获取n个周期前的值
    def ref(series, n):
        return series.shift(n)

    # 辅助函数: 计算两条线的交叉
    def cross(series1, series2):
        return (series1 > series2) & (ref(series1, 1) <= ref(series2, 1))

    # 基础计算 - 严格按照公式
    HHV_HIGH_34 = df['high'].rolling(34).max()  # HHV(HIGH,34)
    LLV_LOW_34 = df['low'].rolling(34).min()    # LLV(LOW,34)
    HHV_HIGH_14 = df['high'].rolling(14).max()  # HHV(HIGH,14)
    LLV_LOW_14 = df['low'].rolling(14).min()    # LLV(LOW,14)

    # A:=MA(-100*(HHV(HIGH,34)-CLOSE)/(HHV(HIGH,34)-LLV(LOW,34)),19)
    A = (-100 * (HHV_HIGH_34 - df['close']) / (HHV_HIGH_34 - LLV_LOW_34)).rolling(19).mean()
    
    # B:=-100*(HHV(HIGH,14)-CLOSE)/(HHV(HIGH,14)-LLV(LOW,14))
    B = -100 * (HHV_HIGH_14 - df['close']) / (HHV_HIGH_14 - LLV_LOW_14)
    
    # d:=EMA(-100*(HHV(HIGH,34)-CLOSE)/(HHV(HIGH,34)-LLV(LOW,34)),4)
    d = (-100 * (HHV_HIGH_34 - df['close']) / (HHV_HIGH_34 - LLV_LOW_34)).ewm(span=4).mean()

    # 长期线:A+100
    df['long_term_curve'] = A + 100
    # 短期线:B+100
    df['short_term_curve'] = B + 100
    # 中期线:d+100
    df['mid_term_curve'] = d + 100

    # 见顶:(ref(中期线,1)>85 and ref(短期线,1)>85 and ref(长期线,1)>65) and cross(长期线,短期线)
    df['见顶'] = (
        (ref(df['mid_term_curve'], 1) > 85) & 
        (ref(df['short_term_curve'], 1) > 85) & 
        (ref(df['long_term_curve'], 1) > 65) & 
        cross(df['long_term_curve'], df['short_term_curve'])
    )

    # 顶部区域:(中期线<ref(中期线,1) and ref(中期线,1)>80) and (ref(短期线,1)>95 or ref(短期线,2)>95) and 
    # 长期线>60 and 短期线<83.5 and 短期线<中期线 and 短期线<长期线+4
    df['顶部区域'] = (
        (df['mid_term_curve'] < ref(df['mid_term_curve'], 1)) & 
        (ref(df['mid_term_curve'], 1) > 80) & 
        ((ref(df['short_term_curve'], 1) > 95) | (ref(df['short_term_curve'], 2) > 95)) & 
        (df['long_term_curve'] > 60) & 
        (df['short_term_curve'] < 83.5) & 
        (df['short_term_curve'] < df['mid_term_curve']) & 
        (df['short_term_curve'] < (df['long_term_curve'] + 4))
    )

    # 顶部:=filter(顶部区域,4) - 模拟过去4个周期内出现顶部区域的情况
    df['顶部'] = df['顶部区域'].rolling(4).max()

    # 底部区域 - 三个条件的组合
    condition1 = (
        (df['long_term_curve'] < 12) & 
        (df['mid_term_curve'] < 8) & 
        ((df['short_term_curve'] < 7.2) | (ref(df['short_term_curve'], 1) < 5)) & 
        ((df['mid_term_curve'] > ref(df['mid_term_curve'], 1)) | (df['short_term_curve'] > ref(df['short_term_curve'], 1)))
    )
    
    condition2 = (
        (df['long_term_curve'] < 8) & 
        (df['mid_term_curve'] < 7) & 
        (df['short_term_curve'] < 15) & 
        (df['short_term_curve'] > ref(df['short_term_curve'], 1))
    )
    
    condition3 = (
        (df['long_term_curve'] < 10) & 
        (df['mid_term_curve'] < 7) & 
        (df['short_term_curve'] < 1)
    )
    
    df['底部区域'] = condition1 | condition2 | condition3

    # 低位金叉:长期线<15 and ref(长期线,1)<15 and 中期线<18 and 短期线>ref(短期线,1) and 
    # cross(短期线,长期线) and 短期线>中期线 and (ref(短期线,1)<5 or ref(短期线,2)<5) and 
    # (中期线>=长期线 or ref(短期线,1)<1)
    df['低位金叉'] = (
        (df['long_term_curve'] < 15) & 
        (ref(df['long_term_curve'], 1) < 15) & 
        (df['mid_term_curve'] < 18) & 
        (df['short_term_curve'] > ref(df['short_term_curve'], 1)) & 
        cross(df['short_term_curve'], df['long_term_curve']) & 
        (df['short_term_curve'] > df['mid_term_curve']) & 
        ((ref(df['short_term_curve'], 1) < 5) | (ref(df['short_term_curve'], 2) < 5)) & 
        ((df['mid_term_curve'] >= df['long_term_curve']) | (ref(df['short_term_curve'], 1) < 1))
    )

    return df


def generate_signals(df):
    """生成买卖信号"""
    # 新增最新收盘价定义
    last_close_price = df['close'].iloc[-1]  # 从df中获取最新收盘价

    # 买入信号条件 - 按照用户定义的金叉状态重写
    # QSDD金叉状态: 中期线 > 长期线; 长期线向上，长期线 < 21
    qsdd_buy = (
        (df['mid_term_curve'].iloc[-1] > df['long_term_curve'].iloc[-1]) and  # 中期线 > 长期线
        (df['long_term_curve'].iloc[-1] > df['long_term_curve'].iloc[-2]) and  # 长期线向上
        (df['long_term_curve'].iloc[-1] < 21)  # 长期线小于21
    )

    # MACD金叉状态: DIF > DEA; DEA向上
    macd_buy = (
        (df['MACD'].iloc[-1] > df['MACD_SIGNAL'].iloc[-1]) and  # DIF > DEA
        (df['MACD_SIGNAL'].iloc[-1] > df['MACD_SIGNAL'].iloc[-2])  # DEA向上
    )

    # KDJ金叉状态: K > D; D向上，J < 110
    kdj_buy = (
        (df['K'].iloc[-1] > df['D'].iloc[-1]) and  # K > D
        (df['D'].iloc[-1] > df['D'].iloc[-2]) and  # D向上
        (df['J'].iloc[-1] < 110)  # J < 110
    )

    # OBV均线向上
    obv_buy = df['OBV_MA'].iloc[-1] > df['OBV_MA'].iloc[-2]
    
    # RSI < 70
    rsi_buy = df['RSI'].iloc[-1] < 70

    # MA金叉状态: MA2 > MA4; MA4向上
    ma_cross_state = (
        (df['MA_short'].iloc[-1] > df['MA_long'].iloc[-1]) and  # MA2 > MA4
        (df['MA_long'].iloc[-1] > df['MA_long'].iloc[-2])  # MA4向上
    )

    # MA金叉时刻: MA2从下方穿过MA4
    ma_cross_moment = (
        (df['MA_short'].iloc[-1] > df['MA_long'].iloc[-1]) and  # 当前MA2 > MA4
        (df['MA_short'].iloc[-2] <= df['MA_long'].iloc[-2])  # 前一时刻MA2 <= MA4
    )

    # MA死叉状态: MA2 < MA4; MA4向下
    ma_dead_cross_state = (
        (df['MA_short'].iloc[-1] < df['MA_long'].iloc[-1]) and  # MA2 < MA4
        (df['MA_long'].iloc[-1] < df['MA_long'].iloc[-2])  # MA4向下
    )
    
    # 第一次买入条件：满足全部6个条件
    first_buy_condition = qsdd_buy and macd_buy and kdj_buy and obv_buy and rsi_buy and ma_cross_state

    # 后续买入条件：金叉时刻，同时满足其他5个条件
    # 金叉时刻本身已经隐含了之前必然处于死叉状态
    subsequent_buy_condition = ma_cross_moment and qsdd_buy and macd_buy and kdj_buy and obv_buy and rsi_buy

    # 根据是否已经买入和是否为首次买入确定最终买入信号
    if not A.has_bought and A.first_buy:
        buy_signal = first_buy_condition
    elif A.has_bought and not A.first_buy:
        buy_signal = False
    elif not A.has_bought and not A.first_buy:
        buy_signal = subsequent_buy_condition
    else:
        buy_signal = False

    # 卖出信号条件
    # MACD死叉状态: DIF < DEA; DEA向下
    macd_sell = (
        (df['MACD'].iloc[-1] < df['MACD_SIGNAL'].iloc[-1]) and  # DIF < DEA
        (df['MACD_SIGNAL'].iloc[-1] < df['MACD_SIGNAL'].iloc[-2])  # DEA向下
    )

    # KDJ死叉状态: K < D; D向下
    kdj_sell = (
        (df['K'].iloc[-1] < df['D'].iloc[-1]) and  # K < D
        (df['D'].iloc[-1] < df['D'].iloc[-2])  # D向下
    )

    # QSDD死叉状态: 中期线 < 长期线; 长期线向下
    qsdd_sell = (
        (df['mid_term_curve'].iloc[-1] < df['long_term_curve'].iloc[-1]) and  # 中期线 < 长期线
        (df['long_term_curve'].iloc[-1] < df['long_term_curve'].iloc[-2])  # 长期线向下
    )

    # 定义卖出原因
    exit_reason = "未触发卖出"
    
    # 修正止盈止损逻辑（使用从df获取的最新价）
    if A.last_buy_price and pd.notnull(last_close_price):
        price_change = (last_close_price - A.last_buy_price) / A.last_buy_price
        profit_sell = price_change >= 0.01  # 获利1%
        loss_sell = price_change <= -0.04  # 止损-4%
        
        # 确定卖出原因
        if qsdd_sell and macd_sell and kdj_sell:
            exit_reason = "三死叉信号"
        elif profit_sell:
            exit_reason = f"止盈: +{price_change*100:.2f}%"
        elif loss_sell:
            exit_reason = f"止损: {price_change*100:.2f}%"
    else:
        profit_sell = False
        loss_sell = False

    sell_signal = (qsdd_sell and macd_sell and kdj_sell) or profit_sell or loss_sell

    return buy_signal, sell_signal, exit_reason


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