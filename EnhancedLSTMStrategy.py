import logging
from functools import reduce
from typing import Dict, Optional
import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.exchange.exchange_utils import *
from freqtrade.strategy import IStrategy, RealParameter, IntParameter
from freqtrade.persistence import Trade
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EnhancedLSTMStrategy(IStrategy):
    """
    增强型LSTM策略 - 趋势跟踪与智能加仓
    - 支持亏损加仓降低成本
    - 趋势跟踪，减少小波动干扰
    - 自适应杠杆和资金管理
    - LSTM模型驱动的进出场决策
    """
    # 超参数配置
    buy_params = {
        "threshold_buy": 0.59453,          # 做多信号阈值（LSTM输出>此值时做多）
        "w0": 0.54347,                     # MA移动平均线权重
        "w1": 0.82226,                     # MACD趋势动量权重（高权重=重视趋势）
        "w2": 0.56675,                     # ROC变化率权重
        "w3": 0.77918,                     # RSI相对强弱权重（高权重=重视超买超卖）
        "w4": 0.98488,                     # 布林带宽度权重（最高权重=重视波动区间）
        "w5": 0.31368,                     # CCI商品通道指数权重
        "w6": 0.75916,                     # OBV成交量权重（高权重=重视量价关系）
        "w7": 0.09226,                     # ATR波动率权重（低权重=不关注短期波动）
        "w8": 0.85667,                     # 随机指标权重（高权重=关注超买超卖）
        "leverage_multiplier": 1.0,        # 杠杆倍数调节器
        "min_leverage": 1,                 # 最小杠杆
        "max_leverage": 15,                # 最大杠杆
        "volatility_threshold": 0.02,      # 波动率阈值
        "confidence_threshold": 0.7,       # 信心度阈值
        "trend_persistence_bars": 5,       # 趋势持续K线数（增加到5根，减少假信号）
        "exit_confidence_threshold": 0.85, # 出场信心阈值（提高到0.85，持有更久）
        # 资金管理参数
        "stake_multiplier": 0.8,           # 基础资金倍数
        "confidence_stake_factor": 2.0,    # 信心度资金因子
        "volatility_stake_factor": 1.5,    # 波动率资金因子
        "max_stake_ratio": 0.4,            # 最大资金使用比例
        "min_stake_ratio": 0.01,           # 最小资金使用比例
    }

    sell_params = {
        "threshold_sell": 0.80573,          # 做空/平多信号阈值
        "risk_reduction_factor": 0.8,      # 风险降低因子
        "trend_reversal_threshold": 0.85,  # 趋势反转阈值（提高到0.85，减少假反转）
        "exit_smoothing_period": 8,        # 出场平滑周期（增加到8，更稳定）
    }

    # ROI表 - 为长期趋势优化（不设置短期止盈）
    minimal_roi = {
        "0": 100,      # 不设置强制止盈，让策略自主决定
        "720": 0.5,    # 12小时后至少50%利润
        "1440": 0.2,   # 24小时后至少20%利润
        "2880": 0.1,   # 48小时后至少10%利润
        "4320": 0      # 72小时后可以平仓
    }

    # 止损设置 - 关闭止损，完全信任LSTM模型
    stoploss = -1  # 关闭止损（-1 = 100%止损）

    # 跟踪止损 - 为大趋势优化
    trailing_stop = True                    # 启用跟踪止损
    trailing_stop_positive = 0.005          # 盈利后跟踪止损距离（0.5%）
    trailing_stop_positive_offset = 0.05    # 盈利5%后才启动跟踪（给趋势空间）
    trailing_only_offset_is_reached = True  # 只有达到偏移后才跟踪
    use_custom_stoploss = True              # 使用自定义止损逻辑
    
    # 趋势跟踪参数 - 优化以捕捉大趋势
    base_trailing_stop_positive = 0.005         # 基础跟踪止损（0.5%）
    base_trailing_stop_positive_offset = 0.05   # 基础触发偏移（5%盈利后启动）
    
    # 大趋势模式参数
    trend_trailing_multiplier = 3.0             # 趋势模式下止损放大3倍（给更多空间）
    trend_profit_threshold = 0.10               # 盈利10%后进入趋势保护模式
    super_trend_threshold = 0.20                # 盈利20%后进入超级趋势模式
    consolidation_trailing_multiplier = 0.3     # 震荡市场止损收紧（快速退出）

    timeframe = "1h"
    can_short = True
    use_exit_signal = True
    process_only_new_candles = True
    leverage_max = 100  # Maximum leverage

    startup_candle_count = 100

    # 保护机制 - 仅基本保护，主要由LSTM决定
    @property
    def protections(self):
        return []  # 不使用内置保护，完全信任LSTM模型

    # Enhanced strategy parameters
    threshold_buy = RealParameter(-1, 1, default=0.5, space='buy')
    threshold_sell = RealParameter(-1, 1, default=-0.5, space='sell')
    
    # Leverage parameters (支持1-100倍完整范围)
    leverage_multiplier = RealParameter(0.5, 2.0, default=1.0, space='buy')
    min_leverage = IntParameter(1, 5, default=1, space='buy')  # 最小杠杆改为1
    max_leverage = IntParameter(50, 100, default=100, space='buy')
    volatility_threshold = RealParameter(0.005, 0.05, default=0.02, space='buy')
    confidence_threshold = RealParameter(0.5, 0.95, default=0.7, space='buy')
    
    # Trend persistence parameters
    trend_persistence_bars = IntParameter(2, 8, default=3, space='buy')
    exit_confidence_threshold = RealParameter(0.6, 0.95, default=0.8, space='sell')
    trend_reversal_threshold = RealParameter(0.5, 0.9, default=0.75, space='sell')
    exit_smoothing_period = IntParameter(3, 10, default=5, space='sell')
    
    # Risk management
    risk_reduction_factor = RealParameter(0.5, 1.0, default=0.8, space='sell')
    
    # 功能开关
    enable_adaptive_leverage = False    # 自适应杠杆（关闭，使用固定杠杆）
    enable_adaptive_stake = False       # 自适应资金管理（关闭）
    enable_percentage_stake = True      # 百分比资金模式（开启，使用20%资金）
    enable_add_position = True          # 加仓功能（开启 - 亏损加仓）
    enable_trend_following = True       # 趋势跟踪模式（开启）
    
    # 固定参数
    fixed_leverage = 15.0               # 固定杠杆倍数（15倍杠杆）
    percentage_stake_ratio = 0.20       # 每次开仓使用20%资金
    
    # 安全杠杆系统参数
    safe_leverage_mode = True           # 启用安全杠杆模式
    base_safe_leverage = 10.0           # 基础安全杠杆
    max_safe_leverage = 15.0            # 最大安全杠杆（降低风险）
    leverage_decay_factor = 0.9         # 杠杆衰减因子（缓慢衰减） 

    # Weights for calculating the aggregate score
    w0 = RealParameter(0, 1, default=0.10, space='buy')
    w1 = RealParameter(0, 1, default=0.15, space='buy')
    w2 = RealParameter(0, 1, default=0.10, space='buy')
    w3 = RealParameter(0, 1, default=0.15, space='buy')
    w4 = RealParameter(0, 1, default=0.10, space='buy')
    w5 = RealParameter(0, 1, default=0.10, space='buy')
    w6 = RealParameter(0, 1, default=0.10, space='buy')
    w7 = RealParameter(0, 1, default=0.05, space='buy')
    w8 = RealParameter(0, 1, default=0.15, space='buy')

    # 完全自适应资金管理参数 (支持1%-100%资金使用)
    stake_multiplier = RealParameter(0.3, 1.5, default=0.8, space='buy')  # 基础资金倍数
    confidence_stake_factor = RealParameter(1.0, 3.0, default=2.0, space='buy')  # 信心度资金因子
    volatility_stake_factor = RealParameter(0.5, 2.5, default=1.5, space='buy')  # 波动率资金因子
    max_stake_ratio = RealParameter(0.1, 0.3, default=0.3, space='buy')  # 最大资金使用比例30%
    min_stake_ratio = RealParameter(0.01, 0.1, default=0.01, space='buy')  # 最小资金使用比例

    # 智能加仓参数 - 亏损加仓降成本
    add_position_ratio = RealParameter(0.5, 1.0, default=0.8, space='buy')      # 加仓比例80%
    max_add_positions = IntParameter(1, 3, default=2, space='buy')              # 最大加仓2次（控制风险）
    min_trend_confidence = RealParameter(0.4, 0.9, default=0.6, space='buy')    # 加仓最小信心度60%
    add_position_volatility_filter = RealParameter(0.01, 0.05, default=0.03, space='buy')  # 波动率过滤
    
    # 加仓策略参数
    pyramid_position_mode = True        # 金字塔式仓位管理
    position_scale_factor = 1.2         # 每次加仓递增20%
    loss_threshold_for_add = -0.05      # 亏损5%开始加仓
    profit_threshold_for_add = 0.08     # 盈利8%追踪趋势加仓
    max_loss_for_add = -0.20           # 亏损20%停止加仓
    add_position_cooldown = 120        # 加仓冷却时间2小时

    def calculate_adaptive_leverage(self, dataframe: DataFrame, current_index: int) -> float:
        """
        智能杠杆计算 - 基于LSTM模型信心度
        综合考虑信心度(70%)和目标强度(30%)
        """
        # 如果自适应杠杆关闭，返回固定杠杆
        if not self.enable_adaptive_leverage:
            return self.fixed_leverage
            
        try:
            if current_index < 20:  # Not enough data
                return self.min_leverage.value  # 默认最小杠杆
                
            # 获取信心度和目标强度
            confidence = dataframe['confidence_smooth'].iloc[current_index] \
                if 'confidence_smooth' in dataframe.columns else 0.5
            target_strength = abs(dataframe['&-target'].iloc[current_index])
            
            # 综合信心度计算: 70%信心度 + 30%信号强度
            combined_confidence = confidence * 0.7 + target_strength * 0.3
            
            # 确保combined_confidence在合理范围内(0-1)
            combined_confidence = max(0, min(1, combined_confidence))
            
            # 映射到杠杆范围：从min_leverage到max_leverage
            leverage_range = self.max_leverage.value - self.min_leverage.value
            leverage = self.min_leverage.value + (combined_confidence * leverage_range * self.leverage_multiplier.value)
            
            # 确保在范围内并四舍五入
            leverage = max(self.min_leverage.value, min(self.max_leverage.value, leverage))
            leverage = round(leverage)
            
            return leverage
            
        except Exception as e:
            logger.warning(f"Error calculating adaptive leverage: {e}")
            return self.min_leverage.value  # 错误时返回最小杠杆

    def calculate_position_size(self, dataframe: DataFrame, current_index: int, 
                              stake_amount: float, current_rate: float) -> float:
        """
        Calculate position size - this method is currently unused
        Note: In futures trading, position size is automatically calculated as:
        Position Size = (Stake Amount × Leverage) / Current Price
        But this is handled by the exchange, not by the strategy
        """
        # This method is not currently used - position sizing is handled by FreqTrade
        # based on stake_amount and leverage returned by respective methods
        return stake_amount / current_rate

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                       metadata: Dict, **kwargs):

        dataframe["%-cci-period"] = ta.CCI(dataframe, timeperiod=20)
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=10)
        dataframe["%-momentum-period"] = ta.MOM(dataframe, timeperiod=4)
        dataframe['%-ma-period'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['%-macd-period'], dataframe['%-macdsignal-period'], dataframe['%-macdhist-period'] = ta.MACD(
            dataframe['close'], slowperiod=12,
            fastperiod=26)
        dataframe['%-roc-period'] = ta.ROC(dataframe, timeperiod=2)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2
        )
        dataframe["bb_lowerband-period"] = bollinger["lower"]
        dataframe["bb_middleband-period"] = bollinger["mid"]
        dataframe["bb_upperband-period"] = bollinger["upper"]
        dataframe["%-bb_width-period"] = (
                                                 dataframe["bb_upperband-period"]
                                                 - dataframe["bb_lowerband-period"]
                                         ) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = (
                dataframe["close"] / dataframe["bb_lowerband-period"]
        )

        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: Dict, **kwargs):

        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        
        # Add leverage-related features
        dataframe["%-volatility"] = dataframe["close"].pct_change().rolling(14).std()
        dataframe["%-trend_strength"] = abs(dataframe["close"].pct_change(10))
        
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: Dict, **kwargs):

        dataframe['date'] = pd.to_datetime(dataframe['date'])
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        
        # Add market session indicators
        dataframe["%-asian_session"] = ((dataframe["%-hour_of_day"] >= 0) & 
                                       (dataframe["%-hour_of_day"] < 8)).astype(int)
        dataframe["%-european_session"] = ((dataframe["%-hour_of_day"] >= 8) & 
                                          (dataframe["%-hour_of_day"] < 16)).astype(int)
        dataframe["%-us_session"] = ((dataframe["%-hour_of_day"] >= 16) & 
                                    (dataframe["%-hour_of_day"] < 24)).astype(int)
        
        return dataframe

    # 移除趋势持续性和平滑退出信号函数 - LSTM完全决定

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:

        dataframe['ma'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=2)
        dataframe['macd'], dataframe['macdsignal'], dataframe['macdhist'] = ta.MACD(dataframe['close'], slowperiod=12,
                                                                                    fastperiod=26)
        dataframe['momentum'] = ta.MOM(dataframe, timeperiod=4)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=10)
        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['bb_upperband'] = bollinger['upperband']
        dataframe['bb_middleband'] = bollinger['middleband']
        dataframe['bb_lowerband'] = bollinger['lowerband']
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        dataframe['stoch'] = ta.STOCH(dataframe)['slowk']
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['obv'] = ta.OBV(dataframe)

        # Enhanced indicators for trend analysis
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=14)
        
        # Add trend strength and persistence indicators
        dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['trend_strength'] = abs(dataframe['ema_21'] - dataframe['ema_50']) / dataframe['close']
        dataframe['price_momentum'] = dataframe['close'].pct_change(5)
        
        # Market structure indicators
        dataframe['higher_high'] = (dataframe['high'] > dataframe['high'].shift(1)) & (dataframe['high'].shift(1) > dataframe['high'].shift(2))
        dataframe['lower_low'] = (dataframe['low'] < dataframe['low'].shift(1)) & (dataframe['low'].shift(1) < dataframe['low'].shift(2))

        # Normalize indicators for leverage calculation
        dataframe['atr_normalized'] = (dataframe['atr'] / dataframe['close']).rolling(20).mean()
        dataframe['bb_width_normalized'] = ((dataframe['bb_upperband'] - dataframe['bb_lowerband']) / 
                                           dataframe['bb_middleband']).rolling(20).mean()

        # Step 1: Normalize Indicators
        dataframe['normalized_stoch'] = (dataframe['stoch'] - dataframe['stoch'].rolling(window=14).mean()) / dataframe[
            'stoch'].rolling(window=14).std()
        dataframe['normalized_atr'] = (dataframe['atr'] - dataframe['atr'].rolling(window=14).mean()) / dataframe[
            'atr'].rolling(window=14).std()
        dataframe['normalized_obv'] = (dataframe['obv'] - dataframe['obv'].rolling(window=14).mean()) / dataframe[
            'obv'].rolling(window=14).std()
        dataframe['normalized_ma'] = (dataframe['close'] - dataframe['close'].rolling(window=10).mean()) / dataframe[
            'close'].rolling(window=10).std()
        dataframe['normalized_macd'] = (dataframe['macd'] - dataframe['macd'].rolling(window=26).mean()) / dataframe[
            'macd'].rolling(window=26).std()
        dataframe['normalized_roc'] = (dataframe['roc'] - dataframe['roc'].rolling(window=2).mean()) / dataframe[
            'roc'].rolling(window=2).std()
        dataframe['normalized_momentum'] = (dataframe['momentum'] - dataframe['momentum'].rolling(window=4).mean()) / \
                                           dataframe['momentum'].rolling(window=4).std()
        dataframe['normalized_rsi'] = (dataframe['rsi'] - dataframe['rsi'].rolling(window=10).mean()) / dataframe[
            'rsi'].rolling(window=10).std()
        dataframe['normalized_bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']).rolling(
            window=20).mean() / (dataframe['bb_upperband'] - dataframe['bb_lowerband']).rolling(window=20).std()
        dataframe['normalized_cci'] = (dataframe['cci'] - dataframe['cci'].rolling(window=20).mean()) / dataframe[
            'cci'].rolling(window=20).std()
        dataframe['normalized_adx'] = (dataframe['adx'] - dataframe['adx'].rolling(window=14).mean()) / dataframe[
            'adx'].rolling(window=14).std()

        # Dynamic Weights with leverage consideration
        trend_strength = abs(dataframe['ma'] - dataframe['close'])
        strong_trend_threshold = trend_strength.rolling(window=14).mean() + 1.5 * trend_strength.rolling(
            window=14).std()
        is_strong_trend = trend_strength > strong_trend_threshold

        # Enhanced dynamic weights
        dataframe['w_momentum'] = np.where(is_strong_trend, self.w3.value * 1.5, self.w3.value)
        dataframe['w_trend'] = np.where(dataframe['adx'] > 25, 1.2, 0.8)  # ADX > 25 indicates strong trend

        # Step 2: Enhanced aggregate score calculation
        w = [self.w0.value, self.w1.value, self.w2.value, self.w3.value, self.w4.value, self.w5.value,
             self.w6.value, self.w7.value, self.w8.value]

        dataframe['S'] = (w[0] * dataframe['normalized_ma'] + w[1] * dataframe['normalized_macd'] + 
                         w[2] * dataframe['normalized_roc'] + w[3] * dataframe['normalized_rsi'] + 
                         w[4] * dataframe['normalized_bb_width'] + w[5] * dataframe['normalized_cci'] + 
                         dataframe['w_momentum'] * dataframe['normalized_momentum'] + 
                         self.w8.value * dataframe['normalized_stoch'] + self.w7.value * dataframe['normalized_atr'] + 
                         self.w6.value * dataframe['normalized_obv'] + 0.1 * dataframe['normalized_adx'])

        # Step 3: Enhanced Market Regime Filter
        dataframe['R'] = 0
        dataframe.loc[(dataframe['close'] > dataframe['bb_middleband']) & (
                dataframe['close'] > dataframe['bb_upperband']), 'R'] = 1
        dataframe.loc[(dataframe['close'] < dataframe['bb_middleband']) & (
                dataframe['close'] < dataframe['bb_lowerband']), 'R'] = -1

        # Additional Market Regime Filter
        dataframe['ma_100'] = ta.SMA(dataframe, timeperiod=100)
        dataframe['R2'] = np.where(dataframe['close'] > dataframe['ma_100'], 1, -1)
        
        # Trend strength regime
        dataframe['R3'] = np.where(dataframe['adx'] > 25, 1.5, 0.8)

        # Step 4: Enhanced Volatility Adjustment
        bb_width = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']
        dataframe['V'] = 1 / (bb_width + 0.001)  # Prevent division by zero
        
        # ATR-based volatility
        dataframe['V2'] = 1 / (dataframe['atr_normalized'] + 0.001)
        
        # Volume-based confirmation
        dataframe['V3'] = np.where(dataframe['volume'] > dataframe['volume'].rolling(20).mean(), 1.2, 0.8)

        # Step 5: Final Target Score with leverage considerations
        dataframe['T'] = (dataframe['S'] * dataframe['R'] * dataframe['V'] * 
                         dataframe['R2'] * dataframe['V2'] * dataframe['R3'] * dataframe['V3'])

        # Enhanced confidence score for trend persistence
        dataframe['confidence'] = abs(dataframe['T']).rolling(8).mean()
        dataframe['confidence_smooth'] = dataframe['confidence'].rolling(3).mean()
        
        # Add trend reversal detection
        dataframe['trend_reversal_score'] = (
            abs(dataframe['T'].diff()) * 
            (dataframe['adx'] / 100) * 
            dataframe['trend_strength'] * 10
        ).rolling(5).mean()

        # Assign the target score T to the AI target column
        dataframe['&-target'] = dataframe['T']

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        self.freqai_info = self.config["freqai"]

        dataframe = self.freqai.start(dataframe, metadata, self)
        
        # Add leverage calculation and exit signals after FreqAI processing
        for i in range(len(dataframe)):
            if i >= 20:  # Ensure we have enough data
                leverage = self.calculate_adaptive_leverage(dataframe, i)
                dataframe.loc[dataframe.index[i], 'calculated_leverage'] = leverage
                
                # 不再计算平滑退出信号 - LSTM直接决定
                dataframe.loc[dataframe.index[i], 'smooth_exit_signal'] = dataframe['&-target'].iloc[i]
            else:
                dataframe.loc[dataframe.index[i], 'calculated_leverage'] = self.min_leverage.value
                dataframe.loc[dataframe.index[i], 'smooth_exit_signal'] = 0
                
        return dataframe

    def calculate_adaptive_stake_ratio(self, dataframe: DataFrame, current_index: int) -> float:
        """
        LSTM模型完全决定仓位 - 无风险管理限制
        直接基于模型输出计算1%-100%资金使用
        如果enable_adaptive_stake为False，返回None表示使用固定stake
        """
        # 如果自适应资金管理关闭，返回None
        if not self.enable_adaptive_stake:
            return None
            
        try:
            if current_index < 20:
                return 0.5  # 默认50%
                
            # 直接使用LSTM模型输出
            target_strength = abs(dataframe['&-target'].iloc[current_index])
            confidence = dataframe['confidence_smooth'].iloc[current_index] if 'confidence_smooth' in dataframe.columns else 0.5
            
            # 简单线性映射：模型输出越强，仓位越大
            # 结合信心度和目标强度
            stake_ratio = target_strength * confidence * self.stake_multiplier.value
            
            # 确保在最小和最大比例范围内
            stake_ratio = max(self.min_stake_ratio.value, min(self.max_stake_ratio.value, stake_ratio))
            
            return stake_ratio
            
        except Exception as e:
            logger.warning(f"Error calculating adaptive stake ratio: {e}")
            return 0.5  # 错误时默认50%
            
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                             current_rate: float, current_profit: float,
                             min_stake: float | None, max_stake: float,
                             current_entry_rate: float, current_exit_rate: float,
                             entry_tag: str | None, exit_tag: str | None,
                             **kwargs) -> float | None:
        """
        简化加仓逻辑：只要亏损+LSTM同方向信号=加仓，最多2次
        """
        if not self.enable_add_position:
            return None
            
        try:
            # 检查加仓次数限制
            if trade.nr_of_successful_entries >= self.max_add_positions.value:
                return None
                
            # 只有亏损时才加仓
            if current_profit >= 0:
                return None
                
            # 获取最新数据
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            if len(dataframe) < 20:
                return None

            # 获取当前信号强度
            current_signal = dataframe['&-target'].iloc[-1]
            previous_signal = dataframe['&-target'].iloc[-2]

            # 检查信号方向是否与持仓一致, 且为新的入场信号
            should_add_position = False
            if trade.is_short:
                if (current_signal < -self.threshold_sell.value and
                        previous_signal >= -self.threshold_sell.value):
                    should_add_position = True
            else:
                if (current_signal > self.threshold_buy.value and
                        previous_signal <= self.threshold_buy.value):
                    should_add_position = True

            if not should_add_position:
                return None
                
            # 计算加仓金额
            available_balance = self.wallets.get_free(self.config['stake_currency'])
            
            # 金字塔式加仓: 每次递增
            position_count = trade.nr_of_successful_entries
            scale_factor = self.position_scale_factor ** position_count
            
            # 基础加仓金额
            base_add_amount = trade.stake_amount * self.add_position_ratio.value
            add_stake = base_add_amount * scale_factor
            
            # 限制单次加仓不超过可用余额的30%
            add_stake = min(add_stake, available_balance * 0.3)
            
            # 确保满足最小金额要求
            if min_stake and add_stake < min_stake:
                return None
                
            logger.info(f"加仓信号 {trade.pair}: 亏损加仓 {current_profit:.2%}, "
                       f"信号强度: {current_signal:.3f}, "
                       f"加仓次数: {position_count + 1}/{self.max_add_positions.value}, "
                       f"加仓金额: {add_stake:.2f} USDT")
            
            return add_stake
            
        except Exception as e:
            logger.warning(f"加仓计算错误: {e}")
            return None

    def custom_stake_amount(self, pair: str, current_time, current_rate: float,
                          proposed_stake: float, min_stake: float | None, max_stake: float,
                          leverage: float, entry_tag: str | None, side: str,
                          **kwargs) -> float:
        """
        计算初始仓位大小 - 为加仓预留空间
        """
        # 获取可用余额
        available_balance = self.wallets.get_free(self.config['stake_currency'])
        
        # 模式1: 百分比模式 - 使用总资金的固定百分比
        if self.enable_percentage_stake:
            calculated_stake = available_balance * self.percentage_stake_ratio
            
            # 应用系统限制
            if min_stake:
                calculated_stake = max(calculated_stake, min_stake)
            if max_stake:
                calculated_stake = min(calculated_stake, max_stake)
                
            logger.info(f"Percentage stake for {pair}: {calculated_stake:.2f} USDT "
                       f"({self.percentage_stake_ratio:.1%} of {available_balance:.2f} USDT)")
            return calculated_stake
        
        # 模式2: 如果自适应资金管理也关闭，使用config中的固定stake
        if not self.enable_adaptive_stake:
            return proposed_stake
        
        # 模式3: 自适应模式 - 根据市场条件动态调整资金比例    
        try:
            # Get the latest dataframe
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            
            if len(dataframe) < 20:
                # Not enough data, use minimal stake
                return max(available_balance * self.min_stake_ratio.value, min_stake or 0)
            
            # Calculate adaptive stake ratio (just like adaptive leverage)
            current_index = len(dataframe) - 1
            stake_ratio = self.calculate_adaptive_stake_ratio(dataframe, current_index)
            
            # Calculate stake amount
            calculated_stake = available_balance * stake_ratio
            
            # Apply system bounds
            if min_stake:
                calculated_stake = max(calculated_stake, min_stake)
            if max_stake:
                calculated_stake = min(calculated_stake, max_stake)
            
            # Get market info for logging
            target_strength = abs(dataframe['&-target'].iloc[-1])
            confidence = dataframe['confidence_smooth'].iloc[-1] if 'confidence_smooth' in dataframe.columns else 0.5
            calculated_leverage = dataframe['calculated_leverage'].iloc[-1] if 'calculated_leverage' in dataframe.columns else 1
            
            logger.info(f"Adaptive stake for {pair}: {calculated_stake:.2f} USDT "
                       f"({stake_ratio:.1%} of {available_balance:.2f} USDT) - "
                       f"Target: {target_strength:.3f}, Confidence: {confidence:.3f}, "
                       f"Leverage: {calculated_leverage}x")
            
            return calculated_stake
            
        except Exception as e:
            logger.warning(f"Error in adaptive stake calculation: {e}")
            # Fallback to minimal risk
            try:
                available_balance = self.wallets.get_free(self.config['stake_currency'])
                fallback_stake = max(available_balance * self.min_stake_ratio.value, min_stake or 0)
                return min(fallback_stake, max_stake) if max_stake else fallback_stake
            except:
                return proposed_stake


    def leverage(self, pair: str, current_time, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Dynamic leverage calculation based on market conditions
        支持1-100倍完整范围
        如果enable_adaptive_leverage为False，返回固定杠杆
        """
        # 如果自适应杠杆关闭，返回固定杠杆
        if not self.enable_adaptive_leverage:
            # 从config中获取固定杠杆，如果没有则使用类属性
            fixed_lev = self.config.get('leverage', self.fixed_leverage)
            return min(fixed_lev, max_leverage)
            
        try:
            # Get the latest dataframe
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            
            if len(dataframe) < 20:
                return min(self.min_leverage.value, max_leverage)
                
            # Get calculated leverage
            calculated_leverage = dataframe['calculated_leverage'].iloc[-1]
            
            # Ensure it's within bounds
            final_leverage = min(calculated_leverage, max_leverage, self.max_leverage.value)
            final_leverage = max(final_leverage, 1)
            
            logger.info(f"LSTM leverage for {pair}: {final_leverage}x")
            
            return final_leverage
            
        except Exception as e:
            logger.warning(f"Error calculating leverage: {e}")
            return min(self.min_leverage.value, max_leverage)

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        生成入场信号 - 支持初始进场和亏损加仓
        """
        # 基础做多条件
        enter_long_conditions = [
            df["do_predict"] == 1,
            df['&-target'] > self.threshold_buy.value,
            df['volume'] > 0,
        ]

        # 基础做空条件
        enter_short_conditions = [
            df["do_predict"] == 1,
            df['&-target'] < -self.threshold_sell.value,  # 注意负值
            df["volume"] > 0,
        ]

        # 趋势强度条件 - 用于大趋势确认
        if len(df) > self.trend_persistence_bars.value:
            # 计算趋势持续性得分
            df['trend_score'] = (
                df['&-target'].rolling(self.trend_persistence_bars.value).mean().abs()
            )
            
            # 强趋势条件: 趋势得分高且信心度高
            # 安全获取confidence_smooth列，如果不存在使用默认值
            confidence_smooth = df.get('confidence_smooth', pd.Series(0.5, index=df.index))
            
            strong_trend_long = [
                df['trend_score'] > 0.3,  # 强趋势阈值
                confidence_smooth > self.confidence_threshold.value,
                df['&-target'] > self.threshold_buy.value * 1.2,  # 更强的信号
            ]
            
            strong_trend_short = [
                df['trend_score'] > 0.3,
                confidence_smooth > self.confidence_threshold.value,
                df['&-target'] < -self.threshold_sell.value * 1.2,
            ]
            
            # 对于强趋势，使用特殊标签（便于后续处理）
            df.loc[
                reduce(lambda x, y: x & y, strong_trend_long), 
                ["enter_long", "enter_tag"]
            ] = (1, "long_strong_trend")
            
            df.loc[
                reduce(lambda x, y: x & y, strong_trend_short), 
                ["enter_short", "enter_tag"]
            ] = (1, "short_strong_trend")
        
        # 标准入场信号
        df.loc[
            reduce(lambda x, y: x & y, enter_long_conditions), 
            ["enter_long", "enter_tag"]
        ] = (1, "long_adaptive")

        df.loc[
            reduce(lambda x, y: x & y, enter_short_conditions), 
            ["enter_short", "enter_tag"]
        ] = (1, "short_adaptive")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # LSTM模型完全决定退出 - 无风险管理
        
        # 多头退出条件 - 仅基于LSTM信号
        exit_long_conditions = [
            df["do_predict"] == 1,
            df['&-target'] < self.threshold_sell.value,  # LSTM说卖就卖
        ]

        # 空头退出条件 - 仅基于LSTM信号
        exit_short_conditions = [
            df["do_predict"] == 1,
            df['&-target'] > self.threshold_buy.value,  # LSTM说买就平空
        ]

        if exit_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, exit_long_conditions), ["exit_long", "exit_tag"]
            ] = (1, "exit_long_lstm")

        if exit_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, exit_short_conditions), ["exit_short", "exit_tag"]
            ] = (1, "exit_short_lstm")

        return df

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, after_fill: bool,
                       **kwargs) -> float | None:
        """
        智能止损系统 - 为大趋势优化
        - 初期给予更多空间
        - 盈利后逐步收紧
        - 大盈利时保护利润
        """
        try:
            # 获取持仓时间（小时）
            trade_duration = (current_time - trade.open_date_utc).total_seconds() / 3600
            
            # 阶段1: 初始阶段 - 关闭止损
            if trade_duration < 2:  # 前2小时
                return -1  # 关闭止损
            
            # 阶段2: 小幅盈利 - 保护成本
            if current_profit > 0.02 and current_profit < self.trend_profit_threshold:
                # 盈利2%-10%之间，使用动态止损
                return -0.05  # 5%回撤止损
            
            # 阶段3: 趋势盈利 - 宽松跟踪
            if current_profit >= self.trend_profit_threshold and current_profit < self.super_trend_threshold:
                # 盈利10%-20%，给趋势空间
                trailing_distance = 0.08  # 允许8%回撤
                return -(trailing_distance * self.trend_trailing_multiplier)
            
            # 阶段4: 超级趋势 - 保护利润
            if current_profit >= self.super_trend_threshold:
                # 盈利超过20%，收紧止损保护利润
                # 动态计算：利润越高，止损越紧
                base_trailing = 0.05
                profit_factor = min(current_profit / 0.5, 1.0)  # 最多到50%利润
                trailing_distance = base_trailing * (1 - profit_factor * 0.5)  # 利润越高越紧
                return -max(trailing_distance, 0.02)  # 最紧不低于2%
            
            # 默认止损
            return None
            
        except Exception as e:
            logger.warning(f"止损计算错误: {e}")
            return -1  # 错误时关闭止损

    def get_position_info(self, pair: str) -> dict:
        """
        获取当前持仓信息（模拟实现）
        在实际应用中，这需要通过Freqtrade API获取真实持仓信息
        """
        # 这是一个模拟方法，实际应用中需要通过API获取真实持仓信息
        return {
            'is_position_open': False,
            'position_size': 0,
            'entry_price': 0,
            'current_profit': 0
        }
