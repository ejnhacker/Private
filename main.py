#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ULTIMATE FOREX TRADING BOT
- Multi-API data aggregation
- Pure Python technical indicators
- Advanced LSTM model
- Professional Telegram alerts
- Fault-tolerant design
"""

import os
import time
import pickle
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from telegram import Bot, ParseMode
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from dotenv import load_dotenv

# =====================
# CONFIGURATION
# =====================
load_dotenv()

class Config:
    # API Keys
    TD_API_KEY = os.getenv('TWELVEDATA_API_KEY')
    AV_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY')
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    # Trading Parameters
    SYMBOLS = ['EUR/USD', 'GBP/USD', 'USD/JPY']
    TIMEFRAMES = {
        'fast': '5min',
        'medium': '15min',
        'slow': '1h'
    }
    BUY_THRESHOLD = 0.68
    SELL_THRESHOLD = 0.32
    
    # Risk Management
    MAX_DAILY_TRADES = 5
    STOP_LOSS_PCT = 1.5
    TAKE_PROFIT_PCT = 3.0

# =====================
# TECHNICAL INDICATORS (Pure Python)
# =====================
class TechnicalAnalysis:
    @staticmethod
    def rsi(series, window=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/window).mean()
        avg_loss = loss.ewm(alpha=1/window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series, fast=12, slow=26, signal=9):
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line

    @staticmethod
    def bollinger_bands(series, window=20, std_dev=2):
        sma = series.rolling(window).mean()
        std = series.rolling(window).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

# =====================
# DATA FETCHER (Multi-API)
# =====================
class DataFetcher:
    API_CONFIG = {
        'twelvedata': {
            'url': 'https://api.twelvedata.com/time_series',
            'params': {
                'apikey': Config.TD_API_KEY,
                'type': 'forex'
            }
        },
        'alphavantage': {
            'url': 'https://www.alphavantage.co/query',
            'params': {
                'apikey': Config.AV_API_KEY,
                'function': 'FX_INTRADAY'
            }
        }
    }

    @staticmethod
    def _format_symbol(api_name, symbol):
        if api_name == 'alphavantage':
            base, quote = symbol.split('/')
            return {'from_symbol': base, 'to_symbol': quote}
        return {'symbol': symbol.replace('/', '')}

    @classmethod
    def fetch_concurrent(cls, symbol, lookback, timeframe):
        """Fetch data from all APIs simultaneously"""
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            for api_name, config in cls.API_CONFIG.items():
                params = {
                    **config['params'],
                    **cls._format_symbol(api_name, symbol),
                    'interval': timeframe,
                    'outputsize': lookback
                }
                futures[api_name] = executor.submit(
                    requests.get,
                    config['url'],
                    params=params,
                    timeout=10
                )
            
            results = {}
            for api_name, future in futures.items():
                try:
                    res = future.result()
                    results[api_name] = cls._process_data(api_name, res.json())
                except Exception as e:
                    print(f"{api_name.upper()} Error: {str(e)}")
            return results

    @staticmethod
    def _process_data(api_name, data):
        """Standardize data from different APIs"""
        if api_name == 'twelvedata':
            df = pd.DataFrame(data.get('values', []))
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                return df[['open', 'high', 'low', 'close']].astype(float)
        
        elif api_name == 'alphavantage':
            ts_key = 'Time Series FX (5min)' if '5min' in data else 'Time Series FX (15min)'
            df = pd.DataFrame(data.get(ts_key, {})).T
            if not df.empty:
                df.index = pd.to_datetime(df.index)
                df.columns = ['open', 'high', 'low', 'close']
                return df.astype(float)
        
        return pd.DataFrame()

# =====================
# CORE TRADING ENGINE
# =====================
class ForexTradingBot:
    def __init__(self):
        self.bot = self._init_telegram()
        self.model = self._init_model()
        self.scaler = MinMaxScaler()
        self.today_trades = 0
        self.trade_history = []
        self.last_trade_time = None

    def _init_telegram(self):
        try:
            bot = Bot(token=Config.TELEGRAM_TOKEN)
            bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text=f"🚀 *Forex Bot Started* at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                parse_mode=ParseMode.MARKDOWN
            )
            return bot
        except Exception as e:
            print(f"Telegram Init Error: {str(e)}")
            return None

    def _init_model(self):
        try:
            if os.path.exists('forex_model.keras'):
                return load_model('forex_model.keras')
            
            model = Sequential([
                Input(shape=(60, 8)),  # 60 timesteps, 8 features
                LSTM(128, return_sequences=True),
                Dropout(0.4),
                LSTM(64),
                Dropout(0.3),
                Dense(1, activation='sigmoid')
            ])
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            return model
        except Exception as e:
            print(f"Model Init Error: {str(e)}")
            return None

    def _calculate_features(self, df):
        """Generate all technical features"""
        # Price Features
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)
        
        # Momentum Indicators
        df['rsi'] = TechnicalAnalysis.rsi(df['close'])
        df['macd'], df['signal_line'] = TechnicalAnalysis.macd(df['close'])
        
        # Trend Indicators
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma50'] = df['close'].rolling(50).mean()
        
        # Volatility Indicators
        df['upper_bb'], df['middle_bb'], df['lower_bb'] = TechnicalAnalysis.bollinger_bands(df['close'])
        
        return df.dropna()

    def _prepare_data(self, dataframes):
        """Combine and validate multi-API data"""
        valid_dfs = [df for df in dataframes.values() if not df.empty]
        if len(valid_dfs) < 1:
            return pd.DataFrame()
            
        combined = pd.concat(valid_dfs)
        aggregated = combined.groupby(combined.index).agg({
            'open': 'mean',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })
        
        return self._calculate_features(aggregated)

    def _check_trading_rules(self):
        """Risk management checks"""
        if self.today_trades >= Config.MAX_DAILY_TRADES:
            return False
            
        if self.last_trade_time and (datetime.now() - self.last_trade_time) < timedelta(minutes=30):
            return False
            
        return True

    def generate_signal(self, df):
        """Generate trading decision"""
        if len(df) < 60 or not self._check_trading_rules():
            return None
            
        # Prepare features
        features = df[['open', 'high', 'low', 'close', 'rsi', 'macd', 'ma20', 'volatility']]
        scaled = self.scaler.fit_transform(features)
        
        # Make prediction
        prediction = self.model.predict(np.array([scaled[-60:]]), verbose=0)[0][0]
        
        # Determine action
        if prediction >= Config.BUY_THRESHOLD:
            direction = 'BUY'
            confidence = prediction
            stop_loss = df['close'].iloc[-1] * (1 - Config.STOP_LOSS_PCT/100)
            take_profit = df['close'].iloc[-1] * (1 + Config.TAKE_PROFIT_PCT/100)
        elif prediction <= Config.SELL_THRESHOLD:
            direction = 'SELL'
            confidence = 1 - prediction
            stop_loss = df['close'].iloc[-1] * (1 + Config.STOP_LOSS_PCT/100)
            take_profit = df['close'].iloc[-1] * (1 - Config.TAKE_PROFIT_PCT/100)
        else:
            return None
            
        return {
            'direction': direction,
            'confidence': confidence,
            'price': df['close'].iloc[-1],
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'time': datetime.now(),
            'indicators': {
                'rsi': df['rsi'].iloc[-1],
                'macd': df['macd'].iloc[-1],
                'ma20': df['ma20'].iloc[-1],
                'ma50': df['ma50'].iloc[-1]
            }
        }

    def send_alert(self, signal, symbol):
        """Send professional trading alert"""
        if not signal or not self.bot:
            return
            
        # Prepare analysis message
        analysis = []
        if signal['indicators']['rsi'] > 70:
            analysis.append("⚠️ Overbought (RSI > 70)")
        elif signal['indicators']['rsi'] < 30:
            analysis.append("⚠️ Oversold (RSI < 30)")
            
        if signal['indicators']['macd'] > 0:
            analysis.append("📈 Bullish MACD")
        else:
            analysis.append("📉 Bearish MACD")
            
        if signal['indicators']['ma20'] > signal['indicators']['ma50']:
            analysis.append("⬆️ MA20 > MA50")
        else:
            analysis.append("⬇️ MA20 < MA50")
        
        # Format message
        emoji = "🚀" if signal['direction'] == 'BUY' else "🔻"
        message = (
            f"{emoji} *{symbol} {signal['direction']} SIGNAL*\n"
            f"━━━━━━━━━━━━━━\n"
            f"• Price: `{signal['price']:.5f}`\n"
            f"• Confidence: `{signal['confidence']*100:.1f}%`\n"
            f"• Stop Loss: `{signal['stop_loss']:.5f}`\n"
            f"• Take Profit: `{signal['take_profit']:.5f}`\n"
            f"• RSI: `{signal['indicators']['rsi']:.2f}`\n"
            f"• MACD: `{signal['indicators']['macd']:.4f}`\n"
            f"• Analysis: {' | '.join(analysis)}\n"
            f"• Time: `{signal['time'].strftime('%H:%M:%S')}`"
        )
        
        try:
            self.bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text=message,
                parse_mode=ParseMode.MARKDOWN
            )
            self.today_trades += 1
            self.last_trade_time = datetime.now()
            self.trade_history.append(signal)
        except Exception as e:
            print(f"Telegram Error: {str(e)}")

    def run(self):
        """Main trading loop"""
        print("🚀 Starting Multi-API Forex Trading Bot")
        while True:
            try:
                current_hour = datetime.now().hour
                if current_hour >= 22 or current_hour < 5:  # Skip late night trading
                    time.sleep(3600)
                    continue
                    
                for symbol in Config.SYMBOLS:
                    # Fetch data from multiple APIs
                    api_data = DataFetcher.fetch_concurrent(
                        symbol=symbol,
                        lookback=200,
                        timeframe=Config.TIMEFRAMES['medium']
                    )
                    
                    # Process and analyze
                    df = self._prepare_data(api_data)
                    if not df.empty:
                        signal = self.generate_signal(df)
                        if signal:
                            self.send_alert(signal, symbol)
                            print(f"Signal generated for {symbol}")
                
                # Reset daily counter at midnight
                if datetime.now().hour == 0 and self.today_trades > 0:
                    self.today_trades = 0
                
                print(f"⏳ Next analysis in {Config.TIMEFRAMES['medium']}...")
                time.sleep(int(timedelta(minutes=15).total_seconds()))
                
            except KeyboardInterrupt:
                print("\n🛑 Bot stopped by user")
                break
            except Exception as e:
                print(f"⚠️ Critical Error: {str(e)}")
                time.sleep(300)

if __name__ == "__main__":
    bot = ForexTradingBot()
    bot.run()
