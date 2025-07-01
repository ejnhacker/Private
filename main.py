import os
import time
import pickle
import requests
import pandas as pd
import numpy as np
import pandas_ta as ta  # Replaced TA-Lib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from telegram import Bot, ParseMode
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# Disable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

# Load configuration
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')

SYMBOLS = ['EUR/USD', 'GBP/USD']
TIMEFRAME = '15min'
CHECK_INTERVAL = 300  # 5 minutes

class ForexDataFetcher:
    def __init__(self):
        self.base_url = "https://api.twelvedata.com"
        self.request_count = 0
        self.last_request_time = 0
        
    def _rate_limit(self):
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < 8:  # 8 requests per minute limit
            time.sleep(8 - elapsed)
        self.last_request_time = time.time()
        self.request_count += 1
        
    def _fetch_data(self, symbol, lookback):
        self._rate_limit()
        params = {
            "symbol": symbol,
            "interval": TIMEFRAME,
            "outputsize": lookback,
            "apikey": TWELVEDATA_API_KEY,
            "type": "forex"
        }
        try:
            response = requests.get(f"{self.base_url}/time_series", params=params, timeout=15)
            response.raise_for_status()
            return response.json().get('values', None)
        except Exception as e:
            print(f"[ERROR] API fetch failed: {str(e)}")
            return None
            
    def _add_technical_indicators(self, df):
        """Calculate indicators using pandas-ta"""
        # Price Transformations
        df['log_ret'] = np.log(df['close']/df['close'].shift(1))
        
        # Momentum Indicators
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # Trend Indicators
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['macd'] = macd['MACD_12_26_9']
        df['signal_line'] = macd['MACDs_12_26_9']
        
        # Volatility Indicators
        bbands = ta.bbands(df['close'], length=20)
        df['upper_band'] = bbands['BBU_20_2.0']
        df['lower_band'] = bbands['BBL_20_2.0']
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        return df.dropna()

    def _process_data(self, data):
        if not data:
            return pd.DataFrame()
            
        try:
            df = pd.DataFrame(data)
            required_cols = ['datetime', 'open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                return pd.DataFrame()
                
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            
            # Convert to numeric
            numeric_cols = ['open', 'high', 'low', 'close']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
            
            # Add volume if available
            df['volume'] = 0 if 'volume' not in df.columns else df['volume'].astype(float)
            
            return self._add_technical_indicators(df)
        except Exception as e:
            print(f"[ERROR] Data processing failed: {str(e)}")
            return pd.DataFrame()

    def get_historical_data(self, symbol, days=14):
        data = self._fetch_data(symbol, days*96)  # 96 candles/day for 15min
        return self._process_data(data)
        
    def get_live_data(self, symbol, lookback=100):
        data = self._fetch_data(symbol, lookback)
        return self._process_data(data)

class ModelManager:
    def __init__(self):
        os.makedirs('models', exist_ok=True)
        
    def get_model(self, symbol, data):
        if data.empty or len(data) < 100:
            print(f"[WARN] Insufficient data for {symbol}")
            return None, None
            
        model_key = symbol.replace('/', '_')
        model_path = f"models/{model_key}_model.keras"
        scaler_path = f"models/{model_key}_scaler.pkl"
        
        try:
            if os.path.exists(model_path):
                model = load_model(model_path)
                scaler = self._load_scaler(scaler_path)
                print(f"[INFO] Loaded existing model for {symbol}")
                return model, scaler
        except Exception as e:
            print(f"[ERROR] Model load failed: {str(e)}")
            
        return self._train_model(data, model_path, scaler_path)
        
    def _train_model(self, data, model_path, scaler_path):
        try:
            # Features: open, high, low, close, rsi, macd, signal_line, atr
            feature_cols = ['open', 'high', 'low', 'close', 'rsi', 'macd', 'signal_line', 'atr']
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(data[feature_cols])
            
            # Prepare sequences
            X, y = [], []
            for i in range(30, len(scaled)):
                X.append(scaled[i-30:i])
                y.append(1 if scaled[i, 3] > scaled[i-1, 3] else 0)  # 3 = close price index
                
            # Enhanced model architecture
            model = Sequential([
                Input(shape=(30, len(feature_cols))),
                LSTM(64, return_sequences=True),
                Dropout(0.3),
                LSTM(32),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(np.array(X), np.array(y), epochs=10, batch_size=32, verbose=0)
            
            model.save(model_path)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
                
            print(f"[INFO] Trained new model for {model_path}")
            return model, scaler
        except Exception as e:
            print(f"[ERROR] Model training failed: {str(e)}")
            return None, None
            
    def _load_scaler(self, path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[ERROR] Scaler load failed: {str(e)}")
            return None

class SignalProcessor:
    def __init__(self):
        self.fetcher = ForexDataFetcher()
        self.models = ModelManager()
        self.bot = self._init_telegram()
        
    def _init_telegram(self):
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            print("[WARN] Telegram credentials missing")
            return None
            
        try:
            bot = Bot(token=TELEGRAM_TOKEN)
            bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=f"✅ Bot Initialized at {datetime.now().strftime('%H:%M:%S')}",
                parse_mode=ParseMode.MARKDOWN
            )
            return bot
        except Exception as e:
            print(f"[ERROR] Telegram init failed: {str(e)}")
            return None
            
    def _calculate_dynamic_thresholds(self, df):
        """Volatility-adjusted thresholds"""
        volatility = df['close'].pct_change().std() * 100
        buy_thresh = 0.60 + (volatility * 0.005)
        sell_thresh = 0.40 - (volatility * 0.005)
        return min(buy_thresh, 0.70), max(sell_thresh, 0.30)
            
    def process_symbol(self, symbol):
        try:
            # Get data
            hist_data = self.fetcher.get_historical_data(symbol, days=14)
            live_data = self.fetcher.get_live_data(symbol, lookback=100)
            
            if hist_data.empty or live_data.empty:
                print(f"[WARN] Empty data for {symbol}")
                return None
                
            # Get model
            model, scaler = self.models.get_model(symbol, hist_data)
            if model is None or scaler is None:
                return None
                
            # Dynamic thresholds
            buy_thresh, sell_thresh = self._calculate_dynamic_thresholds(live_data)
            
            # Prepare prediction
            feature_cols = ['open', 'high', 'low', 'close', 'rsi', 'macd', 'signal_line', 'atr']
            scaled = scaler.transform(live_data[feature_cols])
            
            if len(scaled) < 30:
                print(f"[WARN] Not enough data points for {symbol}")
                return None
                
            prediction = model.predict(np.array([scaled[-30:]]), verbose=0)[0][0]
            price = live_data['close'].iloc[-1]
            
            # Generate signal
            signal = {
                'symbol': symbol,
                'price': price,
                'prediction': prediction,
                'thresholds': (buy_thresh, sell_thresh),
                'indicators': {
                    'rsi': live_data['rsi'].iloc[-1],
                    'macd': live_data['macd'].iloc[-1],
                    'atr': live_data['atr'].iloc[-1]
                }
            }
            
            print(f"[INFO] {symbol} - Price: {price:.5f}, Pred: {prediction:.2%}, "
                  f"RSI: {signal['indicators']['rsi']:.2f}")
            return signal
            
        except Exception as e:
            print(f"[ERROR] Processing failed for {symbol}: {str(e)}")
            return None
            
    def send_signal(self, signal):
        if not signal or not self.bot:
            return
            
        try:
            buy_thresh, sell_thresh = signal['thresholds']
            direction = None
            confidence = 0
            
            if signal['prediction'] >= buy_thresh:
                direction = 'BUY'
                confidence = signal['prediction']
            elif signal['prediction'] <= sell_thresh:
                direction = 'SELL'
                confidence = 1 - signal['prediction']
                
            if not direction:
                return
                
            emoji = "🚀" if direction == 'BUY' else "📉"
            analysis = []
            
            # RSI analysis
            rsi = signal['indicators']['rsi']
            if rsi > 70:
                analysis.append("⚠️ Overbought (RSI > 70)")
            elif rsi < 30:
                analysis.append("⚠️ Oversold (RSI < 30)")
                
            # MACD analysis
            macd = signal['indicators']['macd']
            if macd > 0:
                analysis.append("📈 Bullish MACD")
            else:
                analysis.append("📉 Bearish MACD")
                
            message = (
                f"{emoji} *{signal['symbol']} {direction} SIGNAL*\n"
                f"━━━━━━━━━━━━━━\n"
                f"• Price: `{signal['price']:.5f}`\n"
                f"• Confidence: `{confidence*100:.1f}%`\n"
                f"• RSI: `{rsi:.2f}`\n"
                f"• MACD: `{macd:.4f}`\n"
                f"• ATR: `{signal['indicators']['atr']:.5f}`\n"
                f"• Analysis: {' | '.join(analysis)}\n"
                f"• Time: `{datetime.now().strftime('%H:%M:%S')}`"
            )
            
            self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message,
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception as e:
            print(f"[ERROR] Telegram send failed: {str(e)}")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("💹 ENHANCED FOREX TRADING BOT (pandas-ta Version)")
    print("="*50 + "\n")
    
    processor = SignalProcessor()
    
    # Initial model training if needed
    if not os.path.exists('models'):
        print("[INFO] Training initial models...")
        for symbol in SYMBOLS:
            data = processor.fetcher.get_historical_data(symbol, days=14)
            if not data.empty:
                processor.models.get_model(symbol, data)
    
    # Main loop
    print("\n[INFO] Starting monitoring...")
    while True:
        try:
            print("\n" + "-"*50)
            print(f"[CYCLE] {datetime.now().strftime('%H:%M:%S')}")
            
            for symbol in SYMBOLS:
                signal = processor.process_symbol(symbol)
                processor.send_signal(signal)
                
            print(f"\n[INFO] Next check in {CHECK_INTERVAL//60} minutes...")
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n[INFO] Bot stopped by user")
            break
        except Exception as e:
            print(f"[ERROR] Main loop failed: {str(e)}")
            time.sleep(60)
