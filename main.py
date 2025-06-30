import os
# ===================== GPU/CUDA WARNING FIXES =====================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"               # Disable GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"                # Suppress TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"               # Disable OneDNN warnings
os.environ["KMP_AFFINITY"] = "noverbose,disabled"       # Disable OpenMP warnings
import warnings
warnings.filterwarnings("ignore")                       # Silence all warnings

# ===================== MAIN IMPORTS =====================
import requests
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from telegram import Bot, ParseMode
import time
from datetime import datetime
import pickle
from dotenv import load_dotenv

# ===================== CONFIGURATION =====================
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')

SYMBOLS = ['EUR/USD', 'GBP/USD']
TIMEFRAME = '15min'
CHECK_INTERVAL = 300  # 5 minutes

# ===================== CORE CLASSES =====================
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
        except Exception:
            return None
            
    def _add_technical_indicators(self, df):
        """Add RSI and MACD to dataframe"""
        # RSI Calculation
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD Calculation
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
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
            df['volume'] = 0  # Add empty volume column
            
            # Convert to numeric and add indicators
            df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            return self._add_technical_indicators(df)
        except Exception:
            return pd.DataFrame()

    def get_historical_data(self, symbol, days=14):
        data = self._fetch_data(symbol, days*96)  # 96 candles/day for 15min
        return self._process_data(data)
        
    def get_live_data(self, symbol, lookback=100):  # Increased for volatility calc
        data = self._fetch_data(symbol, lookback)
        return self._process_data(data)

class ModelManager:
    def __init__(self):
        os.makedirs('models', exist_ok=True)
        
    def get_model(self, symbol, data):
        if data.empty or len(data) < 100:
            return None, None
            
        model_key = symbol.replace('/', '_')
        model_path = f"models/{model_key}_model.keras"
        scaler_path = f"models/{model_key}_scaler.pkl"
        
        try:
            if os.path.exists(model_path):
                return load_model(model_path), self._load_scaler(scaler_path)
        except Exception:
            pass
            
        return self._train_model(data, model_path, scaler_path)
        
    def _train_model(self, data, model_path, scaler_path):
        try:
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(data)
            
            # Prepare sequences
            X, y = [], []
            for i in range(30, len(scaled)):
                X.append(scaled[i-30:i])
                y.append(1 if scaled[i, 3] > scaled[i-1, 3] else 0)  # 3 = close price index
                
            # Enhanced model architecture
            model = Sequential([
                Input(shape=(30, data.shape[1])),  # Now uses all features
                LSTM(64, return_sequences=True),
                Dropout(0.3),
                LSTM(32),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam', loss='binary_crossentropy')
            model.fit(np.array(X), np.array(y), epochs=10, batch_size=16, verbose=0)
            
            model.save(model_path)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
                
            return model, scaler
        except Exception:
            return None, None
            
    def _load_scaler(self, path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

class SignalProcessor:
    def __init__(self):
        self.fetcher = ForexDataFetcher()
        self.models = ModelManager()
        self.bot = self._init_bot()
        
    def _init_bot(self):
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            return None
            
        try:
            bot = Bot(token=TELEGRAM_TOKEN)
            bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=f"✅ Bot Started at {datetime.now().strftime('%H:%M:%S')}",
                parse_mode=ParseMode.MARKDOWN
            )
            return bot
        except Exception:
            return None
            
    def _calculate_dynamic_thresholds(self, df):
        """Adjust thresholds based on market volatility"""
        volatility = df['close'].pct_change().std() * 100  # Current volatility %
        buy_thresh = 0.60 + (volatility * 0.005)          # More aggressive in high volatility
        sell_thresh = 0.40 - (volatility * 0.005)
        return min(buy_thresh, 0.70), max(sell_thresh, 0.30)  # Clamped values
            
    def process_symbol(self, symbol):
        try:
            # Get data with enough history for volatility calculation
            live_data = self.fetcher.get_live_data(symbol, lookback=100)
            if live_data.empty or len(live_data) < 50:
                return None
                
            # Calculate dynamic thresholds
            buy_thresh, sell_thresh = self._calculate_dynamic_thresholds(live_data)
            
            # Get model (use last 14 days for training)
            hist_data = self.fetcher.get_historical_data(symbol, days=14)
            model, scaler = self.models.get_model(symbol, hist_data)
            if model is None or scaler is None:
                return None
                
            # Prepare prediction
            scaled = scaler.transform(live_data)
            if len(scaled) < 30:
                return None
                
            prediction = model.predict(np.array([scaled[-30:]]), verbose=0)[0][0]
            price = live_data['close'].iloc[-1]
            
            print(f"{symbol} - Price: {price:.5f}, Prediction: {prediction:.2%}, Volatility: {live_data['close'].pct_change().std()*100:.2f}%")
            
            # Generate signal
            if prediction >= buy_thresh:
                return {
                    'symbol': symbol,
                    'direction': 'BUY',
                    'confidence': prediction,
                    'price': price,
                    'threshold': buy_thresh
                }
            elif prediction <= sell_thresh:
                return {
                    'symbol': symbol,
                    'direction': 'SELL',
                    'confidence': 1-prediction,
                    'price': price,
                    'threshold': sell_thresh
                }
            return None
        except Exception:
            return None
            
    def send_signal(self, signal):
        if not self.bot or not signal:
            return
            
        try:
            emoji = "🚀" if signal['direction'] == 'BUY' else "📉"
            self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=f"""
{emoji} *{signal['symbol']} {signal['direction']} Signal* 
━━━━━━━━━━━━━━
• Confidence: `{signal['confidence']*100:.1f}%` (Threshold: `{signal['threshold']*100:.1f}%`)
• Price: `{signal['price']:.5f}`
• Time: `{datetime.now().strftime('%H:%M:%S')}`
""",
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception:
            pass

# ===================== MAIN EXECUTION =====================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("💹 FOREX SIGNAL BOT - ENHANCED VERSION")
    print("="*50 + "\n")
    
    processor = SignalProcessor()
    
    # Initial model training
    if not os.path.exists('models'):
        print("⚙️ Training initial models...")
        for symbol in SYMBOLS:
            data = processor.fetcher.get_historical_data(symbol, days=14)
            if not data.empty:
                processor.models.get_model(symbol, data)
    
    # Main loop
    print("\n🔍 Starting monitoring...")
    while True:
        try:
            print("\n" + "-"*50)
            print(f"⏳ Cycle at {datetime.now().strftime('%H:%M:%S')}")
            
            for symbol in SYMBOLS:
                signal = processor.process_symbol(symbol)
                processor.send_signal(signal)
                
            print(f"\n🕒 Next check in {CHECK_INTERVAL//60} minutes...")
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
            break
        except Exception as e:
            print(f"⚠️ Error: {str(e)}")
            time.sleep(300)
