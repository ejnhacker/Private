import os
# COMPLETE GPU DISABLE AND WARNING SUPPRESSION
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

# RADICAL WARNING SUPPRESSION
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# IMPORT REDIRECTION TO SUPPRESS INIT MESSAGES
import sys
import logging
logging.disable(sys.maxsize)

# NOW IMPORT TENSORFLOW
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

# REST OF IMPORTS
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

# Load configuration
load_dotenv()

# CONFIGURATION
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')

SYMBOLS = ['EUR/USD', 'GBP/USD']
BUY_THRESHOLD = 0.65
SELL_THRESHOLD = 0.35
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
        if elapsed < 8:
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
            df['volume'] = 0
            return df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        except Exception:
            return pd.DataFrame()

    def get_historical_data(self, symbol, days=14):
        data = self._fetch_data(symbol, days*96)
        return self._process_data(data)
        
    def get_live_data(self, symbol, lookback=48):
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
            
            X, y = [], []
            for i in range(30, len(scaled)):
                X.append(scaled[i-30:i])
                y.append(1 if scaled[i, 3] > scaled[i-1, 3] else 0)
                
            model = Sequential([
                Input(shape=(30, 5)),
                LSTM(32, return_sequences=True),
                Dropout(0.2),
                LSTM(16),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam', loss='binary_crossentropy')
            model.fit(np.array(X), np.array(y), epochs=8, batch_size=8, verbose=0)
            
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
            
    def process_symbol(self, symbol):
        try:
            hist_data = self.fetcher.get_historical_data(symbol, 14)
            live_data = self.fetcher.get_live_data(symbol, 48)
            
            if hist_data.empty or live_data.empty:
                return None
                
            model, scaler = self.models.get_model(symbol, hist_data)
            if model is None or scaler is None:
                return None
                
            scaled = scaler.transform(live_data)
            if len(scaled) < 30:
                return None
                
            prediction = model.predict(np.array([scaled[-30:]]), verbose=0)[0][0]
            price = live_data['close'].iloc[-1]
            
            print(f"{symbol} - Price: {price:.5f}, Prediction: {prediction:.2%}")
            
            if prediction >= BUY_THRESHOLD:
                return {'symbol': symbol, 'direction': 'BUY', 'confidence': prediction, 'price': price}
            elif prediction <= SELL_THRESHOLD:
                return {'symbol': symbol, 'direction': 'SELL', 'confidence': 1-prediction, 'price': price}
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
                text=f"{emoji} *{signal['symbol']} {signal['direction']} Signal*\n━━━━━━━━━━━━━━\n• Confidence: `{signal['confidence']*100:.1f}%`\n• Price: `{signal['price']:.5f}`\n• Time: `{datetime.now().strftime('%H:%M:%S')}`",
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception:
            pass

if __name__ == "__main__":
    print("\nFOREX SIGNAL BOT - SILENT MODE\n")
    processor = SignalProcessor()
    
    if not os.path.exists('models'):
        for symbol in SYMBOLS:
            data = processor.fetcher.get_historical_data(symbol, 14)
            if not data.empty:
                processor.models.get_model(symbol, data)
    
    while True:
        try:
            print(f"Cycle at {datetime.now().strftime('%H:%M:%S')}")
            
            for symbol in SYMBOLS:
                signal = processor.process_symbol(symbol)
                processor.send_signal(signal)
                
            time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            print("\nBot stopped")
            break
        except Exception:
            time.sleep(300)
