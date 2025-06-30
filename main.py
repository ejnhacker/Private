import os
import requests
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from telegram import Bot, ParseMode
import time
from datetime import datetime
import pickle
from dotenv import load_dotenv

# Load configuration
load_dotenv()

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')

SYMBOLS = ['EUR/USD', 'GBP/USD']  # Correct format with forward slash
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
        """Enforce API rate limits"""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < 8:  # 8 requests per minute limit
            time.sleep(8 - elapsed)
        self.last_request_time = time.time()
        self.request_count += 1
        
    def _fetch_data(self, symbol, lookback):
        """Core data fetching with error handling"""
        self._rate_limit()
        
        params = {
            "symbol": symbol,
            "interval": TIMEFRAME,
            "outputsize": lookback,
            "apikey": TWELVEDATA_API_KEY,
            "type": "forex"  # Specify forex data type
        }
        
        try:
            response = requests.get(f"{self.base_url}/time_series", params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if 'code' in data and data['code'] == 429:
                raise ValueError(f"API Limit: {data.get('message', 'Rate limit exceeded')}")
                
            if 'values' not in data:
                raise ValueError(f"No price data in response for {symbol}")
                
            return data['values']
            
        except Exception as e:
            print(f"🚨 API Error for {symbol}: {str(e)}")
            return None
            
    def _process_data(self, data):
        """Convert API data to DataFrame with volume handling"""
        try:
            df = pd.DataFrame(data)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            
            # Standard columns we always expect
            base_cols = ['open', 'high', 'low', 'close']
            
            # Add volume if available, otherwise create zero-filled column
            if 'volume' in df.columns:
                return df[base_cols + ['volume']].astype(float)
            else:
                df['volume'] = 0  # Add volume column with zeros
                return df[base_cols + ['volume']].astype(float)
                
        except Exception as e:
            print(f"📊 Data processing error: {str(e)}")
            return pd.DataFrame()

    def get_historical_data(self, symbol, days=7):
        """Get historical data for training"""
        data = self._fetch_data(symbol, days*96)  # 96 candles/day for 15min
        return self._process_data(data) if data else pd.DataFrame()
        
    def get_live_data(self, symbol, lookback=24):
        """Get recent data for prediction"""
        data = self._fetch_data(symbol, lookback)
        return self._process_data(data) if data else pd.DataFrame()

class ModelManager:
    def __init__(self):
        os.makedirs('models', exist_ok=True)
        
    def get_model(self, symbol, data):
        """Get or create model with validation"""
        if data.empty or len(data) < 100:
            print(f"⚠️ Insufficient data for {symbol}")
            return None, None
            
        model_key = symbol.replace('/', '_')
        model_path = f"models/{model_key}_model.h5"
        scaler_path = f"models/{model_key}_scaler.pkl"
        
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                return load_model(model_path), self._load_scaler(scaler_path)
        except Exception as e:
            print(f"⚠️ Model load error: {str(e)}")
            
        print(f"🛠️ Training new model for {symbol}")
        return self._train_model(data, model_path, scaler_path)
        
    def _train_model(self, data, model_path, scaler_path):
        """Train new model with error handling"""
        try:
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(data)
            
            X, y = [], []
            for i in range(30, len(scaled)):
                X.append(scaled[i-30:i])
                y.append(1 if scaled[i, 3] > scaled[i-1, 3] else 0)  # 3 = close price index
                
            model = Sequential([
                LSTM(32, return_sequences=True, input_shape=(30, 5)),  # 5 features
                Dropout(0.2),
                LSTM(16),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam', loss='binary_crossentropy')
            model.fit(np.array(X), np.array(y), epochs=8, batch_size=16, verbose=0)
            
            model.save(model_path)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
                
            return model, scaler
            
        except Exception as e:
            print(f"🚨 Model training failed: {str(e)}")
            return None, None
            
    def _load_scaler(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

class SignalProcessor:
    def __init__(self):
        self.fetcher = ForexDataFetcher()
        self.models = ModelManager()
        self.bot = self._init_bot()
        
    def _init_bot(self):
        """Initialize Telegram bot safely"""
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            print("⚠️ Telegram credentials missing")
            return None
            
        try:
            bot = Bot(token=TELEGRAM_TOKEN)
            bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=f"✅ Bot Started at {datetime.now().strftime('%H:%M:%S')}",
                parse_mode=ParseMode.MARKDOWN
            )
            return bot
        except Exception as e:
            print(f"🚨 Telegram init failed: {str(e)}")
            return None
            
    def process_symbol(self, symbol):
        """Full processing pipeline for one symbol"""
        try:
            print(f"\n🔎 Processing {symbol}...")
            
            # Get data
            hist_data = self.fetcher.get_historical_data(symbol)
            live_data = self.fetcher.get_live_data(symbol)
            
            if hist_data.empty or live_data.empty:
                print(f"⚠️ Missing data for {symbol}")
                return None
                
            # Get model
            model, scaler = self.models.get_model(symbol, hist_data)
            if model is None or scaler is None:
                return None
                
            # Prepare prediction
            scaled = scaler.transform(live_data)
            if len(scaled) < 30:
                print(f"⚠️ Not enough data points for {symbol}")
                return None
                
            X = np.array([scaled[-30:]])
            prediction = model.predict(X, verbose=0)[0][0]
            price = live_data['close'].iloc[-1]
            
            print(f"📊 {symbol} - Price: {price:.5f}, Prediction: {prediction:.2%}")
            
            # Generate signal
            if prediction >= BUY_THRESHOLD:
                return {
                    'symbol': symbol,
                    'direction': 'BUY',
                    'confidence': prediction,
                    'price': price
                }
            elif prediction <= SELL_THRESHOLD:
                return {
                    'symbol': symbol,
                    'direction': 'SELL',
                    'confidence': 1-prediction,
                    'price': price
                }
                
        except Exception as e:
            print(f"⚠️ Processing error: {str(e)}")
            return None
            
    def send_signal(self, signal):
        """Send formatted signal to Telegram"""
        if not self.bot or not signal:
            return
            
        try:
            emoji = "🚀" if signal['direction'] == 'BUY' else "📉"
            self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=f"""
{emoji} *{signal['symbol']} {signal['direction']} Signal*
━━━━━━━━━━━━━━
• Confidence: `{signal['confidence']*100:.1f}%`
• Price: `{signal['price']:.5f}`
• Time: `{datetime.now().strftime('%H:%M:%S')}`
""",
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception as e:
            print(f"⚠️ Telegram send failed: {str(e)}")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("💹 FOREX SIGNAL BOT - FINAL WORKING VERSION")
    print("="*50 + "\n")
    
    processor = SignalProcessor()
    
    # Initial training if needed
    if not os.path.exists('models'):
        print("\n⚙️ Initial model training...")
        for symbol in SYMBOLS:
            data = processor.fetcher.get_historical_data(symbol)
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
            print(f"⚠️ Critical error: {str(e)}")
            print("🔄 Restarting in 5 minutes...")
            time.sleep(300)
