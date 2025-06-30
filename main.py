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

# ======================
# 1. CONFIGURATION
# ======================
load_dotenv()

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')

# Validate configuration
print("\n🔍 Configuration Check:")
print(f"TELEGRAM_TOKEN: {'*****' if TELEGRAM_TOKEN else 'NOT FOUND'}")
print(f"TELEGRAM_CHAT_ID: {TELEGRAM_CHAT_ID or 'NOT FOUND'}")
print(f"TWELVEDATA_API_KEY: {'*****' if TWELVEDATA_API_KEY else 'NOT FOUND'}")

SYMBOLS = os.getenv('SYMBOLS', 'EUR/USD').split(',')  # Single symbol for API limits
BUY_THRESHOLD = float(os.getenv('BUY_THRESHOLD', '0.7'))
SELL_THRESHOLD = float(os.getenv('SELL_THRESHOLD', '0.3'))
TIMEFRAME = os.getenv('TIMEFRAME', '30min')  # Higher timeframe
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '1800'))  # 30 minutes

# ======================
# 2. API-CONSCIOUS DATA FETCHER
# ======================
class ForexDataFetcher:
    def __init__(self):
        self.base_url = "https://api.twelvedata.com"
        self.last_request = 0
        
    def _rate_limit(self):
        """Enforce strict rate limiting"""
        now = time.time()
        elapsed = now - self.last_request
        if elapsed < 7.5:  # 8 requests/minute (free tier limit)
            time.sleep(7.5 - elapsed)
        self.last_request = time.time()
        
    def get_data(self, symbol, lookback):
        """Safe data fetcher with rate limiting"""
        self._rate_limit()
        
        params = {
            "symbol": symbol,
            "interval": TIMEFRAME,
            "outputsize": lookback,
            "apikey": TWELVEDATA_API_KEY
        }
        
        try:
            response = requests.get(f"{self.base_url}/time_series", params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if 'code' in data and data['code'] == 429:
                raise ValueError(f"API Limit: {data.get('message', 'Daily limit reached')}")
                
            if 'values' not in data:
                raise ValueError(f"No market data in response")
                
            return data['values']
            
        except Exception as e:
            print(f"🚨 API Error for {symbol}: {str(e)}")
            return None

    def get_historical_data(self, symbol, days=5):  # Reduced history
        return self._process_data(self.get_data(symbol, days*48))  # 48 candles/day for 30min
        
    def get_live_data(self, symbol, lookback=24):  # Reduced lookback (~12 hours)
        return self._process_data(self.get_data(symbol, lookback))
        
    def _process_data(self, data):
        """Convert API data to DataFrame"""
        if not data:
            return None
            
        try:
            df = pd.DataFrame(data)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            return df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        except Exception as e:
            print(f"📊 Data processing error: {str(e)}")
            return None

# ======================
# 3. MODEL MANAGER
# ======================
class ModelManager:
    def __init__(self):
        os.makedirs('models', exist_ok=True)
        
    def get_model(self, symbol, data):
        """Get or create model with validation"""
        if data is None or len(data) < 50:
            print(f"⚠️ Insufficient data for {symbol}")
            return None, None
            
        model_path = f"models/{symbol.replace('/', '_')}_model.h5"
        scaler_path = f"models/{symbol.replace('/', '_')}_scaler.pkl"
        
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                return load_model(model_path), self._load_scaler(scaler_path)
        except Exception as e:
            print(f"⚠️ Model load error: {str(e)}")
            
        return self._train_model(data, model_path, scaler_path)
        
    def _train_model(self, data, model_path, scaler_path):
        """Train new model with error handling"""
        try:
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(data)
            
            X, y = [], []
            for i in range(30, len(scaled)):
                X.append(scaled[i-30:i])
                y.append(1 if scaled[i, 3] > scaled[i-1, 3] else 0)
                
            model = Sequential([
                LSTM(32, return_sequences=True, input_shape=(30, 5)),
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

# ======================
# 4. SIGNAL PROCESSOR
# ======================
class SignalProcessor:
    def __init__(self):
        self.fetcher = ForexDataFetcher()
        self.models = ModelManager()
        self.bot = self._init_bot()
        
    def _init_bot(self):
        """Initialize Telegram bot safely"""
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
        except Exception as e:
            print(f"⚠️ Telegram init failed: {str(e)}")
            return None
            
    def process_symbol(self, symbol):
        """Full processing pipeline for one symbol"""
        try:
            print(f"\n🔎 Processing {symbol}...")
            
            # Get data
            hist_data = self.fetcher.get_historical_data(symbol)
            live_data = self.fetcher.get_live_data(symbol)
            
            if hist_data is None or live_data is None:
                return None
                
            # Get model
            model, scaler = self.models.get_model(symbol, hist_data)
            if model is None or scaler is None:
                return None
                
            # Predict
            scaled = scaler.transform(live_data)
            prediction = model.predict(np.array([scaled[-30:]]), verbose=0)[0][0]
            price = live_data['close'].iloc[-1]
            
            print(f"📊 Prediction: {prediction:.2%} at {price:.5f}")
            
            # Generate signal
            if prediction >= BUY_THRESHOLD:
                return {'symbol': symbol, 'direction': 'BUY', 
                        'confidence': prediction, 'price': price}
            elif prediction <= SELL_THRESHOLD:
                return {'symbol': symbol, 'direction': 'SELL',
                        'confidence': 1-prediction, 'price': price}
                        
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

# ======================
# 5. MAIN EXECUTION
# ======================
if __name__ == "__main__":
    print("\n" + "="*40)
    print("💹 FOREX SIGNAL BOT - STABLE VERSION")
    print("="*40 + "\n")
    
    if not TWELVEDATA_API_KEY:
        print("❌ Error: Missing API Key")
        exit(1)
        
    processor = SignalProcessor()
    
    # Initial training if needed
    if not os.path.exists('models'):
        print("\n⚙️ Initial model training...")
        for symbol in SYMBOLS:
            data = processor.fetcher.get_historical_data(symbol)
            if data is not None:
                processor.models.get_model(symbol, data)
    
    # Main loop
    print("\n🔍 Starting monitoring...")
    while True:
        try:
            print("\n" + "-"*40)
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
