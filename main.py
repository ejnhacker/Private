import os
import requests
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from telegram import Bot, TelegramError
import time
from datetime import datetime
import pickle
from dotenv import load_dotenv

# ======================
# 1. ENHANCED CONFIGURATION
# ======================
load_dotenv()

def validate_config():
    """Validate all required configurations"""
    configs = {
        'TELEGRAM_TOKEN': os.getenv('TELEGRAM_TOKEN'),
        'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID'),
        'TWELVEDATA_API_KEY': os.getenv('TWELVEDATA_API_KEY')
    }
    
    print("\n🔍 Configuration Check:")
    for key, value in configs.items():
        if not value:
            print(f"❌ Missing: {key}")
        else:
            print(f"✅ Found: {key}")
    
    return configs

config = validate_config()

TELEGRAM_TOKEN = config['TELEGRAM_TOKEN']
TELEGRAM_CHAT_ID = config['TELEGRAM_CHAT_ID']
TWELVEDATA_API_KEY = config['TWELVEDATA_API_KEY']

# Customizable from .env
SYMBOLS = os.getenv('SYMBOLS', 'EUR/USD,GBP/USD,USD/JPY').split(',')
BUY_THRESHOLD = float(os.getenv('BUY_THRESHOLD', '0.7'))
SELL_THRESHOLD = float(os.getenv('SELL_THRESHOLD', '0.3'))
TIMEFRAME = os.getenv('TIMEFRAME', '5min')

# ======================
# 2. ENHANCED DATA FETCHER
# ======================
class ForexDataFetcher:
    def __init__(self):
        self.base_url = "https://api.twelvedata.com"
        self.session = requests.Session()
        
    def _make_request(self, symbol, lookback):
        """Centralized request handler with error checking"""
        params = {
            "symbol": symbol,
            "interval": TIMEFRAME,
            "outputsize": lookback,
            "apikey": TWELVEDATA_API_KEY
        }
        
        try:
            response = self.session.get(f"{self.base_url}/time_series", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'values' not in data:
                print(f"⚠️ No 'values' in API response for {symbol}. Full response: {data}")
                return None
                
            return data['values']
            
        except Exception as e:
            print(f"🚨 API Error for {symbol}: {str(e)}")
            return None

    def get_historical_data(self, symbol, days=30):
        """Get historical data with robust error handling"""
        data = self._make_request(symbol, days*288)
        if not data:
            return None
            
        try:
            df = pd.DataFrame(data)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            return df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        except Exception as e:
            print(f"📊 Data processing error for {symbol}: {str(e)}")
            return None

    def get_live_data(self, symbol, lookback=50):
        """Get live market data"""
        return self.get_historical_data(symbol, lookback)

# ======================
# 3. MODEL CACHING SYSTEM
# ======================
class ModelCache:
    def __init__(self):
        os.makedirs('models', exist_ok=True)
        self.scalers = {}
        self.models = {}
    
    def load_or_train(self, symbol, data):
        """Improved model loading with validation"""
        if data is None:
            print(f"⚠️ No data available for {symbol}")
            return None, None
            
        model_path = f"models/{symbol.replace('/', '_')}_model.h5"
        scaler_path = f"models/{symbol.replace('/', '_')}_scaler.pkl"
        
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                model = load_model(model_path)
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"♻️ Loaded cached model for {symbol}")
                return model, scaler
        except Exception as e:
            print(f"⚠️ Error loading cached model for {symbol}: {str(e)}")
            
        print(f"🛠️ Training new model for {symbol}")
        return self._train_new_model(data)

    def _train_new_model(self, data):
        """Enhanced model training"""
        try:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            
            X, y = [], []
            for i in range(30, len(scaled_data)):
                X.append(scaled_data[i-30:i])
                y.append(1 if scaled_data[i, 3] > scaled_data[i-1, 3] else 0)
            
            model = Sequential([
                LSTM(32, return_sequences=True, input_shape=(30, 5)),
                Dropout(0.2),
                LSTM(16),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(np.array(X), np.array(y), epochs=10, batch_size=32, verbose=0)
            
            return model, scaler
        except Exception as e:
            print(f"🚨 Model training failed: {str(e)}")
            return None, None

# ======================
# 4. ROBUST SIGNAL GENERATOR
# ======================
class SignalGenerator:
    def __init__(self):
        self.data_fetcher = ForexDataFetcher()
        self.cache = ModelCache()
        self.bot = self._init_telegram()
        
    def _init_telegram(self):
        """Initialize Telegram bot with verification"""
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            print("⚠️ Telegram credentials not configured")
            return None
            
        try:
            bot = Bot(token=TELEGRAM_TOKEN)
            # Verify bot can send messages
            test_msg = "🤖 Forex Bot Initialized Successfully\n" \
                     f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=test_msg)
            print("✅ Telegram connection verified")
            return bot
        except TelegramError as e:
            print(f"🚨 Telegram initialization failed: {str(e)}")
            return None
        except Exception as e:
            print(f"🚨 Unexpected Telegram error: {str(e)}")
            return None

    def generate_signals(self):
        """Enhanced signal generation"""
        signals = []
        
        for symbol in SYMBOLS:
            try:
                print(f"\n🔎 Processing {symbol}...")
                
                # Get data
                hist_data = self.data_fetcher.get_historical_data(symbol)
                live_data = self.data_fetcher.get_live_data(symbol)
                
                if hist_data is None or live_data is None:
                    print(f"⚠️ Skipping {symbol} due to data issues")
                    continue
                
                # Get model
                model, scaler = self.cache.load_or_train(symbol, hist_data)
                if model is None or scaler is None:
                    continue
                
                # Prepare prediction
                scaled_data = scaler.transform(live_data)
                X = np.array([scaled_data[-30:]])
                
                # Predict
                prediction = model.predict(X, verbose=0)[0][0]
                price = live_data['close'].iloc[-1]
                
                print(f"📊 {symbol} - Price: {price:.5f}, Prediction: {prediction:.2%}")
                
                # Generate signal
                if prediction >= BUY_THRESHOLD:
                    signals.append({
                        'symbol': symbol,
                        'direction': "BUY",
                        'confidence': float(prediction),
                        'price': float(price)
                    })
                elif prediction <= SELL_THRESHOLD:
                    signals.append({
                        'symbol': symbol,
                        'direction': "SELL", 
                        'confidence': float(1 - prediction),
                        'price': float(price)
                    })
                    
            except Exception as e:
                print(f"⚠️ Error processing {symbol}: {str(e)}")
                continue
                
        return signals
    
    def send_signals(self, signals):
        """Reliable signal sending with rich formatting"""
        if not self.bot:
            print("⚠️ Cannot send signals - Telegram not initialized")
            return
            
        if not signals:
            print("🔍 No signals to send")
            return
            
        print(f"\n📨 Sending {len(signals)} signals to Telegram...")
        
        for signal in signals:
            try:
                emoji = "🚀" if signal['direction'] == "BUY" else "📉"
                message = (
                    f"{emoji} *{signal['symbol']} {signal['direction']} Signal*\n"
                    f"━━━━━━━━━━━━━━\n"
                    f"• Confidence: `{signal['confidence']*100:.1f}%`\n"
                    f"• Price: `{signal['price']:.5f}`\n"
                    f"• Time: `{datetime.now().strftime('%H:%M:%S')}`"
                )
                
                self.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=message,
                    parse_mode='Markdown'
                )
                print(f"✅ Sent {signal['symbol']} {signal['direction']} signal")
                
            except Exception as e:
                print(f"⚠️ Failed to send {signal['symbol']} signal: {str(e)}")

# ======================
# 5. MAIN EXECUTION
# ======================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("💹 FOREX SIGNAL BOT - ENHANCED DEBUG VERSION")
    print("="*50 + "\n")
    
    # Initial checks
    if not TWELVEDATA_API_KEY:
        print("❌ CRITICAL: No TwelveData API Key found")
        exit(1)
        
    bot = SignalGenerator()
    
    # First-time setup
    if not os.path.exists('models'):
        print("\n⚙️ Initial model training...")
        for symbol in SYMBOLS:
            print(f"🛠️ Training model for {symbol}")
            data = bot.data_fetcher.get_historical_data(symbol)
            if data is not None:
                bot.cache.load_or_train(symbol, data)
    
    # Main loop
    print("\n🔍 Starting signal monitoring...")
    while True:
        try:
            print("\n" + "-"*50)
            print(f"🔄 Cycle started at {datetime.now().strftime('%H:%M:%S')}")
            
            signals = bot.generate_signals()
            bot.send_signals(signals)
            
            print(f"⏳ Next check in 3 minutes...")
            time.sleep(180)
            
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
            break
        except Exception as e:
            print(f"⚠️ Unexpected error in main loop: {str(e)}")
            print("🔄 Retrying in 1 minute...")
            time.sleep(60)
