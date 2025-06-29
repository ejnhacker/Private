import os
import requests
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from telegram import Bot
import time
from datetime import datetime
import pickle

# ======================
# 1. CONFIGURATION (FROM .ENV)
# ======================
# Add this near other imports
from dotenv import load_dotenv
load_dotenv()  # Load .env or Replit secrets

# Modify the configuration section to:
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN') or os.environ['TELEGRAM_TOKEN']
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID') or os.environ['TELEGRAM_CHAT_ID']
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY') or os.environ['TWELVEDATA_API_KEY']

# Customizable from .env (comma-separated)
SYMBOLS = os.getenv('SYMBOLS', 'EURUSD,GBPUSD,USDJPY').split(',') 
BUY_THRESHOLD = float(os.getenv('BUY_THRESHOLD', '0.7'))  # 70% confidence
SELL_THRESHOLD = float(os.getenv('SELL_THRESHOLD', '0.3'))  # 30% confidence
TIMEFRAME = os.getenv('TIMEFRAME', '5min')

# ======================
# 2. DATA FETCHER CLASS
# ======================
class ForexDataFetcher:
    def __init__(self):
        self.base_url = "https://api.twelvedata.com"
    
    def get_historical_data(self, symbol, days=30):
        """Get historical data for training"""
        params = {
            "symbol": symbol,
            "interval": TIMEFRAME,
            "outputsize": days*288,  # 288 5-min candles/day
            "apikey": TWELVEDATA_API_KEY
        }
        response = requests.get(f"{self.base_url}/time_series", params=params)
        data = response.json()['values']
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    
    def get_live_data(self, symbol, lookback=50):
        """Get latest candles for prediction"""
        params = {
            "symbol": symbol,
            "interval": TIMEFRAME,
            "outputsize": lookback,
            "apikey": TWELVEDATA_API_KEY
        }
        response = requests.get(f"{self.base_url}/time_series", params=params)
        data = response.json()['values']
        df = pd.DataFrame(data)
        df.set_index('datetime', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']].astype(float)

# ======================
# 3. MODEL CACHING SYSTEM
# ======================
class ModelCache:
    def __init__(self):
        os.makedirs('models', exist_ok=True)
        self.scalers = {}
        self.models = {}
    
    def load_or_train(self, symbol, data):
        """Load cached model or train new one"""
        model_path = f"models/{symbol}_model.h5"
        scaler_path = f"models/{symbol}_scaler.pkl"
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = load_model(model_path)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"♻️ Loaded cached model for {symbol}")
        else:
            print(f"🛠️ Training new model for {symbol}")
            model, scaler = self._train_new_model(data)
            model.save(model_path)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        
        self.models[symbol] = model
        self.scalers[symbol] = scaler
        return model, scaler
    
    def _train_new_model(self, data):
        """Train fresh model"""
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(30, len(scaled_data)):
            X.append(scaled_data[i-30:i])
            y.append(1 if scaled_data[i, 3] > scaled_data[i-1, 3] else 0)
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(30, 5)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.fit(np.array(X), np.array(y), epochs=10, batch_size=32, verbose=0)
        return model, scaler

# ======================
# 4. SIGNAL GENERATOR
# ======================
class SignalGenerator:
    def __init__(self):
        self.data_fetcher = ForexDataFetcher()
        self.cache = ModelCache()
        self.bot = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None
    
    def generate_signals(self):
        """Generate signals for all symbols"""
        signals = []
        for symbol in SYMBOLS:
            try:
                # Get model and data
                model, scaler = self.cache.load_or_train(
                    symbol,
                    self.data_fetcher.get_historical_data(symbol)
                )
                live_data = self.data_fetcher.get_live_data(symbol)
                
                # Prepare prediction
                scaled_data = scaler.transform(live_data)
                X = np.array([scaled_data[-30:]])
                
                # Predict
                prediction = model.predict(X, verbose=0)[0][0]
                price = live_data['close'].iloc[-1]
                
                # Check thresholds
                if prediction >= BUY_THRESHOLD:
                    signals.append({
                        'symbol': symbol,
                        'direction': "BUY",
                        'confidence': prediction,
                        'price': price
                    })
                elif prediction <= SELL_THRESHOLD:
                    signals.append({
                        'symbol': symbol,
                        'direction': "SELL", 
                        'confidence': 1 - prediction,
                        'price': price
                    })
                        
            except Exception as e:
                print(f"⚠️ Error processing {symbol}: {str(e)}")
        
        return signals
    
    def send_signals(self, signals):
        """Send formatted alerts via Telegram"""
        if not signals or not self.bot:
            return
            
        for signal in signals:
            emoji = "🚀" if signal['direction'] == "BUY" else "📉"
            self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=f"""
{emoji} *{signal['symbol']} {signal['direction']} Signal*
━━━━━━━━━━━━━━
• Confidence: `{signal['confidence']*100:.1f}%`
• Price: `{signal['price']:.5f}`
• Time: `{datetime.now().strftime("%H:%M:%S")}`
""",
                parse_mode='Markdown'
            )

# ======================
# 5. MAIN EXECUTION
# ======================
if __name__ == "__main__":
    print("""
    ███████╗ ██████╗ ██████╗ ███████╗██╗  ██╗
    ██╔════╝██╔═══██╗██╔══██╗██╔════╝╚██╗██╔╝
    █████╗  ██║   ██║██████╔╝█████╗   ╚███╔╝ 
    ██╔══╝  ██║   ██║██╔══██╗██╔══╝   ██╔██╗ 
    ██║     ╚██████╔╝██║  ██║███████╗██╔╝ ██╗
    ╚═╝      ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
    """)
    
    bot = SignalGenerator()
    
    # First-time model training
    if not os.path.exists('models'):
        print("⚙️ Training models for all symbols...")
        for symbol in SYMBOLS:
            bot.cache.load_or_train(
                symbol,
                bot.data_fetcher.get_historical_data(symbol)
            )
    
    # Main loop
    print("🔍 Starting signal monitoring...")
    while True:
        try:
            signals = bot.generate_signals()
            bot.send_signals(signals)
            time.sleep(180)  # Check every 3 minutes
        except Exception as e:
            print(f"⚠️ Error in main loop: {str(e)}")
            time.sleep(60)