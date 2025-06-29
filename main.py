import os
import asyncio
import requests
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from telegram import Bot
from telegram.error import TelegramError
import time
from datetime import datetime
import pickle
from dotenv import load_dotenv

# ======================
# 1. CONFIGURATION
# ======================
load_dotenv()

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN') or os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID') or os.environ.get('TELEGRAM_CHAT_ID')
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY') or os.environ.get('TWELVEDATA_API_KEY')

SYMBOLS = os.getenv('SYMBOLS', 'EURUSD,GBPUSD,USDJPY').split(',') 
BUY_THRESHOLD = float(os.getenv('BUY_THRESHOLD', '0.7'))
SELL_THRESHOLD = float(os.getenv('SELL_THRESHOLD', '0.3'))
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
            "outputsize": days*288,
            "apikey": TWELVEDATA_API_KEY
        }
        try:
            response = requests.get(f"{self.base_url}/time_series", params=params)
            response.raise_for_status()
            data = response.json()
            if 'values' not in data:
                print(f"⚠️ No 'values' in API response for {symbol}")
                return None
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            return df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        except Exception as e:
            print(f"⚠️ Error fetching data for {symbol}: {str(e)}")
            return None
    
    def get_live_data(self, symbol, lookback=50):
        """Get latest candles for prediction"""
        params = {
            "symbol": symbol,
            "interval": TIMEFRAME,
            "outputsize": lookback,
            "apikey": TWELVEDATA_API_KEY
        }
        try:
            response = requests.get(f"{self.base_url}/time_series", params=params)
            response.raise_for_status()
            data = response.json()
            if 'values' not in data:
                print(f"⚠️ No 'values' in API response for {symbol}")
                return None
            df = pd.DataFrame(data['values'])
            df.set_index('datetime', inplace=True)
            return df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        except Exception as e:
            print(f"⚠️ Error fetching live data for {symbol}: {str(e)}")
            return None

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
        if data is None:
            return None, None
            
        model_path = f"models/{symbol}_model.h5"
        scaler_path = f"models/{symbol}_scaler.pkl"
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                model = load_model(model_path)
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"♻️ Loaded cached model for {symbol}")
                return model, scaler
            except Exception as e:
                print(f"⚠️ Error loading model for {symbol}: {str(e)}")
                return self._train_new_model(data)
        else:
            return self._train_new_model(data)
    
    def _train_new_model(self, data):
        """Train fresh model"""
        try:
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
        except Exception as e:
            print(f"⚠️ Error training model: {str(e)}")
            return None, None

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
                hist_data = self.data_fetcher.get_historical_data(symbol)
                if hist_data is None:
                    continue
                    
                model, scaler = self.cache.load_or_train(symbol, hist_data)
                if model is None or scaler is None:
                    continue
                
                live_data = self.data_fetcher.get_live_data(symbol)
                if live_data is None:
                    continue
                
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
    
    async def send_signals(self, signals):
        """Send formatted alerts via Telegram"""
        if not signals or not self.bot:
            return
            
        for signal in signals:
            try:
                emoji = "🚀" if signal['direction'] == "BUY" else "📉"
                await self.bot.send_message(
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
            except TelegramError as e:
                print(f"⚠️ Telegram Error: {str(e)}")
            except Exception as e:
                print(f"⚠️ Unexpected Error sending message: {str(e)}")

# ======================
# 5. MAIN EXECUTION
# ======================
async def main():
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
            hist_data = bot.data_fetcher.get_historical_data(symbol)
            if hist_data is not None:
                bot.cache.load_or_train(symbol, hist_data)
    
    # Main loop
    print("🔍 Starting signal monitoring...")
    while True:
        try:
            signals = bot.generate_signals()
            await bot.send_signals(signals)
            await asyncio.sleep(180)  # Check every 3 minutes
        except Exception as e:
            print(f"⚠️ Error in main loop: {str(e)}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
