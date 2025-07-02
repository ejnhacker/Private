import os
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from telegram import Bot, ParseMode
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# Disable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

# Configuration
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')

SYMBOLS = ['EUR/USD', 'GBP/USD']
TIMEFRAME = '15min'
CHECK_INTERVAL = 300  # 5 minutes

# Technical Indicators (Pure Python Implementation)
def calculate_rsi(series, window=14):
    """Manual RSI calculation without external dependencies"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Manual MACD calculation"""
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line, signal_line

class ForexDataFetcher:
    def __init__(self):
        self.base_url = "https://api.twelvedata.com/time_series"
        self.last_request = 0

    def _rate_limit(self):
        """Ensure 8 requests/minute limit"""
        elapsed = time.time() - self.last_request
        if elapsed < 8:
            time.sleep(8 - elapsed)
        self.last_request = time.time()

    def get_data(self, symbol, lookback):
        """Fetch OHLC data from API"""
        self._rate_limit()
        params = {
            "symbol": symbol,
            "interval": TIMEFRAME,
            "outputsize": lookback,
            "apikey": TWELVEDATA_API_KEY,
            "type": "forex"
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json().get('values', [])
            return self._process_data(data)
        except Exception as e:
            print(f"API Error: {e}")
            return pd.DataFrame()

    def _process_data(self, data):
        """Convert API data to DataFrame with indicators"""
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        # Convert to numeric
        cols = ['open', 'high', 'low', 'close']
        df[cols] = df[cols].apply(pd.to_numeric)
        
        # Calculate indicators
        df['rsi'] = calculate_rsi(df['close'])
        df['macd'], df['signal'] = calculate_macd(df['close'])
        
        return df.dropna()

class TradingBot:
    def __init__(self):
        self.fetcher = ForexDataFetcher()
        self.scaler = MinMaxScaler()
        self.model = self._build_model()
        self.bot = self._init_telegram()

    def _init_telegram(self):
        """Initialize Telegram bot"""
        try:
            bot = Bot(token=TELEGRAM_TOKEN)
            bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=f"🤖 Bot Started at {datetime.now().strftime('%H:%M:%S')}",
                parse_mode=ParseMode.MARKDOWN
            )
            return bot
        except Exception as e:
            print(f"Telegram Error: {e}")
            return None

    def _build_model(self):
        """Create LSTM model architecture"""
        model = Sequential([
            Input(shape=(30, 5)),  # open, high, low, close, rsi
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def generate_signal(self, df):
        """Generate trading signal"""
        if len(df) < 30:
            return None
            
        # Prepare features
        features = df[['open', 'high', 'low', 'close', 'rsi']]
        scaled = self.scaler.fit_transform(features)
        
        # Predict
        prediction = self.model.predict(np.array([scaled[-30:]]), verbose=0)[0][0]
        
        # Determine signal
        if prediction >= 0.65:
            return {'action': 'BUY', 'confidence': prediction, 'price': df['close'].iloc[-1]}
        elif prediction <= 0.35:
            return {'action': 'SELL', 'confidence': 1-prediction, 'price': df['close'].iloc[-1]}
        return None

    def send_alert(self, signal, symbol):
        """Send Telegram alert"""
        if not signal or not self.bot:
            return
            
        emoji = "🚀" if signal['action'] == 'BUY' else "🔻"
        message = (
            f"{emoji} *{symbol} {signal['action']} Signal*\n"
            f"━━━━━━━━━━━━━━\n"
            f"• Price: `{signal['price']:.5f}`\n"
            f"• Confidence: `{signal['confidence']*100:.1f}%`\n"
            f"• RSI: `{df['rsi'].iloc[-1]:.2f}`\n"
            f"• Time: `{datetime.now().strftime('%H:%M:%S')}`"
        )
        try:
            self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message,
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception as e:
            print(f"Telegram Send Error: {e}")

    def run(self):
        """Main trading loop"""
        print("🚀 Starting Forex Trading Bot")
        while True:
            try:
                for symbol in SYMBOLS:
                    data = self.fetcher.get_data(symbol, lookback=100)
                    if not data.empty:
                        signal = self.generate_signal(data)
                        if signal:
                            self.send_alert(signal, symbol)
                
                print(f"⏳ Next check in {CHECK_INTERVAL//60} minutes...")
                time.sleep(CHECK_INTERVAL)
                
            except KeyboardInterrupt:
                print("\n🛑 Bot stopped by user")
                break
            except Exception as e:
                print(f"⚠️ Error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
