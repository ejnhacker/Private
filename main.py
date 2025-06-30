import os
import requests
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from telegram import Bot
from telegram.constants import ParseMode
import time
from datetime import datetime
import pickle
from dotenv import load_dotenv

# ======================
# 1. कॉन्फिगरेशन
# ======================
load_dotenv()

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')

SYMBOLS = ['EUR/USD', 'GBP/USD']  # सही फॉरमैट
BUY_THRESHOLD = 0.65
SELL_THRESHOLD = 0.35
TIMEFRAME = '15min'
CHECK_INTERVAL = 300  # 5 मिनट

# ======================
# 2. डेटा फेचर (बिल्कुल सही इंडेंटेशन)
# ======================
class ForexDataFetcher:
    def __init__(self):
        self.base_url = "https://api.twelvedata.com"
        self.request_count = 0  # ✅ सही इंडेंटेशन
        self.last_request_time = 0  # ✅ __init__ के अंदर
    
    def get_historical_data(self, symbol, days=7):
        try:
            params = {
                "symbol": symbol,
                "interval": TIMEFRAME,
                "outputsize": days*96,
                "apikey": TWELVEDATA_API_KEY
            }
            response = requests.get(f"{self.base_url}/time_series", params=params)
            data = response.json()
            
            if 'values' not in data:
                print(f"❌ {symbol} के लिए डेटा नहीं मिला")
                return pd.DataFrame()
                
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            return df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            
        except Exception as e:
            print(f"🔥 डेटा फेच करने में त्रुटि: {e}")
            return pd.DataFrame()

# ======================
# 3. मॉडल मैनेजर
# ======================
class ModelManager:
    def __init__(self):
        os.makedirs('models', exist_ok=True)
        self.models = {}  # ✅ सही इंडेंटेशन
        self.scalers = {}  # ✅ समान लेवल
    
    def load_or_train(self, symbol, data):
        model_path = f"models/{symbol.replace('/','_')}_model.h5"
        scaler_path = f"models/{symbol.replace('/','_')}_scaler.pkl"
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                model = load_model(model_path)
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"♻️ {symbol} के लिए कैश्ड मॉडल लोड किया")
                return model, scaler
            except Exception as e:
                print(f"⚠️ मॉडल लोडिंग त्रुटि: {e}")
        
        # नया मॉडल ट्रेनिंग
        print(f"🛠️ {symbol} के लिए नया मॉडल ट्रेन कर रहा हूँ")
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(30, len(scaled_data)):
            X.append(scaled_data[i-30:i])
            y.append(1 if scaled_data[i,3] > scaled_data[i-1,3] else 0)
            
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=(30,5)),
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

# ======================
# 4. सिग्नल जनरेटर
# ======================
class SignalGenerator:
    def __init__(self):
        self.data_fetcher = ForexDataFetcher()
        self.model_manager = ModelManager()
        self.bot = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None
    
    def generate_signals(self):
        signals = []
        for symbol in SYMBOLS:
            try:
                # डेटा प्राप्त करें
                hist_data = self.data_fetcher.get_historical_data(symbol)
                live_data = self.data_fetcher.get_historical_data(symbol)  # हिस्टोरिकल डेटा का उपयोग करें
                
                if hist_data.empty or live_data.empty:
                    continue
                    
                # मॉडल प्राप्त करें
                model, scaler = self.model_manager.load_or_train(symbol, hist_data)
                
                # भविष्यवाणी करें
                scaled = scaler.transform(live_data)
                prediction = model.predict(np.array([scaled[-30:]]), verbose=0)[0][0]
                price = live_data['close'].iloc[-1]
                
                print(f"📊 {symbol} - मूल्य: {price:.5f}, भविष्यवाणी: {prediction:.2%}")
                
                # सिग्नल जनरेट करें
                if prediction >= BUY_THRESHOLD:
                    signals.append({
                        'symbol': symbol,
                        'direction': 'BUY',
                        'confidence': prediction,
                        'price': price
                    })
                elif prediction <= SELL_THRESHOLD:
                    signals.append({
                        'symbol': symbol,
                        'direction': 'SELL',
                        'confidence': 1-prediction,
                        'price': price
                    })
                    
            except Exception as e:
                print(f"⚠️ {symbol} प्रोसेस करने में त्रुटि: {e}")
                
        return signals
    
    def send_signals(self, signals):
        if not signals or not self.bot:
            return
            
        for signal in signals:
            try:
                emoji = "🚀" if signal['direction'] == 'BUY' else "📉"
                message = f"""
{emoji} *{signal['symbol']} {signal['direction']} सिग्नल*
━━━━━━━━━━━━━━
• विश्वास: `{signal['confidence']*100:.1f}%`
• मूल्य: `{signal['price']:.5f}`
• समय: `{datetime.now().strftime('%H:%M:%S')}`
"""
                self.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=message,
                    parse_mode=ParseMode.MARKDOWN
                )
                print(f"✅ {signal['symbol']} सिग्नल भेजा गया")
            except Exception as e:
                print(f"❌ सिग्नल भेजने में विफल: {e}")

# ======================
# 5. मुख्य एक्जीक्यूशन
# ======================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("💹 फॉरेक्स सिग्नल बॉट शुरू हो रहा है...")
    print("="*50 + "\n")
    
    bot = SignalGenerator()
    
    # टेलीग्राम टेस्ट
    if bot.bot:
        try:
            bot.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text="✅ बॉट सफलतापूर्वक शुरू हुआ!",
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception as e:
            print(f"⚠️ टेलीग्राम टेस्ट विफल: {e}")
    
    # मुख्य लूप
    while True:
        try:
            print("\n" + "-"*50)
            print(f"⏳ {datetime.now().strftime('%H:%M:%S')} पर जाँच की जा रही है...")
            
            signals = bot.generate_signals()
            bot.send_signals(signals)
            
            print(f"\n🕒 अगली जाँच {CHECK_INTERVAL//60} मिनट में...")
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n🛑 बॉट उपयोगकर्ता द्वारा रोका गया")
            break
        except Exception as e:
            print(f"🔥 मुख्य लूप में त्रुटि: {e}")
            time.sleep(60)
