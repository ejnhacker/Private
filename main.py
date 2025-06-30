import os
import time
import pickle
import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from telegram import Bot, ParseMode
import talib

# API Configurations
APIS = {
    'TWELVEDATA': {
        'url': 'https://api.twelvedata.com/time_series',
        'params': {'apikey': os.getenv('TWELVEDATA_KEY'), 'interval': '15min', 'type': 'forex'},
        'symbol_format': lambda s: {'symbol': s.replace('/', '')}
    },
    'ALPHAVANTAGE': {
        'url': 'https://www.alphavantage.co/query',
        'params': {'apikey': os.getenv('ALPHAVANTAGE_KEY'), 'function': 'FX_INTRADAY', 'interval': '15min'},
        'symbol_format': lambda s: {'from_symbol': s.split('/')[0], 'to_symbol': s.split('/')[1]}
    },
    'OANDA': {
        'url': 'https://api-fxpractice.oanda.com/v3/candles',
        'headers': {'Authorization': f"Bearer {os.getenv('OANDA_KEY')}"},
        'symbol_format': lambda s: {'instrument': s.replace('/', '_'), 'original_symbol': s}  # Store original
    }
}

class ProductionForexBot:
    def __init__(self):
        self.bot = Bot(token=os.getenv('TELEGRAM_TOKEN'))
        try:
            self.model = load_model('forex_model.keras')
            self.scaler = pickle.load(open('scaler.pkl', 'rb'))
        except Exception as e:
            self._send_alert(f"⚠️ Model Load Failed: {str(e)}")
            raise

    def _send_alert(self, message):
        """Send alerts to Telegram with error handling"""
        try:
            self.bot.send_message(
                chat_id=os.getenv('TELEGRAM_CHAT_ID'),
                text=message,
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception as e:
            print(f"Telegram alert failed: {str(e)}")

    def fetch_concurrent_data(self, symbol):
        """Fetch data with API-specific symbol formatting"""
        with ThreadPoolExecutor() as executor:
            futures = {}
            for api, config in APIS.items():
                params = {**config['params'], **config['symbol_format'](symbol)}
                futures[api] = executor.submit(
                    self._fetch_api_data,
                    api,
                    config['url'],
                    params,
                    config.get('headers')
                )
            return {api: future.result() for api, future in futures.items()}

    def _fetch_api_data(self, api_name, url, params=None, headers=None):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            data = response.json()
            
            # Store original symbol for OANDA
            if api_name == 'OANDA' and 'original_symbol' in params:
                data['original_symbol'] = params['original_symbol']
                
            return self._standardize_data(data, api_name)
        except Exception as e:
            print(f"[ERROR] {api_name} API failed: {url} - {str(e)}")
            return pd.DataFrame()

    def _standardize_data(self, data, api_name):
        """Convert API data to common format with symbol preservation"""
        df = pd.DataFrame()
        
        # TwelveData
        if api_name == 'TWELVEDATA' and 'values' in data:
            df = pd.DataFrame(data['values'])
            df['symbol'] = data.get('meta', {}).get('symbol', '').replace('', '/')
        
        # AlphaVantage
        elif api_name == 'ALPHAVANTAGE' and 'Time Series FX (15min)' in data:
            df = pd.DataFrame(data['Time Series FX (15min)']).T.reset_index()
            df.columns = ['datetime', 'open', 'high', 'low', 'close']
            df['symbol'] = f"{data.get('Meta Data', {}).get('2. From Symbol', '')}/{data.get('Meta Data', {}).get('3. To Symbol', '')}"
        
        # OANDA
        elif api_name == 'OANDA' and 'candles' in data:
            df = pd.DataFrame([{
                'datetime': c['time'],
                'open': c['mid']['o'],
                'high': c['mid']['h'],
                'low': c['mid']['l'],
                'close': c['mid']['c']
            } for c in data['candles']])
            df['symbol'] = data.get('original_symbol', 'UNKNOWN')
        
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
            return df[['symbol', 'open', 'high', 'low', 'close']]
        return pd.DataFrame()

    def generate_signal(self, df, dataframes):
        """Generate trading signal with proper symbol handling"""
        if len(df) < 30 or df.empty:
            return None
            
        # Get symbol from the first valid dataframe
        symbol = next((d['symbol'].iloc[-1] for d in dataframes if not d.empty), 'UNKNOWN')
        
        # Technical Analysis
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['macd'], _, _ = talib.MACD(df['close'])
        
        # Model Prediction
        features = self.scaler.transform(df[['open', 'high', 'low', 'close', 'rsi', 'macd']])
        prediction = self.model.predict(np.array([features[-30:]]), verbose=0)[0][0]
        
        # API Agreement Score
        api_agreement = min(3, len([d for d in dataframes if not d.empty])) / 3
        confidence = prediction * api_agreement
        
        signal = {
            'symbol': symbol,
            'direction': 'BUY' if confidence >= 0.65 else 'SELL' if confidence <= 0.35 else 'HOLD',
            'confidence': f"{confidence*100:.1f}%",
            'agreement': f"{api_agreement*100:.0f}%",
            'price': df['close'].iloc[-1],
            'rsi': df['rsi'].iloc[-1],
            'macd': df['macd'].iloc[-1]
        }
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {signal['symbol']} - "
              f"{signal['direction']} ({signal['confidence']} confidence)")
        return signal

    def run(self):
        """Main execution loop with enhanced error handling"""
        while True:
            try:
                for symbol in ['EUR/USD', 'GBP/USD']:
                    api_data = self.fetch_concurrent_data(symbol)
                    valid_data = [df for df in api_data.values() if not df.empty]
                    
                    if len(valid_data) >= 2:
                        combined = pd.concat(valid_data).groupby(level=0).agg({
                            'open': 'mean',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last'
                        })
                        
                        signal = self.generate_signal(combined, valid_data)
                        if signal and signal['direction'] != 'HOLD':
                            self._send_alert(
                                f"🚀 *{signal['symbol']} {signal['direction']}*\n"
                                f"Confidence: {signal['confidence']}\n"
                                f"Price: {signal['price']:.5f}\n"
                                f"RSI: {signal['rsi']:.2f}"
                            )
                
                time.sleep(300)
                
            except KeyboardInterrupt:
                print("\nBot stopped by user")
                break
                
            except Exception as e:
                error_msg = f"⚠️ Bot Error:\n{str(e)}"
                print(error_msg)
                self._send_alert(error_msg)
                time.sleep(60)  # Wait before retry

if __name__ == "__main__":
    bot = ProductionForexBot()
    bot.run()
