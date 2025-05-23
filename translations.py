"""
Translation module for the stock analysis application
"""

# Dictionary containing translations for different languages
TRANSLATIONS = {
    'en': {  # English (default)
        'app_title': '📈 Stock Analysis Platform',
        'app_subtitle': 'Compare multiple stocks with charts, financial data, and technical indicators',
        'settings': 'Settings',
        'language': 'Language',
        'enter_symbols': 'Enter stock symbols (comma-separated)',
        'symbols_help': 'Example: AAPL,MSFT,GOOGL or 2330.TW,3008.TW for Taiwan stocks',
        'time_period': 'Select time period',
        'interval': 'Select interval',
        'tech_indicators': 'Technical Indicators',
        'sma': 'Simple Moving Average (SMA)',
        'ema': 'Exponential Moving Average (EMA)',
        'rsi': 'Relative Strength Index (RSI)',
        'macd': 'MACD',
        'bollinger': 'Bollinger Bands',
        'period': 'Period',
        'std_dev': 'Standard Deviation',
        'price_comparison': 'Price Comparison',
        'financial_data': 'Financial Data',
        'tech_analysis': 'Technical Analysis',
        'raw_data': 'Raw Data',
        'favorites': 'Favorites',
        'add_to_favorites': 'Add to Favorites',
        'remove_from_favorites': 'Remove from Favorites',
        'favorite_stocks': 'Favorite Stocks',
        'no_favorites': 'No favorite stocks yet. Add some stocks to your favorites!',
        'loading_data': 'Loading data for {}...',
        'no_data': 'No data available for {}. Please check the symbol.',
        'error_fetching': 'Error fetching data for {}: {}',
        'current_price': 'Current Price',
        'change': 'Change',
        'pct_change': 'Percentage Change',
        'pct_change_period': 'Percentage Change Since Start of Period',
        'price_with_indicators': '{} Price with Indicators',
        'volume': 'Volume',
        'select_stock': 'Select stock for technical analysis',
        'download': 'Download {} data as CSV',
        'download_all': 'Download All Stock Data as CSV',
        'enter_to_start': 'Enter stock symbols in the sidebar to get started.',
        'valid_symbol': 'Please enter at least one valid stock symbol.',
        'compare_stocks': 'Stock Price Comparison',
        'date': 'Date',
        'price_usd': 'Price (USD)',
        'stocks': 'Stocks',
        'financial_comparison': 'Financial Data Comparison',
        'company_name': 'Company Name',
        'market_cap': 'Market Cap',
        'pe_ratio': 'P/E Ratio',
        'forward_pe': 'Forward P/E',
        'peg_ratio': 'PEG Ratio',
        'dividend_yield': 'Dividend Yield (%)',
        'week_high': '52 Week High',
        'week_low': '52 Week Low',
        'day_ma_50': '50-Day MA',
        'day_ma_200': '200-Day MA',
        'revenue': 'Revenue',
        'profit_margin': 'Profit Margin (%)',
        'roe': 'Return on Equity (%)',
        'roa': 'Return on Assets (%)',
        'total_debt': 'Total Debt',
        'total_cash': 'Total Cash',
        'free_cash_flow': 'Free Cash Flow',
    },
    'zh-tw': {  # Traditional Chinese
        'app_title': '📈 股票分析平台',
        'app_subtitle': '比較多支股票的圖表、財務數據和技術指標',
        'settings': '設定',
        'language': '語言',
        'enter_symbols': '輸入股票代碼（以逗號分隔）',
        'symbols_help': '範例：AAPL,MSFT,GOOGL 或台灣股票 2330.TW,3008.TW',
        'time_period': '選擇時間範圍',
        'interval': '選擇間隔',
        'tech_indicators': '技術指標',
        'sma': '簡單移動平均線 (SMA)',
        'ema': '指數移動平均線 (EMA)',
        'rsi': '相對強弱指數 (RSI)',
        'macd': 'MACD指標',
        'bollinger': '布林帶',
        'period': '週期',
        'std_dev': '標準差',
        'price_comparison': '價格比較',
        'financial_data': '財務數據',
        'tech_analysis': '技術分析',
        'raw_data': '原始數據',
        'favorites': '收藏',
        'add_to_favorites': '加入收藏',
        'remove_from_favorites': '移除收藏',
        'favorite_stocks': '收藏的股票',
        'no_favorites': '尚未有收藏的股票。請將一些股票加入您的收藏！',
        'loading_data': '正在加載 {} 的數據...',
        'no_data': '{} 沒有可用數據。請檢查股票代碼。',
        'error_fetching': '獲取 {} 數據時出錯：{}',
        'current_price': '當前價格',
        'change': '漲跌',
        'pct_change': '漲跌幅',
        'pct_change_period': '自時期開始以來的百分比變化',
        'price_with_indicators': '{} 價格與指標',
        'volume': '成交量',
        'select_stock': '選擇股票進行技術分析',
        'download': '下載 {} 數據為CSV',
        'download_all': '下載所有股票數據為CSV',
        'enter_to_start': '在側邊欄輸入股票代碼以開始。',
        'valid_symbol': '請輸入至少一個有效的股票代碼。',
        'compare_stocks': '股票價格比較',
        'date': '日期',
        'price_usd': '價格（美元）',
        'stocks': '股票',
        'financial_comparison': '財務數據比較',
        'company_name': '公司名稱',
        'market_cap': '市值',
        'pe_ratio': '市盈率',
        'forward_pe': '預期市盈率',
        'peg_ratio': 'PEG比率',
        'dividend_yield': '股息收益率 (%)',
        'week_high': '52週最高價',
        'week_low': '52週最低價',
        'day_ma_50': '50日移動平均線',
        'day_ma_200': '200日移動平均線',
        'revenue': '營收',
        'profit_margin': '利潤率 (%)',
        'roe': '股本回報率 (%)',
        'roa': '資產回報率 (%)',
        'total_debt': '總債務',
        'total_cash': '總現金',
        'free_cash_flow': '自由現金流',
    }
}

def get_translation(key, language='en'):
    """
    Get the translation for a specific key in the specified language
    
    Parameters:
        key (str): The translation key
        language (str): The language code (default: 'en')
        
    Returns:
        str: The translated text, or the key itself if translation not found
    """
    if language not in TRANSLATIONS:
        language = 'en'  # Default to English if language not supported
        
    if key not in TRANSLATIONS[language]:
        # Return the English version if key not found in specified language
        return TRANSLATIONS['en'].get(key, key)
        
    return TRANSLATIONS[language][key]