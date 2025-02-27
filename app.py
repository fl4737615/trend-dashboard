import os
import io
import nltk
import heapq
import logging
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from flask import Flask
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from collections import defaultdict
from tenacity import retry, stop_after_attempt, wait_exponential
from defusedxml.lxml import parse as defused_parse

# ========== SERVER CONFIGURATION ==========
server = Flask(__name__)
app = Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.LUX],
    assets_folder="assets"  # CRITICAL FIX: Explicit assets path
)

@server.route('/health')
def health_check():
    return "OK", 200

# ========== NLTK SETUP ==========
NLTK_DIR = os.path.join(os.getenv("NLTK_DATA", "/opt/render/nltk_data"))
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', download_dir=NLTK_DIR)

# ========== TREND ENGINE ==========
class TrendRadar:
    _model = None
    
    @classmethod
    def get_model(cls):
        if not cls._model:
            cls._model = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._model
    
    def __init__(self):
        self.model = self.get_model()
        self.sia = nltk.sentiment.vader.SentimentIntensityAnalyzer()
        self.session = requests.Session()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1))
    def get_trends(self):
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._get_wiki_trends): "web",
                executor.submit(self._get_news_entities): "news",
                executor.submit(self._get_stock_signals): "stocks",
                executor.submit(self._get_research_pulse): "research"
            }
            
            trends = defaultdict(lambda: {'count':0, 'sources':set()})
            for future in futures:
                try:
                    for entity in future.result():
                        trends[entity]['count'] += 1
                        trends[entity]['sources'].add(futures[future])
                except Exception as e:
                    logging.error(f"Data source failed: {str(e)}")
        
        return self._analyze_trends(trends)

    def _analyze_trends(self, trends):
        try:
            entities = list(trends.keys())
            embeddings = self.model.encode(entities)
            clusters = DBSCAN(eps=0.35, min_samples=2).fit_predict(embeddings)
            
            scored = []
            for idx, cluster in enumerate(clusters):
                if cluster != -1:
                    entity = entities[idx]
                    sentiment = self.sia.polarity_scores(entity)['compound']
                    score = trends[entity]['count'] * 0.7 + (sentiment + 1) * 0.3
                    scored.append((entity, score, trends[entity]['sources']))
            
            return heapq.nlargest(25, scored, key=lambda x: x[1])
        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}")
            return []

    def _get_wiki_trends(self):
        try:
            date = (datetime.now() - timedelta(days=1)).strftime('%Y/%m/%d')
            url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{date}"
            response = self.session.get(url, timeout=10)
            return [item['article'] for item in response.json()['items'][0]['articles'][:25]]
        except Exception as e:
            logging.warning(f"Wikipedia failed: {str(e)}")
            return []

    def _get_news_entities(self):
        entities = []
        try:
            data = requests.get("https://api.gdeltproject.org/api/v2/doc/doc?query=sourcecountry:US&format=json", timeout=15).json()
            entities += [art['title'] for art in data.get('articles', [])[:30]]
        except Exception as e:
            logging.warning(f"GDELT failed: {str(e)}")
        
        newsapi_key = os.getenv('NEWSAPI_KEY', '')
        if newsapi_key and not newsapi_key.startswith('your_'):
            try:
                url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={newsapi_key}"
                newsapi_data = requests.get(url, timeout=10).json()
                entities += [art['title'] for art in newsapi_data.get('articles', [])[:20]]
            except Exception as e:
                logging.warning(f"NewsAPI failed: {str(e)}")
        
        return entities

    def _get_stock_signals(self):
        entities = []
        av_key = os.getenv('ALPHAVANTAGE_KEY', '')
        
        if av_key and not av_key.startswith('your_'):
            try:
                url = f"https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={av_key}"
                data = requests.get(url, timeout=10).json()
                gainers = data.get('top_gainers', [])[:3]
                losers = data.get('top_losers', [])[:3]
                entities += [entry['ticker'] for entry in gainers + losers]
            except Exception as e:
                logging.warning(f"AlphaVantage failed: {str(e)}")
        
        try:
            tickers = ['TSLA', 'AAPL', 'AMZN', 'GOOG', 'META']
            data = yf.download(tickers, period='1d', group_by='ticker')
            entities += [f"{ticker} Stock" for ticker in tickers]
        except Exception as e:
            logging.warning(f"Yahoo Finance failed: {str(e)}")
        
        return entities

    def _get_research_pulse(self):
        try:
            response = self.session.get("http://export.arxiv.org/api/query?search_query=all&max_results=50", timeout=15)
            tree = defused_parse(io.BytesIO(response.content))
            return [entry.findtext('title') for entry in tree.findall('.//entry')[:30]]
        except Exception as e:
            logging.warning(f"arXiv failed: {str(e)}")
            return []

# ========== UI ==========
app.layout = dbc.Container([
    dcc.Interval(id='refresh', interval=300*1000),
    dbc.Row(dbc.Col(html.H1("Global Trend Radar", className="header"))),
    dbc.Row(id='trend-grid', className="g-2")
], fluid=True)

@app.callback(
    Output('trend-grid', 'children'),
    Input('refresh', 'n_intervals')
)
def update_interface(n):
    try:
        engine = TrendRadar()
        trends = engine.get_trends()
        return create_cards(trends)
    except Exception as e:
        logging.error(f"Critical failure: {str(e)}")
        return error_card()

def create_cards(trends):
    ICON_MAP = {'web': '🌐', 'news': '📰', 'stocks': '💹', 'research': '📚'}
    return [
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(
                    html.Div([
                        html.Div(entity, className="trend-title"),
                        html.Div(
                            className="confidence-bar",
                            style={'width': f'{min(score*25, 100)}%'}
                        )
                    ])
                ),
                dbc.CardBody([
                    html.Div([
                        html.Span(ICON_MAP[src], className="source-badge")
                        for src in sources
                    ])
                ])
            ], className="trend-card"),
            xs=12, sm=6, md=4, lg=3
        ) for entity, score, sources in trends
    ] if trends else error_card()

def error_card():
    return dbc.Col(
        dbc.Card([
            dbc.CardBody("⚠️ Temporary data outage - next scan in 5 minutes")
        ], className="error-card"),
        width=12
    )

if __name__ == "__main__":
    app.run_server(debug=False)
