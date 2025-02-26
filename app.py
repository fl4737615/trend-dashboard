import os
import nltk
import heapq
import logging
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from collections import defaultdict
from tenacity import retry, stop_after_attempt, wait_exponential
from defusedxml import defused_parse  # Security fix

# Configure logging
logging.basicConfig(level=logging.INFO)

# ========== NLTK CONFIG ========== #
NLTK_DIR = os.path.join(os.getenv("NLTK_DATA", "/opt/render/nltk_data"))
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)
nltk.download('vader_lexicon', download_dir=NLTK_DIR)

# ========== TREND ENGINE ========== #
class TrendRadar:
    _model = None  # Singleton pattern
    
    @classmethod
    def get_model(cls):
        if not cls._model:
            cls._model = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._model
    
    def __init__(self):
        self.model = self.get_model()
        self.sia = nltk.sentiment.vader.SentimentIntensityAnalyzer()
        self.session = requests.Session()  # Connection pooling
    
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
                    scored.append((
                        entity,
                        score,
                        trends[entity]['sources']
                    ))
            
            return heapq.nlargest(25, scored, key=lambda x: x[1])
        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}")
            return []

    # Data methods updated with session and security
    def _get_wiki_trends(self):
        try:
            date = (datetime.now() - timedelta(days=1)).strftime('%Y/%m/%d')
            url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{date}"
            response = self.session.get(url, timeout=10)
            return [item['article'] for item in response.json()['items'][0]['articles'][:25]]
        except Exception as e:
            logging.warning(f"Wikipedia failed: {str(e)}")
            return []

    def _get_research_pulse(self):
        try:
            response = self.session.get("http://export.arxiv.org/api/query?search_query=all&max_results=50", timeout=15)
            soup = defused_parse(response.content)  # XXE protection
            return [entry.title.text for entry in soup.find_all('entry')[:30]]
        except Exception as e:
            logging.warning(f"arXiv failed: {str(e)}")
            return []

    # Other data methods use self.session instead of raw requests...

# ========== UI ========== #
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

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
    return [
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(
                    html.Div([
                        html.Div(entity, className="trend-title"),
                        html.Div(
                            className="confidence-bar",
                            **{'data-score': str(len(sources))}
                        )
                    ])
                ),
                dbc.CardBody([
                    html.Div([
                        html.Span(src, className="source-badge") 
                        for src in sources
                    ])
                ])
            ], className="trend-card"),
            xs=12, sm=6, md=4, lg=3
        ) for entity, score, sources in trends
    ] if trends else error_card()

# Error card remains same...

if __name__ == "__main__":
    app.run_server(debug=False)
