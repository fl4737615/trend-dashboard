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
import dash  # Needed for dash.callback_context
from dash import Dash, html, dcc, Input, Output, no_update, State
import dash_bootstrap_components as dbc
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from collections import defaultdict
from tenacity import retry, stop_after_attempt, wait_exponential
from defusedxml.lxml import parse as defused_parse
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # Explicit import for sentiment analyzer

# ========== EXTERNAL CONFIGURATION ==========
# Externalize parameters with defaults for flexibility and performance tuning.
DBSCAN_EPS = float(os.getenv('DBSCAN_EPS', '0.35'))
DBSCAN_MIN_SAMPLES = int(os.getenv('DBSCAN_MIN_SAMPLES', '2'))
THREADPOOL_MAX_WORKERS = int(os.getenv('THREADPOOL_MAX_WORKERS', '4'))
REFRESH_INTERVAL_MS = int(os.getenv('REFRESH_INTERVAL_MS', str(300 * 1000)))  # default 300000 ms (5 minutes)

# ========== INITIALIZATION SEQUENCE ==========
print("Application bootstrap sequence started")
logging.basicConfig(level=logging.INFO)

# ========== SERVER CONFIGURATION ==========
server = Flask(__name__)
app = Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.LUX],
    assets_folder="assets",
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}]
)

@server.route('/health')
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}, 200

# ========== NLTK SETUP ==========
try:
    NLTK_DIR = os.path.join(os.getenv("NLTK_DATA", "/opt/render/nltk_data"))
    os.makedirs(NLTK_DIR, exist_ok=True)
    nltk.data.path.append(NLTK_DIR)
    nltk.download('vader_lexicon', download_dir=NLTK_DIR, quiet=True)
    print("NLTK initialized successfully")
except Exception as e:
    print(f"NLTK failed: {str(e)}")
    raise

# ========== TREND ENGINE ==========
class TrendRadar:
    _model = None
    _shared_session = None  # Shared requests session

    @classmethod
    def get_model(cls):
        if not cls._model:
            try:
                print("Loading ML model...")
                cls._model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Model loaded successfully")
            except Exception as e:
                print(f"Model failed to load: {str(e)}")
                raise
        return cls._model

    @classmethod
    def get_shared_session(cls):
        if not cls._shared_session:
            cls._shared_session = requests.Session()
            cls._shared_session.headers.update({'User-Agent': 'Mozilla/5.0'})
        return cls._shared_session

    def __init__(self):
        self.model = self.get_model()
        self.sia = SentimentIntensityAnalyzer()  # Use the explicitly imported analyzer
        self.session = self.get_shared_session()
        self.start_time = datetime.now()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def get_trends(self):
        debug_data = {
            'sources': defaultdict(dict),
            'errors': [],
            'performance': {}
        }
        try:
            with ThreadPoolExecutor(max_workers=THREADPOOL_MAX_WORKERS) as executor:
                futures = {
                    executor.submit(self._get_wiki_trends): "web",
                    executor.submit(self._get_news_entities): "news",
                    executor.submit(self._get_stock_signals): "stocks",
                    executor.submit(self._get_research_pulse): "research"
                }

                trends = defaultdict(lambda: {'count': 0, 'sources': set()})
                for future in futures:
                    source_type = futures[future]
                    try:
                        result = future.result(timeout=12)
                        debug_data['sources'][source_type] = {
                            'status': 'success',
                            'count': len(result)
                        }
                        for entity in result:
                            trends[entity]['count'] += 1
                            trends[entity]['sources'].add(source_type)
                    except Exception as e:
                        debug_data['sources'][source_type] = {
                            'status': 'failed',
                            'error': str(e)
                        }
                        debug_data['errors'].append(str(e))
                        logging.error(f"{source_type} failed: {str(e)}")

            debug_data['performance']['fetch_duration'] = (
                datetime.now() - self.start_time
            ).total_seconds()

            analysis_start = datetime.now()
            final_trends = self._analyze_trends(trends)
            debug_data['performance']['analysis_duration'] = (
                datetime.now() - analysis_start
            ).total_seconds()

            return final_trends, debug_data
        except Exception as e:
            debug_data['errors'].append(str(e))
            raise

    def _analyze_trends(self, trends):
        try:
            entities = list(trends.keys())
            if not entities:
                return []
                
            embeddings = self.model.encode(entities)
            clusters = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(embeddings)
            
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
            response.raise_for_status()
            return [item['article'] for item in response.json()['items'][0]['articles'][:25]]
        except Exception as e:
            logging.warning(f"Wikipedia failed: {str(e)}")
            return []

    def _get_news_entities(self):
        entities = []
        try:
            data = self.session.get("https://api.gdeltproject.org/api/v2/doc/doc?query=sourcecountry:US&format=json", timeout=15).json()
            entities += [art['title'] for art in data.get('articles', [])[:30]]
        except Exception as e:
            logging.warning(f"GDELT failed: {str(e)}")
        
        newsapi_key = os.getenv('NEWSAPI_KEY', '')
        if newsapi_key and not newsapi_key.startswith('your_'):
            try:
                url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={newsapi_key}"
                newsapi_data = self.session.get(url, timeout=10).json()
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
                data = self.session.get(url, timeout=10).json()
                gainers = data.get('top_gainers', [])[:3]
                losers = data.get('top_losers', [])[:3]
                entities += [entry['ticker'] for entry in gainers + losers]
            except Exception as e:
                logging.warning(f"AlphaVantage failed: {str(e)}")
        
        try:
            tickers = ['TSLA', 'AAPL', 'AMZN', 'GOOG', 'META']
            _ = yf.download(tickers, period='1d', group_by='ticker')
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

# Create a shared instance of TrendRadar for all callbacks
trend_radar_instance = TrendRadar()

# ========== UI COMPONENTS ==========
app.layout = dbc.Container([
    dcc.Interval(id='refresh', interval=REFRESH_INTERVAL_MS, disabled=False),
    dcc.Store(id='debug-store', data={'logs': [], 'status': 'init'}),
    
    dbc.Row([
        dbc.Col(html.H1("Global Trend Radar", className="header"), width=8),
        dbc.Col(
            dbc.Switch(
                id="debug-toggle",
                label="Diagnostics Mode",
                value=False,
                persistence=True
            ), width=4
        )
    ]),
    
    dbc.Row(
        dbc.Col(
            dbc.Button("Refresh Trends", id="refresh-button", color="primary", className="refresh-button"),
            width=12
        )
    ),
    
    dbc.Row(
        dbc.Col(
            dbc.Alert(
                "Initializing trend detection engine...",
                id="init-alert",
                color="info",
                className="mb-3"
            )
        )
    ),
    
    dbc.Row(id='trend-grid'),
    
    dbc.Row(
        dbc.Col(
            html.Div(
                id='debug-console',
                className="debug-console",
                style={'display': 'none'}
            )
        )
    ),
    
    dcc.Loading(
        id="loading-spinner",
        type="circle",
        children=[html.Div(id="loading-output")],
        className="loading-spinner"
    )
], fluid=True)

# ========== CALLBACKS ==========
@app.callback(
    [Output('trend-grid', 'children'),
     Output('debug-store', 'data'),
     Output('init-alert', 'children'),
     Output('init-alert', 'color'),
     Output('init-alert', 'style')],
    [Input('refresh', 'n_intervals'),
     Input('refresh-button', 'n_clicks')],
    [State('debug-store', 'data')],
    prevent_initial_call=False
)
def update_interface(n_intervals, refresh_clicks, stored_data):
    ctx = dash.callback_context
    # Determine if this is the initial call or a manual/interval trigger
    initial_call = not ctx.triggered

    try:
        if initial_call:
            print("First load initialization")
            return (
                no_update,
                {'logs': [], 'status': 'loading'},
                "Warming up analysis engine...",
                "warning",
                {'display': 'block'}
            )

        trends, debug_data = trend_radar_instance.get_trends()
        logs = stored_data.get('logs', [])[-9:] + [{
            'timestamp': datetime.now().isoformat(),
            'status': 'success' if trends else 'empty',
            'sources': {k: v for k, v in debug_data['sources'].items()},
            'errors': debug_data['errors'],
            'performance': debug_data['performance']
        }]

        if not trends:
            return (
                error_card("No trends found - check data sources"),
                {'logs': logs, 'status': 'empty'},
                "No trends detected - running diagnostics...",
                "danger",
                {'display': 'block'}
            )

        return (
            create_cards(trends),
            {'logs': logs, 'status': 'ready'},
            "System operational - trends updating",
            "success",
            {'display': 'none'}
        )

    except Exception as e:
        error_msg = f"Critical failure: {str(e)[:200]}"
        print(f"Error: {error_msg}")
        return (
            error_card(error_msg),
            {'logs': stored_data.get('logs', []), 'status': 'error'},
            "System malfunction - check debug console",
            "danger",
            {'display': 'block'}
        )

@app.callback(
    Output('debug-console', 'children'),
    Output('debug-console', 'style'),
    Input('debug-store', 'data'),
    Input('debug-toggle', 'value')
)
def update_debug_panel(logs, show_debug):
    console_style = {'display': 'block'} if show_debug else {'display': 'none'}
    entries = []
    
    for entry in logs.get('logs', []):
        entry_html = html.Div(
            [
                html.Span(f"[{entry['timestamp']}] ", className="debug-time"),
                html.Span(entry.get('status', 'unknown').upper(), 
                          className=f"status-{entry.get('status', 'unknown')}"),
                html.Div([
                    html.Div([
                        html.Span(src, className="source-tag"),
                        html.Span(f"{info['status']} ({info.get('count',0)})", 
                                  className=f"source-{info['status']}")
                    ]) for src, info in entry.get('sources', {}).items()
                ], className="source-status"),
                html.Div([
                    html.Div([
                        html.Div(f"{metric['fetch_duration']:.1f}s", className="metric-value"),
                        html.Div("Data Fetch", className="metric-label")
                    ], className="metric-card"),
                    html.Div([
                        html.Div(f"{metric['analysis_duration']:.1f}s", className="metric-value"),
                        html.Div("Analysis", className="metric-label")
                    ], className="metric-card")
                ], className="performance-metrics") if 'performance' in entry else None
            ],
            className="debug-entry"
        )
        entries.append(entry_html)
    
    return entries, console_style

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
                            style={'width': f'{min(score * 25, 100)}%'}
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

def error_card(message="Temporary data outage - next scan in 5 minutes"):
    return dbc.Col(
        dbc.Card([
            dbc.CardBody(message)
        ], className="error-card"),
        width=12
    )

if __name__ == "__main__":
    print("Starting application server")
    app.run_server(debug=False)