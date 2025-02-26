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

# ========== EXPLICIT SERVER CONFIGURATION ========== #
server = Flask(__name__)  # Must be declared first
app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.LUX])

# Rest of your TrendRadar class and other code remains unchanged...
# [Keep all previous TrendRadar and UI code exactly as before]
