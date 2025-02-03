import streamlit as st
st.set_page_config(page_title="AI Assistant", layout="wide")

import streamlit as st
from PIL import Image, ImageEnhance
import json
import os
import io
from datetime import datetime
import requests
import firebase_admin
from firebase_admin import credentials, db
import anthropic
from openai import OpenAI
import pygame
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import queue
import threading
import time
import numpy as np
from collections import defaultdict
import atexit
import random
import speech_recognition as sr
import pyttsx3
import tempfile
import wave
import pyaudio
import google.generativeai as genai
import replicate
from enum import Enum
from typing import Dict, Optional
import logging
import nltk
nltk.download('punkt')  # Download the punkt tokenizer data
from nltk.tokenize import sent_tokenize
import hashlib
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
openai_api_key = st.secrets["OPENAI_API_KEY"]
anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"]
REPLICATE_API_KEY = st.secrets["REPLICATE_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]

# Initialize API clients
openai_client = OpenAI(api_key=openai_api_key)
claude_client = anthropic.Anthropic(api_key=anthropic_api_key)
together_client = OpenAI(api_key=TOGETHER_API_KEY, base_url="https://api.together.xyz/v1")
genai.configure(api_key=GEMINI_API_KEY)
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_KEY

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize pygame mixer
pygame.mixer.init()

# File paths for data persistence - using Streamlit's session state for cloud deployment
if 'liminality_backlogs' not in st.session_state:
    st.session_state.liminality_backlogs = []
if 'astral_plane_backlogs' not in st.session_state:
    st.session_state.astral_plane_backlogs = []
if 'conversation_backlogs' not in st.session_state:
    st.session_state.conversation_backlogs = []
if 'key_learnings' not in st.session_state:
    st.session_state.key_learnings = []

def load_json_file(filepath):
    """Load JSON file safely"""
    if filepath.endswith('liminality_backlogs.json'):
        return st.session_state.liminality_backlogs
    elif filepath.endswith('astral_plane_backlogs.json'):
        return st.session_state.astral_plane_backlogs
    elif filepath.endswith('conversation_backlogs.json'):
        return st.session_state.conversation_backlogs
    elif filepath.endswith('key_learnings.json'):
        return st.session_state.key_learnings
    return []

def save_to_session_state(data, file_type):
    """Save data to session state"""
    if file_type == 'liminality':
        st.session_state.liminality_backlogs = data
    elif file_type == 'astral':
        st.session_state.astral_plane_backlogs = data
    elif file_type == 'conversation':
        st.session_state.conversation_backlogs = data
    elif file_type == 'key_learnings':
        st.session_state.key_learnings = data

def display_liminality_backlogs():
    """Display stored conversations in Liminality Backlogs"""
    try:
        backlogs = st.session_state.liminality_backlogs
            
        if not backlogs:
            st.warning("No training conversations found in Liminality Backlogs.")
            return
            
        st.subheader(" Liminality Backlogs")
        
        # Sort by timestamp (newest first)
        backlogs.sort(key=lambda x: x['timestamp'], reverse=True)
        
        for conv in backlogs:
            timestamp = conv.get('timestamp', 'Unknown time')
            messages = conv.get('messages', [])
            
            # Filter out missing trainer responses
            messages = [msg for msg in messages if "[Missing" not in msg.get('content', '')]
            
            if messages:  # Only show conversations that have actual messages
                # Create expandable section showing only timestamp when collapsed
                with st.expander(f"{timestamp}"):
                    # Display messages without speaker labels
                    for msg in messages:
                        content = msg.get('content', '')
                        
                        # Remove any "Speaker:" prefix if present
                        content = re.sub(r'^[^:]*:\s*', '', content)
                        
                        # Display the content without speaker label
                        st.markdown(content)
                            
    except Exception as e:
        st.error(f"Error displaying Liminality Backlogs: {str(e)}")
        logging.error(f"Error in display_liminality_backlogs: {e}")

def display_key_learnings():
    """Display key learnings in a separate tab"""
    try:
        learnings = st.session_state.key_learnings
        
        if not learnings:
            st.warning("No key learnings recorded yet.")
            return
            
        st.subheader(" Key Learnings")
        
        # Sort by timestamp (newest first)
        learnings.sort(key=lambda x: x['timestamp'], reverse=True)
        
        for learning in learnings:
            timestamp = learning.get('timestamp', 'Unknown time')
            key_points = learning.get('key_points', [])
            
            # Create expandable section showing only timestamp when collapsed
            with st.expander(f"{timestamp}"):
                # Display key points as bullet points
                for point in key_points:
                    st.markdown(f"• {point}")
            
    except Exception as e:
        st.error(f"Error displaying key learnings: {e}")
        logging.error(f"Error in display_key_learnings: {e}")
