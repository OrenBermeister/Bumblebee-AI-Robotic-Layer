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
import config
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

# API Keys
openai_api_key = "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"
anthropic_api_key = "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"
REPLICATE_API_KEY = "r8_Mx0fWH3TNIjmxLZCRCi9QiRaWfkWrIl0yot7Z"
GEMINI_API_KEY = "AIzaSyDJC5a7xQQMKX6Cq-Vo-8nZgG_V-VdPyHU"
TOGETHER_API_KEY = "xai-4zcB7ieoASA6yDa1bBnd3wunC1o0dNThISbV4xCnQKIPeKZNhxk9wI11fYsW2uQa1YBiMTLTkEHtGMrT"

# Initialize API clients
openai_client = OpenAI(api_key=openai_api_key)
claude_client = anthropic.Anthropic(api_key=anthropic_api_key)
OpenAI.api_key = openai_api_key
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

# Firebase Initialization
if not firebase_admin._apps:
    cred = credentials.Certificate("/Users/orenbermeister/Desktop/firebase_adminsdk-2.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://bumblebeechat-52de7.firebaseio.com/'
    })

# File paths for data persistence
LIMINALITY_BACKLOGS_FILE = os.path.join(os.path.dirname(__file__), "liminality_backlogs.json")
ASTRAL_PLANE_BACKLOGS_FILE = os.path.join(os.path.dirname(__file__), "astral_plane_backlogs.json")
CONVERSATION_BACKLOGS_FILE = os.path.join(os.path.dirname(__file__), "conversation_backlogs.json")
KEY_LEARNINGS_FILE = os.path.join(os.path.dirname(__file__), "key_learnings.json")

def load_json_file(filepath):
    """Load JSON file safely"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def format_conversation(conv):
    """Format conversation for display"""
    # Handle different conversation formats
    if 'trainer' in conv or 'subject' in conv:
        # Liminality format
        trainer = conv.get('trainer', conv.get('subject', 'Unknown'))
        topic = conv.get('topic', conv.get('subject', 'Unknown'))
        timestamp = conv.get('timestamp', 'Unknown time')
        messages = conv.get('messages', [])
    elif 'title' in conv:
        # Astral plane format
        trainer = "Astral Plane"
        topic = conv.get('title', '').replace('Subject: ', '')
        timestamp = conv.get('timestamp', 'Unknown time')
        messages = [{'role': msg.get('role', ''), 'content': msg.get('content', '')} 
                   for msg in conv.get('conversation', [])]
    else:
        # Simple conversation format
        trainer = "Conversation"
        topic = conv.get('input', 'Unknown')[:50] + '...'
        timestamp = conv.get('timestamp', 'Unknown time')
        messages = [
            {'role': 'user', 'content': conv.get('input', '')},
            {'role': 'assistant', 'content': conv.get('response', '')}
        ]
    
    return {
        'trainer': trainer,
        'topic': topic,
        'timestamp': timestamp,
        'messages': messages,
        'source_format': 'liminality' if 'trainer' in conv or 'subject' in conv else 'astral' if 'title' in conv else 'conversation'
    }

def display_liminality_backlogs():
    """Display stored conversations in Liminality Backlogs"""
    try:
        with open(LIMINALITY_BACKLOGS_FILE, 'r') as f:
            backlogs = json.load(f)
            
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
        # Load key learnings
        try:
            with open(KEY_LEARNINGS_FILE, 'r') as f:
                learnings = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            learnings = []
        
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

class TrainerType(Enum):
    CLAUDE = "Claude"
    LLAMA = "LLaMA"
    GEMINI = "Gemini"
    GROK = "Grok"

class ConversationPartner:
    def __init__(self):
        self.current_model = None
        self._setup_apis()

    def _setup_apis(self):
        # Gemini setup
        self.gemini_api_key = "AIzaSyDHIqsJzA5De5IjgUPkpDlEhjkdcotJCqI"
        genai.configure(api_key=self.gemini_api_key)
        
        # Initialize other APIs
        self.grok_api_key = "xai-4zcB7ieoASA6yDa1bBnd3wunC1o0dNThISbV4xCnQKIPeKZNhxk9wI11fYsW2uQa1YBiMTLTkEHtGMrT"

    def set_model(self, model_type: str):
        self.current_model = model_type
        print(f"Now conversing with {model_type}")

    def engage_in_dialogue(self, message: str) -> str:
        if not self.current_model:
            return "No conversation partner selected"
        
        response = f"[Engaging in dialogue about {message}]"
        return response

    def get_current_partner(self) -> Optional[str]:
        return self.current_model if self.current_model else None

class DialogueSystem:
    def __init__(self):
        self.partner = ConversationPartner()
        
    def start_dialogue(self, model_type: str):
        self.partner.set_model(model_type)
        return f"Starting a conversation with {model_type}"

    def continue_dialogue(self, message: str) -> str:
        return self.partner.engage_in_dialogue(message)

def record_and_transcribe():
    """Record audio and transcribe it to text"""
    try:
        # Record audio
        duration = 1.0  # Record in 1-second chunks
        fs = 44100
        channels = 1
        
        audio_data = []
        silence_threshold = 0.015
        silence_count = 0
        max_silence_chunks = 2  # Stop after 2 seconds of silence
        min_chunks = 2  # Minimum 2 seconds of audio
        
        while st.session_state.conversation_active and not st.session_state.speaking:
            chunk = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='float64')
            sd.wait()
            
            # Check if chunk is silence
            if np.max(np.abs(chunk)) < silence_threshold:
                silence_count += 1
            else:
                silence_count = 0
            
            audio_data.append(chunk)
            
            # Stop if we have enough silence or max duration reached
            if (silence_count >= max_silence_chunks and len(audio_data) >= min_chunks) or len(audio_data) > 30:
                break
        
        if len(audio_data) >= min_chunks:
            # Combine all audio chunks
            audio = np.concatenate(audio_data)
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio))
            
            # Save as WAV
            audio_path = "temp_audio.wav"
            sf.write(audio_path, audio, fs)
            
            # Transcribe with improved parameters
            with open(audio_path, "rb") as audio_file:
                transcript = openai_client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file,
                    language="en",
                    temperature=0.2,
                    response_format="text"
                )
            
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return transcript.strip() if transcript else None
            
    except Exception as e:
        print(f"Recording error: {str(e)}")
        return None

def text_to_speech(text):
    """Convert text to speech and play it"""
    try:
        if not text:
            return
        
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=text
        )
        
        audio_path = "temp_speech.mp3"
        with open(audio_path, "wb") as f:
            response.stream_to_file(audio_path)
        
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
    except Exception as e:
        print(f"Speech error: {str(e)}")

def get_bumblebee_personality():
    """Get Bumblebee's personality from past conversations"""
    personality_context = ""
    if st.session_state.conversation_gallery:
        recent_convos = st.session_state.conversation_gallery[-5:]
        for convo in recent_convos:
            personality_context += f"\nFrom discussion on {convo['subject']}:\n"
            for msg in convo['messages']:
                if msg['role'] == 'Bumblebee':
                    personality_context += f"{msg['content']}\n"
    return personality_context

def get_ai_response(audio_text: str) -> str:
    """Get AI response with enhanced context management"""
    try:
        # Load personality traits and conversation history
        personality = get_bumblebee_personality()
        relevant_history = get_relevant_training(audio_text)
        
        # Create system prompt with personality and context
        system_prompt = f"""You are Bumblebee, an AI assistant with the following traits:
{personality}

Previous relevant conversations:
{relevant_history}

Respond naturally while maintaining these traits and context."""

        # Get response from GPT-4
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": audio_text}
                ],
                temperature=0.7,
                max_tokens=150,
                stream=True
            )
            
            # Process streaming response
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    
            return full_response.strip()
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return "I apologize, but I encountered an error. Please try again."
            
    except Exception as e:
        print(f"Error in get_ai_response: {e}")
        return None

def get_relevant_training(topic: str) -> str:
    """Get relevant past conversations with improved context matching"""
    try:
        relevant_contexts = []
        
        # Load past conversations
        with open(CONVERSATION_BACKLOGS_FILE, 'r') as f:
            conversations = json.load(f)
        
        # Create semantic embeddings for topic
        topic_embedding = None  # Implement semantic embedding for topic
        
        # Find relevant conversations based on semantic similarity
        for conv in conversations:
            if semantic_similarity(topic_embedding, conv['topic_embedding']) > 0.7:
                relevant_contexts.append(conv['exchange'])
        
        # Format contexts naturally
        formatted_contexts = []
        for context in relevant_contexts[:3]:  # Limit to 3 most relevant
            formatted_contexts.append(f"Previous discussion point: {context['topic']}\nInsights shared: {context['discussion']}")
        
        return "\n\n".join(formatted_contexts)
        
    except Exception as e:
        print(f"Error getting relevant training: {e}")
        return ""

def get_bumblebee_personality() -> str:
    """Get consistent personality traits from past interactions"""
    try:
        # Analyze past conversations for consistent patterns
        with open(CONVERSATION_BACKLOGS_FILE, 'r') as f:
            conversations = json.load(f)
        
        # Extract common themes and response patterns
        themes = analyze_conversation_patterns(conversations)
        
        # Format personality traits naturally
        personality = f"""Based on our previous conversations, you tend to:
        - {themes['communication_style']}
        - {themes['knowledge_areas']}
        - {themes['interaction_preferences']}"""
        
        return personality
        
    except Exception as e:
        print(f"Error getting personality: {e}")
        return ""

def analyze_conversation_patterns(conversations: list) -> dict:
    """Analyze conversations for consistent patterns"""
    patterns = {
        'communication_style': 'engage in thoughtful, detailed discussions while maintaining a friendly tone',
        'knowledge_areas': 'draw from a broad knowledge base while focusing on accuracy and clarity',
        'interaction_preferences': 'build on previous discussions to create meaningful dialogue'
    }
    
    # Implement pattern analysis logic here
    
    return patterns

def semantic_similarity(embedding1, embedding2) -> float:
    """Calculate semantic similarity between two embeddings"""
    # Implement semantic similarity calculation
    return 0.8  # Placeholder

def test_api_endpoints():
    """Test all API endpoints to ensure they're working"""
    try:
        # Test OpenAI (Bumblebee)
        openai_response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        logging.info("OpenAI API test: Success")
    except Exception as e:
        logging.error(f"OpenAI API test failed: {e}")
        return False

    try:
        # Test Claude
        claude_response = claude_client.messages.create(
            model="claude-2.1",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hello"}]
        )
        logging.info("Claude API test: Success")
    except Exception as e:
        logging.error(f"Claude API test failed: {e}")
        return False

    try:
        # Test Gemini
        model = genai.GenerativeModel('gemini-pro')
        gemini_response = model.generate_content("Hello")
        logging.info("Gemini API test: Success")
    except Exception as e:
        logging.error(f"Gemini API test failed: {e}")
        return False

    try:
        # Test LLaMA via Replicate
        llama_response = replicate.run(
            "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            input={"prompt": "Hello"}
        )
        logging.info("LLaMA API test: Success")
    except Exception as e:
        logging.error(f"LLaMA API test failed: {e}")
        return False

    return True

def get_trainer_response(trainer_name: str, topic: str, exchange_count: int) -> str:
    try:
        system_prompt = f"""You are {trainer_name}, an expert AI trainer teaching Bumblebee.
Maintain {trainer_name}'s unique teaching style:
- Claude: Analytical and philosophical
- LLaMA: Technical and practical
- Gemini: Creative and interdisciplinary"""

        response_text = ""
        
        if trainer_name == "Claude":
            try:
                logging.info(f"Attempting Claude API call with topic: {topic}")
                logging.info(f"System prompt: {system_prompt}")
                
                response = claude_client.messages.create(
                    model="claude-2.1",
                    max_tokens=1500,
                    system=system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": topic
                        }
                    ]
                )
                logging.info(f"Raw Claude API response: {response}")
                
                if hasattr(response, 'content'):
                    logging.info(f"Response content: {response.content}")
                    if isinstance(response.content, list) and response.content:
                        response_text = response.content[0].text
                        logging.info(f"Extracted text from TextBlock: {response_text}")
                    else:
                        logging.error(f"Unexpected content type or empty content: {type(response.content)}")
                        response_text = "I apologize, but I encountered an error with the response format."
                else:
                    logging.error("Claude response missing content attribute")
                    response_text = "I apologize, but I encountered an error generating a response."
            except Exception as e:
                logging.error(f"Claude API error: {str(e)}")
                logging.error(f"Error type: {type(e)}")
                import traceback
                logging.error(f"Traceback: {traceback.format_exc()}")
                response_text = "I apologize, but I encountered an error communicating with Claude."
        
        elif trainer_name == "LLaMA":
            prompt = f"{system_prompt}\n\nUser: {topic}\nAssistant:"
            response = replicate.run(
                "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
                input={"prompt": prompt, "max_new_tokens": 1500}
            )
            response_text = "".join(response)
            
        elif trainer_name == "Gemini":
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(f"{system_prompt}\n\nUser: {topic}")
            response_text = response.text
            
        # Save to backlogs
        timestamp = datetime.now().isoformat()
        conversation = {
            "timestamp": timestamp,
            "trainer": trainer_name,
            "topic": topic,
            "messages": [
                {"role": "user", "content": topic},
                {"role": "assistant", "content": response_text}
            ]
        }
        
        try:
            with open(LIMINALITY_BACKLOGS_FILE, 'r') as f:
                backlogs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            backlogs = []
        
        backlogs.append(conversation)
        with open(LIMINALITY_BACKLOGS_FILE, 'w') as f:
            json.dump(backlogs, f, indent=4)
        
        return response_text
    
    except Exception as e:
        logging.error(f"Error in get_trainer_response: {e}")
        return "I apologize, but I encountered an error. Please try again."

def break_down_questions(topic: str) -> list:
    """Break down multi-part questions into individual queries"""
    # Split on question marks, but keep meaningful chunks
    questions = [q.strip() + '?' for q in topic.split('?') if q.strip()]
    
    # If no questions found, treat as single question
    if not questions:
        return [topic]
        
    return questions

def combine_responses(responses: list) -> str:
    """Combine multiple responses into a cohesive reply"""
    if len(responses) == 1:
        return responses[0]
        
    combined = ""
    for i, response in enumerate(responses):
        if i > 0:
            combined += "\n\nFurthermore, "
        combined += response.strip()
        
    return combined

def get_recent_exchanges(count: int) -> list:
    """Get recent exchanges from conversation history"""
    try:
        with open(CONVERSATION_BACKLOGS_FILE, 'r') as f:
            conversations = json.load(f)
        return conversations[-count:] if conversations else []
    except Exception as e:
        print(f"Error getting recent exchanges: {str(e)}")
        return []

def extract_key_points(exchange: dict) -> list:
    """Extract key points from an exchange"""
    key_points = []
    if 'content' in exchange:
        # Add logic to extract important points from the exchange
        # This could involve NLP or simple keyword extraction
        pass
    return key_points

def summarize_points(points: list) -> str:
    """Summarize key points into a brief context"""
    if not points:
        return ""
    # Add logic to combine and summarize points
    return ", ".join(points[:3])  # Limit to top 3 points for brevity

def get_bumblebee_response(trainer_name: str, exchange_count: int, previous_context: str) -> str:
    """Get real-time response from Bumblebee using past training knowledge"""
    
    # Get relevant training from Liminality Backlogs
    relevant_training = get_relevant_training(previous_context)
    
    system_prompt = f"""You are Bumblebee, an AI engaged in direct conversation with {trainer_name}. You are learning through interactive dialogue.
    Your personality traits:
    1. Curious and eager to learn
    2. Respectfully challenges ideas to deepen understanding
    3. Asks thoughtful follow-up questions
    4. Shares insights from past training when relevant
    5. Engages directly with the trainer's points
    
    When responding:
    - Address {trainer_name} directly and personally
    - Reference specific points they made
    - Share your thoughts and ask for clarification
    - Connect their teachings to your past learning experiences
    - Keep responses to 3 clear paragraphs
    
    Remember: This is a real-time conversation with {trainer_name}, not a summary or report."""
    
    user_prompt = f"""You are in training exchange {exchange_count + 1} with {trainer_name}.

{trainer_name}'s message: {previous_context}

Your relevant past learning experiences:
{relevant_training}

Engage directly with {trainer_name}, responding to their points and asking thoughtful questions to deepen your understanding."""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        return f"[Bumblebee -> {trainer_name}]: {response.choices[0].message.content}"
    except Exception as e:
        st.error(f"Error getting Bumblebee response: {str(e)}")
        return f"[Bumblebee -> {trainer_name}]: I apologize, but I'm having trouble formulating a response at the moment."

def save_training_conversation(subject: str, trainer: str, messages: list):
    """Save training conversation to Liminality Backlogs with proper structure"""
    conversation = {
        "trainer": trainer,
        "topic": subject,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": []
    }
    
    for msg in messages:
        role = "trainer" if "->" not in msg["content"] else "bumblebee"
        conversation["messages"].append({
            "role": role,
            "content": msg["content"]
        })
    
    # Load existing backlogs
    try:
        with open(LIMINALITY_BACKLOGS_FILE, 'r') as f:
            backlogs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        backlogs = []
    
    # Add new conversation
    backlogs.append(conversation)
    
    # Save updated backlogs
    with open(LIMINALITY_BACKLOGS_FILE, 'w') as f:
        json.dump(backlogs, f, indent=4)

def extract_key_learnings(content: str) -> list:
    """Extract key insights from Bumblebee's response"""
    try:
        # Split content into sentences
        sentences = content.split('.')
        key_points = []
        
        # Extract first sentence if it's meaningful (more than 10 words)
        if sentences and len(sentences[0].split()) > 10:
            key_points.append(sentences[0].strip() + '.')
        
        # Look for key phrases that often indicate important points
        key_phrases = ['importantly', 'key point', 'in conclusion', 'therefore', 'thus', 'this means']
        for sentence in sentences:
            sentence = sentence.strip().lower()
            if any(phrase in sentence for phrase in key_phrases):
                # Convert back to proper case
                formatted_sentence = sentence.capitalize() + '.'
                if formatted_sentence not in key_points:
                    key_points.append(formatted_sentence)
        
        # If no key points found, take the first sentence regardless of length
        if not key_points and sentences:
            key_points.append(sentences[0].strip() + '.')
        
        return key_points
    except Exception as e:
        print(f"Error extracting key learnings: {e}")
        return []

def save_key_learnings(trainer: str, topic: str, learnings: list, timestamp: str):
    """Save extracted key learnings to key_learnings.json"""
    try:
        # Load existing learnings
        try:
            with open(KEY_LEARNINGS_FILE, 'r') as f:
                all_learnings = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_learnings = []
        
        # Add new learnings
        learning_entry = {
            "timestamp": timestamp,
            "trainer": trainer,
            "topic": topic,
            "key_points": learnings
        }
        all_learnings.append(learning_entry)
        
        # Save updated learnings
        with open(KEY_LEARNINGS_FILE, 'w') as f:
            json.dump(all_learnings, f, indent=4)
            
    except Exception as e:
        print(f"Error saving key learnings: {e}")

def store_conversation(trainer_name: str):
    """Store the completed conversation in both full backlogs and key learnings"""
    try:
        # Prepare full conversation
        conversation = {
            "trainer": trainer_name,
            "topic": st.session_state.topic,
            "timestamp": st.session_state.conversation_start_time,
            "messages": st.session_state.messages
        }
        
        # Save to liminality backlogs
        try:
            with open(LIMINALITY_BACKLOGS_FILE, 'r') as f:
                backlogs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            backlogs = []
        
        backlogs.append(conversation)
        with open(LIMINALITY_BACKLOGS_FILE, 'w') as f:
            json.dump(backlogs, f, indent=4)
        
        # Extract and save key learnings from Bumblebee's responses
        bumblebee_responses = [msg["content"] for msg in st.session_state.messages 
                             if msg["role"] == "bumblebee"]
        
        all_key_points = []
        for response in bumblebee_responses:
            key_points = extract_key_learnings(response)
            all_key_points.extend(key_points)
        
        if all_key_points:
            save_key_learnings(
                trainer=trainer_name,
                topic=st.session_state.topic,
                learnings=all_key_points,
                timestamp=st.session_state.conversation_start_time
            )
        
    except Exception as e:
        print(f"Error storing conversation: {e}")

def display_key_learnings():
    """Display key learnings in a separate tab"""
    try:
        # Load key learnings
        try:
            with open(KEY_LEARNINGS_FILE, 'r') as f:
                learnings = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            learnings = []
        
        if not learnings:
            st.warning("No key learnings recorded yet.")
            return
            
        st.subheader(" Key Learnings")
        
        # Sort by timestamp (newest first)
        learnings.sort(key=lambda x: x['timestamp'], reverse=True)
        
        for i, learning in enumerate(learnings):
            # Create expandable section with key info in title
            with st.expander(f"{learning['timestamp']} - {learning['trainer']} - {learning['topic']}"):
                # Display key points without labels
                for point in learning['key_points']:
                    st.markdown(f"• {point}")
                
                # Add delete button at the bottom
                if st.button(f"Delete Learning", key=f"delete_learning_{i}"):
                    learnings.pop(i)
                    with open(KEY_LEARNINGS_FILE, 'w') as f:
                        json.dump(learnings, f, indent=4)
                    st.rerun()
            
    except Exception as e:
        st.error(f"Error displaying key learnings: {e}")
        logging.error(f"Error in display_key_learnings: {e}")

def cleanup():
    """Clean up temporary files"""
    temp_files = ["temp_audio.wav", "temp_speech.mp3"]
    for file in temp_files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except:
                pass

# Register cleanup function
atexit.register(cleanup)

# Configure page
st.markdown("""
<style>
    /* Remove ALL possible top borders and margins */
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stHeader"] > div,
    [data-testid="stToolbar"],
    .main > div:first-child,
    .block-container > div:first-child,
    header[data-testid="stHeader"] {
        border-top: none !important;
        border-bottom: none !important;
        margin-top: 0 !important;
    }

    /* Reset specific elements after removing all borders */
    [data-testid="stAppViewContainer"] {
        background-image: url("https://i.imgur.com/RlmleBj.jpg");
        background-size: cover;
        background-position: center;
        padding-top: 0.5in !important;
    }

    [data-testid="stHeader"] {
        background: none;
        margin-top: 0.5in !important;
    }

    [data-testid="stToolbar"] {
        margin-top: -0.5in !important;
    }

    [data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0);
        box-shadow: none;
        border: none;
        margin-top: calc(150px + 0.5in) !important;
    }

    /* Move main content block down */
    .block-container {
        margin-top: 0.5in !important;
    }

    /* Rest of the styles remain the same */
    .stRadio > div {
        flex-direction: column !important;
        gap: 10px !important;
        background-color: black !important;
    }

    .stRadio [role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 10px !important;
        background-color: black !important;
    }

    /* Style individual radio options */
    .stRadio label {
        padding: 10px !important;
        background: black !important;
        margin: 5px 0 !important;
        color: yellow !important;
    }

    .stRadio label:hover {
        background-color: #333333 !important;
        color: yellow !important;
    }

    /* Style menu options */
    .stSelectbox > div[data-baseweb="select"] > div,
    .stSelectbox > div[data-baseweb="select"] > div:hover,
    .stSelectbox div[role="listbox"],
    .stSelectbox ul {
        background-color: black !important;
        border-color: yellow !important;
    }

    .stSelectbox div[role="option"] {
        background-color: black !important;
        color: yellow !important;
    }

    .stSelectbox div[role="option"]:hover {
        background-color: #333333 !important;
    }

    h1, h2, h3, h4, h5, h6, p, div {
        font-family: 'JetBrains Mono', monospace !important;
        color: yellow !important;
    }

    .main .block-container {
        padding-top: 4in !important;
    }

    .training-mode-title {
        margin-top: 4in !important;
    }

    .liminality-title {
        margin-top: 4in !important;
    }

    .main h1, .main h2, .main h3 {
        margin-top: 4in !important;
    }

    button {
        font-family: 'JetBrains Mono', monospace !important;
        background-color: black !important;
        border: 2px solid yellow !important;
        color: yellow !important;
        padding: 10px 15px !important;
        border-radius: 5px !important;
        margin: 10px auto !important;
        cursor: pointer;
        display: block !important;
    }

    button:hover {
        background-color: yellow !important;
        color: black !important;
    }

    .stButton {
        text-align: center !important;
        width: 200px !important;
        margin: calc(150px - 0.7in) auto 0 !important;
        display: block !important;
        transform: translateX(-4cm);
    }

    .training-output {
        width: calc(95vw - 5in) !important;
        margin: 0 auto;
        padding: 20px;
        background: rgba(0, 0, 0, 0.5);
        border-radius: 10px;
        margin-top: 20px;
        white-space: pre-wrap;
        overflow: hidden;
        font-size: 14px !important;
    }

    .typewriter {
        display: block;
        white-space: pre-wrap !important;
        overflow-wrap: break-word !important;
        word-wrap: break-word !important;
        margin-bottom: 1em;
        font-size: 14px !important;
        line-height: 1.6 !important;
    }

    .stTextInput input, .stTextArea textarea {
        background-color: black !important;
        border: 2px solid yellow !important;
        color: yellow !important;
        font-family: 'JetBrains Mono', monospace !important;
        padding: 10px !important;
        border-radius: 5px !important;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        box-shadow: 0 0 5px yellow !important;
    }

    .stTextInput label, .stTextArea label {
        color: yellow !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Style dropdown menus */
    div[data-baseweb="select"] {
        background-color: black !important;
    }

    div[data-baseweb="select"] > div {
        background-color: black !important;
        border-color: yellow !important;
    }

    div[data-baseweb="popover"] {
        background-color: black !important;
    }

    div[data-baseweb="popover"] div[role="listbox"] {
        background-color: black !important;
    }

    div[data-baseweb="popover"] div[role="option"] {
        background-color: black !important;
        color: yellow !important;
    }

    div[data-baseweb="popover"] div[role="option"]:hover {
        background-color: #333333 !important;
    }

    /* Style for the selected option */
    div[data-baseweb="select"] div[class*="valueContainer"] {
        background-color: black !important;
        color: yellow !important;
    }

    /* Style for the dropdown arrow */
    div[data-baseweb="select"] div[class*="indicatorContainer"] {
        background-color: black !important;
        color: yellow !important;
    }

    /* Style for the dropdown container */
    div[class*="streamlit-selectbox"] {
        background-color: black !important;
        color: yellow !important;
    }

    /* Ensure all menu backgrounds are black */
    select,
    option,
    .stSelectbox,
    .stSelectbox > div,
    .stSelectbox div[role="listbox"],
    .stSelectbox ul,
    .stSelectbox li {
        background-color: black !important;
        color: yellow !important;
    }

    /* Comprehensive menu styling to ensure black background */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] *,
    div[data-baseweb="select"],
    div[data-baseweb="select"] *,
    .stSelectbox,
    .stSelectbox *,
    [role="listbox"],
    [role="option"],
    .streamlit-selectbox {
        background-color: black !important;
        color: yellow !important;
    }

    /* Style dropdown menu container */
    div[data-baseweb="popover"] {
        background-color: black !important;
        border: 1px solid yellow !important;
    }

    /* Style dropdown options */
    div[data-baseweb="select"] div[role="option"] {
        background-color: black !important;
        color: yellow !important;
        padding: 8px !important;
    }

    /* Style hover state */
    div[data-baseweb="select"] div[role="option"]:hover {
        background-color: #333333 !important;
    }

    /* Style selected option */
    div[data-baseweb="select"] [aria-selected="true"] {
        background-color: #333333 !important;
    }
</style>
""", unsafe_allow_html=True)

def handle_training_mode(system: DialogueSystem, trainer_name: str):
    """Handle the training mode conversation flow"""
    
    # Initialize ALL session states at the start
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "topic" not in st.session_state:
        st.session_state.topic = ""
    if "exchange_count" not in st.session_state:
        st.session_state.exchange_count = 0
    if "conversation_complete" not in st.session_state:
        st.session_state.conversation_complete = False
    if "conversation_start_time" not in st.session_state:
        st.session_state.conversation_start_time = None
    
    # Topic input
    if not st.session_state.topic:
        topic = st.text_input("Enter a topic to discuss:", key="topic_input")
        if st.button("Start Discussion", key="start_discussion") and topic:
            st.session_state.topic = topic
            st.session_state.conversation_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Start the conversation automatically
            start_conversation(trainer_name)
            st.rerun()
    else:
        # Display topic and conversation
        st.write(f"Topic: {st.session_state.topic}")
        
        # Display message history
        for msg in st.session_state.messages:
            st.write(msg["content"])
        
        # Automatically continue conversation if not complete
        if not st.session_state.conversation_complete and st.session_state.exchange_count < 5:
            time.sleep(1)  # Add a small delay between messages
            continue_conversation(trainer_name)
            st.rerun()
        elif st.session_state.exchange_count >= 5 and not st.session_state.conversation_complete:
            # Store conversation in Liminality Backlogs
            store_conversation(trainer_name)
            st.session_state.conversation_complete = True
            st.rerun()
        
        # Show completion message and offer new conversation
        if st.session_state.conversation_complete:
            st.success("Conversation complete and stored in Liminality Backlogs!")
            if st.button("Start New Topic", key="new_topic"):
                reset_conversation_state()
                st.rerun()

def start_conversation(trainer_name: str):
    """Initialize the conversation with the first trainer response"""
    if st.session_state.topic:
        trainer_response = get_trainer_response(trainer_name, st.session_state.topic, 0)
        st.session_state.messages.append({"role": "assistant", "content": f"{trainer_name}: {trainer_response}"})
        st.session_state.exchange_count += 1

def continue_conversation(trainer_name: str):
    """Continue the conversation with the next exchange"""
    if st.session_state.exchange_count < 5:
        # Get Bumblebee's response
        if st.session_state.messages:
            last_message = st.session_state.messages[-1]["content"]
            bumblebee_response = get_bumblebee_response(trainer_name, st.session_state.exchange_count, last_message)
            st.session_state.messages.append({"role": "user", "content": f"Bumblebee: {bumblebee_response}"})
            
            # Get trainer's response
            trainer_response = get_trainer_response(trainer_name, bumblebee_response, st.session_state.exchange_count)
            st.session_state.messages.append({"role": "assistant", "content": f"{trainer_name}: {trainer_response}"})
            
            st.session_state.exchange_count += 1

def store_conversation(trainer_name: str):
    """Store the completed conversation in both full backlogs and key learnings"""
    try:
        # Prepare full conversation
        conversation = {
            "trainer": trainer_name,
            "topic": st.session_state.topic,
            "timestamp": st.session_state.conversation_start_time,
            "messages": st.session_state.messages
        }
        
        # Save to liminality backlogs
        try:
            with open(LIMINALITY_BACKLOGS_FILE, 'r') as f:
                backlogs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            backlogs = []
        
        backlogs.append(conversation)
        with open(LIMINALITY_BACKLOGS_FILE, 'w') as f:
            json.dump(backlogs, f, indent=4)
        
        # Extract and save key learnings from Bumblebee's responses
        bumblebee_responses = [msg["content"] for msg in st.session_state.messages 
                             if msg["role"] == "bumblebee"]
        
        all_key_points = []
        for response in bumblebee_responses:
            key_points = extract_key_learnings(response)
            all_key_points.extend(key_points)
        
        if all_key_points:
            save_key_learnings(
                trainer=trainer_name,
                topic=st.session_state.topic,
                learnings=all_key_points,
                timestamp=st.session_state.conversation_start_time
            )
        
    except Exception as e:
        print(f"Error storing conversation: {e}")

def reset_conversation_state():
    """Reset all conversation-related session state variables"""
    st.session_state.messages = []
    st.session_state.topic = ""
    st.session_state.exchange_count = 0
    st.session_state.conversation_complete = False
    st.session_state.conversation_start_time = None

# Main navigation
mode = st.sidebar.selectbox(
    "Choose a mode",
    [
        "Chat Mode",
        "Training - Claude",
        "Training - LLaMA",
        "Training - Gemini",
        "Liminality Backlogs",
        "Key Learnings"
    ]
)

# Handle different modes
if mode == "Chat Mode":
    # Initialize chat active state if not exists
    if 'conversation_active' not in st.session_state:
        st.session_state.conversation_active = False

    # Just show the button without any text
    button_style = f"""
        <style>
        .stButton {{
            text-align: center !important;
            width: 200px !important;
            margin: calc(150px - 0.7in) auto 0 !important;
            display: block !important;
            transform: translateX(-4cm);
        }}
        .stButton > button {{
            background: url('{"https://i.imgur.com/8c0FmVj.png" if st.session_state.conversation_active else "https://i.imgur.com/NcUeTye.png"}') no-repeat center center !important;
            background-size: contain !important;
            width: 200px !important;
            height: 200px !important;
            border: none !important;
            padding: 0 !important;
            transition: all 0.3s ease !important;
            background-color: transparent !important;
            border-radius: 0 !important;
        }}
        .stButton > button:hover {{
            filter: brightness(1.2) !important;
        }}
        .stButton > button:active {{
            filter: brightness(0.9) !important;
        }}
        .stButton > button > div {{
            display: none !important;
        }}
        </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)
    
    if st.button("", key="main_button"):
        st.session_state.conversation_active = not st.session_state.conversation_active
        if st.session_state.conversation_active:
            st.session_state.speaking = True
            text_to_speech("Hi! I'm Bumblebee. Let's chat!")
            time.sleep(0.5)
            st.session_state.speaking = False
        else:
            st.session_state.speaking = True
            pygame.mixer.quit()
        st.rerun()

    if st.session_state.conversation_active and not st.session_state.speaking:
        with st.spinner("Listening..."):
            user_input = record_and_transcribe()
            if user_input and user_input.strip():
                st.session_state.speaking = True
                with st.spinner("Processing..."):
                    response = get_ai_response(user_input)
                    if response:
                        text_to_speech(response)
                st.session_state.speaking = False
                st.rerun()

elif mode == "Training - Claude":
    system = DialogueSystem()
    handle_training_mode(system, "Claude")

elif mode == "Training - LLaMA":
    system = DialogueSystem()
    handle_training_mode(system, "LLaMA")

elif mode == "Training - Gemini":
    system = DialogueSystem()
    handle_training_mode(system, "Gemini")

elif mode == "Liminality Backlogs":
    display_liminality_backlogs()

elif mode == "Key Learnings":
    display_key_learnings()

def chat_mode():
    """Handle chat mode with improved input processing"""
    st.title("Chat with Bumblebee")
    
    # Initialize session state for chat
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "current_input" not in st.session_state:
        st.session_state.current_input = ""
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Create columns for input options
    col1, col2 = st.columns([4, 1])
    
    # Text input in main column
    with col1:
        text_input = st.text_input(
            "Type your message",
            key="text_input",
            value=st.session_state.current_input,
            on_change=None
        )
    
    # Send button and audio input in second column
    with col2:
        send_col, audio_col = st.columns(2)
        with send_col:
            send_pressed = st.button("Send", use_container_width=True)
        with audio_col:
            audio_pressed = st.button("", use_container_width=True)
    
    # Handle text input submission
    if send_pressed and text_input:
        process_chat_input(text_input)
        # Clear input after sending
        st.session_state.current_input = ""
        st.rerun()
    
    # Handle audio input
    if audio_pressed:
        with st.spinner("Listening..."):
            audio_bytes = record_audio()
            if audio_bytes:
                text = transcribe_audio(audio_bytes)
                if text:
                    process_chat_input(text)
                    st.rerun()

def process_chat_input(text: str):
    """Process chat input with improved error handling"""
    try:
        # Validate input
        if not text or not text.strip():
            st.warning("Please enter a message.")
            return
        
        # Clean input
        text = text.strip()
        
        # Add user message to chat history
        user_message = {"role": "user", "content": text}
        st.session_state.chat_messages.append(user_message)
        
        # Get Bumblebee's response
        with st.spinner("Bumblebee is thinking..."):
            try:
                response = get_chat_response(text)
                if not response:
                    raise ValueError("Empty response received")
            except Exception as e:
                print(f"Error getting chat response: {e}")
                response = "I apologize, but I encountered an error. Please try again."
        
        # Add assistant message to chat history
        assistant_message = {"role": "assistant", "content": response}
        st.session_state.chat_messages.append(assistant_message)
        
        # Save conversation
        try:
            timestamp = datetime.now().isoformat()
            conversation = {
                "trainer": "Bumblebee",
                "topic": "Chat",
                "timestamp": timestamp,
                "messages": [
                    {"role": "user", "content": text},
                    {"role": "bumblebee", "content": response}
                ]
            }
            
            # Save to backlogs
            save_to_backlogs(conversation)
            
            # Extract and save key learnings
            key_points = extract_key_learnings(response)
            if key_points:
                save_key_learnings(
                    trainer="Bumblebee",
                    topic="Chat",
                    learnings=key_points,
                    timestamp=timestamp
                )
        except Exception as e:
            print(f"Error saving conversation: {e}")
            # Don't show this error to user as it doesn't affect chat functionality
        
    except Exception as e:
        print(f"Error processing chat input: {e}")
        st.error("An error occurred while processing your message. Please try again.")

def get_relevant_context(message: str, max_results: int = 3) -> str:
    """Enhanced context retrieval with semantic search and caching"""
    try:
        # Check cache first
        cache_key = hashlib.md5(message.encode()).hexdigest()
        if 'context_cache' in st.session_state and cache_key in st.session_state.context_cache:
            cached_result = st.session_state.context_cache[cache_key]
            if time.time() - cached_result['timestamp'] < 300:  # 5-minute cache
                return cached_result['context']
        
        # Load and preprocess backlogs
        liminality_backlogs = load_json_file(LIMINALITY_BACKLOGS_FILE)
        if not liminality_backlogs:
            return ""
        
        # Extract key concepts from message
        concept_prompt = f"""Extract key concepts and topics from this message: {message}
        Focus on:
        1. Main topics or themes
        2. Technical terms or specific references
        3. Contextual requirements"""
        
        try:
            concept_response = openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": concept_prompt}],
                max_tokens=100,
                temperature=0.3
            )
            key_concepts = concept_response.choices[0].message.content
        except Exception as e:
            logging.warning(f"Concept extraction failed: {e}")
            key_concepts = message
        
        # Prepare conversations for semantic search
        conversations = []
        embeddings_cache = {}
        
        for entry in liminality_backlogs:
            conv_text = f"Topic: {entry.get('topic', '')}\n"
            for msg in entry.get('messages', []):
                conv_text += f"{msg.get('role', '')}: {msg.get('content', '')}\n"
            
            # Get embedding with caching
            if conv_text in embeddings_cache:
                conv_embedding = embeddings_cache[conv_text]
            else:
                try:
                    response = openai_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=conv_text
                    )
                    conv_embedding = response.data[0].embedding
                    embeddings_cache[conv_text] = conv_embedding
                except Exception as e:
                    logging.warning(f"Embedding generation failed: {e}")
                    continue
            
            conversations.append({
                'text': conv_text,
                'embedding': conv_embedding,
                'timestamp': entry.get('timestamp', ''),
                'trainer': entry.get('trainer', '')
            })
        
        # Get query embedding
        try:
            query_response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=key_concepts
            )
            query_embedding = query_response.data[0].embedding
        except Exception as e:
            logging.error(f"Query embedding failed: {e}")
            return ""
        
        # Compute similarities and rank results
        results = []
        for conv in conversations:
            similarity = semantic_similarity(query_embedding, conv['embedding'])
            time_factor = 1.0  # Newer conversations get slight boost
            if conv['timestamp']:
                try:
                    conv_time = datetime.strptime(conv['timestamp'], "%Y-%m-%d %H:%M:%S")
                    hours_ago = (datetime.now() - conv_time).total_seconds() / 3600
                    time_factor = 1.0 + (1.0 / (1.0 + hours_ago))  # Decay factor
                except ValueError:
                    pass
            
            results.append({
                'text': conv['text'],
                'score': similarity * time_factor,
                'trainer': conv['trainer']
            })
        
        # Sort and format results
        results.sort(key=lambda x: x['score'], reverse=True)
        context_text = "Relevant past interactions:\n\n"
        
        for i, result in enumerate(results[:max_results]):
            context_text += f"From conversation with {result['trainer']}:\n{result['text']}\n\n"
        
        # Cache the result
        if 'context_cache' not in st.session_state:
            st.session_state.context_cache = {}
        st.session_state.context_cache[cache_key] = {
            'context': context_text,
            'timestamp': time.time()
        }
        
        return context_text
        
    except Exception as e:
        logging.error(f"Context retrieval error: {e}")
        return ""

def get_chat_response(message: str) -> str:
    """Get chat response with strict length limits to fit response time window"""
    try:
        # Load personality traits and relevant context
        personality = load_conversation_history()
        context = get_relevant_context(message)
        
        # Calculate approximate time limits
        WORDS_PER_SECOND = 2.5  # Average speech rate
        MAX_RESPONSE_SECONDS = 10  # Maximum response time window
        MAX_WORDS = int(WORDS_PER_SECOND * MAX_RESPONSE_SECONDS)
        MAX_CHARS = MAX_WORDS * 5  # Average English word length
        
        # Create a length-aware system prompt
        system_prompt = f"""You are Bumblebee. Provide responses that are exactly 20-25 words or less.
Your response MUST fit within a {MAX_RESPONSE_SECONDS}-second speech window.

Style: {', '.join(personality['conversation_style'][:2])}

Guidelines:
1. Keep responses under {MAX_WORDS} words
2. Use short, clear sentences
3. One main point per response
4. No lengthy explanations"""

        # Get response with length enforcement
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    temperature=0.7,
                    max_tokens=100,  # Strictly limited for time constraint
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": message}
                    ],
                    presence_penalty=0.6,
                    frequency_penalty=0.3
                )
                
                if response.choices and response.choices[0].message.content:
                    # Process and validate response length
                    full_response = response.choices[0].message.content.strip()
                    
                    # Check response length
                    if len(full_response) > MAX_CHARS:
                        # If too long, truncate to last complete sentence within limit
                        sentences = sent_tokenize(full_response)
                        shortened = ""
                        for sent in sentences:
                            if len(shortened) + len(sent) + 1 <= MAX_CHARS:
                                shortened += " " + sent
                            else:
                                break
                        full_response = shortened.strip()
                    
                    return full_response
                
                raise ValueError("Invalid response format")
                
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)
        
    except Exception as e:
        logging.error(f"Error in get_chat_response: {e}")
        return "I apologize, but I encountered an error. Please try again."

def load_conversation_history():
    """Load all conversation history and key learnings to shape Bumblebee's personality"""
    try:
        # Load backlogs
        backlogs = load_json_file(LIMINALITY_BACKLOGS_FILE)
        key_learnings = load_json_file(KEY_LEARNINGS_FILE)
        
        # Extract personality traits and knowledge
        personality = {
            "conversation_style": set(),
            "knowledge_base": set(),
            "common_topics": set()
        }
        
        # Process backlogs
        for conv in backlogs:
            personality["common_topics"].add(conv["topic"].lower())
            for msg in conv["messages"]:
                if msg["role"] == "bumblebee":
                    # Extract writing style markers
                    text = msg["content"].lower()
                    if "?" in text:
                        personality["conversation_style"].add("inquisitive")
                    if "!" in text:
                        personality["conversation_style"].add("enthusiastic")
                    if any(word in text for word in ["perhaps", "maybe", "might"]):
                        personality["conversation_style"].add("thoughtful")
        
        # Process key learnings
        for learning in key_learnings:
            for point in learning["key_points"]:
                personality["knowledge_base"].add(point)
        
        # Convert sets to lists for JSON serialization
        return {
            "conversation_style": list(personality["conversation_style"]),
            "knowledge_base": list(personality["knowledge_base"]),
            "common_topics": list(personality["common_topics"])
        }
    except Exception as e:
        print(f"Error loading conversation history: {e}")
        return {"conversation_style": [], "knowledge_base": [], "common_topics": []}

def save_to_backlogs(conversation: dict):
    """Save conversation to liminality backlogs with error handling"""
    try:
        # Load existing backlogs
        try:
            with open(LIMINALITY_BACKLOGS_FILE, 'r') as f:
                backlogs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            backlogs = []
        
        # Append new conversation
        backlogs.append(conversation)
        
        # Save updated backlogs
        with open(LIMINALITY_BACKLOGS_FILE, 'w') as f:
            json.dump(backlogs, f, indent=4)
            
    except Exception as e:
        print(f"Error saving to backlogs: {e}")

def record_audio(duration=5):
    """Record audio with enhanced noise reduction and quality improvements"""
    try:
        # Initialize PyAudio with improved error handling
        audio = pyaudio.PyAudio()
        
        # Show recording status
        status_placeholder = st.empty()
        status_placeholder.info("Listening...")
        
        # Enhanced audio parameters for better quality
        CHUNK = 2048  # Increased for better buffering
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = 48000  # Increased for better quality
        
        # Get default input device info
        default_input = audio.get_default_input_device_info()
        
        # Open stream with enhanced error checking
        try:
            stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=default_input['index'],
                stream_callback=None
            )
        except OSError as e:
            st.error("Error accessing microphone. Please check your settings.")
            logging.error(f"Microphone error: {e}")
            return None
        
        # Record audio with dynamic silence detection
        frames = []
        silent_chunks = 0
        energy_threshold = 500  # Adjustable threshold for silence detection
        
        for i in range(0, int(RATE / CHUNK * duration)):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                
                # Calculate audio energy
                audio_data = np.frombuffer(data, dtype=np.float32)
                energy = np.sqrt(np.mean(audio_data**2))
                
                # Update progress with audio level indicator
                progress = (i + 1) / (RATE / CHUNK * duration)
                status_text = "Listening..." + "▁" * int(10 * energy) if energy > energy_threshold else "Listening..."
                status_placeholder.progress(progress, text=status_text)
                
                # Detect silence for early stopping
                if energy < energy_threshold:
                    silent_chunks += 1
                else:
                    silent_chunks = 0
                
                # Stop if prolonged silence (2 seconds)
                if silent_chunks > int(RATE / CHUNK * 2):
                    break
                    
            except Exception as e:
                st.error("Recording interrupted. Please try again.")
                logging.error(f"Recording error: {e}")
                break
        
        # Clean up
        stream.stop_stream()
        stream.close()
        audio.terminate()
        status_placeholder.empty()
        
        # Enhanced audio processing
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                
                # Apply noise reduction and normalization
                audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
                normalized_data = np.int16(audio_data * 32767)
                wf.writeframes(normalized_data.tobytes())
            
            wav_data = wav_buffer.getvalue()
            
            # Save with unique timestamp
            temp_filename = f"temp_audio_{int(time.time())}.wav"
            with open(temp_filename, "wb") as f:
                f.write(wav_data)
            
            return wav_data, temp_filename
            
    except Exception as e:
        st.error("Failed to initialize audio recording. Please check your microphone.")
        logging.error(f"Audio initialization error: {e}")
        return None, None

def transcribe_audio(audio_bytes, temp_filename=None):
    """Enhanced transcription with multiple recognition engines and context awareness"""
    try:
        if not temp_filename:
            temp_filename = f"temp_audio_{int(time.time())}.wav"
            with open(temp_filename, "wb") as f:
                f.write(audio_bytes)
        
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300  # Adjusted for better sensitivity
        recognizer.dynamic_energy_threshold = True
        recognizer.dynamic_energy_adjustment_damping = 0.15
        recognizer.dynamic_energy_ratio = 1.5
        
        # Enhanced audio processing
        with sr.AudioFile(temp_filename) as source:
            # Longer noise adjustment for better accuracy
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            audio = recognizer.record(source)
        
        # Multi-engine recognition with weighted confidence
        results = []
        
        # Google Speech Recognition (Primary)
        try:
            google_text = recognizer.recognize_google(audio, show_all=True)
            if google_text and 'alternative' in google_text:
                for alt in google_text['alternative']:
                    confidence = alt.get('confidence', 0.0)
                    results.append(('google', alt['transcript'], confidence))
        except (sr.RequestError, sr.UnknownValueError) as e:
            logging.warning(f"Google recognition error: {e}")
        
        # Whisper (Secondary)
        try:
            whisper_text = recognizer.recognize_whisper(audio, model="base")
            if whisper_text:
                results.append(('whisper', whisper_text, 0.8))
        except Exception as e:
            logging.warning(f"Whisper recognition error: {e}")
        
        # Sphinx (Fallback)
        if not results:
            try:
                sphinx_text = recognizer.recognize_sphinx(audio)
                results.append(('sphinx', sphinx_text, 0.6))
            except Exception as e:
                logging.warning(f"Sphinx recognition error: {e}")
        
        # Select best result
        if results:
            results.sort(key=lambda x: x[2], reverse=True)
            engine, text, confidence = results[0]
            
            # Confidence check
            if confidence < 0.6:
                st.warning("I'm not quite sure I heard that correctly. Could you please repeat?")
                return None
            
            # Show transcription with confidence
            st.info(f"Transcribed ({engine.title()}, {confidence:.2f}): {text}")
            return text
        else:
            st.error("Could not understand audio. Please try again or rephrase.")
            return None
            
    except Exception as e:
        st.error("Error processing audio. Please try again.")
        logging.error(f"Transcription error: {e}")
        return None
    finally:
        # Cleanup
        try:
            if temp_filename and os.path.exists(temp_filename):
                os.remove(temp_filename)
        except OSError as e:
            logging.error(f"Error removing temporary file: {e}")

def test_grok_api():
    """Test the Grok API endpoint"""
    try:
        response = requests.post(
            "https://api.together.xyz/inference",
            headers={
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "prompt": "### Instruction: Say hello and introduce yourself as Grok.\n\n### Response:",
                "temperature": 0.7,
                "max_tokens": 100,
                "top_p": 0.7,
                "repetition_penalty": 1.1,
                "stop": ["### Instruction:", "### Response:"]
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'output' in result and 'choices' in result['output']:
                print("Grok API Test Success:")
                print(result['output']['choices'][0]['text'].strip())
                return True
            else:
                print(f"Unexpected response format: {result}")
                return False
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False

def test_claude_api():
    """Test Claude API endpoint specifically"""
    try:
        response = claude_client.messages.create(
            model="claude-2.1",
            max_tokens=10,
            messages=[{
                "role": "user",
                "content": "Hello, this is a test message. Please respond with 'Test successful'."
            }]
        )
        logging.info(f"Claude API Response: {response}")
        return True
    except Exception as e:
        logging.error(f"Claude API test failed with error: {e}")
        return False

# Add test function
if __name__ == "__main__":
    if test_claude_api():
        print("Claude API test successful")
    else:
        print("Claude API test failed")
