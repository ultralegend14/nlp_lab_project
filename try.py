# ==============================================================================
# 📈 SALES DOCUMENT ANALYZER - FINAL, COMPLETE, AND UNABRIDGED VERSION
# ==============================================================================
#
# DESCRIPTION:
# This is the full, unabridged code as requested. It takes the user's original
# 752-line script and ONLY adds the necessary code to implement a stable
# voice recording feature using 'streamlit-webrtc', fixing the previous
# component error. No original code has been removed or shortened.
#
# REQUIRED LIBRARIES:
# pip install streamlit pandas numpy plotly transformers torch textstat PyPDF2 PyMuPDF pdfplumber wordcloud matplotlib requests streamlit-webrtc av
#
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from transformers.pipelines import pipeline
import plotly.graph_objects as go
import requests
import matplotlib.pyplot as plt
from io import BytesIO
import json
import re
from streamlit_webrtc import AudioProcessorBase
from datetime import datetime
from io import BytesIO
import time
from typing import Dict, List, Optional, Tuple, Any
import warnings
import threading # Required for streamlit-webrtc processing
warnings.filterwarnings('ignore')

# --- Library Availability Checks ---

# PDF Processing Libraries
try:
    import PyPDF2
    import fitz  # PyMuPDF
    import pdfplumber
    PDF_LIBRARIES_AVAILABLE = True
except ImportError:
    PDF_LIBRARIES_AVAILABLE = False

# NLP Libraries (HuggingFace Transformers, textstat)
try:
    from transformers import pipeline
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    NLP_LIBRARIES_AVAILABLE = True
except ImportError:
    NLP_LIBRARIES_AVAILABLE = False

# Word Cloud Visualization
try:
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Voice Input Libraries (Using the stable streamlit-webrtc)
try:
    from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
    import av # Required for audio frame processing by streamlit-webrtc
    VOICE_LIBRARIES_AVAILABLE = True
except ImportError:
    VOICE_LIBRARIES_AVAILABLE = False

# ========== CONFIGURATION ==========
st.set_page_config(
    page_title="Sales Document Analyzer",
    page_icon="📈",
    layout="wide"
)

# API Configuration
import streamlit as st
ALL_GROQ_API_KEYS = st.secrets["ALL_GROQ_API_KEYS"]

GROQ_CHAT_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# NEW: Added Whisper API URL
GROQ_WHISPER_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


# ========== NEW: AUDIO RECORDER CLASS FOR STREAMLIT-WEBRTC ==========
# This class replaces the faulty st_audiorec component. It captures audio
# frames from the browser and converts them into a processable WAV format.
class AudioRecorder(AudioProcessorBase):
    def __init__(self) -> None:
        self._frames = []
        self._lock = threading.Lock()

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # This method receives audio frames from the browser in real-time.
        with self._lock:
            self._frames.append(frame)
        return frame

    def get_recorded_audio(self) -> Optional[bytes]:
        """
        Combines all received audio frames into a single WAV byte object.
        This is called after the user stops the recording.
        """
        with self._lock:
            if not self._frames:
                return None

            # Use the sample rate from the first frame
            sample_rate = self._frames[0].sample_rate
            
            # Concatenate all audio frame data into one numpy array
            audio_data = np.concatenate([f.to_ndarray() for f in self._frames], axis=1)

            # Create an in-memory buffer to write the WAV file to
            output_buffer = BytesIO()
            
            # Use PyAV to encode the raw audio data into WAV format
            with av.open(output_buffer, 'w', 'wav') as out_container:
                # Add a mono audio stream with the correct sample rate
                out_stream = out_container.add_stream('pcm_s16le', rate=sample_rate, layout='mono')
                
                # Reshape data for mono and create a new AudioFrame
                reshaped_data = audio_data.T.reshape(1, -1)
                frame = av.AudioFrame.from_ndarray(reshaped_data, format='s16', layout='mono')
                frame.sample_rate = sample_rate
                
                # Encode and write the frame to the buffer
                for packet in out_stream.encode(frame):
                    out_container.mux(packet)
            
            # Clear the frames for the next recording session
            self._frames = []

            # Return the complete WAV file as bytes
            return output_buffer.getvalue()


# ========== PDF PROCESSING ==========
class PDFProcessor:
    @staticmethod
    def extract_text_pypdf2(file) -> str:
        try:
            reader = PyPDF2.PdfReader(file)
            text = "".join(page.extract_text() + "\n" for page in reader.pages if page.extract_text())
            return text
        except Exception as e: st.error(f"PyPDF2 extraction failed: {e}"); return ""

    @staticmethod
    def extract_text_pymupdf(file) -> str:
        try:
            file_bytes = file.read()
            file.seek(0)
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text = "".join(page.get_text() + "\n" for page in doc)
            doc.close()
            return text
        except Exception as e: st.error(f"PyMuPDF extraction failed: {e}"); return ""

    @staticmethod
    def extract_text_pdfplumber(file) -> str:
        try:
            with pdfplumber.open(file) as pdf:
                text = "".join(page.extract_text() + "\n" for page in pdf.pages if page.extract_text())
            return text
        except Exception as e: st.error(f"pdfplumber extraction failed: {e}"); return ""

    @classmethod
    def process_pdf(cls, file) -> str:
        if not PDF_LIBRARIES_AVAILABLE:
            st.error("PDF libraries missing."); return ""
        
        all_texts = []
        extractors = [("PyPDF2", cls.extract_text_pypdf2), ("PyMuPDF", cls.extract_text_pymupdf), ("pdfplumber", cls.extract_text_pdfplumber)]
        for name, func in extractors:
            try:
                file.seek(0)
                text = func(file)
                if text and text.strip(): all_texts.append(text)
            except Exception as e: st.warning(f"{name} failed: {e}")
        
        return max(all_texts, key=len) if all_texts else ""

# ========== GROQ API INTERFACE (AUGMENTED) ==========
class GroqInterface:
    def __init__(self, api_keys_list: List[str]):
        if not api_keys_list:
            raise ValueError("API keys list cannot be empty for GroqInterface.")
        
        self.api_keys = [key for key in api_keys_list if key and key.startswith("gsk_") and "YOUR_GROQ_API_KEY" not in key]
        
        if not self.api_keys:
            raise ValueError("No valid API keys (starting with gsk_ and not placeholders) were found in the provided list.")

        self.current_key_index = 0
        self.chat_url = GROQ_CHAT_API_URL
        # NEW: Added Whisper URL attribute
        self.whisper_url = GROQ_WHISPER_API_URL
        
        # Define prompts within the class or as a class attribute
        self.prompts = {
            "comprehensive": """
            You are an expert sales analyst. Analyze this sales document and provide a comprehensive report summary covering whatever is available in the document from the following sections:
            1. Executive Sales Summary. 2. Sales Performance Analysis (overall, trends). 3. Product/Service/Model Performance. 4. Key Sales Metrics & KPIs (explained). 5. Regional/Channel Performance. 6. Sales Team & Strategy Effectiveness. 7. Market & Competitive Landscape. 8. Sales Risk Assessment. 9. Strategic Sales Insights & Outlook. 10. Recommendations for Sales Improvement.
            Be specific, cite numbers, and use Markdown and ensure the summary is proper and has relevant data given in the document only.
            """,
            "kpi": """
            You are an expert sales data extractor. From the provided sales document, meticulously extract Key Performance Indicators (KPIs) and sales metrics.
            Your primary goal is to structure the output in **easily parsable Markdown tables for generating charts.**

            **CRITICAL OUTPUT FORMATTING RULES:**
            1.  **ONLY LIST KPIs FOUND:** If a KPI or data point is not explicitly mentioned or clearly calculable from the document, **OMIT IT ENTIRELY.** Do NOT state "Not found," "Not available," or "N/A."
            2.  **USE MARKDOWN TABLES:** For any series of related data (e.g., monthly sales, sales by product, sales by region), YOU MUST use Markdown tables. Use appropriate Markdown H3 headings (e.g., `### Monthly Sales Performance`) before each table to indicate its content.
            3.  **SPECIFIC TABLE COLUMNS & EXAMPLES:**

                *   For **Time-Series Data (e.g., Monthly/Quarterly/Annual Overall Sales):**
                    Use H3 heading like: `### Overall Sales Trend by Period`
                    Columns: `| Metric Name         | Period   | Value | Unit        |`
                    Example:
                    ```markdown
                    ### Overall Sales Trend by Period
                    | Metric Name         | Period   | Value | Unit        |
                    |---------------------|----------|-------|-------------|
                    | Total Sales Revenue | Q1 2023  | 1.2   | million USD |
                    | Total Sales Revenue | Q2 2023  | 1.5   | million USD |
                    | Monthly Sales       | Jan 2024 | 450   | k USD       |
                    | Monthly Sales       | Feb 2024 | 480   | k USD       |
                    ```

                *   For **Categorical Data (e.g., Sales by Product/Model/Service):**
                    Use H3 heading like: `### Sales Performance by Product/Model`
                    Columns: `| Product/Model Name | Metric Type     | Value | Unit    | Period (if known) |`
                    Example:
                    ```markdown
                    ### Sales Performance by Product/Model
                    | Product/Model Name | Metric Type     | Value | Unit        | Period   |
                    |--------------------|-----------------|-------|-------------|----------|
                    | Product Alpha      | Sales Revenue   | 750   | k USD       | Q3 2023  |
                    | Product Alpha      | Units Sold      | 1500  | units       | Q3 2023  |
                    | Product Beta       | Sales Revenue   | 900   | k USD       | Q3 2023  |
                    ```

                *   For **Regional/Segment Data:**
                    Use H3 heading like: `### Sales Performance by Region/Segment`
                    Columns: `| Region/Segment | Metric Type     | Value | Unit    | Period (if known) |`
                    Example:
                    ```markdown
                    ### Sales Performance by Region/Segment
                    | Region/Segment | Metric Type   | Value | Unit        | Period   |
                    |----------------|---------------|-------|-------------|----------|
                    | North America  | Sales Revenue | 2.1   | million USD | FY 2023  |
                    | Europe         | Sales Revenue | 1.8   | million USD | FY 2023  |
                    ```

                *   For **Other Individual KPIs (not part of a clear series for the above tables):**
                    Use H3 heading like: `### Key Standalone Metrics`
                    Columns: `| KPI Name                  | Value | Unit      | Context/Period       |`
                    Example:
                    ```markdown
                    ### Key Standalone Metrics
                    | KPI Name                  | Value | Unit      | Context/Period       |
                    |---------------------------|-------|-----------|----------------------|
                    | Average Deal Size         | 25    | k USD     | FY 2023              |
                    | Customer Churn Rate       | 5     | %         | Q3 2023              |
                    ```
            4.  **VALUE & UNIT:** The `Value` column should contain ONLY the numeric part (e.g., "1.2", "450"). The `Unit` column should specify the unit (e.g., "million USD", "k USD", "%", "units").
            5.  **ACCURACY:** Transcribe numbers precisely. If a period or category applies to multiple rows, repeat it for clarity in parsing.

            **FOCUS ON EXTRACTING PLOTTABLE DATA:** Prioritize data series like monthly sales, product performance over time or category, regional comparisons.
            If data is presented in the document that fits one of these table structures, reformat it accordingly.
            """,
            "risk": """
            Conduct a sales risk assessment. Identify risks impacting sales: Market, Operational, Customer, Product, Strategic. Describe each, potential impact, and mitigations if mentioned. Use Markdown.
            """,
            # NEW: Added prompt for voice correction
            "voice_correction": """
            You are an expert AI assistant specializing in correcting speech-to-text transcriptions. The following text was transcribed from a user's voice and may contain errors. Your task is to correct any spelling mistakes, fix grammatical errors, and rephrase it into a clear, coherent, and natural-sounding question or command.
            - Focus on the user's intent.
            - Do not add any information that isn't present in the original text.
            - Output ONLY the corrected text, with no preamble, explanation, or quotation marks.

            Example 1:
            Original: what was the total sell for product alpha in quarter tree
            Corrected: What was the total sales for Product Alpha in Quarter 3?

            Example 2:
            Original: show me the risk assess meant for the european market
            Corrected: Show me the risk assessment for the European market.
            """
        }

    def call_api(self, messages: List[Dict], model: str = "llama3-70b-8192", 
                 temperature: float = 0.05, max_tokens: int = 3000) -> str:
        
        num_valid_keys = len(self.api_keys)
        if num_valid_keys == 0:
            return "❌ API Error: No valid Groq API keys are available in the GroqInterface instance."

        last_known_error = f"❌ API Error: All {num_valid_keys} configured valid Groq API key(s) failed after trying each once."

        for _ in range(num_valid_keys):
            current_key = self.api_keys[self.current_key_index]
            
            headers = {"Authorization": f"Bearer {current_key}", "Content-Type": "application/json"}
            data = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
            
            try:
                response = requests.post(self.chat_url, headers=headers, json=data, timeout=180)
                response.raise_for_status()
                
                self.current_key_index = (self.current_key_index + 1) % num_valid_keys
                return response.json()['choices'][0]['message']['content'].strip()

            except requests.exceptions.HTTPError as http_err:
                err_content = http_err.response.text
                status_code = http_err.response.status_code
                
                is_retriable_key_error = False
                if status_code == 429:
                    is_retriable_key_error = True
                else:
                    try:
                        err_json = http_err.response.json()
                        if "error" in err_json and "type" in err_json["error"]:
                            error_type = err_json["error"]["type"].lower()
                            if error_type in ["insufficient_quota", "rate_limit_exceeded", 
                                              "api_key_revoked", "invalid_api_key", "permission_denied"]:
                                is_retriable_key_error = True
                    except ValueError:
                        pass 
                    
                    if not is_retriable_key_error:
                         lc_err_content = err_content.lower()
                         if any(phrase in lc_err_content for phrase in 
                                ["rate limit", "quota exceeded", "insufficient quota", 
                                 "invalid api key", "key revoked", "api key validation failed"]):
                             is_retriable_key_error = True
                
                if is_retriable_key_error:
                    last_known_error = (f"API Key ...{current_key[-4:]} failed (Status {status_code}, retriable). "
                                          f"Error details: {err_content[:250]}")
                    self.current_key_index = (self.current_key_index + 1) % num_valid_keys
                    time.sleep(0.25) 
                else:
                    return f"❌ API HTTP Error (Key ...{current_key[-4:]}): {http_err}. Response: {err_content}"
            
            except requests.exceptions.RequestException as req_err:
                last_known_error = f"Request Error with key ...{current_key[-4:]}: {str(req_err)}. Trying next key."
                self.current_key_index = (self.current_key_index + 1) % num_valid_keys
                time.sleep(0.25)
            
            except Exception as e:
                return f"❌ Unexpected API/Request Error (Key ...{current_key[-4:]}): {str(e)}"

        return last_known_error
    
    def analyze_document(self, text: str, analysis_type: str = "comprehensive") -> str:
        system_message = self.prompts.get(analysis_type, self.prompts["comprehensive"])
        max_input_text_length = 28000 # Approx 7k tokens
        
        messages = [{"role": "system", "content": system_message},
                    {"role": "user", "content": f"Document Content:\n{text[:max_input_text_length]}"}]
        
        # Increased max_tokens for KPI specifically due to detailed table format.
        output_tokens = 4096 if analysis_type == "kpi" else 3000
        return self.call_api(messages, max_tokens=output_tokens)
    
    # NEW: Method to transcribe audio
    def transcribe_audio(self, audio_bytes: bytes) -> str:
        """Transcribes audio using the Groq Whisper API with key rotation."""
        num_keys = len(self.api_keys)
        last_known_error = f"❌ Transcription Error: All {num_keys} API keys failed."

        for _ in range(num_keys):
            current_key = self.api_keys[self.current_key_index]
            headers = {"Authorization": f"Bearer {current_key}"}
            files = {'file': ('audio.wav', audio_bytes, 'audio/wav')}
            data = {'model': 'whisper-large-v3'}

            try:
                response = requests.post(self.whisper_url, headers=headers, files=files, data=data, timeout=60)
                response.raise_for_status()
                # Rotate key only on success
                self.current_key_index = (self.current_key_index + 1) % num_keys
                return response.json()['text']
            except requests.exceptions.RequestException as req_err:
                last_known_error = f"Request Error during transcription with key ...{current_key[-4:]}: {req_err}. Trying next key."
                self.current_key_index = (self.current_key_index + 1) % num_keys
                time.sleep(0.25)
            except Exception as e:
                return f"❌ Unexpected Error during transcription: {e}"
        return last_known_error

    # NEW: Method to correct transcription
    def get_corrected_transcription(self, raw_text: str) -> str:
        """Uses an LLM to correct a raw transcription."""
        if not raw_text or not raw_text.strip():
            return ""
        
        system_message = self.prompts["voice_correction"]
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": raw_text}
        ]
        
        # Use a smaller, faster model for this simple correction task
        return self.call_api(messages, model="llama3-8b-8192", temperature=0.1, max_tokens=300)

# ========== NLP ANALYZER ==========
class NLPAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = None
        if NLP_LIBRARIES_AVAILABLE: self.load_models()
    
    def load_models(self):
        try:
            # Force PyTorch backend
            import torch
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                framework="pt"  # Force use of PyTorch
            )
        except Exception as e:
            import traceback
            st.warning("Sentiment model load failed.")
            st.error(f"Error: {e}")
            st.code(traceback.format_exc())
    
    def analyze_sentiment(self, text: str) -> Dict:
        if not NLP_LIBRARIES_AVAILABLE or not self.sentiment_analyzer: return {"error": "Sentiment analyzer unavailable."}
        try:
            chunks = [text[i:i+500] for i in range(0, len(text), 500)][:20] 
            results = [self.sentiment_analyzer(c)[0] for c in chunks if len(c.strip()) > 20] 
            if not results: return {"error": "No text suitable for sentiment analysis."}
            
            s_labels = [r['label'].upper() for r in results]
            s_scores = [r['score'] for r in results]
            
            s_counts = pd.Series(s_labels).value_counts(normalize=True)
            
            overall_sentiment = "NEUTRAL" 
            if not s_counts.empty:
                overall_sentiment = s_counts.index[0]
                if len(s_counts) > 1 and abs(s_counts.iloc[0] - s_counts.iloc[1]) < 0.1: 
                    if "POSITIVE" in s_counts.index and "NEGATIVE" in s_counts.index:
                        overall_sentiment = "MIXED" 
                    elif s_counts.iloc[0] < 0.5: 
                        overall_sentiment = "NEUTRAL"

            avg_confidence = np.mean(s_scores) if s_scores else 0
            
            return {'overall_sentiment': overall_sentiment, 
                    'confidence': avg_confidence, 
                    'distribution': pd.Series(s_labels).value_counts().to_dict()}
        except Exception as e: return {"error": f"Sentiment analysis error: {e}"}

    def extract_relevant_terms(self, text: str) -> Dict:
        patterns = {
            'revenue_sales_figures': r'(?:revenue|sales|income|earnings|turnover)[s]?\s*(?:of|is|was|:|)\s*\$?\s*[\d,]+(?:\.\d{1,2})?\s*(?:million|billion|trillion|m|b|t|k)?',
            'growth_percentage_metrics': r'\b\d{1,3}(?:\.\d{1,2})?%\s*(?:increase|decrease|growth|decline|margin|YoY|QoQ|MoM|rate|share)?\b',
            'dates_periods': r'\b(?:Q[1-4]\s*\d{4}|FY\s*\d{2,4}|H[12]\s*\d{4}|\d{4}-\d{4}|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?)\b',
            'sales_kpi_terms': r'\b(?:CAC|LTV|ARPU|ARPC|churn rate|conversion rate|win rate|average deal size|sales cycle|market share|customer acquisition|lead generation|units sold|ASP|average selling price)\b(?:\s*of\s*[\d\.\%\$]+)?'
        }
        extracted = {}
        for name, pattern in patterns.items():
            try:
                matches = re.findall(pattern, text, re.IGNORECASE)
                unique_matches = sorted(list(set(m.strip() for m in matches if m.strip())), key=len, reverse=True)
                extracted[name] = unique_matches[:15]
            except Exception: 
                extracted[name] = []
        return extracted
    
    def get_document_stats(self, text: str) -> Dict:
        words = text.split()
        num_words = len(words)
        num_chars = len(text)
        est_reading_time = round(num_words / 200, 1) if num_words > 0 else 0.0
        
        stats = {
            'word_count': num_words, 
            'character_count': num_chars, 
            'estimated_reading_time': est_reading_time
        }
        
        if NLP_LIBRARIES_AVAILABLE and len(text.strip()) > 100: 
            try:
                stats['readability_score'] = round(flesch_reading_ease(text), 1)
                stats['grade_level'] = round(flesch_kincaid_grade(text), 1)
            except Exception: 
                stats.update({'readability_score': 'N/A', 'grade_level': 'N/A'})
        else:
            stats.update({'readability_score': 'N/A (text too short or NLP lib unavailable)', 
                          'grade_level': 'N/A (text too short or NLP lib unavailable)'})
        return stats

# ========== KPI PARSING AND CHARTING (ORIGINAL UNTOUCHED VERSION) ==========
def convert_value_unit_v2(value_str: Optional[str], unit_str: Optional[str]) -> Tuple[Optional[float], str]:
    if value_str is None or str(value_str).strip() == "": return None, str(unit_str or "UNKNOWN").strip().upper()
    
    val_str_cleaned = str(value_str).lower().replace(',', '').strip()
    unit_final = str(unit_str or "").strip()

    currency_prefixes = {"usd": "$", "eur": "€", "gbp": "£", "inr": "₹"} 
    for k, v in currency_prefixes.items():
        if k in unit_final.lower():
            if v not in val_str_cleaned and not any(c.isdigit() for c in v if c not in val_str_cleaned): 
                 val_str_cleaned = v + val_str_cleaned 
            unit_final = re.sub(r'\b' + k + r'\b', '', unit_final.lower(), flags=re.IGNORECASE).strip()


    currency_symbol = ""
    if '$' in val_str_cleaned or 'usd' in val_str_cleaned: currency_symbol = "USD"
    elif '€' in val_str_cleaned or 'eur' in val_str_cleaned: currency_symbol = "EUR"
    elif '£' in val_str_cleaned or 'gbp' in val_str_cleaned: currency_symbol = "GBP"
    elif '₹' in val_str_cleaned or 'inr' in val_str_cleaned: currency_symbol = "INR"

    val_str_cleaned = val_str_cleaned.replace('$', '').replace('€', '').replace('£', '').replace('₹', '').replace('usd','').replace('eur','').replace('gbp','').replace('inr','').strip()
    
    multiplier = 1.0
    string_to_check_multiplier = (unit_final + " " + val_str_cleaned).lower()

    if 'billion' in string_to_check_multiplier or ' bn' in string_to_check_multiplier or ' b' == val_str_cleaned[-2:]: multiplier = 1e9
    elif 'million' in string_to_check_multiplier or ' mn' in string_to_check_multiplier or ' m' == val_str_cleaned[-2:]: multiplier = 1e6
    elif 'k ' in string_to_check_multiplier or val_str_cleaned.endswith('k') or 'thousand' in string_to_check_multiplier:
        if val_str_cleaned.endswith('k'): val_str_cleaned = val_str_cleaned[:-1]
        multiplier = 1e3
        
    unit_final = unit_final.lower().replace("billion", "").replace("million", "").replace("thousand","").replace("bn","").replace("mn","").replace("k","").strip()

    try:
        val_str_cleaned = re.sub(r'\s*\.\s*', '.', val_str_cleaned) 
        val_str_cleaned = re.sub(r'\.{2,}', '.', val_str_cleaned) 
        if val_str_cleaned.startswith('.'): val_str_cleaned = '0' + val_str_cleaned 
        
        numeric_val = float(val_str_cleaned) * multiplier
    except ValueError:
        return None, (currency_symbol + " " + unit_final).strip().upper() or "UNKNOWN"

    final_unit_parts = []
    if currency_symbol: final_unit_parts.append(currency_symbol)
    
    if unit_final and unit_final.lower() not in (cs.lower() for cs in ["usd","eur","gbp","inr"]):
        final_unit_parts.append(unit_final)
    
    original_full_unit_info = (str(unit_str or "") + str(value_str or "")).lower()
    if "%" in original_full_unit_info and "%" not in " ".join(final_unit_parts):
        if not final_unit_parts or final_unit_parts[-1]!="%": final_unit_parts.append("%")
    
    unit_to_return = " ".join(final_unit_parts).strip().upper()
    if not unit_to_return and numeric_val is not None:
        if isinstance(numeric_val, int) or numeric_val.is_integer():
            unit_to_return = "UNITS" 
        else:
            unit_to_return = "VALUE"
    elif not unit_to_return:
        unit_to_return = "UNKNOWN"

    return numeric_val, unit_to_return


def parse_kpi_data_from_ai_text_v2(ai_text: str) -> List[Dict[str, Any]]:
    parsed_kpis = []
    lines = ai_text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    current_table_type = "Unknown"
    current_headers_raw: List[str] = [] 
    header_indices: Dict[str, int] = {} 
    
    col_patterns = {
        "metric_name": ["metric name", "kpi name", "metric"],
        "period": ["period", "time period", "date"],
        "value": ["value", "amount", "figure"],
        "unit": ["unit", "units", "currency"],
        "product_model_name": ["product/model name", "product name", "model name", "item"],
        "category": ["category", "type", "segment"], 
        "metric_type": ["metric type", "value type", "data type"], 
        "region_segment": ["region/segment", "region", "market", "segment name"],
        "context_period": ["context/period", "context", "notes", "description", "period (if known)"]
    }

    for line_num, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped: continue

        h3_match = re.match(r"^\s*###\s*(.+)", line_stripped)
        if h3_match:
            heading_text = h3_match.group(1).lower().strip()
            if "overall sales trend" in heading_text or "by period" in heading_text or "time series" in heading_text: current_table_type = "time_series"
            elif "product/model" in heading_text or "by product" in heading_text or "by model" in heading_text : current_table_type = "product_model"
            elif "region/segment" in heading_text or "by region" in heading_text or "by segment" in heading_text: current_table_type = "region_segment"
            elif "standalone metrics" in heading_text or "key individual kpis" in heading_text: current_table_type = "standalone"
            else: current_table_type = "generic_table" 
            current_headers_raw = [] 
            header_indices = {}
            continue

        if line_stripped.startswith("|") and "|---" not in line_stripped and not current_headers_raw:
            headers_from_line = [h.strip().lower() for h in line_stripped.strip('|').split('|')]
            if len(headers_from_line) < 2: continue 

            temp_header_indices = {}
            has_value_col = False
            for i, header_text in enumerate(headers_from_line):
                for semantic_name, patterns in col_patterns.items():
                    if any(p in header_text for p in patterns):
                        if semantic_name not in temp_header_indices: 
                            temp_header_indices[semantic_name] = i
                        if semantic_name == 'value': has_value_col = True
            
            if has_value_col and (temp_header_indices.keys() - {'value', 'unit'}): 
                current_headers_raw = headers_from_line
                header_indices = temp_header_indices
            continue

        if line_stripped.startswith("|") and all(c in '-| ' for c in line_stripped):
            continue

        if line_stripped.startswith("|") and current_headers_raw and header_indices:
            cols = [c.strip() for c in line_stripped.strip('|').split('|')]
            
            if len(cols) == len(current_headers_raw):
                kpi_entry: Dict[str, Any] = {"table_type": current_table_type, "source_line": line_num}
                
                raw_value_str: Optional[str] = None
                raw_unit_str: Optional[str] = None

                for semantic_name, col_idx in header_indices.items():
                    if col_idx < len(cols): 
                        cell_value = cols[col_idx]
                        if semantic_name == "value":
                            raw_value_str = cell_value
                        elif semantic_name == "unit":
                            raw_unit_str = cell_value
                        else: 
                            kpi_entry[semantic_name] = cell_value
                
                parsed_value, parsed_unit = convert_value_unit_v2(raw_value_str, raw_unit_str)
                
                if parsed_value is not None:
                    kpi_entry["value"] = parsed_value
                    kpi_entry["unit"] = parsed_unit
                    
                    if current_table_type == "product_model" and "product_model_name" in kpi_entry:
                        kpi_entry["category_key"] = kpi_entry["product_model_name"]
                    elif current_table_type == "region_segment" and "region_segment" in kpi_entry:
                        kpi_entry["category_key"] = kpi_entry["region_segment"]
                    
                    if 'metric_name' not in kpi_entry:
                        if current_table_type == "product_model" and "product_model_name" in kpi_entry:
                            kpi_entry["metric_name"] = kpi_entry.get("metric_type", kpi_entry["product_model_name"])
                        elif current_table_type == "region_segment" and "region_segment" in kpi_entry:
                            kpi_entry["metric_name"] = kpi_entry.get("metric_type", kpi_entry["region_segment"])
                        elif "category_key" in kpi_entry : 
                            kpi_entry["metric_name"] = kpi_entry["category_key"]
                    
                    has_identifier = any(kpi_entry.get(key_name) for key_name in ["metric_name", "category_key", "product_model_name", "region_segment"])

                    if has_identifier:
                        parsed_kpis.append(kpi_entry)
    
    return parsed_kpis


def create_dynamic_kpi_charts_v2(parsed_kpis: List[Dict[str, Any]]):
    if not parsed_kpis:
        st.info("No plottable KPI data was successfully parsed from the response after filtering.")
        return

    df_kpis = pd.DataFrame(parsed_kpis)
    if df_kpis.empty or 'value' not in df_kpis.columns:
        st.info("KPI data is empty or missing 'value' column after DataFrame conversion.")
        return
        
    df_kpis['value'] = pd.to_numeric(df_kpis['value'], errors='coerce')
    df_kpis.dropna(subset=['value'], inplace=True)

    if df_kpis.empty:
        st.info("No valid numeric KPI values found for plotting after parsing and cleaning.")
        return

    st.markdown("---")
    st.markdown("### 📊 Visualized Sales KPIs")
    
    plotted_indices = set() 

    time_series_data = df_kpis[df_kpis['table_type'] == 'time_series'].copy()
    if not time_series_data.empty and 'period' in time_series_data.columns and \
       ('metric_name' in time_series_data.columns or 'metric_type' in time_series_data.columns):
        
        name_col_ts = 'metric_name' if 'metric_name' in time_series_data.columns and time_series_data['metric_name'].nunique() > 1 else 'metric_type'
        if name_col_ts not in time_series_data.columns: name_col_ts = 'unit' 

        try:
            def create_sort_key_period(period_str_val):
                period_str = str(period_str_val).strip()
                for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%b %Y", "%B %Y", "%Y %b", "%Y %B"):
                    try: return datetime.strptime(period_str, fmt).strftime("%Y%m%d")
                    except ValueError: pass
                q_match = re.match(r"(Q[1-4])\s*(\d{4})", period_str, re.IGNORECASE)
                if q_match: return f"{q_match.group(2)}{q_match.group(1).upper()}"
                if re.match(r"^\d{4}$", period_str): return f"{period_str}0000" 
                return period_str 
            
            time_series_data['sortable_period'] = time_series_data['period'].apply(create_sort_key_period)
            time_series_data.sort_values(by=[name_col_ts, 'sortable_period'], inplace=True)
        except Exception: 
            time_series_data.sort_values(by=[name_col_ts, 'period'], inplace=True)

        for metric_identifier, group in time_series_data.groupby(name_col_ts):
            group = group.dropna(subset=['period', 'value']) 
            if len(group) > 1: 
                title = f"Trend: {metric_identifier}"
                if 'unit' in group.columns and group['unit'].nunique() == 1:
                    title += f" ({group['unit'].iloc[0]})"
                
                fig = px.line(group, x='period', y='value', color_discrete_sequence=px.colors.qualitative.Plotly,
                              markers=True, title=title,
                              labels={'value': 'Value', 'period': 'Period'})
                fig.update_traces(texttemplate='%{y:,.2s}', textposition='top center')
                st.plotly_chart(fig, use_container_width=True)
                plotted_indices.update(group.index)
            elif len(group) == 1: 
                 entry = group.iloc[0]
                 label_text = f"{entry.get(name_col_ts, 'Metric')} ({entry.get('period','Period')})"
                 st.metric(label=label_text, 
                           value=f"{entry['value']:,.2f} {entry.get('unit','')}")
                 plotted_indices.update(group.index)

    categorical_chart_configs = [
        {"table_type": "product_model", "id_col": "product_model_name", "chart_title_prefix": "Performance by Product/Model"},
        {"table_type": "region_segment", "id_col": "region_segment", "chart_title_prefix": "Performance by Region/Segment"},
        {"table_type": "generic_table", "id_col": "category", "chart_title_prefix": "Performance by Category"} 
    ]

    for config in categorical_chart_configs:
        cat_data = df_kpis[df_kpis['table_type'] == config["table_type"]].copy()
        id_column = config["id_col"]
        
        if not cat_data.empty and id_column in cat_data.columns and \
           ('metric_type' in cat_data.columns or cat_data['metric_name'].nunique() > 0) :
            
            metric_group_col = 'metric_type' if 'metric_type' in cat_data.columns and cat_data['metric_type'].nunique() > 1 else 'metric_name'
            if metric_group_col not in cat_data.columns: continue 

            st.subheader(config["chart_title_prefix"])
            
            if 'period' not in cat_data.columns: cat_data['period'] = 'Overall'
            else: cat_data['period'] = cat_data['period'].fillna('Overall').astype(str)

            for (metric_id, period_val), group in cat_data.groupby([metric_group_col, 'period']):
                group = group.dropna(subset=[id_column, 'value'])
                if len(group) > 1 : 
                    title = f"{metric_id}"
                    if period_val != 'Overall': title += f" ({period_val})"
                    
                    unique_units = group['unit'].dropna().unique()
                    if len(unique_units) == 1: title += f" ({unique_units[0]})"

                    fig = px.bar(group, x=id_column, y='value', 
                                 color=id_column if group[id_column].nunique() <= 10 else None, 
                                 title=title, text_auto='.2s',
                                 labels={'value': 'Value', id_column: config["id_col"].replace("_", " ").title()})
                    fig.update_layout(xaxis_tickangle=-30)
                    st.plotly_chart(fig, use_container_width=True)
                    plotted_indices.update(group.index)
                elif len(group) == 1: 
                    entry = group.iloc[0]
                    label_text = f"{entry.get(metric_group_col, id_column)}: {entry.get(id_column, 'Item')}"
                    if period_val != 'Overall': label_text += f" ({period_val})"
                    st.metric(label=label_text, value=f"{entry['value']:,.2f} {entry.get('unit','')}")
                    plotted_indices.update(group.index)

    remaining_kpis_df = df_kpis[~df_kpis.index.isin(plotted_indices)].copy()
    if not remaining_kpis_df.empty:
        st.subheader("Other Key Metrics / Standalone KPIs")
        
        standalone_first = pd.concat([
            remaining_kpis_df[remaining_kpis_df['table_type'] == 'standalone'],
            remaining_kpis_df[remaining_kpis_df['table_type'] != 'standalone']
        ]).drop_duplicates()

        num_cols_metrics = min(len(standalone_first), 4) 
        if num_cols_metrics > 0:
            metric_display_cols = st.columns(num_cols_metrics)
            col_idx = 0
            for _, row_data in standalone_first.iterrows():
                label_parts = []
                if row_data.get('metric_name'): label_parts.append(str(row_data['metric_name']))
                elif row_data.get('kpi_name'): label_parts.append(str(row_data['kpi_name'])) 
                
                context_parts = []
                if row_data.get('period') and str(row_data['period']) not in " ".join(label_parts): context_parts.append(str(row_data['period']))
                if row_data.get('context_period') and str(row_data['context_period']) not in " ".join(label_parts): context_parts.append(str(row_data['context_period']))
                if row_data.get('category_key') and str(row_data['category_key']) not in " ".join(label_parts): context_parts.append(str(row_data['category_key']))
                
                final_label = " ".join(label_parts)
                if context_parts:
                    final_label += f" ({', '.join(filter(None, context_parts))})"
                if not final_label: final_label = "Metric" 

                value_display = f"{row_data['value']:,.2f}"
                unit_display = str(row_data.get('unit','')).strip()
                if unit_display and unit_display not in ["UNKNOWN", "VALUE", "UNITS"]: 
                    value_display += f" {unit_display}"
                elif unit_display == "UNITS" and (isinstance(row_data['value'], int) or row_data['value'].is_integer()):
                     value_display += f" {unit_display}" 

                with metric_display_cols[col_idx % num_cols_metrics]:
                    st.metric(label=final_label, value=value_display)
                col_idx += 1

# ========== VISUALIZATION ==========
def create_wordcloud_visualization(text: str) -> Optional[plt.Figure]:
    if not WORDCLOUD_AVAILABLE:
        st.warning("WordCloud library not found. Please install it: `pip install wordcloud matplotlib`")
        return None
    if not text or len(text.strip()) < 100:
        st.info("Not enough text to generate a meaningful word cloud.")
        return None
    
    try:
        stopwords = set(STOPWORDS)
        stopwords.update(["s", "will", "report", "sales", "document", "company", "revenue", "product", "market", "data", "figure", "table"])
        
        wordcloud = WordCloud(width=800, height=400, 
                              background_color='white',
                              stopwords=stopwords,
                              min_font_size=10,
                              colormap='viridis',
                              collocations=False).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        plt.tight_layout(pad=0)
        
        return fig
    except Exception as e:
        st.error(f"Failed to generate word cloud: {e}")
        return None

def create_sentiment_chart(sentiment_data: Dict) -> Optional[go.Figure]:
    if 'distribution' not in sentiment_data or not sentiment_data['distribution']:
        st.info("Sentiment distribution data is not available for charting.")
        return None 
    
    labels = list(sentiment_data['distribution'].keys())
    values = list(sentiment_data['distribution'].values())
    
    color_map = {'POSITIVE': '#27ae60', 'NEGATIVE': '#c0392b', 'NEUTRAL': '#f39c12', 'MIXED': '#8e44ad'}
    colors = [color_map.get(l, '#7f8c8d') for l in labels] 

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4, 
                                 marker_colors=colors, textinfo='label+percent', 
                                 insidetextorientation='radial')])
    fig.update_layout(title_text="Document Sentiment Distribution", height=350, 
                      legend_title_text="Sentiments", 
                      margin=dict(t=50, b=0, l=0, r=0))
    return fig

def create_terms_chart(terms_data: Dict) -> Optional[go.Figure]:
    if not terms_data or not any(terms_data.values()): 
        st.info("No relevant terms were extracted by regex for charting.")
        return None

    term_counts = {k.replace('_', ' ').title(): len(v) for k, v in terms_data.items() if v} 
    
    if not term_counts: 
        st.info("No relevant terms found after processing for chart.")
        return None
        
    fig = go.Figure(data=[go.Bar(x=list(term_counts.keys()), 
                                 y=list(term_counts.values()), 
                                 marker_color='#2980b9', 
                                 text=list(term_counts.values()), 
                                 textposition='outside')])
    fig.update_layout(title_text="Regex-Found Term Mentions by Category", 
                      xaxis_title="Term Category", yaxis_title="Count", 
                      height=400, xaxis_tickangle=-30, 
                      margin=dict(t=50, b=100, l=0, r=0)) 
    return fig

# ========== MAIN APPLICATION ==========
def main():
    st.title("📈 Sales Document Analyzer")
    st.markdown("Upload a sales-related PDF to extract insights, KPIs, and generate visualizations using advanced NLP.")
    
    if not PDF_LIBRARIES_AVAILABLE: 
        st.error("CRITICAL: PDF processing libraries are not installed.")
        st.stop()

    if not VOICE_LIBRARIES_AVAILABLE:
        st.error("CRITICAL: Voice libraries (`streamlit-webrtc`, `av`) are not installed.")
        st.stop()

    initial_valid_keys_present = False
    if ALL_GROQ_API_KEYS: 
        for key_val in ALL_GROQ_API_KEYS:
            if isinstance(key_val, str) and key_val.startswith("gsk_") and "YOUR_GROQ_API_KEY" not in key_val:
                initial_valid_keys_present = True
                break 
    
    if not initial_valid_keys_present:
        st.error(
            "CRITICAL: No valid Groq API Keys are configured in `ALL_GROQ_API_KEYS` list."
        )
        st.stop()
        
    try:
        groq = GroqInterface(ALL_GROQ_API_KEYS)
    except ValueError as e: 
        st.error(f"CRITICAL: Could not initialize Groq Interface: {e}")
        st.stop()

    nlp = NLPAnalyzer()
    
    with st.sidebar:
        st.header("⚙️ Analysis Options")
        opt_sentiment = st.checkbox("Sentiment Analysis", True, help="Uses HuggingFace Transformers locally.")
        opt_regex_terms = st.checkbox("Extract Common Terms (Regex)", True, help="Uses regular expressions to find common sales-related terms.")
        opt_wordcloud = st.checkbox("Generate Word Cloud", True, help="Creates a visual representation of the most frequent words.")

    uploaded_file = st.file_uploader("Upload Sales PDF Document", type=["pdf"], help="Text-based PDFs work best. Scanned PDFs (images) will not work well.")
    
    if uploaded_file:
        with st.spinner("📄 Extracting text from PDF... This may take a moment."):
            text = PDFProcessor.process_pdf(uploaded_file)
        
        if not text or len(text.strip()) < 50: 
            st.error("❌ No significant text could be extracted from the PDF. ")
            return 
        
        st.success(f"✅ Text extracted successfully (~{len(text):,} characters).")
        
        doc_stats = nlp.get_document_stats(text)
        st.subheader("📄 Document Overview")
        cols_stats = st.columns(4)
        cols_stats[0].metric("Word Count", f"{doc_stats['word_count']:,}")
        cols_stats[1].metric("Character Count", f"{doc_stats['character_count']:,}")
        cols_stats[2].metric("Est. Read Time", f"{doc_stats['estimated_reading_time']} min")
        cols_stats[3].metric("Flesch Readability", str(doc_stats['readability_score']))
        st.markdown("---")

        tab_names = ["📄 Detailed Analysis", "📊 KPI & Keyword Analysis", "💬 Document Q&A"]
        tab1, tab2, tab3 = st.tabs(tab_names)

        with tab1:
            st.header(tab_names[0])
            col_comp, col_risk = st.columns(2)

            with col_comp:
                if st.button("🚀 Generate Comprehensive Analysis", key="btn_comp_analysis", type="primary"):
                    with st.spinner("Performing a comprehensive analysis... This might take a minute or two."):
                        st.session_state.comp_analysis = groq.analyze_document(text, "comprehensive") 
                
                if 'comp_analysis' in st.session_state and st.session_state.comp_analysis:
                    st.markdown("### Comprehensive Analysis Result:")
                    if "❌" in st.session_state.comp_analysis: 
                        st.error(st.session_state.comp_analysis)
                    else:
                        st.markdown(st.session_state.comp_analysis)

            with col_risk:
                if st.button("⚠️ Generate Sales Risk Assessment", key="btn_risk", type="primary"):
                    with st.spinner("Assessing sales risks..."):
                        st.session_state.risk_ai = groq.analyze_document(text, "risk")
                if 'risk_ai' in st.session_state and st.session_state.risk_ai:
                    st.markdown("### Sales Risk Assessment:")
                    if "❌" in st.session_state.risk_ai: st.error(st.session_state.risk_ai)
                    else: st.markdown(st.session_state.risk_ai)
            
            st.markdown("---")
            if opt_sentiment and NLP_LIBRARIES_AVAILABLE:
                st.subheader("Sentiment Analysis")
                with st.spinner("Analyzing sentiment locally..."):
                    sentiment_results = nlp.analyze_sentiment(text)
                
                if 'error' not in sentiment_results:
                    sc1, sc2 = st.columns([1, 2])
                    sc1.metric("Overall Sentiment", sentiment_results.get('overall_sentiment', 'N/A').capitalize())
                    sc1.metric("Avg. Confidence", f"{sentiment_results.get('confidence', 0):.1%}")
                    sentiment_chart = create_sentiment_chart(sentiment_results)
                    if sentiment_chart:
                        sc2.plotly_chart(sentiment_chart, use_container_width=True)
                    else:
                        sc2.info("Could not generate sentiment chart based on available data.")
                else:
                    st.warning(f"Sentiment Analysis Issue: {sentiment_results['error']}")
            elif opt_sentiment and not NLP_LIBRARIES_AVAILABLE:
                 st.warning("NLP libraries for sentiment analysis are not available.")

        with tab2:
            st.header(tab_names[1])
            if st.button("📊 Extract Sales KPIs (for Charts)", key="btn_kpi_extraction", type="primary"):
                with st.spinner("Extracting Key Performance Indicators... This can take some time."):
                    st.session_state.kpi_text_ai = groq.analyze_document(text, "kpi") 
            
            if 'kpi_text_ai' in st.session_state and st.session_state.kpi_text_ai:
                st.markdown("### Raw Extracted KPI Data (for verification):")
                if "❌" in st.session_state.kpi_text_ai:
                    st.error(st.session_state.kpi_text_ai)
                else: 
                    st.markdown(f"```markdown\n{st.session_state.kpi_text_ai}\n```") 
                    with st.spinner("Parsing extracted data and generating charts..."):
                        parsed_kpi_data = parse_kpi_data_from_ai_text_v2(st.session_state.kpi_text_ai)
                    
                    if parsed_kpi_data:
                        create_dynamic_kpi_charts_v2(parsed_kpi_data)
                    else:
                        st.warning("Could not parse plottable KPIs from the response.")
            
            st.markdown("---")
            
            keyword_cols_1, keyword_cols_2 = st.columns(2)
            with keyword_cols_1:
                if opt_wordcloud and WORDCLOUD_AVAILABLE:
                    st.subheader("☁️ Document Word Cloud")
                    with st.spinner("Generating word cloud..."):
                        wordcloud_fig = create_wordcloud_visualization(text)
                    if wordcloud_fig:
                        st.pyplot(wordcloud_fig)
                    else:
                        st.info("Could not generate word cloud.")
            
            with keyword_cols_2:
                if opt_regex_terms:
                    st.subheader("⚙️ Regex-Based Common Term Extraction")
                    with st.spinner("Extracting common terms using regex..."):
                        regex_extracted_terms = nlp.extract_relevant_terms(text)
                    
                    terms_chart_fig = create_terms_chart(regex_extracted_terms)
                    if terms_chart_fig:
                        st.plotly_chart(terms_chart_fig, use_container_width=True)
                        with st.expander("View Extracted Terms Details"):
                            for term_category, term_values in regex_extracted_terms.items():
                                if term_values: 
                                    st.markdown(f"**{term_category.replace('_',' ').title()}:** {', '.join(term_values)}")
                    else:
                        st.info("No common sales terms were found by the regex patterns.")

        with tab3:
            st.header(tab_names[2])
            qa_context_limit = 7000 
            qa_context = text[:qa_context_limit]

            # --- VOICE INPUT SECTION USING STREAMLIT-WEBRTC ---
            st.subheader("🎙️ Ask by Voice")
            st.markdown("Click **START** to record your question, and **STOP** when you are finished. The AI will process your voice and put the corrected question in the text box below.")

            webrtc_ctx = webrtc_streamer(
                key="audio-recorder",
                mode=WebRtcMode.SENDONLY,
                audio_processor_factory=AudioRecorder,
                media_stream_constraints={"video": False, "audio": True},
            )

            # This block runs when the user clicks "STOP"
            if not webrtc_ctx.state.playing and webrtc_ctx.audio_processor:
                audio_processor = webrtc_ctx.audio_processor
                # Get the recorded audio bytes from the processor
                recorded_audio = audio_processor.get_recorded_audio()

                # Process only if there is new audio data
                if recorded_audio and st.session_state.get("last_audio_chunk") != recorded_audio:
                    st.session_state["last_audio_chunk"] = recorded_audio
                    
                    with st.spinner("Transcribing audio..."):
                        raw_transcript = groq.transcribe_audio(recorded_audio)
                        st.session_state.raw_transcript = raw_transcript

                    if "❌" not in raw_transcript and raw_transcript.strip():
                        st.info(f"**Raw Transcription:** *{st.session_state.raw_transcript}*")
                        with st.spinner("AI is correcting the transcription..."):
                            corrected_transcript = groq.get_corrected_transcription(raw_transcript)
                            st.session_state.corrected_transcript = corrected_transcript
                    else:
                        st.session_state.corrected_transcript = "" 
                        st.error(f"Audio transcription failed or was empty. Error: {raw_transcript}")
            
            if 'corrected_transcript' in st.session_state and st.session_state.corrected_transcript:
                st.success(f"**AI-Corrected Question:** *{st.session_state.corrected_transcript}*")
            
            # --- TEXT INPUT Q&A SECTION ---
            st.subheader("❓ Ask Your Own Question About the Document")
            custom_question = st.text_area("Enter your question here:", 
                                           value=st.session_state.get('corrected_transcript', ''),
                                           placeholder="e.g., What were the total sales for Region X in Q3 2023?", 
                                           height=100)
            if st.button("🔍 Get Answer", key="custom_q_btn", type="primary"):
                if custom_question.strip():
                    with st.spinner("Processing your custom question..."):
                        custom_qa_messages = [
                            {"role": "system", "content": "You are an expert Q&A assistant. Answer based ONLY on the provided document context. If the information is not found, state that clearly."},
                            {"role": "user", "content": f"Document Context:\n```\n{qa_context}\n```\n\nQuestion: {custom_question}"}
                        ]
                        custom_answer = groq.call_api(custom_qa_messages, max_tokens=1536) 
                        st.success(f"**Your Question:** {custom_question}\n\n**💡 Answer:**\n{custom_answer}")
                else:
                    st.warning("Please enter a question before submitting.")
            
            st.markdown("---")
            st.subheader("💾 Export Generated Content & Preview")
            
            exp_col, prev_col = st.columns(2)
            with exp_col:
                st.markdown("#### Download Generated Reports")
                exportable_content = {
                    "Comprehensive Analysis": "comp_analysis", 
                    "KPI Extraction (Raw)": "kpi_text_ai", 
                    "Sales Risk Assessment": "risk_ai"
                }
                
                export_buttons_made = 0
                for display_label, session_key in exportable_content.items():
                    if session_key in st.session_state and st.session_state[session_key] and "❌" not in str(st.session_state[session_key]):
                        file_name_prefix = display_label.replace(" ", "_").replace("(", "").replace(")", "")
                        st.download_button(
                            label=f"📄 Download {display_label}",
                            data=st.session_state[session_key],
                            file_name=f"{file_name_prefix}_{datetime.now():%Y%m%d_%H%M}.md",
                            mime="text/markdown",
                            key=f"download_{session_key}"
                        )
                        export_buttons_made += 1
                if export_buttons_made == 0 : st.info("No content generated yet to export for this session.")

            with prev_col:
                st.markdown("#### Document Preview (First part of extracted text)")
                preview_len = st.slider("Adjust preview length (characters):", 
                                        min_value=500, 
                                        max_value=min(5000, len(text)), 
                                        value=min(1500, len(text)), 
                                        step=100,
                                        key="preview_slider")
                st.text_area("Beginning of extracted text:", text[:preview_len], height=200, disabled=True, key="preview_text")

    else: 
        st.info("👋 Welcome! Please upload a sales-related PDF document using the browser above to begin analysis.")
        if not initial_valid_keys_present: 
            st.error(
                "Reminder: Groq API Key configuration needs attention."
            )

if __name__ == "__main__":
    # Initialize session state keys
    for key_name in ['comp_analysis', 'kpi_text_ai', 'risk_ai', 'raw_transcript', 'corrected_transcript', 'last_audio_chunk']: 
        if key_name not in st.session_state: 
            st.session_state[key_name] = None
    try:
        main()
    except Exception as e:
        st.error(f"An critical application error occurred: {e}")
        st.exception(e)
