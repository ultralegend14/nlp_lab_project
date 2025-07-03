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
from datetime import datetime
from io import BytesIO
import time
from typing import Dict, List, Optional, Tuple, Any
import warnings
import speech_recognition as sr

warnings.filterwarnings('ignore')

# PDF Processing
try:
    import PyPDF2
    import fitz  # PyMuPDF
    import pdfplumber
    PDF_LIBRARIES_AVAILABLE = True
except ImportError:
    PDF_LIBRARIES_AVAILABLE = False

# NLP Libraries
try:
    from transformers import pipeline
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    NLP_LIBRARIES_AVAILABLE = True
except ImportError:
    NLP_LIBRARIES_AVAILABLE = False

# Word Cloud
try:
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# ========== CONFIGURATION ==========
st.set_page_config(
    page_title="Sales Document Analyzer",
    page_icon="📈",
    layout="wide"
)

# --- IMPORTANT: PASTE YOUR GROQ API KEYS HERE ---
# Get your free keys from: https://console.groq.com/keys
ALL_GROQ_API_KEYS = [
    "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",
]
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

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

# ========== GROQ API INTERFACE ==========
class GroqInterface:
    def __init__(self, api_keys_list: List[str]):
        if not api_keys_list:
            raise ValueError("API keys list cannot be empty for GroqInterface.")
        
        self.api_keys = [key for key in api_keys_list if key and key.startswith("gsk_") and "YOUR_GROQ_API_KEY" not in key]
        
        if not self.api_keys:
            raise ValueError("No valid API keys (starting with gsk_ and not placeholders) were found in the provided list.")

        self.current_key_index = 0
        self.base_url = GROQ_API_URL
        
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
            "text_enhancement": """
            You are a text enhancement assistant. Fix any spelling or grammar errors in the following text, but preserve the meaning and intent. Only return the corrected text without any additional comments or explanations.
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
                response = requests.post(self.base_url, headers=headers, json=data, timeout=180)
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
    
    def enhance_text(self, text: str) -> str:
        """Enhance text by correcting spelling and grammar errors"""
        system_message = self.prompts.get("text_enhancement", "")
        if not system_message:
            return text
        
        messages = [{"role": "system", "content": system_message},
                   {"role": "user", "content": f"Text to correct: {text}"}]
        
        return self.call_api(messages, max_tokens=len(text) + 100)

# ========== SPEECH RECOGNITION ==========
def transcribe_audio():
    """Transcribe audio from microphone input"""
    try:
        r = sr.Recognizer()
        
        # Configure recognizer for better accuracy
        r.energy_threshold = 300
        r.dynamic_energy_threshold = True
        r.pause_threshold = 0.8
        r.phrase_threshold = 0.3
        r.non_speaking_duration = 0.8
        
        with sr.Microphone() as source:
            # Create placeholder for listening status
            status_placeholder = st.empty()
            status_placeholder.markdown('<div style="text-align: center; color: #ff6b6b; font-weight: bold; padding: 1rem; background: rgba(255, 107, 107, 0.1); border-radius: 10px; margin: 1rem 0;">🎤 Listening... Speak clearly!</div>', unsafe_allow_html=True)
            
            # Adjust for ambient noise
            r.adjust_for_ambient_noise(source, duration=1.0)
            
            # Listen for audio
            audio = r.listen(source, timeout=10, phrase_time_limit=15)
            status_placeholder.empty()
            
            # Show processing status
            processing_placeholder = st.empty()
            processing_placeholder.markdown('<div style="text-align: center; color: #4834d4; font-weight: bold; padding: 1rem; background: rgba(72, 52, 212, 0.1); border-radius: 10px; margin: 1rem 0;">🔄 Processing speech...</div>', unsafe_allow_html=True)
            
            # Try multiple recognition services for better accuracy
            text = ""
            try:
                # Primary: Google Speech Recognition
                text = r.recognize_google(audio, language='en-US')
            except sr.UnknownValueError:
                try:
                    # Fallback: Google with alternative language model
                    text = r.recognize_google(audio, language='en-IN')
                except sr.UnknownValueError:
                    processing_placeholder.empty()
                    st.error("❌ Could not understand the audio. Please speak more clearly.")
                    return ""
            
            processing_placeholder.empty()
            
            if text:
                st.success(f"✅ Transcribed: \"{text}\"")
                return text.strip()
                
    except sr.WaitTimeoutError:
        st.warning("⏰ No speech detected within 10 seconds. Please try again.")
    except sr.RequestError as e:
        st.error(f"❌ Speech recognition service error: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected error during transcription: {e}")
    
    return ""

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

# ========== KPI PARSING AND CHARTING ==========
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
        "region_segment": ["region/segment", "region", "market", "segment"],
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
        st.error("CRITICAL: PDF processing libraries (PyPDF2, PyMuPDF, pdfplumber) are not installed. "
                 "Please install them: `pip install PyPDF2 PyMuPDF pdfplumber`")
        st.stop()

    initial_valid_keys_present = False
    if ALL_GROQ_API_KEYS: 
        for key_val in ALL_GROQ_API_KEYS:
            if isinstance(key_val, str) and key_val.startswith("gsk_") and "YOUR_GROQ_API_KEY" not in key_val:
                initial_valid_keys_present = True
                break 
    
    if not initial_valid_keys_present:
        st.error(
            "CRITICAL: No valid Groq API Keys are configured in the `ALL_GROQ_API_KEYS` list. "
            "Please add at least one valid key (e.g., 'gsk_xxxxxxxx') to the script."
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
        opt_auto_correct = st.checkbox("Auto-correct speech transcription", True, help="Automatically correct transcribed speech using AI")

    uploaded_file = st.file_uploader("Upload Sales PDF Document", type=["pdf"], help="Text-based PDFs work best. Scanned PDFs (images) will not work well.")
    
    if uploaded_file:
        with st.spinner("📄 Extracting text from PDF... This may take a moment."):
            text = PDFProcessor.process_pdf(uploaded_file)
        
        if not text or len(text.strip()) < 50: 
            st.error("❌ No significant text could be extracted from the PDF. "
                     "The PDF might be image-based, empty, or corrupted. Please try a different, text-based PDF.")
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
                 st.warning("NLP libraries for sentiment analysis are not available. Please install `transformers` and `torch`/`tensorflow`.")

        with tab2:
            st.header(tab_names[1])
            if st.button("📊 Extract Sales KPIs (for Charts)", key="btn_kpi_extraction", type="primary"):
                with st.spinner("Extracting Key Performance Indicators... This can take some time due to detailed formatting requirements."):
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
                        st.warning("Could not parse plottable KPIs from the response, or no specific data series were found in the required format. "
                                   "Please check the raw data output above.")
            
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
                elif opt_wordcloud and not WORDCLOUD_AVAILABLE:
                    st.warning("WordCloud library not installed. Please install it (`pip install wordcloud matplotlib`) to see the visualization.")
            
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
                        st.info("No common sales terms were found by the regex patterns in this document.")

        with tab3:
            st.header(tab_names[2])
            st.subheader("🎯 Quick Predefined Questions")
            qa_context_limit = 7000 
            qa_context = text[:qa_context_limit]

            predefined_questions = [
                "What are the main product or model sales figures mentioned?", 
                "List the key sales metrics and their values found in the document.", 
                "What are the major sales risks identified?", 
                "Provide a brief overall sales performance summary based on this document."
            ]
            
            q_cols = st.columns(len(predefined_questions))
            for idx, question_text in enumerate(predefined_questions):
                if q_cols[idx].button(question_text, key=f"quick_q_{idx}"):
                    with st.spinner(f"Answering: \"{question_text[:30]}...\""):
                        qa_messages = [
                            {"role": "system", "content": "You are a helpful Q&A assistant. Answer the user's question based ONLY on the provided document context. If the information is not found in the context, explicitly state that. Be concise."},
                            {"role": "user", "content": f"Document Context:\n```\n{qa_context}\n```\n\nQuestion: {question_text}"}
                        ]
                        answer = groq.call_api(qa_messages, max_tokens=1024) 
                        st.info(f"**Question:** {question_text}\n\n**Answer:**\n{answer}")
            
            st.subheader("❓ Ask Your Own Question About the Document")
            
            # Initialize session state for transcribed question
            if 'transcribed_question' not in st.session_state:
                st.session_state.transcribed_question = ''
            
            # Check if we need to force a rerun after transcription
            if 'force_rerun' not in st.session_state:
                st.session_state.force_rerun = False
            
            col1, col2 = st.columns([4, 1])
            
            with col2:
                st.write("")
                st.write("")
                if st.button("🎤 Record", key="custom_q_mic", help="Record your question using microphone"):
                    with st.spinner("🎤 Recording and transcribing..."):
                        transcribed_text = transcribe_audio()
                        if transcribed_text:
                            if opt_auto_correct:
                                with st.spinner("Enhancing transcription..."):
                                    try:
                                        enhanced_text = groq.enhance_text(transcribed_text)
                                        st.session_state.transcribed_question = enhanced_text
                                        st.success(f"✅ Transcribed and enhanced: '{enhanced_text}'")
                                    except Exception as e:
                                        st.error(f"Enhancement failed: {e}")
                                        st.session_state.transcribed_question = transcribed_text
                                        st.success(f"✅ Transcribed: '{transcribed_text}'")
                            else:
                                st.session_state.transcribed_question = transcribed_text
                                st.success(f"✅ Transcribed: '{transcribed_text}'")
                            
                            # Force a rerun to update the text area
                            st.session_state.force_rerun = True
                            st.rerun()
                        else:
                            st.error("❌ No audio transcribed. Please try again.")
                            st.session_state.transcribed_question = ""
            
            with col1:
                # Use the current value from session state
                current_question = st.session_state.transcribed_question
                
                custom_question = st.text_area(
                    "Enter your question here:", 
                    value=current_question,
                    placeholder="e.g., What were the total sales for Region X in Q3 2023?", 
                    height=100,
                    key="custom_question_input"
                )
                
                # Update session state if user manually edits the text area
                if custom_question != current_question:
                    st.session_state.transcribed_question = custom_question
            
            # Reset force rerun flag
            if st.session_state.force_rerun:
                st.session_state.force_rerun = False
            
            if st.button("🔍 Get Answer for Custom Question", key="custom_q_btn", type="primary"):
                question_to_process = custom_question.strip()
                if question_to_process:
                    with st.spinner("Processing your custom question..."):
                        custom_qa_messages = [
                            {"role": "system", "content": "You are an expert Q&A assistant. Answer the user's question based ONLY on the provided document context. If the information is not found, state that clearly. Provide specific details if available."},
                            {"role": "user", "content": f"Document Context:\n```\n{qa_context}\n```\n\nQuestion: {question_to_process}"}
                        ]
                        custom_answer = groq.call_api(custom_qa_messages, max_tokens=1536) 
                        st.success(f"**Your Question:** {question_to_process}\n\n**💡 Answer:**\n{custom_answer}")
                        
                        # Clear the transcribed question after processing
                        st.session_state.transcribed_question = ""
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
                if export_buttons_made == 0: 
                    st.info("No content generated yet to export for this session.")

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
                "Reminder: Groq API Key configuration needs attention. "
                "Ensure `ALL_GROQ_API_KEYS` in the script contains valid keys."
            )

if __name__ == "__main__":
    # Initialize session state keys
    for key_name in ['comp_analysis', 'kpi_text_ai', 'risk_ai', 'transcribed_question', 'force_rerun']: 
        if key_name not in st.session_state: 
            st.session_state[key_name] = None if key_name != 'force_rerun' else False
    try:
        main()
    except Exception as e:
        st.error(f"An critical application error occurred: {e}")
        st.exception(e)
