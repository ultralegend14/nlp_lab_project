import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import re
from datetime import datetime
from io import BytesIO
import time
from typing import Dict, List, Optional, Tuple, Any
import warnings
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
    # from wordcloud import WordCloud # Not used in this version, can be added back if needed
    # import matplotlib.pyplot as plt # Not used in this version
    NLP_LIBRARIES_AVAILABLE = True
except ImportError:
    NLP_LIBRARIES_AVAILABLE = False

# ========== CONFIGURATION ==========
st.set_page_config(
    page_title="Sales Document Analyzer",
    page_icon="📈",
    layout="wide"
)

# API Configuration
GROQ_API_KEY = "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"  # <--- IMPORTANT: REPLACE WITH YOUR ACTUAL GROQ API KEY
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
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = GROQ_API_URL
    
    def call_api(self, messages: List[Dict], model: str = "llama3-70b-8192", 
                 temperature: float = 0.05, max_tokens: int = 3000) -> str: # Temp very low for structured output
        if not self.api_key or "YOUR_GROQ_API_KEY" in self.api_key:
            return "❌ API Error: GROQ_API_KEY is not configured. Please set your API key."
        
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=180) # Increased timeout
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except requests.exceptions.HTTPError as http_err:
            err_content = response.text
            try: err_content = response.json()
            except: pass
            return f"❌ API HTTP Error: {http_err}. Response: {err_content}"
        except Exception as e: return f"❌ API/Request Error: {str(e)}"
    
    def analyze_document(self, text: str, analysis_type: str = "comprehensive") -> str:
        prompts = {
            "comprehensive": """
            You are an expert sales analyst. Analyze this sales document and provide a comprehensive report covering:
            1. Executive Sales Summary. 2. Sales Performance Analysis (overall, trends). 3. Product/Service/Model Performance. 4. Key Sales Metrics & KPIs (explained). 5. Regional/Channel Performance. 6. Sales Team & Strategy Effectiveness. 7. Market & Competitive Landscape. 8. Sales Risk Assessment. 9. Strategic Sales Insights & Outlook. 10. Recommendations for Sales Improvement.
            Be specific, cite numbers, and use Markdown.
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
            "summary": """
            Provide a concise executive summary of this sales document. Highlight key sales figures (total revenue, product/model sales), performance metrics (growth, KPIs), achievements, challenges, and strategic outlook in under 300 words. Use Markdown.
            """
        }
        system_message = prompts.get(analysis_type, prompts["comprehensive"])
        max_input_text_length = 28000 # Approx 7k tokens
        
        messages = [{"role": "system", "content": system_message},
                    {"role": "user", "content": f"Document Content:\n{text[:max_input_text_length]}"}]
        
        # Increased max_tokens for KPI specifically due to detailed table format.
        output_tokens = 4096 if analysis_type == "kpi" else 3000
        return self.call_api(messages, max_tokens=output_tokens)

# ========== NLP ANALYZER ==========
class NLPAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = None
        if NLP_LIBRARIES_AVAILABLE: self.load_models()
    
    def load_models(self):
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        except Exception as e: st.warning(f"Sentiment model load failed: {e}")
    
    def analyze_sentiment(self, text: str) -> Dict: # Same as before
        if not NLP_LIBRARIES_AVAILABLE or not self.sentiment_analyzer: return {"error": "Sentiment analyzer unavailable."}
        try:
            chunks = [text[i:i+500] for i in range(0, len(text), 500)][:20]
            results = [self.sentiment_analyzer(c)[0] for c in chunks if len(c.strip()) > 20]
            if not results: return {"error": "No text for sentiment analysis."}
            s_counts = pd.Series([r['label'].upper() for r in results]).value_counts()
            overall = s_counts.index[0] if not s_counts.empty else "NEUTRAL"
            if len(s_counts) > 1 and s_counts.iloc[0] == s_counts.iloc[1]: overall = "NEUTRAL"
            return {'overall_sentiment': overall, 'confidence': np.mean([r['score'] for r in results]) if results else 0, 'distribution': s_counts.to_dict()}
        except Exception as e: return {"error": f"Sentiment analysis error: {e}"}

    def extract_relevant_terms(self, text: str) -> Dict: # Same as before
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
                extracted[name] = sorted(list(set(m.strip() for m in matches if m.strip())), key=len, reverse=True)[:15]
            except: extracted[name] = []
        return extracted
    
    def get_document_stats(self, text: str) -> Dict: # Same as before
        words = text.split()
        stats = {'word_count': len(words), 'character_count': len(text), 'estimated_reading_time': round(len(words)/200,1) or 0}
        if NLP_LIBRARIES_AVAILABLE and len(text.strip()) > 100:
            try:
                stats['readability_score'] = round(flesch_reading_ease(text),1)
                stats['grade_level'] = round(flesch_kincaid_grade(text),1)
            except: stats.update({'readability_score': 'N/A', 'grade_level': 'N/A'})
        else: stats.update({'readability_score': 'N/A', 'grade_level': 'N/A'})
        return stats

# ========== KPI PARSING AND CHARTING (NEW - V2) ==========
def convert_value_unit_v2(value_str: Optional[str], unit_str: Optional[str]) -> Tuple[Optional[float], str]:
    if value_str is None: return None, str(unit_str or "UNKNOWN").strip().upper()
    
    val_str_cleaned = str(value_str).lower().replace(',', '').strip()
    unit_final = str(unit_str or "").strip()

    # Prepend currency from unit_str if value is just numeric
    # E.g. Value: 1.2, Unit: million USD -> unit_final should become USD and multiplier applied
    currency_prefixes = {"usd": "$", "eur": "€", "gbp": "£"}
    for k, v in currency_prefixes.items():
        if k in unit_final.lower():
            if v not in val_str_cleaned: # If symbol not already in value
                 val_str_cleaned = v + val_str_cleaned # Prepend for logic below
            unit_final = unit_final.lower().replace(k, "").strip() # Remove currency keyword from unit string

    currency_symbol = ""
    if '$' in val_str_cleaned: currency_symbol = "USD"; val_str_cleaned = val_str_cleaned.replace('$', '')
    elif '€' in val_str_cleaned: currency_symbol = "EUR"; val_str_cleaned = val_str_cleaned.replace('€', '')
    elif '£' in val_str_cleaned: currency_symbol = "GBP"; val_str_cleaned = val_str_cleaned.replace('£', '')
    val_str_cleaned = val_str_cleaned.strip()

    multiplier = 1.0
    # Check unit_str first for multipliers, then val_str_cleaned
    # This allows value to be "1.2" and unit "million USD"
    string_to_check_multiplier = (unit_final + " " + val_str_cleaned).lower()

    if 'billion' in string_to_check_multiplier: multiplier = 1e9
    elif 'million' in string_to_check_multiplier: multiplier = 1e6
    elif 'k ' in string_to_check_multiplier or val_str_cleaned.endswith('k'): # Check for "k " or "100k"
        if val_str_cleaned.endswith('k'): val_str_cleaned = val_str_cleaned[:-1] # Remove k from value if it's there
        multiplier = 1e3
        
    # Remove multiplier words from unit_final if they were there
    unit_final = unit_final.lower().replace("billion", "").replace("million", "").replace("k ", " ").replace("thousand","").strip()

    try:
        numeric_val = float(val_str_cleaned) * multiplier
    except ValueError:
        # st.warning(f"Could not convert value string '{value_str}' to float.") # Debug
        return None, (currency_symbol + " " + unit_final).strip().upper() or "UNKNOWN"

    final_unit_parts = []
    if currency_symbol: final_unit_parts.append(currency_symbol)
    if unit_final: final_unit_parts.append(unit_final)
    
    # Check for % in original unit string if not already part of unit_final
    if "%" in str(unit_str or "") and "%" not in unit_final :
        if not final_unit_parts or final_unit_parts[-1]!="%": final_unit_parts.append("%")
    elif not final_unit_parts: # If no currency, no unit string remnants
        if "%" in str(value_str or "") : final_unit_parts.append("%") # Check original value for %

    return numeric_val, " ".join(final_unit_parts).strip().upper() or "VALUE"


def parse_kpi_data_from_ai_text_v2(ai_text: str) -> List[Dict[str, Any]]:
    parsed_kpis = []
    lines = ai_text.split('\n')
    current_table_type = "Unknown"
    current_headers: List[str] = []
    header_indices: Dict[str, int] = {}
    
    # Expected column name patterns (lowercase for matching)
    col_patterns = {
        "metric_name": ["metric name", "kpi name"],
        "period": ["period"],
        "value": ["value"],
        "unit": ["unit"],
        "product_model_name": ["product/model name", "category/model name"],
        "category": ["category"], # Generic category
        "metric_type": ["metric type"], # E.g. 'Sales Revenue', 'Units Sold' for product/region tables
        "region_segment": ["region/segment"],
        "context_period": ["context/period", "context", "notes"] # For standalone KPIs
    }

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped: continue

        # Detect H3 table type headers
        h3_match = re.match(r"^\s*###\s*(.+)", line_stripped)
        if h3_match:
            heading_text = h3_match.group(1).lower()
            if "overall sales trend" in heading_text or "by period" in heading_text: current_table_type = "time_series"
            elif "product/model" in heading_text: current_table_type = "product_model"
            elif "region/segment" in heading_text: current_table_type = "region_segment"
            elif "standalone metrics" in heading_text: current_table_type = "standalone"
            else: current_table_type = "generic_table"
            current_headers = [] # Reset headers for new table
            header_indices = {}
            continue

        # Detect and parse table header row
        if line_stripped.startswith("|") and "|---" not in line_stripped and not current_headers:
            headers_raw = [h.strip().lower() for h in line_stripped.strip('|').split('|')]
            temp_header_indices = {}
            valid_header = False
            for i, header_text in enumerate(headers_raw):
                for semantic_name, patterns in col_patterns.items():
                    if any(p in header_text for p in patterns):
                        temp_header_indices[semantic_name] = i
                        valid_header = True # Found at least one recognizable header
                        break 
            if valid_header and ('value' in temp_header_indices): # Value column is essential
                current_headers = headers_raw # Store raw headers for data row length check
                header_indices = temp_header_indices
            continue

        # Parse table data row
        if line_stripped.startswith("|") and "|---" not in line_stripped and current_headers and header_indices:
            cols = [c.strip() for c in line_stripped.strip('|').split('|')]
            if len(cols) == len(current_headers): # Ensure column count matches header
                kpi_entry: Dict[str, Any] = {"table_type": current_table_type}
                
                # Extract known fields using header_indices
                raw_value_str: Optional[str] = None
                raw_unit_str: Optional[str] = None

                for semantic_name, idx in header_indices.items():
                    if semantic_name == "value":
                        raw_value_str = cols[idx]
                    elif semantic_name == "unit":
                        raw_unit_str = cols[idx]
                    else: # Other text fields
                        kpi_entry[semantic_name] = cols[idx]
                
                # Convert value and unit
                parsed_value, parsed_unit = convert_value_unit_v2(raw_value_str, raw_unit_str)
                
                if parsed_value is not None:
                    kpi_entry["value"] = parsed_value
                    kpi_entry["unit"] = parsed_unit
                    
                    # Consolidate naming for simpler charting logic later
                    if current_table_type == "product_model" and "product_model_name" in kpi_entry:
                        kpi_entry["category_key"] = kpi_entry["product_model_name"]
                    elif current_table_type == "region_segment" and "region_segment" in kpi_entry:
                        kpi_entry["category_key"] = kpi_entry["region_segment"]
                    
                    # Ensure a primary name/identifier exists
                    if "metric_name" not in kpi_entry and "product_model_name" in kpi_entry:
                        kpi_entry["metric_name"] = kpi_entry["product_model_name"] # Use product name if main metric name missing
                    elif "metric_name" not in kpi_entry and "region_segment" in kpi_entry:
                         kpi_entry["metric_name"] = kpi_entry["region_segment"]


                    if "metric_name" in kpi_entry or "category_key" in kpi_entry: # Must have some identifier
                        parsed_kpis.append(kpi_entry)
                # else:
                    # st.warning(f"Skipping row due to unparsable value: {line_stripped}") # Debug

    return parsed_kpis


def create_dynamic_kpi_charts_v2(parsed_kpis: List[Dict[str, Any]]):
    if not parsed_kpis:
        st.info("No plottable KPI data was successfully parsed from the AI's response after filtering.")
        return

    df_kpis = pd.DataFrame(parsed_kpis)
    if df_kpis.empty or 'value' not in df_kpis.columns:
        st.info("KPI data is empty or missing 'value' column after parsing.")
        return
        
    df_kpis['value'] = pd.to_numeric(df_kpis['value'], errors='coerce')
    df_kpis.dropna(subset=['value'], inplace=True) # Critical for plotting

    if df_kpis.empty:
        st.info("No valid numeric KPI values found for plotting after parsing.")
        return

    st.markdown("---")
    st.markdown("### 📊 Visualized Sales KPIs")
    
    plotted_indices = set() # To track which rows are plotted to avoid double plotting

    # Chart 1: Time-Series Data (e.g., Overall Sales Trend by Period)
    time_series_data = df_kpis[df_kpis['table_type'] == 'time_series'].copy()
    if not time_series_data.empty and 'period' in time_series_data.columns and 'metric_name' in time_series_data.columns:
        # Attempt to sort by period (basic, can be improved with date parsing)
        try:
            # Create a sortable period if possible (YYYYMM or YYYYQ)
            def create_sort_key(period_str):
                period_str = str(period_str)
                q_match = re.match(r"(Q[1-4])\s*(\d{4})", period_str, re.IGNORECASE)
                if q_match: return f"{q_match.group(2)}{q_match.group(1).upper()}"
                m_match = re.match(r"(\w{3,})\s*(\d{4})", period_str, re.IGNORECASE) # Jan 2024
                if m_match:
                    try: return datetime.strptime(period_str, "%b %Y").strftime("%Y%m")
                    except: pass
                    try: return datetime.strptime(period_str, "%B %Y").strftime("%Y%m")
                    except: pass
                if re.match(r"^\d{4}$", period_str): return period_str # Year only
                return period_str # Fallback to string sort
            
            time_series_data['sortable_period'] = time_series_data['period'].apply(create_sort_key)
            time_series_data.sort_values(by=['metric_name', 'sortable_period'], inplace=True)
        except Exception: # Fallback if sorting fails
            time_series_data.sort_values(by=['metric_name', 'period'], inplace=True)

        for metric, group in time_series_data.groupby('metric_name'):
            if len(group) > 1: # Only plot if there's a series
                st.subheader(f"Trend: {metric}")
                fig = px.line(group, x='period', y='value', color='unit', markers=True,
                              title=f"{metric} Over Time",
                              labels={'value': 'Value', 'period': 'Period', 'unit': 'Unit'})
                fig.update_traces(texttemplate='%{y:,.2s}', textposition='top center')
                st.plotly_chart(fig, use_container_width=True)
                plotted_indices.update(group.index)
            elif len(group) == 1: # Single data point for this time-series metric
                 entry = group.iloc[0]
                 st.metric(label=f"{entry.get('metric_name','Metric')} ({entry.get('period','Period')})", 
                           value=f"{entry['value']:,.2f} {entry.get('unit','')}")
                 plotted_indices.update(group.index)


    # Chart 2: Product/Model Performance
    product_model_data = df_kpis[df_kpis['table_type'] == 'product_model'].copy()
    if not product_model_data.empty and 'product_model_name' in product_model_data.columns and 'metric_type' in product_model_data.columns:
        st.subheader("Performance by Product/Model")
        # Ensure 'period' exists for faceting, or use a placeholder
        if 'period' not in product_model_data.columns: product_model_data['period'] = 'Overall'
        else: product_model_data['period'] = product_model_data['period'].fillna('Overall')

        for (metric_type, period), group in product_model_data.groupby(['metric_type', 'period']):
            if len(group) > 1 : # Plot if multiple products/models for this metric_type/period
                title = f"{metric_type} by Product/Model"
                if period != 'Overall': title += f" ({period})"
                
                fig = px.bar(group, x='product_model_name', y='value', color='unit',
                             title=title, text_auto='.2s',
                             labels={'value': 'Value', 'product_model_name': 'Product/Model', 'unit': 'Unit'})
                fig.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)
                plotted_indices.update(group.index)

    # Chart 3: Regional/Segment Performance
    region_segment_data = df_kpis[df_kpis['table_type'] == 'region_segment'].copy()
    if not region_segment_data.empty and 'region_segment' in region_segment_data.columns and 'metric_type' in region_segment_data.columns:
        st.subheader("Performance by Region/Segment")
        if 'period' not in region_segment_data.columns: region_segment_data['period'] = 'Overall'
        else: region_segment_data['period'] = region_segment_data['period'].fillna('Overall')
        
        for (metric_type, period), group in region_segment_data.groupby(['metric_type', 'period']):
            if len(group) > 1 :
                title = f"{metric_type} by Region/Segment"
                if period != 'Overall': title += f" ({period})"
                fig = px.bar(group, x='region_segment', y='value', color='unit',
                             title=title, text_auto='.2s',
                             labels={'value': 'Value', 'region_segment': 'Region/Segment', 'unit': 'Unit'})
                fig.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)
                plotted_indices.update(group.index)

    # Display Standalone KPIs & any remaining unplotted KPIs as st.metric
    remaining_kpis = df_kpis[~df_kpis.index.isin(plotted_indices)].copy()
    if not remaining_kpis.empty:
        st.subheader("Other Key Metrics / Standalone KPIs")
        # Prioritize 'standalone' table type if available
        standalone_metrics = remaining_kpis[remaining_kpis['table_type'] == 'standalone']
        other_metrics = remaining_kpis[remaining_kpis['table_type'] != 'standalone']
        
        metrics_to_display_asis = pd.concat([standalone_metrics, other_metrics])

        num_cols = min(len(metrics_to_display_asis), 4) # Max 4 columns for metrics
        if num_cols > 0:
            metric_cols = st.columns(num_cols)
            col_idx = 0
            for _, row in metrics_to_display_asis.iterrows():
                label = str(row.get('metric_name') or row.get('kpi_name', 'Metric')) # Use kpi_name if from 'standalone' table
                context = str(row.get('period') or row.get('context_period') or row.get('category_key','')).strip()
                if context : label += f" ({context})"
                
                value_display = f"{row['value']:,.2f}"
                unit_display = str(row.get('unit','')).strip()
                if unit_display and unit_display != "UNKNOWN" and unit_display != "VALUE":
                    value_display += f" {unit_display}"
                
                with metric_cols[col_idx % num_cols]:
                    st.metric(label=label, value=value_display)
                col_idx += 1


# ========== VISUALIZATION (Existing - Sentiment, Regex Terms) ==========
def create_sentiment_chart(sentiment_data: Dict) -> go.Figure: # Same as before
    fig = go.Figure()
    if 'distribution' not in sentiment_data or not sentiment_data['distribution']:
        fig.update_layout(title="Sentiment Data Not Available", height=350); return fig
    labels, values = list(sentiment_data['distribution'].keys()), list(sentiment_data['distribution'].values())
    colors = [({'POSITIVE': '#27ae60', 'NEGATIVE': '#c0392b', 'NEUTRAL': '#f39c12'}).get(l, '#7f8c8d') for l in labels]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4, marker_colors=colors, textinfo='label+percent', insidetextorientation='radial')])
    fig.update_layout(title_text="Document Sentiment Distribution", height=350, legend_title_text="Sentiments", margin=dict(t=50, b=0, l=0, r=0))
    return fig

def create_terms_chart(terms_data: Dict) -> go.Figure: # Same as before
    fig = go.Figure()
    if not terms_data or not any(terms_data.values()):
        fig.update_layout(title="No Regex Terms Found", height=350); return fig
    term_counts = {k.replace('_', ' ').title(): len(v) for k, v in terms_data.items() if v}
    if not term_counts: fig.update_layout(title="No Regex Terms Found", height=350); return fig
    fig = go.Figure(data=[go.Bar(x=list(term_counts.keys()), y=list(term_counts.values()), marker_color='#2980b9', text=list(term_counts.values()), textposition='outside')])
    fig.update_layout(title_text="Regex-Found Term Mentions by Category", xaxis_title="Term Category", yaxis_title="Count", height=400, xaxis_tickangle=-30, margin=dict(t=50, b=100, l=0, r=0))
    return fig

# ========== MAIN APPLICATION ==========
def main():
    st.title("📈 AI Sales Document Analyzer")
    st.markdown("Upload a sales-related PDF to extract insights, KPIs, and generate visualizations.")
    
    if not PDF_LIBRARIES_AVAILABLE: st.error("PDF libraries missing. Install: `pip install PyPDF2 PyMuPDF pdfplumber`"); return
    if "YOUR_GROQ_API_KEY" in GROQ_API_KEY or not GROQ_API_KEY:
        st.error("CRITICAL: Groq API Key is not configured. Edit the script and set `GROQ_API_KEY`."); st.stop()
        
    groq = GroqInterface(GROQ_API_KEY)
    nlp = NLPAnalyzer()
    
    with st.sidebar:
        st.header("⚙️ Analysis Options")
        opt_sentiment = st.checkbox("Sentiment Analysis", True)
        opt_regex_terms = st.checkbox("Extract Terms (Regex)", True)
        # opt_visualizations = st.checkbox("Show Visualizations", True) # Visuals are now more integrated

    uploaded_file = st.file_uploader("Upload Sales PDF Document", type=["pdf"], help="Text-based PDFs work best.")
    
    if uploaded_file:
        with st.spinner("📄 Extracting text..."): text = PDFProcessor.process_pdf(uploaded_file)
        if not text or len(text.strip()) < 50: st.error("❌ No text extracted. Try another PDF."); return
        
        st.success(f"✅ Text extracted (~{len(text):,} chars).")
        stats = nlp.get_document_stats(text)
        st.subheader("📄 Document Overview"); c1,c2,c3,c4=st.columns(4)
        c1.metric("Words",f"{stats['word_count']:,}"); c2.metric("Chars",f"{stats['character_count']:,}")
        c3.metric("Read Time",f"{stats['estimated_reading_time']}m"); c4.metric("Flesch Score",str(stats['readability_score']))
        st.markdown("---")

        tab_names = ["🧠 Comprehensive AI Analysis", "🎯 AI KPI Extraction & Charts", "💡 AI Quick Insights", "💬 Interactive Q&A"]
        tab1, tab2, tab3, tab4 = st.tabs(tab_names)

        with tab1: # Comprehensive Analysis
            st.header(tab_names[0])
            if st.button("🚀 Generate Comprehensive Sales Analysis", key="btn_comp"):
                with st.spinner("AI analyzing (comprehensive)..."): st.session_state.comp_analysis = groq.analyze_document(text, "comprehensive")
            if 'comp_analysis' in st.session_state and st.session_state.comp_analysis:
                st.markdown("### AI Comprehensive Analysis:")
                if "❌" in st.session_state.comp_analysis: st.error(st.session_state.comp_analysis)
                else: st.markdown(st.session_state.comp_analysis)
            
            if opt_sentiment and NLP_LIBRARIES_AVAILABLE:
                st.markdown("---"); st.subheader("😊 Sentiment Analysis")
                s_data = nlp.analyze_sentiment(text)
                if 'error' not in s_data:
                    sc1,sc2=st.columns([1,2]); sc1.metric("Overall Sentiment",s_data.get('overall_sentiment','N/A').capitalize())
                    sc1.metric("Confidence",f"{s_data.get('confidence',0):.1%}")
                    if s_data.get('distribution'): sc2.plotly_chart(create_sentiment_chart(s_data),use_container_width=True)
                else: st.warning(s_data['error'])

        with tab2: # KPI Extraction
            st.header(tab_names[1])
            if st.button("📊 Extract Sales KPIs with AI (for Charts)", key="btn_kpi"):
                with st.spinner("AI extracting KPIs (this may take a while)..."): st.session_state.kpi_text_ai = groq.analyze_document(text, "kpi")
            
            if 'kpi_text_ai' in st.session_state and st.session_state.kpi_text_ai:
                st.markdown("### Raw AI Output for KPIs:")
                if "❌" in st.session_state.kpi_text_ai: st.error(st.session_state.kpi_text_ai)
                else: 
                    st.markdown(f"```markdown\n{st.session_state.kpi_text_ai}\n```") # Show raw output
                    with st.spinner("Parsing AI output and generating charts..."):
                        parsed_kpis = parse_kpi_data_from_ai_text_v2(st.session_state.kpi_text_ai)
                    if parsed_kpis:
                        create_dynamic_kpi_charts_v2(parsed_kpis)
                    else:
                        st.warning("Could not parse plottable KPIs from the AI's response, or no specific data series were found in the required format. Check raw output above.")
            
            if opt_regex_terms:
                st.markdown("---"); st.subheader("⚙️ Regex-Based Term Extraction")
                terms_regex = nlp.extract_relevant_terms(text)
                if any(terms_regex.values()):
                    st.plotly_chart(create_terms_chart(terms_regex), use_container_width=True)
                    with st.expander("Details: Regex Terms"):
                        for type, vals in terms_regex.items():
                            if vals: st.markdown(f"**{type.replace('_',' ').title()}:** {', '.join(vals)}")
                else: st.info("No common sales terms found by regex.")

        with tab3: # Quick Insights
            st.header(tab_names[2])
            qc1, qc2 = st.columns(2)
            with qc1:
                if st.button("📋 Generate AI Sales Summary", key="btn_summary"):
                    with st.spinner("AI generating summary..."): st.session_state.summary_ai = groq.analyze_document(text, "summary")
                if 'summary_ai' in st.session_state and st.session_state.summary_ai:
                    st.markdown("### AI Sales Summary:")
                    if "❌" in st.session_state.summary_ai: st.error(st.session_state.summary_ai)
                    else: st.markdown(st.session_state.summary_ai)
            with qc2:
                if st.button("⚠️ Generate AI Sales Risk Assessment", key="btn_risk"):
                    with st.spinner("AI assessing risks..."): st.session_state.risk_ai = groq.analyze_document(text, "risk")
                if 'risk_ai' in st.session_state and st.session_state.risk_ai:
                    st.markdown("### AI Sales Risk Assessment:")
                    if "❌" in st.session_state.risk_ai: st.error(st.session_state.risk_ai)
                    else: st.markdown(st.session_state.risk_ai)
            st.markdown("---"); st.subheader("📄 Document Preview")
            st.text_area("Text Start:", text[:st.slider("Preview Chars",500,min(5000,len(text)),1500)], height=200, disabled=True)

        with tab4: # Q&A
            st.header(tab_names[3])
            st.subheader("🎯 Quick Sales Questions")
            quick_qs = ["Main product/model sales figures?", "Key sales metrics & values?", "Major sales risks?", "Overall sales performance summary?"]
            qcols = st.columns(len(quick_qs))
            for i, q_txt in enumerate(quick_qs):
                if qcols[i].button(q_txt, key=f"qq_{i}"):
                    with st.spinner("AI answering..."):
                        ans = groq.call_api([{"role":"system","content":"Answer sales questions based ONLY on provided context. If not found, state that."},
                                             {"role":"user","content":f"Context:\n{text[:7500]}\n\nQ: {q_txt}"}], max_tokens=1024)
                        st.info(f"**Q:** {q_txt}\n\n**A:** {ans}")
            st.subheader("❓ Ask Your Own Question")
            custom_q = st.text_area("Your question:", placeholder="e.g., What were sales for Model X in Q3?", height=100)
            if st.button("🔍 Get AI Answer", key="custom_q"):
                if custom_q.strip():
                    with st.spinner("AI processing..."):
                        ans = groq.call_api([{"role":"system","content":"Expert sales Q&A. Answer based ONLY on document. If not found, state that."},
                                             {"role":"user","content":f"Document:\n{text[:7500]}\n\nQ: {custom_q}"}], max_tokens=1536)
                        st.success(f"**💡 AI Answer:**\n{ans}")
                else: st.warning("Enter a question.")
            
            st.markdown("---"); st.subheader("💾 Export AI Content")
            exports = {"Comprehensive Analysis": "comp_analysis", "KPI Text (Raw AI)": "kpi_text_ai", "Summary": "summary_ai", "Risk Assessment": "risk_ai"}
            for label, skey in exports.items():
                if skey in st.session_state and st.session_state[skey] and "❌" not in st.session_state[skey]:
                    st.download_button(f"📄 Download {label}", st.session_state[skey], f"{skey}_{datetime.now():%Y%m%d_%H%M}.md", "text/markdown")
    else:
        st.info("👋 Welcome! Upload a sales PDF document using the browser above to get started.")
        if "YOUR_GROQ_API_KEY" in GROQ_API_KEY or not GROQ_API_KEY:
            st.error("Reminder: Groq API Key is not configured correctly in the script.")


if __name__ == "__main__":
    for key in ['comp_analysis', 'kpi_text_ai', 'summary_ai', 'risk_ai']: # Init session state
        if key not in st.session_state: st.session_state[key] = None
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected application error occurred: {e}")
        st.exception(e) 