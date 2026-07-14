import re
import os, json, random, threading, io, datetime
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for,flash
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, FileField, SubmitField, SelectField
from flask_wtf.file import FileAllowed
from concurrent.futures import ThreadPoolExecutor
from google import genai
from groq import Groq
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from fpdf import FPDF
import requests
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchRun
import threading
from datetime import timedelta

os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'msap_exclusive'
external_db_path = r"D:\Other competitive exams"

# --- UPDATED CONFIGURATION ---
CONFIG_PATH = os.path.join(external_db_path, "scraper_config.json")
# Define as string to avoid the TypeError you encountered
START_DATE_STR = "26/05/2026"


bank_folder_path = os.path.join(external_db_path, "Question Bank")
bank_file_path = os.path.join(bank_folder_path, "bank_data.json")
BANK_PATH = bank_file_path
bank_lock = threading.Lock()
# 2. Create the folder automatically if it doesn't exist yet
if not os.path.exists(external_db_path):
    os.makedirs(external_db_path)


if not os.path.exists(bank_folder_path):
    os.makedirs(bank_folder_path)


# 3. Point the database URI to the new location
# We use os.path.join to handle the slashes correctly
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(external_db_path, 'data.sqlite')

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
# --- THE THREE SEPARATE DATABASES ---
class DailyData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    source_id = db.Column(db.String(10))
    content = db.Column(db.Text)

class WeeklyData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    source_id = db.Column(db.String(10))
    content = db.Column(db.Text)

class MonthlyData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    source_id = db.Column(db.String(10))
    content = db.Column(db.Text)

with app.app_context():
    db.create_all()

# --- THE UPLOAD FORM ---
class DataUploadForm(FlaskForm):
    category = SelectField('Category', choices=[('Daily', 'Daily'), ('Weekly', 'Weekly'), ('Monthly', 'Monthly')])
    source_id = StringField('Source ID')
    pdf_file = FileField('PDF File', validators=[FileAllowed(['pdf'])])
    
    # ADD THIS LINE:
    url_link = StringField('Or URL') 
    
    submit = SubmitField('Upload Content')
# --- CONFIGURATION & API KEYS ---
# Replace with your actual keys
GEMINI_KEYS=["AIzaSyA0CBOynF_jPwmn7MGw1SIR0Guw1D8zGfI", "AIzaSyDwENySKDkZJEr67Qec633KrbOMzmkyjwo", "AIzaSyD1Hoxd-Dj5LeapV7BTtWQhV94QkjuoBMs"]  # 3 Gemini Keys for parallel processing
GROQ_KEY_CLEANER = "YOUR_GROQ_KEY"

# Initialize Engines
client = genai.Client(api_key=GEMINI_KEYS[0]) # Default config
groq_client = Groq(api_key=GROQ_KEY_CLEANER)

# Mock Databases (In a real app, these would be your uploaded text files)
# Structure: { "ID": "Text content" }
db_daily = {"11": "Source 1 text...", "12": "Source 2 text..."}
db_weekly = {"21": "Source 1 text...", "22": "Source 2 text..."}
db_monthly = {"31": "Source 1 text...", "345": "Source 45 text..."}

def load_question_bank():
    """Loads the question bank from the JSON file on disk."""
    if os.path.exists(bank_file_path):
        try:
            with open(bank_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] Could not load bank: {e}")
            return {}
    return {}

def save_question_bank():
    """Saves the current QUESTION_BANK dictionary to the JSON file."""
    try:
        with open(bank_file_path, 'w', encoding='utf-8') as f:
            json.dump(QUESTION_BANK, f, indent=4)
        print(f"[SUCCESS] Question Bank saved to {bank_file_path}")
    except Exception as e:
        print(f"[ERROR] Save failed: {e}")

# Initialize by loading existing data
QUESTION_BANK = load_question_bank()


print("[DEBUG] System Initialized. 4 API Keys configured.")
print(f"[DEBUG] Databases loaded: Daily({len(db_daily)}), Weekly({len(db_weekly)}), Monthly({len(db_monthly)})")

# --- UTILITY: SIMILARITY CLEANER ---
def get_embedding(text):
    """Uses the stable 005 model to avoid 404 errors."""
    try:
        # Use text-embedding-005 as it is globally available on v1beta
        clean_text = " ".join(text.split())[:2000]
        result = client.embed_content(model="models/gemini-embedding-001", content=clean_text)
        return result.embeddings[0].values
    except Exception as e:
        print(f"[EMBEDDING ERROR] {e}")
        return [0] * 768 # Return dummy vector so the app doesn't crash

def is_unique(new_question_text):
    """Checks for both exact string matches and semantic similarity."""
    # 1. String Normalization Check (Immediate)
    normalized_new = " ".join(new_question_text.lower().split())
    
    if not QUESTION_BANK:
        return True

    # Check for exact matches first
    for old_text in QUESTION_BANK.keys():
        if normalized_new == " ".join(old_text.lower().split()):
            print(f"[CLEANER] REJECTED: Exact string match.")
            return False

    # 2. Semantic Check
    new_raw_vec = get_embedding(new_question_text)
    if new_raw_vec is None: 
        return True # If API fails, allow it but it might cause a dupe later
    
    new_vec = np.array(new_raw_vec).reshape(1, -1)
    
    for old_text, data in QUESTION_BANK.items():
        if 'embedding' not in data or data['embedding'] is None:
            continue
            
        old_vec = np.array(data['embedding']).reshape(1, -1)
        
        # Guard against zero vectors
        if np.count_nonzero(old_vec) == 0: continue

        similarity = cosine_similarity(new_vec, old_vec)[0][0]
        
        # Threshold: 0.85-0.90 is usually better for detecting "same question, different wording"
        if similarity > 0.88:
            print(f"[CLEANER] REJECTED: Semantic Similarity {similarity:.2f}")
            return False
            
    print(f"[CLEANER] ACCEPTED: Unique question.")
    return True

# --- THE ENGINES ---
# --- NEW: DELETE ROUTE ---
@app.route('/delete/<category>/<int:item_id>')
def delete_item(category, item_id):
    if category == 'Daily': item = DailyData.query.get(item_id)
    elif category == 'Weekly': item = WeeklyData.query.get(item_id)
    else: item = MonthlyData.query.get(item_id)
    
    if item:
        db.session.delete(item)
        db.session.commit()
    return redirect('/')
import re
def run_engine(api_key, engine_name, sources, num_q, exam_type):
    print(f"[{engine_name} ENGINE] Starting. Target: {num_q} questions.")
    
    # Configure specific key for this thread
    client = genai.Client(api_key=api_key)
    
    source_text = "\n".join([f"SOURCE ID {k}: {v}" for k, v in sources.items()])
    source_count = len(sources)
    q_per_file = num_q // source_count if source_count > 0 else 0

    search = DuckDuckGoSearchRun()
    pattern = search.run(f"actual UPSC Prelims GS 1 MCQ patterns 2023 2024 statement based questions framing logic")

    prompt = f"""
        ROLE: You are an expert Exam Designer specialized in {exam_type}.
        CONTEXT: I have provided {source_count} sources below, labeled with 'SOURCE ID'.
        {source_text}
        

        STRICT TASK INSTRUCTIONS:
        1. Generate EXACTLY {num_q} questions.
        2. MATHEMATICAL QUOTA: Approx {q_per_file} questions from EACH SOURCE ID.
        3. Use CONTEXT MATERIAL only. No outside knowledge.
        4. Torture the candidate with difficult, tricky questions (UPSC style).
        5. Mimic the exact pattern of questions as per the search results below:{pattern}
        EXHAUSTIVE TYPE DISTRIBUTION:
        1. 45% - "HOW MANY" PAIRS:
        - Format: Give 3 or 4 statements or pairs.
        - Options MUST BE: (a) Only one (b) Only two (c) Only three (d) All four / None.
        
        2. 20% - STATEMENT-BASED:
        - Format: "Consider the following statements:" followed by 2-3 numbered statements.
        - Options: Traditional combinations (e.g., 1 and 2 only).
        
        3. 15% - ASSERTION-REASON (NEW PATTERN):
        - Format:
            Statement-I: [Fact]
            Statement-II: [Explanation/Reasoning]
        - Options: 
            (a) Both Statement-I and Statement-II are correct and Statement-II is the correct explanation for Statement-I.
            (b) Both Statement-I and Statement-II are correct and Statement-II is not the correct explanation for Statement-I.
            (c) Statement-I is correct but Statement-II is incorrect.
            (d) Statement-I is incorrect but Statement-II is correct.

        4. 10% - MATCH THE FOLLOWING:
        - Format: Two columns (List-I and List-II).
        
        5. 10% - CONCEPTUAL DIRECT:
        - Format: "Which one of the following best describes..." or "With reference to..."

        STRICT FORMATTING RULES:
        - For Statement/Pair questions: Each statement MUST start on a new line with a number (1., 2.).
        - For Match the Following: Use a clear text-based table format.
        - Source Attribution: The "explanation" MUST quote the source text and cite the SOURCE ID.

        OUTPUT FORMAT (Raw JSON):
        {{
        "questions": [
            {{
            "question": "Question text with proper line breaks (\\n) for statements",
            "options": ["(a) ...", "(b) ...", "(c) ...", "(d) ..."],
            "answer": "(a) ...", 
            "explanation": "Text from source + SOURCE ID"
            }}
        ]
        }}
        CRITICAL: Ensure the "answer" field is an EXACT STRING MATCH to one of the 4 options.
        CRITICAL: The "answer" must match one option EXACTLY.
        CRITICAL: Ensure answers are correct to the best of your ability, as they will be banked if unique.
        The explanation may contain the exact text snippet from the source that supports the correct answer.(if possible) and must mention the SOURCE ID it came from.
        CRITICAL: The explanation should contain the exact text snippet from the source that supports the correct answer.
        Torture the candidate with the most difficult,tricky question and make it hard to guess the answer.
        The questions should be designed to test deep understanding and critical thinking, not just surface-level recall.
        The choice of statements you make for statement type questions should very well reflect the pyq's of relevant paper especially for UPSC.
        ENCOURAGED: You are encouraged to frame 1% questions on your own relevant to the source but from internet search.
    """

    try:
        response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
        )
        print(f"[{engine_name} ENGINE] Response received from AI.")
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            raw_data = json.loads(json_match.group(0))
        else:
            raw_data = json.loads(response.text)
        return raw_data.get("questions", [])
    except Exception as e:
        print(f"[{engine_name} ENGINE] ERROR: {e}")
        return []



def get_last_run_date():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                data = json.load(f)
                # Converts the saved string from JSON back into a date object
                return datetime.datetime.strptime(data['last_date'], "%d/%m/%Y").date()
        except Exception as e:
            print(f"Config error: {e}")
    
    # Fallback if no config exists: Parse the string we defined at the top
    return datetime.datetime.strptime(START_DATE_STR, "%d/%m/%Y").date()

def save_run_date(date_obj):
    with open(CONFIG_PATH, 'w') as f:
        json.dump({'last_date': date_obj.strftime("%d/%m/%Y")}, f)

def scrape_daily_sources(target_date):
    """Scrapes EVERYTHING from PIB (all releases) and Vajiram (reverse date URL)."""
    headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    combined_content = f"--- DATA FOR {target_date.strftime('%d/%m/%Y')} ---\n"
    session = requests.Session()

    # 1. VAJIRAM & RAVI (Format: .../current-affairs/upsc-prelims-current-affair/2026/5/27)
    vaj_url = f"https://vajiramandravi.com/current-affairs/upsc-prelims-current-affair/{target_date.year}/{target_date.month}/{target_date.day}"
    try:
        v_res = session.get(vaj_url, headers=headers, timeout=15)
        if v_res.status_code == 200:
            v_soup = BeautifulSoup(v_res.content, 'html.parser')
            # Scrape the main article content
            article = v_soup.find('article')
            if article:
                combined_content += f"\n[VAJIRAM CONTENT]\n{article.get_text(separator='\n', strip=True)}\n"
    except Exception as e:
        print(f"Vajiram Scrape Error: {e}")

    # 2. PIB (Archive page -> Follow every link)
    pib_archive = f"https://pib.gov.in/allRel.aspx?d={target_date.day}&m={target_date.month}&y={target_date.year}"
    try:
        res = session.get(pib_archive, headers=headers, timeout=15)
        soup = BeautifulSoup(res.content, 'html.parser')
        # Find every press release link
        links = soup.find_all('a', href=re.compile(r'PressReleasePage\.aspx'))
        
        for link in links:
            rel_url = "https://pib.gov.in/" + link['href']
            try:
                p_res = session.get(rel_url, headers=headers, timeout=10)
                p_soup = BeautifulSoup(p_res.content, 'html.parser')
                # Grab the main release text
                rel_body = p_soup.find('div', class_='ReleaseControl')
                if rel_body:
                    combined_content += f"\n\n--- PIB RELEASE: {link.get_text()} ---\n"
                    combined_content += rel_body.get_text(separator='\n', strip=True)
            except:
                continue
    except Exception as e:
        print(f"PIB Archive Error: {e}")

    return combined_content
def auto_update_db():
    """Checks the gap and saves PIB and Vajiram as SEPARATE selectable records."""
    last_run = get_last_run_date()
    # Test date: 28/05/2026 as per your request
    today = datetime.date(2026, 5, 28) 
    current_target = last_run + timedelta(days=1)
    
    headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    while current_target < today:
        date_label = current_target.strftime('%d-%m')
        
        # --- 1. SCRAPE VAJIRAM & RAVI ---
        try:
            vaj_url = f"https://vajiramandravi.com/current-affairs/upsc-prelims-current-affair/{current_target.year}/{current_target.month}/{current_target.day}"
            v_res = requests.get(vaj_url, headers=headers, timeout=10)
            if v_res.status_code == 200:
                v_soup = BeautifulSoup(v_res.content, 'html.parser')
                article = v_soup.find('article')
                if article:
                    with app.app_context():
                        # Saved as a separate record for independent checkbox selection
                        db.session.add(DailyData(source_id=f"vaj{date_label}", 
                                               content=f"VAJIRAM CONTENT: {article.get_text(strip=True)}"))
                        db.session.commit()
        except: pass

        # --- 2. SCRAPE PIB ---
        try:
            pib_url = f"https://pib.gov.in/allRel.aspx?d={current_target.day}&m={current_target.month}&y={current_target.year}"
            res = requests.get(pib_url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.content, 'html.parser')
            links = soup.find_all('a', href=re.compile(r'PressReleasePage\.aspx'))
            if links:
                # Combine all day's titles into one PIB entry
                pib_text = f"PIB RELEASES: " + " | ".join([l.get_text() for l in links])
                with app.app_context():
                    db.session.add(DailyData(source_id=f"pib{date_label}", content=pib_text))
                    db.session.commit()
        except: pass
            
        current_target += timedelta(days=1)
    
    # Save the run state
    with open(CONFIG_PATH, 'w') as f:
        json.dump({'last_date': today.strftime("%d/%m/%Y")}, f)
        
# --- ORCHESTRATOR ---

@app.route('/', methods=['GET', 'POST'])
def index():
    form = DataUploadForm()
    if form.validate_on_submit():
        content_text = ""
        
        # 1. URL Scraping with Browser Headers (Fixes the blocking issue)
        if form.url_link.data:
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                res = requests.get(form.url_link.data, headers=headers, timeout=10)
                if res.status_code == 200:
                    soup = BeautifulSoup(res.content, 'html.parser')
                    # Get text and clean up extra whitespace
                    content_text += ' '.join(soup.get_text().split())
                else:
                    flash(f"Website blocked access (Error {res.status_code})", "danger")
            except Exception as e:
                flash(f"URL Error: {str(e)}", "danger")

        # 2. PDF Extraction
        if form.pdf_file.data:
            try:
                reader = PdfReader(form.pdf_file.data)
                for page in reader.pages:
                    content_text += page.extract_text() + " "
            except Exception as e:
                flash(f"PDF Error: {str(e)}", "danger")

        # 3. Save to Database ONLY if we actually got text
        if content_text.strip():
            # Logic for Auto-ID with prefixes
            if form.category.data == 'Daily':
                count = DailyData.query.count() + 1
                new_data = DailyData(source_id=f"a{count}", content=content_text)
            elif form.category.data == 'Weekly':
                count = WeeklyData.query.count() + 1
                new_data = WeeklyData(source_id=f"c{count}", content=content_text)
            else:
                count = MonthlyData.query.count() + 1
                new_data = MonthlyData(source_id=f"m{count}", content=content_text)
            
            db.session.add(new_data)
            db.session.commit()
            flash(f"Successfully uploaded to {form.category.data}!", "success")
        else:
            flash("No text content could be extracted. Try a different source.", "warning")

        return redirect(url_for('index'))
    
    # Load all items for the template
    daily_items = DailyData.query.all()
    weekly_items = WeeklyData.query.all()
    monthly_items = MonthlyData.query.all()
    
    return render_template('index.html', upload_form=form, 
                           daily_list=daily_items, 
                           weekly_list=weekly_items, 
                           monthly_list=monthly_items)
import threading

def process_bank_update(new_results):
    print(f"[BACKGROUND] Audit starting for {len(new_results)} questions...")
    updated = False
    for q in new_results:
        # Check uniqueness
        if is_unique(q['question']):
            print(f"[BACKGROUND] Banking unique question...")
            # STORE EVERYTHING: embedding + full question details
            QUESTION_BANK[q['question']] = {
                "embedding": get_embedding(q['question']),
                "full_data": q  # This includes options, answer, and explanation
            }
            updated = True
    
    if updated:
        with bank_lock:
            save_question_bank()

def clean_text_for_pdf(text):
    """Replaces Unicode characters with PDF-safe equivalents."""
    replacements = {
        '\u2013': '-', # en-dash
        '\u2014': '-', # em-dash
        '\u2018': "'", # left single quote
        '\u2019': "'", # right single quote
        '\u201c': '"', # left double quote
        '\u201d': '"', # right double quote
        '\u2022': '*', # bullet point
    }
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    
    # Final fallback: encode to latin-1 and ignore characters that still won't fit
    return text.encode('latin-1', 'replace').decode('latin-1')

@app.route('/download_bank')
def download_bank():
    pdf = FPDF()
    pdf.add_page()
    
    # Header logic with Timestamp
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Question Bank", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, f"Generated on: {timestamp_str}", ln=True, align='C')
    pdf.ln(10)

    # Loop through the Bank
    for i, (q_text, data) in enumerate(QUESTION_BANK.items(), 1):
        q_item = data['full_data']
        
        # 1. Question (Bold)
        pdf.set_font("Arial", 'B', 11)
        pdf.multi_cell(0, 8, clean_text_for_pdf(f"Q{i}. {q_item['question']}"))
        
        # 2. Options (Normal)
        pdf.set_font("Arial", size=10)
        for opt in q_item.get('options', []):
            pdf.multi_cell(0, 7, clean_text_for_pdf(f"   {opt}"))
        
        # 3. Answer & Explanation (Italic/Bold)
        pdf.set_font("Arial", 'B', 10)
        pdf.multi_cell(0, 7, clean_text_for_pdf(f"Correct Answer: {q_item.get('answer', 'N/A')}"))
        pdf.set_font("Arial", 'I', 10)
        pdf.multi_cell(0, 7, clean_text_for_pdf(f"Explanation: {q_item.get('explanation', 'N/A')}"))
        pdf.ln(5)

    pdf_output = pdf.output(dest='S').encode('latin-1') 
    buffer = io.BytesIO(pdf_output)
    buffer.seek(0)
    
    # Filename with timestamp
    # Use the 'now' variable defined at the top of the function
    file_name = f"Question_Bank_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(buffer, as_attachment=True, download_name=file_name, mimetype='application/pdf')

@app.route('/generate_quiz', methods=['POST'])
def generate_quiz():
    data = request.json
    n = int(data.get('n', 10))
    exam_type = data.get('exam_type', 'UPSC')

    selected_ids = data.get('selected_ids', []) # Get list of IDs user checked

    # Filter database based on checkboxes. If none selected, default to all.
    if selected_ids:
        db_daily = {r.source_id: r.content for r in DailyData.query.filter(DailyData.source_id.in_(selected_ids)).all()}
        db_weekly = {r.source_id: r.content for r in WeeklyData.query.filter(WeeklyData.source_id.in_(selected_ids)).all()}
        db_monthly = {r.source_id: r.content for r in MonthlyData.query.filter(MonthlyData.source_id.in_(selected_ids)).all()}
    else:
        db_daily = {r.source_id: r.content for r in DailyData.query.all()}
        db_weekly = {r.source_id: r.content for r in WeeklyData.query.all()}
        db_monthly = {r.source_id: r.content for r in MonthlyData.query.all()}

    # ... (Keep existing engine distribution logic below) ...
    # results = []
    # with ThreadPoolExecutor...
    print(f"\n[ORCHESTRATOR] Starting Quiz Generation for n={n}")
    
    # 2:3:5 Distribution Logic
    n_daily = (2 * n) // 10
    n_weekly = (3 * n) // 10
    n_monthly = n - n_daily - n_weekly
    
    # 1. RUN ENGINES IN PARALLEL
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Key 1 -> Daily (DB 1X), Key 2 -> Weekly (DB 2X), Key 3 -> Monthly (DB 3X)
        f1 = executor.submit(run_engine, GEMINI_KEYS[0], "DAILY", db_daily, n_daily, exam_type)
        f2 = executor.submit(run_engine, GEMINI_KEYS[1], "WEEKLY", db_weekly, n_weekly, exam_type)
        f3 = executor.submit(run_engine, GEMINI_KEYS[2], "MONTHLY", db_monthly, n_monthly, exam_type)
        
        results.extend(f1.result())
        results.extend(f2.result())
        results.extend(f3.result())

    print(f"[ORCHESTRATOR] Total results received from Gemini: {len(results)}")

    # 2. FINAL JUMBLE (BEFORE returning to user)
    random.shuffle(results)
    
    # 3. START BACKGROUND BANK UPDATE
    # We trigger the bank processing but DO NOT 'join' the thread.
    # This allows the return statement to execute immediately.
    print("[ORCHESTRATOR] Starting background bank audit...")
    background_thread = threading.Thread(target=process_bank_update, args=(results,))
    background_thread.start()

    # 4. JSONIFY & RETURN TO USER (IMMEDIATE)
    print(f"[ORCHESTRATOR] Returning {len(results)} questions to user.")
    return jsonify(results)


import subprocess

@app.route('/shutdown', methods=['POST'])
def shutdown():
    print("[SYSTEM] Score 50%+ verified. Terminating Kiosk...")
    # Kill Edge browser
    subprocess.run(["taskkill", "/F", "/IM", "msedge.exe", "/T"], capture_output=True)
    # Kill the Flask server (this also closes the minimized CMD window from your .bat)
    os._exit(0)

def clean_duplicates():
    if not os.path.exists(BANK_PATH):
        print("File not found.")
        return

    with open(BANK_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    original_count = len(data)
    cleaned_bank = {}
    
    print(f"Starting cleanup of {original_count} items...")

    for q_text, q_info in data.items():
        # 1. String Normalization Check
        is_duplicate = False
        norm_new = " ".join(q_text.lower().split())
        
        for existing_text in cleaned_bank.keys():
            norm_existing = " ".join(existing_text.lower().split())
            if norm_new == norm_existing:
                is_duplicate = True
                break
        
        if is_duplicate:
            continue

        # 2. Semantic Similarity Check (Optional but thorough)
        # Convert embedding to numpy if it exists
        if 'embedding' in q_info:
            new_vec = np.array(q_info['embedding']).reshape(1, -1)
            for _, existing_info in cleaned_bank.items():
                if 'embedding' in existing_info:
                    old_vec = np.array(existing_info['embedding']).reshape(1, -1)
                    similarity = cosine_similarity(new_vec, old_vec)[0][0]
                    if similarity > 0.90: # Very strict threshold for cleaning
                        is_duplicate = True
                        break
        
        if not is_duplicate:
            cleaned_bank[q_text] = q_info

    # Save the cleaned data back
    with open(BANK_PATH, 'w', encoding='utf-8') as f:
        json.dump(cleaned_bank, f, indent=4)

    print(f"Cleanup complete!")
    print(f"Removed: {original_count - len(cleaned_bank)} duplicates.")
    print(f"Remaining: {len(cleaned_bank)} unique questions.")


if __name__ == '__main__':
    # This runs the gap-check every time you start the .py file
    threading.Thread(target=auto_update_db).start()
    
    app.run(debug=False, port=5000)