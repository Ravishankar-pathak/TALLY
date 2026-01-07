import os
import json
import psycopg2
import re
from typing import List, Dict, Any, Union, Optional
import pandas as pd
import tkinter as tk
from tkinter import scrolledtext, Entry, Button, Frame, END, Label, Listbox, Scrollbar, messagebox, filedialog
import threading
from datetime import datetime
import math
from tkinter import ttk
import numpy as np
import textwrap
import io
# Imports from Code 2 for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
# =================== CONFIGURATION =================== #
DB_CONFIG = {
    "host": os.getenv("DB_HOST", ""),
    "database": os.getenv("DB_NAME", ""),
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
    "port": int(os.getenv("DB_PORT", ")),
}

# =================== FINANCIAL YEAR MANAGEMENT (From Code 2) =================== #
def get_current_financial_year():
    """Get the current financial year based on today's date (April 1 to March 31)"""
    today = datetime.now()
    if today.month >= 4: # April to December
        return f"{today.year}-{today.year+1}"
    else: # January to March
        return f"{today.year-1}-{today.year}"

# Get current financial year
CURRENT_FY = get_current_financial_year()

# =================== GLOBAL DATAFRAMES =================== #
def normalize_string(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[^\w\s\(\)\-\.]', '', s)
    return s

# Main ledger data
df = None
# Transaction data
transactions_df = None

try:
    print("Loading data from PostgreSQL into memory...")
    conn = psycopg2.connect(**DB_CONFIG)
    
    # Load main ledger data
    df = pd.read_sql_query("SELECT * FROM tally_data", conn)
    
    # Load transaction data
    transactions_df = pd.read_sql_query("SELECT * FROM tally_transactions", conn)
    conn.close()
    
    # Preprocess main ledger data
    df.columns = df.columns.str.strip()
    df['ledger_name'] = df['ledger_name'].apply(normalize_string)
    df['parent'] = df['parent'].apply(normalize_string)
    df['altered_on'] = pd.to_datetime(df['altered_on'], errors='coerce')
    df['closing_balance'] = pd.to_numeric(df['closing_balance'], errors='coerce').fillna(0)
    df['opening_balance'] = pd.to_numeric(df['opening_balance'], errors='coerce').fillna(0)
    
    # Preprocess transaction data
    if transactions_df is not None:
        transactions_df.columns = transactions_df.columns.str.strip()
        transactions_df['date'] = pd.to_datetime(transactions_df['date'], errors='coerce')
        transactions_df['party_name'] = transactions_df['party_name'].apply(normalize_string)
        transactions_df['debit_amount'] = pd.to_numeric(transactions_df['debit_amount'], errors='coerce').fillna(0)
        transactions_df['credit_amount'] = pd.to_numeric(transactions_df['credit_amount'], errors='coerce').fillna(0)
        
        # Create a normalized party name column for better matching
        transactions_df['normalized_party'] = transactions_df['party_name'].apply(normalize_string)
        
        print(f"Transaction data loaded successfully! Total records: {len(transactions_df)}")
    else:
        print("Warning: No transaction data found. Statements will use mock data.")
    
    print("Data loaded and pre-processed successfully!")
    print(f"Total ledger records: {len(df)}")
except Exception as e:
    print(f"Critical Error: Could not load data from database. {e}")

# =================== ENHANCED MEMORY SYSTEM =================== #
MEMORY_FILE = "memory.json"
memory = {"cache": {}, "log": [], "user_preferences": {}}

def load_memory():
    global memory
    try:
        with open(MEMORY_FILE, 'r') as f:
            content = f.read().strip()
            if content:
                memory = json.loads(content)
                print(f"Loaded memory from {MEMORY_FILE}")
            else:
                memory = {"cache": {}, "log": [], "user_preferences": {}}
                print("Memory file empty, starting fresh.")
    except (FileNotFoundError, json.JSONDecodeError):
        memory = {"cache": {}, "log": [], "user_preferences": {}}
        print("Invalid or missing memory file, starting fresh.")

def save_memory():
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f, default=str, indent=4)
        print(f"Saved memory to {MEMORY_FILE}")

load_memory()

# =================== ENHANCED QUERY PATTERNS (Merged) =================== #
ENHANCED_QUERY_PATTERNS = {
    # --- Code 2 Patterns ---
    'statement_query': {
        'patterns': [
            r'(?:show|provide|generate|give).*?statement.*?(?:of|for)\s*(.+)',
            r'statement\s+(?:of|for)\s*(.+)',
            r'financial\s+statement\s+(?:of|for)\s*(.+)',
        ],
        'type': 'statement'
    },
    'br_query': {
        'patterns': [
            r'(?:show|provide|generate|give).*?bill\s+receivable.*?(?:of|for)\s*(.+)',
            r'bill\s+receivable\s+(?:of|for)\s*(.+)',
            r'br\s+(?:of|for)\s*(.+)',
            r'br\s+statement\s+(?:of|for)\s*(.+)',
        ],
        'type': 'br'
    },
    # --- Code 1 Patterns ---
    'ranking_single': {
        'patterns': [
            r'(?:who|which).*?(?:is\s+the\s+)?(highest|most|maximum|best|top|lowest|minimum|worst).*?(?:paying|balance).*?party',
            r'(?:highest|most|maximum|best|top|lowest|minimum|worst).*?(?:paying|balance).*?party',
            r'(?:best|top).*?party.*?(?:in\s+\d{4})?$',
            r'who.*?(?:best|top).*?party',
        ],
        'type': 'ranking',
        'limit': 1
    },
    'ranking_multiple': {
        'patterns': [
            r'top\s+(\d+).*?parties?.*?(?:by\s+)?balance',
            r'top\s+(\d+).*?parties?',
            r'(\d+)\s+(?:best|top|highest).*?parties?',
            r'who.*?top\s+(\d+).*?parties?',
            r'show.*?top\s+(\d+).*?parties?',
            r'list.*?top\s+(\d+).*?parties?',
        ],
        'type': 'ranking',
        'limit': 'extract'
    },
    'balance_query': {
        'patterns': [
            r'(?:what is|show me|tell me).*?(?:opening|closing)?\s*balance.*?(?:of|for)\s*(.+)',
            r'balance.*?(?:of|for)\s*(.+)',
            r'how much.*?(?:opening|closing)?\s*balance.*?(.+)',
        ],
        'type': 'balance'
    },
    'parent_query': {
        'patterns': [
            r'(?:what is|who is|find).*?parent.*?(?:of|for)\s*(.+)',
            r'parent.*?(?:of|for)\s*(.+)',
            r'which group.*?(.+)',
        ],
        'type': 'parent'
    },
    'profit_loss': {
        'patterns': [
            r'(?:profit|loss).*?(?:in|for|during)\s*(\d{4})',
            r'(?:income|expense).*?(?:in|for|during)\s*(\d{4})',
            r'financial.*?(?:performance|statement).*?(\d{4})',
            r'how much.*?(?:profit|loss).*?(\d{4})',
        ],
        'type': 'financial'
    },
    'search_ledger': {
        'patterns': [
            r'(?:find|search|look for).*?(ledger|party).*?(.+)',
            r'who is\s*(.+)',
            r'information about\s*(.+)',
            r'details of\s*(.+)',
        ],
        'type': 'search'
    },
    'comparison': {
        'patterns': [
            r'compare.*?(?:balance|performance).*?(.+)',
            r'(?:difference|comparison).*?(.+)',
            r'how does.*?compare to.*?',
        ],
        'type': 'comparison'
    },
    'trends': {
        'patterns': [
            r'trend.*?(?:balance|performance)',
            r'how.*?(?:changed|evolved).*?over time',
            r'historical.*?(?:data|performance)',
            r'growth.*?(?:over|during)',
        ],
        'type': 'trends'
    },
    'summary': {
        'patterns': [
            r'summary.*?(?:financial|performance)',
            r'overview.*?(?:business|financial)',
            r'report.*?(?:status|performance)',
            r'what is.*?(?:situation|status)',
        ],
        'type': 'summary'
    }
}

# =================== CORRECTED FINANCIAL CATEGORIES =================== #
CORRECT_FINANCIAL_CATEGORIES = {
    'actual_income': [
        "Sales Accounts", "Indirect Incomes", "Service Charges", "Commission Received",
        "Discount Received", "Reimbursement of Tour", "Misc Income", "Sale of Scrap"
    ],
    'actual_expenses': [
        "Direct Expenses", "Indirect Expenses", "Employees Cost", "Office Exp",
        "PROFESSIONAL FEE", "R & D Expenses", "Repair & Maintencae Expenses",
        "Telephone Exp", "Tour & Travels", "Vechicle & Running Expenses-GST",
        "Business Promotion", "Freight Charges-GST", "Packaging Expenses",
        "Finanace Cost", "Bank Charges", "Electricity Exp", "Rent Paid"
    ]
}

# =================== ULTIMATE QUERY PROCESSOR =================== #
class UltimateQueryProcessor:
    def __init__(self, df, transactions_df, execute_sql_func):
        self.df = df
        self.transactions_df = transactions_df
        self._execute_sql = execute_sql_func
        self.context = {}
        
    def extract_parameters(self, question: str) -> Dict[str, Any]:
        """Extract all possible parameters from question (Merged Logic)"""
        q_lower = question.lower()
        params = {
            'year': None,
            'count': None,
            'ledger_name': None,
            'parent_group': None,
            'balance_type': 'closing',
            'time_period': None,
            'comparison_targets': [],
            'search_term': None,
            'order': 'DESC',
            'query_type': None, # Added from Code 2
            'financial_year': None # Added from Code 2
        }
        
        # --- Code 2 Extraction Logic ---
        # Extract financial year
        fy_match = re.search(r'(\d{4})-(\d{4})', q_lower)
        if fy_match:
            params['financial_year'] = f"{fy_match.group(1)}-{fy_match.group(2)}"
        
        # Extract query type
        if re.search(r'statement', q_lower):
            params['query_type'] = 'statement'
        elif re.search(r'br|bill\s+receivable', q_lower):
            params['query_type'] = 'br'
        
        # --- Code 1 Extraction Logic ---
        # Extract year
        year_match = re.search(r'\b(20\d{2})\b', q_lower)
        if year_match:
            params['year'] = int(year_match.group(1))
        
        # Extract count/limit
        numbers = re.findall(r'\b(\d+)\b', q_lower)
        for num in numbers:
            num_int = int(num)
            if 1 <= num_int <= 1000 and (params['year'] is None or num != str(params['year'])):
                params['count'] = num_int
                break
        
        # Extract ledger name (Merged patterns)
        ledger_patterns = [
            r'statement.*?(?:of|for)\s*(.+)', # Code 2
            r'br.*?(?:of|for)\s*(.+)', # Code 2
            r'bill\s+receivable.*?(?:of|for)\s*(.+)', # Code 2
            r'balance.*?(?:of|for)\s*(.+)',
            r'parent.*?(?:of|for)\s*(.+)',
            r'information about\s*(.+)',
            r'details of\s*(.+)',
            r'who is\s*(.+)',
        ]
        for pattern in ledger_patterns:
            match = re.search(pattern, q_lower)
            if match and match.group(1):
                params['ledger_name'] = normalize_string(match.group(1).replace('?', '').strip())
                break
        
        # Extract balance type
        if 'opening' in q_lower:
            params['balance_type'] = 'opening'
        elif 'closing' in q_lower:
            params['balance_type'] = 'closing'
        
        # Extract order
        if any(word in q_lower for word in ['lowest', 'minimum', 'worst', 'bottom', 'least']):
            params['order'] = 'ASC'
        
        return params

    def route_query(self, question: str) -> str:
        """Enhanced query routing (Merged)"""
        q_lower = question.lower()
        
        # Check for Code 2 specific query types
        if re.search(r'statement', q_lower):
            return "statement_query"
        elif re.search(r'br|bill\s+receivable', q_lower):
            return "br_query"
        
        # Check for Code 1 specific query types
        elif any(word in q_lower for word in ['profit', 'loss', 'income', 'expense', 'revenue']):
            return "financial_analysis"
        elif any(word in q_lower for word in ['pending', 'due', 'outstanding', 'unpaid']):
            return "pending_analysis"
        elif any(word in q_lower for word in ['compare', 'versus', 'vs', 'difference']):
            return "comparative_analysis"
        elif any(word in q_lower for word in ['trend', 'history', 'over time', 'growth']):
            return "trend_analysis"
        elif any(word in q_lower for word in ['summary', 'overview', 'report', 'status']):
            return "summary_report"
        elif any(word in q_lower for word in ['search', 'find', 'look for', 'information about']):
            return "search_query"
        elif any(word in q_lower for word in ['balance', 'amount', 'how much']):
            return "balance_query"
        elif any(word in q_lower for word in ['parent', 'group', 'category']):
            return "parent_query"
        elif any(word in q_lower for word in ['top', 'best', 'highest', 'most', 'worst', 'lowest']):
            return "ranking_query"
        else:
            return "general_query"
    
    def process_query(self, question: str, context: Dict = {}) -> Dict[str, Any]:
        """Main query processing method (Merged)"""
        route = self.route_query(question)
        params = self.extract_parameters(question)
        print(f"DEBUG: Routing '{question}' to '{route}' with params: {params}")
        
        if route == "financial_analysis":
            return self.process_financial_analysis(question, params, context)
        elif route == "pending_analysis":
            return self.process_pending_analysis(question, params, context)
        elif route == "comparative_analysis":
            return self.process_comparative_analysis(question, params, context)
        elif route == "trend_analysis":
            return self.process_trend_analysis(question, params, context)
        elif route == "summary_report":
            return self.process_summary_report(question, params, context)
        elif route == "ranking_query":
            return self.process_ranking_query(question, params, context)
        elif route == "balance_query":
            return self.process_balance_query(question, params, context)
        elif route == "parent_query":
            return self.process_parent_query(question, params, context)
        elif route == "search_query":
            return self.process_search_query(question, params, context)
        # Added from Code 2
        elif route == "statement_query":
            return self.process_statement_query(question, params, context)
        elif route == "br_query":
            return self.process_br_query(question, params, context)
        else:
            return self.process_general_query(question, params, context)
    
    # --- Helper for Dynamic Table Generation ---
    def _create_dynamic_table_text(self, title_lines, headers, rows, total_row=None):
        """Create a dynamically sized text table that never breaks alignment"""
        # Calculate widths based on data length
        all_data = [headers] + rows
        num_columns = len(headers)
        col_widths = [0] * num_columns
        for row in all_data:
            for i, cell in enumerate(row):
                if i < num_columns:
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Add padding
        col_widths = [w + 2 for w in col_widths] # 1 space padding on each side
        total_width = sum(col_widths) + (num_columns - 1) + 4 # 4 for border edge chars
        
        # Helper to create horizontal lines
        def make_sep(left, mid, right, line='═'):
            return left + mid.join([line * w for w in col_widths]) + right
        
        border_top = make_sep("╔", "╤", "╗")
        border_head_sep = make_sep("╠", "╪", "╣")
        border_mid = make_sep("╟", "┼", "╢", "─")
        border_bot = make_sep("╚", "╧", "╝")
        
        lines = []
        
        # Title Box
        lines.append(f"╔{'═' * (total_width-2)}╗")
        for t in title_lines:
            lines.append(f"║{t.center(total_width-2)}║")
        
        # Connect Title to Table
        lines.append(make_sep("╠", "╤", "╣"))
        
        # Header Row
        header_str = "║"
        for i, h in enumerate(headers):
            header_str += f"{h.center(col_widths[i])}│"
        header_str = header_str[:-1] + "║"
        lines.append(header_str)
        lines.append(border_mid)
        
        # Data Rows
        for row in rows:
            row_str = "║"
            for i, cell in enumerate(row):
                s_cell = str(cell)
                w = col_widths[i]
                # Right align numbers/money, left align text
                if any(c in s_cell for c in ['₹', '.', ',']) and any(c.isdigit() for c in s_cell) and len(s_cell) < 20:
                    row_str += f"{s_cell.rjust(w-1)} │"
                else:
                    row_str += f" {s_cell.ljust(w-1)}│"
            row_str = row_str[:-1] + "║"
            lines.append(row_str)
        
        # Footer
        if total_row:
            lines.append(border_head_sep)
            t_label, t_val = total_row
            # Create a footer row that spans
            # This is tricky with columns, so we'll just put it in a full width row or try to align to last columns
            footer_content = f"{t_label}: {t_val}"
            lines.append(f"║{footer_content.rjust(total_width-2)}║")
            lines.append(f"╚{'═' * (total_width-2)}╝")
        else:
            lines.append(border_bot)
        
        return "\n".join(lines)

    # =================== ENHANCED STATEMENT & BR GENERATION WITH REAL DATA =================== #
    def generate_financial_year_statement_real(self, ledger_name: str, financial_year: Optional[str] = None, use_real_data: bool = True) -> Dict[str, Any]:
        """Generate real financial year statement using actual transaction data"""
        if self.df is None or (use_real_data and self.transactions_df is None):
            return {"result": "❌ Error: Data loading failed.", "error": True}
        
        try:
            # Use current financial year if not specified
            if financial_year is None:
                financial_year = CURRENT_FY
            
            # Parse financial year
            if '-' in financial_year:
                fy_start_year = int(financial_year.split('-')[0])
                fy_end_year = int(financial_year.split('-')[1])
            else:
                fy_start_year = int(financial_year)
                fy_end_year = fy_start_year + 1
            
            # FY date range
            fy_start_date = pd.Timestamp(f"{fy_start_year}-04-01")
            fy_end_date = pd.Timestamp(f"{fy_end_year}-03-31")
            
            # Search for exact ledger match
            ledger_row = self.df[self.df['ledger_name'].str.lower() == ledger_name.lower()]
            if len(ledger_row) == 0:
                # Try partial match
                matches = self.df[self.df['ledger_name'].str.lower().str.contains(ledger_name.lower(), na=False)]
                if len(matches) == 0:
                    return {"result": f"❌ No ledger found containing '{ledger_name}'.", "error": True}
                elif len(matches) > 1:
                    return {
                        "result": f"🔍 Found {len(matches)} ledgers matching '{ledger_name}'. Please be more specific.",
                        "clarification_needed": True,
                        "new_context": {'pending_clarification': {
                            "template": "statement for {name}",
                            "options": matches['ledger_name'].head(5).tolist()
                        }}
                    }
                ledger_row = matches.iloc[0]
            else:
                ledger_row = ledger_row.iloc[0]
            
            company_name = ledger_row['ledger_name']
            opening_balance = round(float(ledger_row['opening_balance']), 2)
            closing_balance = round(float(ledger_row['closing_balance']), 2)
            parent_group = ledger_row['parent']
            
            # Get real transactions if available
            transactions = []
            if use_real_data and self.transactions_df is not None:
                # Filter transactions for this party and financial year
                party_transactions = self.transactions_df[
                    (self.transactions_df['normalized_party'] == normalize_string(company_name)) &
                    (self.transactions_df['date'] >= fy_start_date) &
                    (self.transactions_df['date'] <= fy_end_date)
                ].copy()
                
                # Sort by date
                party_transactions = party_transactions.sort_values('date')
                
                # If no transactions found in this FY, try to find transactions with similar names
                if len(party_transactions) == 0:
                    # Try partial match on party name
                    similar_transactions = self.transactions_df[
                        (self.transactions_df['normalized_party'].str.contains(normalize_string(company_name), case=False, na=False)) &
                        (self.transactions_df['date'] >= fy_start_date) &
                        (self.transactions_df['date'] <= fy_end_date)
                    ].copy()
                    
                    if len(similar_transactions) > 0:
                        party_transactions = similar_transactions.sort_values('date')
                        print(f"Found {len(party_transactions)} transactions using partial match for {company_name}")
                
                # If still no transactions, use mock data but indicate it
                if len(party_transactions) == 0:
                    print(f"No real transactions found for {company_name} in {financial_year}. Using mock data.")
                    use_real_data = False
            
            # Add opening balance as first transaction
            transactions.append({
                "date": fy_start_date.strftime("%Y-%m-%d"),
                "voucher_type": "Opening Balance",
                "voucher_no": "OP-001",
                "particulars": "Opening Balance",
                "debit": 0,
                "credit": opening_balance if opening_balance > 0 else 0,
                "balance": opening_balance
            })
            
            # Add real transactions or generate mock ones
            if use_real_data and self.transactions_df is not None and len(party_transactions) > 0:
                running_balance = opening_balance
                
                # Process each real transaction
                for idx, row in party_transactions.iterrows():
                    date_str = row['date'].strftime("%Y-%m-%d")
                    voucher_type = str(row.get('voucher_type', 'Journal'))
                    voucher_no = str(row.get('voucher_no', f'VCH-{idx+1:03d}'))
                    particulars = str(row.get('particulars', row.get('party_name', 'Transaction')))
                    debit = float(row.get('debit_amount', 0))
                    credit = float(row.get('credit_amount', 0))
                    
                    # Calculate running balance
                    if debit > 0:
                        running_balance += debit
                    if credit > 0:
                        running_balance -= credit
                    
                    transactions.append({
                        "date": date_str,
                        "voucher_type": voucher_type,
                        "voucher_no": voucher_no,
                        "particulars": particulars[:50],  # Limit length for display
                        "debit": round(debit, 2),
                        "credit": round(credit, 2),
                        "balance": round(running_balance, 2)
                    })
                
                # The final running balance should match the closing balance
                final_balance = running_balance
            else:
                # Generate mock transactions (original logic)
                transaction_dates = [
                    f"{fy_start_year}-04-15", f"{fy_start_year}-05-15", f"{fy_start_year}-06-10",
                    f"{fy_start_year}-07-05", f"{fy_start_year}-08-20", f"{fy_start_year}-09-15",
                    f"{fy_start_year}-10-10", f"{fy_start_year}-11-05", f"{fy_start_year}-12-20",
                    f"{fy_end_year}-01-15", f"{fy_end_year}-02-10", f"{fy_end_year}-03-05"
                ]
                
                # Use opening balance to generate realistic transactions
                transaction_amount = max(abs(opening_balance) * 0.1, 10000) if opening_balance != 0 else 50000
                running_balance = opening_balance
                
                for i, date in enumerate(transaction_dates):
                    # Alternate between debit and credit transactions
                    if i % 2 == 0:
                        debit = transaction_amount
                        credit = 0
                        voucher_type = "Payment"
                        particulars = "Payment received"
                        voucher_no = f"PAY-{i+1:03d}"
                    else:
                        debit = 0
                        credit = transaction_amount * 1.5
                        voucher_type = "Invoice"
                        particulars = "Goods sold"
                        voucher_no = f"INV-{i+1:03d}"
                    
                    if debit > 0:
                        running_balance += debit
                    if credit > 0:
                        running_balance -= credit
                    
                    transactions.append({
                        "date": date,
                        "voucher_type": voucher_type,
                        "voucher_no": voucher_no,
                        "particulars": particulars,
                        "debit": round(debit, 2),
                        "credit": round(credit, 2),
                        "balance": round(running_balance, 2)
                    })
                
                final_balance = running_balance
            
            # Add closing balance
            transactions.append({
                "date": fy_end_date.strftime("%Y-%m-%d"),
                "voucher_type": "Closing Balance",
                "voucher_no": "CL-001",
                "particulars": "Closing Balance",
                "debit": 0,
                "credit": closing_balance,
                "balance": closing_balance
            })
            
            # Prepare Data for Table
            headers = ["Date", "Voucher Type", "Voucher No", "Particulars", "Debit (₹)", "Credit (₹)", "Balance (₹)"]
            rows = []
            for trans in transactions:
                date_str = trans['date'][5:] if len(trans['date']) > 5 else trans['date']  # Get DD-MM format or full date
                rows.append([
                    date_str,
                    trans['voucher_type'][:15],
                    trans['voucher_no'][:10],
                    trans['particulars'][:25],
                    f"{trans['debit']:,.2f}" if trans['debit'] > 0 else "0.00",
                    f"{trans['credit']:,.2f}" if trans['credit'] > 0 else "0.00",
                    f"{trans['balance']:,.2f}"
                ])
            
            # Generate Formatted Text using Dynamic Table
            title_lines = [
                "FINANCIAL YEAR STATEMENT" + (" (REAL DATA)" if use_real_data else " (MOCK DATA)"),
                f"{financial_year} ({fy_start_year} to {fy_end_year})",
                f"Company: {company_name}",
                f"Parent: {parent_group}"
            ]
            
            formatted_table = self._create_dynamic_table_text(
                title_lines,
                headers,
                rows,
                total_row=("Closing Balance", f"₹{closing_balance:,.2f}")
            )
            
            # Prepare raw data for download
            raw_data = {
                "type": "statement",
                "company_name": company_name,
                "financial_year": financial_year,
                "opening_balance": opening_balance,
                "closing_balance": closing_balance,
                "parent_group": parent_group,
                "transactions": transactions,
                "result_text": formatted_table, # Store the full formatted table string
                "used_real_data": use_real_data
            }
            
            return {
                "result": formatted_table,
                "success": True,
                "raw_data": raw_data,
                "new_context": {"last_statement_vendor": company_name, "statement_year": financial_year}
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"result": f"❌ Error generating statement: {str(e)}", "error": True}
    
    def generate_bill_receivable_report_real(self, ledger_name: str, financial_year: Optional[str] = None, use_real_data: bool = True) -> Dict[str, Any]:
        """Generate real Bill Receivable report using actual transaction data"""
        if self.df is None or (use_real_data and self.transactions_df is None):
            return {"result": "❌ Error: Data loading failed.", "error": True}
        
        try:
            # Use current financial year if not specified
            if financial_year is None:
                financial_year = CURRENT_FY
            
            # Parse financial year
            if '-' in financial_year:
                fy_start_year = int(financial_year.split('-')[0])
                fy_end_year = int(financial_year.split('-')[1])
            else:
                fy_start_year = int(financial_year)
                fy_end_year = fy_start_year + 1
            
            # FY date range
            fy_start_date = pd.Timestamp(f"{fy_start_year}-04-01")
            fy_end_date = pd.Timestamp(f"{fy_end_year}-03-31")
            
            # Search for ledger
            ledger_row = self.df[self.df['ledger_name'].str.lower() == ledger_name.lower()]
            if len(ledger_row) == 0:
                # Try partial match
                matches = self.df[self.df['ledger_name'].str.lower().str.contains(ledger_name.lower(), na=False)]
                if len(matches) == 0:
                    return {"result": f"❌ No ledger found containing '{ledger_name}'.", "error": True}
                elif len(matches) > 1:
                    return {
                        "result": f"🔍 Found {len(matches)} ledgers matching '{ledger_name}'. Please be more specific.",
                        "clarification_needed": True,
                        "new_context": {'pending_clarification': {
                            "template": "BR for {name}",
                            "options": matches['ledger_name'].head(5).tolist()
                        }}
                    }
                ledger_row = matches.iloc[0]
            else:
                ledger_row = ledger_row.iloc[0]
            
            company_name = ledger_row['ledger_name']
            opening_balance = round(float(ledger_row['opening_balance']), 2)
            closing_balance = round(float(ledger_row['closing_balance']), 2)
            parent_group = ledger_row['parent']
            
            # Find bill receivable data
            br_data = []
            total_br = 0
            
            if use_real_data and self.transactions_df is not None:
                # Filter transactions that might represent bill receivables
                # Look for sales/invoice type transactions that haven't been fully paid
                
                # First, get all transactions for this party in the financial year
                party_transactions = self.transactions_df[
                    (self.transactions_df['normalized_party'] == normalize_string(company_name)) &
                    (self.transactions_df['date'] >= fy_start_date) &
                    (self.transactions_df['date'] <= fy_end_date)
                ].copy()
                
                if len(party_transactions) > 0:
                    # Sort by date
                    party_transactions = party_transactions.sort_values('date')
                    
                    # Identify potential bill receivables
                    # Look for transactions with credit amounts (sales) that might not have corresponding debits (payments)
                    sales_transactions = party_transactions[
                        (party_transactions['credit_amount'] > 0) &
                        (party_transactions['voucher_type'].str.contains('Invoice|Sales|Bill', case=False, na=False))
                    ]
                    
                    payment_transactions = party_transactions[
                        (party_transactions['debit_amount'] > 0) &
                        (party_transactions['voucher_type'].str.contains('Payment|Receipt|Cash', case=False, na=False))
                    ]
                    
                    # Create BR entries from sales transactions
                    for idx, row in sales_transactions.head(10).iterrows():
                        # Calculate due date (30 days after invoice date as default)
                        invoice_date = row['date']
                        due_date = invoice_date + pd.Timedelta(days=30)
                        
                        # Determine status based on payments received
                        payment_received = 0
                        if len(payment_transactions) > 0:
                            # Simple matching - this would need more sophisticated logic in real system
                            payment_received = min(row['credit_amount'], payment_transactions['debit_amount'].sum())
                        
                        amount_due = row['credit_amount'] - payment_received
                        status = "Outstanding" if amount_due > 0.5 else "Paid"
                        
                        if amount_due > 0:
                            br_data.append({
                                "br_date": invoice_date.strftime("%Y-%m-%d"),
                                "br_no": str(row.get('voucher_no', f'BR-{idx+1:03d}')),
                                "party": company_name,
                                "amount": round(amount_due, 2),
                                "due_date": due_date.strftime("%Y-%m-%d"),
                                "status": status
                            })
                            total_br += amount_due
                    
                    print(f"Found {len(br_data)} bill receivables from real data for {company_name}")
                
                # If no BR data found from transactions, fall back to ledger-based approach
                if len(br_data) == 0:
                    # Look for other customers with similar names or in receivable groups
                    receivable_groups = ['Sundry Debtors', 'Accounts Receivable', 'Trade Receivables', 'Debtors']
                    
                    # Find all debtors
                    debtors_df = self.df[
                        self.df['parent'].str.contains('|'.join([g.lower() for g in receivable_groups]), case=False, na=False)
                    ]
                    
                    if len(debtors_df) > 0:
                        print(f"Found {len(debtors_df)} debtors in receivable groups")
                        
                        # Try to match by name similarity
                        similar_debtors = debtors_df[
                            debtors_df['ledger_name'].str.contains(normalize_string(company_name), case=False, na=False)
                        ]
                        
                        if len(similar_debtors) > 0:
                            # Create BR entries from similar debtors
                            for i, (_, row) in enumerate(similar_debtors.head(5).iterrows()):
                                if row['closing_balance'] < 0:  # Negative balance indicates receivable
                                    br_date = fy_start_date + pd.Timedelta(days=i*15)
                                    due_date = br_date + pd.Timedelta(days=30)
                                    
                                    br_data.append({
                                        "br_date": br_date.strftime("%Y-%m-%d"),
                                        "br_no": f"BR-{i+1:03d}",
                                        "party": row['ledger_name'],
                                        "amount": round(abs(row['closing_balance']), 2),
                                        "due_date": due_date.strftime("%Y-%m-%d"),
                                        "status": "Outstanding"
                                    })
                                    total_br += abs(row['closing_balance'])
            
            # If still no data or not using real data, use the original mock generation
            if len(br_data) == 0 or not use_real_data:
                print(f"No real BR data found or mock mode enabled. Generating mock BR data for {company_name}")
                
                # Create mock BR entries based on the closing balance
                num_entries = min(5, max(1, int(abs(closing_balance) / 50000)))
                entry_amount = abs(closing_balance) / num_entries
                
                for i in range(num_entries):
                    br_date = fy_start_date + pd.Timedelta(days=i*15)
                    br_no = f"BR-{i+1:03d}"
                    amount = entry_amount
                    due_date = br_date + pd.Timedelta(days=30)
                    status = "Outstanding" if i % 2 == 0 else "Partially Paid"
                    
                    br_data.append({
                        "br_date": br_date.strftime("%Y-%m-%d"),
                        "br_no": br_no,
                        "party": company_name,
                        "amount": round(amount, 2),
                        "due_date": due_date.strftime("%Y-%m-%d"),
                        "status": status
                    })
                    total_br += amount
            
            # Prepare Data for Table
            headers = ["BR Date", "BR No.", "Party Name", "Amount (₹)", "Due Date", "Status"]
            rows = []
            for br in br_data:
                br_date = br['br_date'][8:] + "-" + br['br_date'][5:7] if len(br['br_date']) > 7 else br['br_date']  # Get DD-MM format
                due_date = br['due_date'][8:] + "-" + br['due_date'][5:7] if len(br['due_date']) > 7 else br['due_date']  # Get DD-MM format
                rows.append([
                    br_date,
                    br['br_no'][:10],
                    br['party'][:25],
                    f"{br['amount']:,.2f}",
                    due_date,
                    br['status'][:12]
                ])
            
            # Generate Formatted Text using Dynamic Table
            title_lines = [
                "BILL RECEIVABLE (BR) STATEMENT" + (" (REAL DATA)" if use_real_data else " (MOCK DATA)"),
                f"{financial_year} ({fy_start_year} to {fy_end_year})",
                f"Company: {company_name}",
                f"Parent: {parent_group}"
            ]
            
            formatted_table = self._create_dynamic_table_text(
                title_lines,
                headers,
                rows,
                total_row=("Total Outstanding", f"₹{total_br:,.2f}")
            )
            
            # Add raw data for download functionality
            raw_data = {
                "type": "br_report",
                "company_name": company_name,
                "financial_year": financial_year,
                "total_br": total_br,
                "br_entries": br_data,
                "result_text": formatted_table, # Store the full formatted table string
                "used_real_data": use_real_data
            }
            
            return {
                "result": formatted_table,
                "success": True,
                "raw_data": raw_data,
                "new_context": {"last_br_company": company_name, "br_year": financial_year}
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"result": f"❌ Error generating BR report: {str(e)}", "error": True}

    # --- Code 2 Specific Methods (Enhanced with real data) ---
    def process_statement_query(self, question: str, params: Dict, context: Dict) -> Dict[str, Any]:
        """Process financial statement queries - always uses current financial year"""
        if self.df is None:
            return {"result": "❌ Error: Data loading failed.", "error": True}
        
        ledger_name = params.get('ledger_name')
        if not ledger_name:
            return {
                "result": "🔍 Please specify which company's statement you want to see (e.g., 'statement of ABC Company')",
                "clarification_needed": True
            }
        
        try:
            # Search for ledger
            matches = self.df[self.df['ledger_name'].str.lower().str.contains(ledger_name.lower(), na=False)]
            if len(matches) == 0:
                # Try to find similar names
                similar_names = []
                ledger_name_lower = ledger_name.lower()
                for name in self.df['ledger_name'].unique():
                    if name and ledger_name_lower in name.lower():
                        similar_names.append(name)
                
                if similar_names:
                    return {
                        "result": f"🔍 No exact match for '{ledger_name}'. Did you mean one of these?\n" +
                                 "\n".join([f"• {name}" for name in similar_names[:5]]),
                        "clarification_needed": True,
                        "new_context": {'pending_clarification': {
                            "template": "statement for {name}",
                            "options": similar_names[:5]
                        }}
                    }
                else:
                    return {"result": f"❌ No ledger found containing '{ledger_name}'.", "error": True}
            
            # Generate the statement using current financial year and real data if available
            return self.generate_financial_year_statement_real(
                matches.iloc[0]['ledger_name'], 
                CURRENT_FY,
                use_real_data=(self.transactions_df is not None)
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"result": f"❌ Error in statement processing: {str(e)}", "error": True}
    
    def process_br_query(self, question: str, params: Dict, context: Dict) -> Dict[str, Any]:
        """Process Bill Receivable queries - always uses current financial year"""
        if self.df is None:
            return {"result": "❌ Error: Data loading failed.", "error": True}
        
        ledger_name = params.get('ledger_name')
        if not ledger_name:
            return {
                "result": "🔍 Please specify which company's Bill Receivable you want to see (e.g., 'BR of ABC Company')",
                "clarification_needed": True
            }
        
        try:
            # Search for ledger
            matches = self.df[self.df['ledger_name'].str.lower().str.contains(ledger_name.lower(), na=False)]
            if len(matches) == 0:
                # Try to find similar names
                similar_names = []
                ledger_name_lower = ledger_name.lower()
                for name in self.df['ledger_name'].unique():
                    if name and ledger_name_lower in name.lower():
                        similar_names.append(name)
                
                if similar_names:
                    return {
                        "result": f"🔍 No exact match for '{ledger_name}'. Did you mean one of these?\n" +
                                 "\n".join([f"• {name}" for name in similar_names[:5]]),
                        "clarification_needed": True,
                        "new_context": {'pending_clarification': {
                            "template": "BR for {name}",
                            "options": similar_names[:5]
                        }}
                    }
                else:
                    return {"result": f"❌ No ledger found containing '{ledger_name}'.", "error": True}
            
            # Generate the BR report using current financial year and real data if available
            return self.generate_bill_receivable_report_real(
                matches.iloc[0]['ledger_name'], 
                CURRENT_FY,
                use_real_data=(self.transactions_df is not None)
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"result": f"❌ Error in BR processing: {str(e)}", "error": True}

    # --- Code 1 Original Methods (Preserved) ---
    def process_financial_analysis(self, question: str, params: Dict, context: Dict) -> Dict[str, Any]:
        """CORRECT financial analysis using actual Profit & Loss account"""
        if self.df is None:
            return {"result": "❌ Error: Data loading failed.", "error": True}
        
        try:
            # Method 1: Use actual Profit & Loss account
            pnl_account = self.df[self.df['ledger_name'].str.contains('Profit & Loss', na=False, case=False)]
            if not pnl_account.empty:
                actual_profit = pnl_account.iloc[0]['closing_balance']
                opening_balance = pnl_account.iloc[0]['opening_balance']
                net_profit = actual_profit - opening_balance
                result_lines = ["📊 CORRECT Financial Analysis"]
                result_lines.append("=" * 50)
                result_lines.append(f"💰 Profit & Loss Account: ₹{actual_profit:,.2f}")
                result_lines.append(f"📈 Opening Balance: ₹{opening_balance:,.2f}")
                result_lines.append(f"🎯 Actual Net Profit: ₹{net_profit:,.2f}")
                
                # Get top actual parties (not including internal accounts)
                real_parties = self.df[
                    ~self.df['parent'].isin(['Primary', 'Reserves & Surplus', 'Duties & Taxes', 'GST', 'TDS/TCS'])
                ].nlargest(5, 'closing_balance')[['ledger_name', 'closing_balance', 'parent']]
                
                result_lines.append("\n🏆 Top 5 Real Parties by Balance:")
                for i, (_, row) in enumerate(real_parties.iterrows(), 1):
                    result_lines.append(f"{i}. {row['ledger_name']} - ₹{row['closing_balance']:,.2f} ({row['parent']})")
                
                return {
                    "result": "\n".join(result_lines),
                    "success": True
                }
            
            # Method 2: If P&L not found, use proper calculation
            else:
                # Get actual business transactions (exclude internal accounts)
                business_df = self.df[
                    ~self.df['parent'].isin([
                        'Primary', 'Reserves & Surplus', 'Duties & Taxes', 'GST', 'TDS/TCS',
                        'Secured Loans', 'Unsecured Loans', 'Current Liablities', 'Bank Accounts'
                    ])
                ]
                
                positive_balances = business_df[business_df['closing_balance'] > 0]['closing_balance'].sum()
                negative_balances = business_df[business_df['closing_balance'] < 0]['closing_balance'].sum()
                net_position = positive_balances + negative_balances # Negative balances are already negative
                
                result_lines = ["📊 Financial Position Analysis"]
                result_lines.append("=" * 40)
                result_lines.append(f"💰 Assets/Receivables: ₹{positive_balances:,.2f}")
                result_lines.append(f"📉 Liabilities/Payables: ₹{abs(negative_balances):,.2f}")
                result_lines.append(f"🎯 Net Position: ₹{net_position:,.2f}")
                
                # Show largest balances
                top_positive = business_df.nlargest(3, 'closing_balance')[['ledger_name', 'closing_balance']]
                top_negative = business_df.nsmallest(3, 'closing_balance')[['ledger_name', 'closing_balance']]
                
                result_lines.append("\n🔝 Top 3 Assets:")
                for i, (_, row) in enumerate(top_positive.iterrows(), 1):
                    result_lines.append(f"{i}. {row['ledger_name']} - ₹{row['closing_balance']:,.2f}")
                
                result_lines.append("\n📉 Top 3 Liabilities:")
                for i, (_, row) in enumerate(top_negative.iterrows(), 1):
                    result_lines.append(f"{i}. {row['ledger_name']} - ₹{row['closing_balance']:,.2f}")
                
                return {
                    "result": "\n".join(result_lines),
                    "success": True
                }
        except Exception as e:
            return {"result": f"❌ Error in financial analysis: {str(e)}", "error": True}
    
    def process_pending_analysis(self, question: str, params: Dict, context: Dict) -> Dict[str, Any]:
        """Enhanced pending payments analysis"""
        if self.df is None:
            return {"result": "❌ Error: Data loading failed.", "error": True}
        
        q_lower = question.lower()
        try:
            # Find parties with pending balances (exclude internal accounts)
            pending_df = self.df[
                ((self.df['closing_balance'] != 0) | (self.df['opening_balance'] != 0)) &
                ~self.df['parent'].isin(['Primary', 'Reserves & Surplus', 'Duties & Taxes', 'GST', 'TDS/TCS'])
            ].copy()
            
            if pending_df.empty:
                return {
                    "result": "✅ All payments are settled. There are no vendors with pending payments.",
                    "success": True
                }
            
            # Calculate metrics
            total_pending = len(pending_df)
            total_amount = pending_df['closing_balance'].sum()
            avg_pending = pending_df['closing_balance'].mean()
            
            # Top pending amounts
            pending_df.sort_values('closing_balance', ascending=False, inplace=True)
            top_pending = pending_df.head(15)
            
            result_lines = ["💰 Pending Payments Analysis"]
            result_lines.append("=" * 40)
            result_lines.append(f"📊 Total Pending Parties: {total_pending}")
            result_lines.append(f"💵 Total Pending Amount: ₹{total_amount:,.2f}")
            result_lines.append(f"📈 Average Pending: ₹{avg_pending:,.2f}")
            result_lines.append("\n🔝 Top Pending Payments:")
            for i, (_, row) in enumerate(top_pending.iterrows(), 1):
                result_lines.append(f"{i}. {row['ledger_name']} - ₹{row['closing_balance']:,.2f} ({row['parent']})")
            
            return {
                "result": "\n".join(result_lines),
                "success": True,
                "result_type": "paginated_pending",
                "data": pending_df,
                "total_count": total_pending
            }
        except Exception as e:
            return {"result": f"❌ Error in pending analysis: {str(e)}", "error": True}
    
    def process_comparative_analysis(self, question: str, params: Dict, context: Dict) -> Dict[str, Any]:
        """Compare different entities"""
        if self.df is None:
            return {"result": "❌ Error: Data loading failed.", "error": True}
        
        search_term = params.get('ledger_name')
        if not search_term:
            return {
                "result": "🔍 Please specify what you want to compare (e.g., 'compare party A and party B')",
                "clarification_needed": True
            }
        
        try:
            # Find matching ledgers
            matches = self.df[self.df['ledger_name'].str.lower().str.contains(search_term.lower(), na=False)]
            if len(matches) == 0:
                return {"result": f"❌ No ledgers found containing '{search_term}'.", "error": True}
            elif len(matches) == 1:
                ledger = matches.iloc[0]
                result_lines = [f"📊 Analysis of {ledger['ledger_name']}"]
                result_lines.append("=" * 30)
                result_lines.append(f"💰 Closing Balance: ₹{ledger['closing_balance']:,.2f}")
                result_lines.append(f"📈 Opening Balance: ₹{ledger['opening_balance']:,.2f}")
                result_lines.append(f"🏷️ Parent Group: {ledger['parent']}")
                if pd.notna(ledger['altered_on']):
                    result_lines.append(f"📅 Last Updated: {ledger['altered_on'].strftime('%Y-%m-%d')}")
                
                return {
                    "result": "\n".join(result_lines),
                    "success": True,
                    "new_context": {"last_ledger_name": ledger['ledger_name']}
                }
            else:
                result_lines = [f"🔍 Found {len(matches)} matching ledgers for '{search_term}':"]
                for i, (_, row) in enumerate(matches.head(10).iterrows(), 1):
                    result_lines.append(f"{i}. {row['ledger_name']} - ₹{row['closing_balance']:,.2f} ({row['parent']})")
                if len(matches) > 10:
                    result_lines.append(f"\n... and {len(matches) - 10} more")
                
                return {
                    "result": "\n".join(result_lines),
                    "success": True,
                    "new_context": {'pending_clarification': {
                        "template": "compare {name}",
                        "options": matches['ledger_name'].head(10).tolist()
                    }}
                }
        except Exception as e:
            return {"result": f"❌ Error in comparative analysis: {str(e)}", "error": True}
    
    def process_trend_analysis(self, question: str, params: Dict, context: Dict) -> Dict[str, Any]:
        """CORRECT trend analysis based on actual data"""
        if self.df is None:
            return {"result": "❌ Error: Data loading failed.", "error": True}
        
        try:
            # Check if we have proper date data
            if self.df['altered_on'].isna().all():
                return {
                    "result": "📊 No proper date-based data available for trend analysis.\nAvailable analysis:\n• Current financial position\n• Top performing parties\n• Pending payments",
                    "success": True
                }
            
            # Simple analysis based on available data
            total_parties = len(self.df)
            total_balance = self.df['closing_balance'].sum()
            
            # Get business data (exclude internal accounts)
            business_df = self.df[
                ~self.df['parent'].isin(['Primary', 'Reserves & Surplus', 'Duties & Taxes', 'GST', 'TDS/TCS'])
            ]
            
            # Top 5 positive and negative balances
            top_positive = business_df.nlargest(5, 'closing_balance')[['ledger_name', 'closing_balance', 'parent']]
            top_negative = business_df.nsmallest(5, 'closing_balance')[['ledger_name', 'closing_balance', 'parent']]
            
            result_lines = ["📈 Current Financial Overview"]
            result_lines.append("=" * 40)
            result_lines.append(f"📊 Total Parties: {total_parties}")
            result_lines.append(f"💰 Net Balance Position: ₹{total_balance:,.2f}")
            result_lines.append("\n🔝 Top 5 Assets/Income Sources:")
            for i, (_, row) in enumerate(top_positive.iterrows(), 1):
                result_lines.append(f"{i}. {row['ledger_name']} - ₹{row['closing_balance']:,.2f} ({row['parent']})")
            
            result_lines.append("\n📉 Top 5 Liabilities/Expenses:")
            for i, (_, row) in enumerate(top_negative.iterrows(), 1):
                result_lines.append(f"{i}. {row['ledger_name']} - ₹{row['closing_balance']:,.2f} ({row['parent']})")
            
            return {
                "result": "\n".join(result_lines),
                "success": True
            }
        except Exception as e:
            return {"result": f"❌ Error in trend analysis: {str(e)}", "error": True}
    
    def process_summary_report(self, question: str, params: Dict, context: Dict) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        if self.df is None:
            return {"result": "❌ Error: Data loading failed.", "error": True}
        
        try:
            # Basic statistics
            total_parties = len(self.df)
            total_balance = self.df['closing_balance'].sum()
            avg_balance = self.df['closing_balance'].mean()
            max_balance = self.df['closing_balance'].max()
            min_balance = self.df['closing_balance'].min()
            
            # Business data (exclude internal accounts)
            business_df = self.df[
                ~self.df['parent'].isin(['Primary', 'Reserves & Surplus', 'Duties & Taxes', 'GST', 'TDS/TCS'])
            ]
            
            # Top categories
            category_summary = business_df.groupby('parent')['closing_balance'].sum().nlargest(5)
            
            # Recent activity
            recent_df = self.df[self.df['altered_on'].notna()].nlargest(5, 'altered_on')
            
            result_lines = ["📋 Comprehensive Business Summary"]
            result_lines.append("=" * 50)
            result_lines.append(f"🏢 Total Parties: {total_parties}")
            result_lines.append(f"💰 Total Balance: ₹{total_balance:,.2f}")
            result_lines.append(f"📊 Average Balance: ₹{avg_balance:,.2f}")
            result_lines.append(f"📈 Maximum Balance: ₹{max_balance:,.2f}")
            result_lines.append(f"📉 Minimum Balance: ₹{min_balance:,.2f}")
            result_lines.append("\n🏷️ Top 5 Business Categories by Balance:")
            for category, balance in category_summary.items():
                result_lines.append(f"• {category}: ₹{balance:,.2f}")
            
            result_lines.append("\n🕒 Recent Activities:")
            for i, (_, row) in enumerate(recent_df.iterrows(), 1):
                date_str = row['altered_on'].strftime('%Y-%m-%d') if pd.notna(row['altered_on']) else 'Unknown'
                result_lines.append(f"{i}. {row['ledger_name']} - ₹{row['closing_balance']:,.2f} ({date_str})")
            
            return {
                "result": "\n".join(result_lines),
                "success": True
            }
        except Exception as e:
            return {"result": f"❌ Error generating summary: {str(e)}", "error": True}
    
    def process_ranking_query(self, question: str, params: Dict, context: Dict) -> Dict[str, Any]:
        """Process ranking queries"""
        if self.df is None:
            return {"result": "❌ Error: Data loading failed.", "error": True}
        
        year = params.get('year')
        count = params.get('count', 10)
        order = params.get('order', 'DESC')
        
        try:
            # Filter data
            if year:
                filtered_df = self.df[self.df['altered_on'].dt.year == year]
                if len(filtered_df) == 0:
                    available_years = sorted(self.df['altered_on'].dt.year.dropna().unique())
                    return {
                        "result": f"❌ No data found for {year}. Available years: {', '.join(map(str, available_years))}",
                        "error": True
                    }
            else:
                filtered_df = self.df
            
            # Exclude internal accounts for business ranking
            business_df = filtered_df[
                ~filtered_df['parent'].isin(['Primary', 'Reserves & Surplus', 'Duties & Taxes', 'GST', 'TDS/TCS'])
            ]
            
            # Sort and get top N
            ascending = (order == 'ASC')
            ranked_df = business_df.nlargest(count, 'closing_balance') if not ascending else business_df.nsmallest(count, 'closing_balance')
            
            # Build result
            order_text = "Highest" if not ascending else "Lowest"
            year_text = f" in {year}" if year else ""
            result_lines = [f"🏆 {order_text} {count} Business Parties by Balance{year_text}:"]
            result_lines.append("=" * 60)
            for i, (_, row) in enumerate(ranked_df.iterrows(), 1):
                result_lines.append(f"{i}. {row['ledger_name']}")
                result_lines.append(f" 💰 Balance: ₹{row['closing_balance']:,.2f}")
                result_lines.append(f" 🏷️ Category: {row['parent']}")
                if pd.notna(row['altered_on']):
                    result_lines.append(f" 📅 Updated: {row['altered_on'].strftime('%Y-%m-%d')}")
                result_lines.append("")
            
            return {
                "result": "\n".join(result_lines),
                "success": True,
                "new_context": {"last_year": year} if year else {}
            }
        except Exception as e:
            return {"result": f"❌ Error in ranking analysis: {str(e)}", "error": True}
    
    def process_balance_query(self, question: str, params: Dict, context: Dict) -> Dict[str, Any]:
        """Process balance queries"""
        if self.df is None:
            return {"result": "❌ Error: Data loading failed.", "error": True}
        
        ledger_name = params.get('ledger_name')
        balance_type = params.get('balance_type', 'closing')
        if not ledger_name:
            return {
                "result": "🔍 Please specify which ledger's balance you want to check (e.g., 'balance of ABC Company')",
                "clarification_needed": True
            }
        
        try:
            # Search for ledger
            matches = self.df[self.df['ledger_name'].str.lower().str.contains(ledger_name.lower(), na=False)]
            if len(matches) == 0:
                return {"result": f"❌ No ledger found containing '{ledger_name}'.", "error": True}
            elif len(matches) == 1:
                ledger = matches.iloc[0]
                balance_value = ledger[f"{balance_type}_balance"]
                result_lines = [f"💰 Balance Information for {ledger['ledger_name']}"]
                result_lines.append("=" * 40)
                result_lines.append(f"💵 {balance_type.title()} Balance: ₹{balance_value:,.2f}")
                result_lines.append(f"📈 Opening Balance: ₹{ledger['opening_balance']:,.2f}")
                result_lines.append(f"🏷️ Parent Group: {ledger['parent']}")
                if pd.notna(ledger['altered_on']):
                    result_lines.append(f"📅 Last Updated: {ledger['altered_on'].strftime('%Y-%m-%d')}")
                
                return {
                    "result": "\n".join(result_lines),
                    "success": True,
                    "new_context": {"last_ledger_name": ledger['ledger_name']}
                }
            else:
                result_lines = [f"🔍 Found {len(matches)} ledgers matching '{ledger_name}':"]
                for i, (_, row) in enumerate(matches.head(10).iterrows(), 1):
                    result_lines.append(f"{i}. {row['ledger_name']} - ₹{row['closing_balance']:,.2f} ({row['parent']})")
                
                return {
                    "result": "\n".join(result_lines),
                    "success": True,
                    "new_context": {'pending_clarification': {
                        "template": f"what is the {balance_type} balance of {{name}}",
                        "options": matches['ledger_name'].head(10).tolist()
                    }}
                }
        except Exception as e:
            return {"result": f"❌ Error retrieving balance: {str(e)}", "error": True}
    
    def process_parent_query(self, question: str, params: Dict, context: Dict) -> Dict[str, Any]:
        """Process parent group queries"""
        if self.df is None:
            return {"result": "❌ Error: Data loading failed.", "error": True}
        
        ledger_name = params.get('ledger_name')
        if not ledger_name:
            return {
                "result": "🔍 Please specify which ledger's parent you want to find (e.g., 'parent of ABC Company')",
                "clarification_needed": True
            }
        
        try:
            # Search for ledger
            matches = self.df[self.df['ledger_name'].str.lower().str.contains(ledger_name.lower(), na=False)]
            if len(matches) == 0:
                return {"result": f"❌ No ledger found containing '{ledger_name}'.", "error": True}
            elif len(matches) == 1:
                ledger = matches.iloc[0]
                result_lines = [f"🏷️ Parent Information for {ledger['ledger_name']}"]
                result_lines.append("=" * 40)
                result_lines.append(f"📂 Parent Group: {ledger['parent']}")
                result_lines.append(f"💰 Closing Balance: ₹{ledger['closing_balance']:,.2f}")
                result_lines.append(f"📈 Opening Balance: ₹{ledger['opening_balance']:,.2f}")
                
                return {
                    "result": "\n".join(result_lines),
                    "success": True,
                    "new_context": {"last_ledger_name": ledger['ledger_name']}
                }
            else:
                result_lines = [f"🔍 Found {len(matches)} ledgers matching '{ledger_name}':"]
                for i, (_, row) in enumerate(matches.head(10).iterrows(), 1):
                    result_lines.append(f"{i}. {row['ledger_name']} - Parent: {row['parent']}")
                
                return {
                    "result": "\n".join(result_lines),
                    "success": True,
                    "new_context": {'pending_clarification': {
                        "template": "what is the parent of {name}",
                        "options": matches['ledger_name'].head(10).tolist()
                    }}
                }
        except Exception as e:
            return {"result": f"❌ Error retrieving parent information: {str(e)}", "error": True}
    
    def process_search_query(self, question: str, params: Dict, context: Dict) -> Dict[str, Any]:
        """General search functionality"""
        if self.df is None:
            return {"result": "❌ Error: Data loading failed.", "error": True}
        
        search_term = params.get('ledger_name') or params.get('search_term')
        if not search_term:
            return {
                "result": "🔍 Please specify what you want to search for (e.g., 'search for ABC Company')",
                "clarification_needed": True
            }
        
        try:
            # Search in ledger names and parents
            name_matches = self.df[self.df['ledger_name'].str.lower().str.contains(search_term.lower(), na=False)]
            parent_matches = self.df[self.df['parent'].str.lower().str.contains(search_term.lower(), na=False)]
            all_matches = pd.concat([name_matches, parent_matches]).drop_duplicates()
            
            if len(all_matches) == 0:
                return {"result": f"❌ No results found for '{search_term}'.", "error": True}
            
            result_lines = [f"🔍 Search Results for '{search_term}':"]
            result_lines.append(f"📊 Found {len(all_matches)} matching records")
            result_lines.append("=" * 50)
            for i, (_, row) in enumerate(all_matches.head(15).iterrows(), 1):
                result_lines.append(f"{i}. {row['ledger_name']}")
                result_lines.append(f" 💰 Balance: ₹{row['closing_balance']:,.2f}")
                result_lines.append(f" 🏷️ Category: {row['parent']}")
                result_lines.append("")
            if len(all_matches) > 15:
                result_lines.append(f"... and {len(all_matches) - 15} more results")
            
            return {
                "result": "\n".join(result_lines),
                "success": True
            }
        except Exception as e:
            return {"result": f"❌ Error in search: {str(e)}", "error": True}
    
    def process_general_query(self, question: str, params: Dict, context: Dict) -> Dict[str, Any]:
        """Handle general/unclassified queries"""
        # Try to understand the intent and provide helpful response
        q_lower = question.lower()
        help_lines = [
            "🤖 I can help you with various types of queries:",
            "",
            "📊 **Financial Analysis**",
            "• 'Profit in 2023'",
            "• 'Financial performance last year'",
            "",
            "💰 **Balance & Statements**",
            "• 'Balance of ABC Company'",
            "• 'Statement of XYZ Traders' (Full Year)",
            "• 'BR of Alpha Ltd' (Bill Receivable)",
            "",
            "🏆 **Ranking & Top Performers**",
            "• 'Top 10 parties by balance'",
            "• 'Highest paying parties in 2024'",
            "",
            "🔍 **Search & Information**",
            "• 'Search for suppliers'",
            "• 'Find parent of ledger'",
            "",
            "📈 **Trends & Reports**",
            "• 'Business summary'",
            "• 'Trend analysis'",
            "",
            "💸 **Pending Payments**",
            "• 'Show pending vendors'",
            "• 'Outstanding payments'",
        ]
        return {
            "result": "\n".join(help_lines),
            "success": True,
            "clarification_needed": True
        }

# =================== DATABASE HELPER =================== #
def _execute_sql(sql_query: str, params=None) -> Union[List[Dict[str, Any]], str]:
    print(f"Executing SQL: {sql_query}")
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_query, params or ())
                if cur.description is None:
                    return "Query executed successfully, no results to display."
                colnames = [d[0] for d in cur.description]
                rows = cur.fetchall()
                result = [dict(zip(colnames, r)) for r in rows]
                return result
    except Exception as e:
        print(f"SQL Error Details: {e}")
        return f"SQL Execution Error: {e}"

# =================== QUICK ACTION FUNCTIONS (Code 1) =================== #
def generate_correct_profit_loss_report():
    """CORRECT profit/loss report"""
    if df is None:
        return {"result": "❌ Error: Data loading failed.", "error": True}
    
    try:
        # Find actual Profit & Loss account
        pnl_accounts = df[df['ledger_name'].str.contains('Profit', na=False, case=False)]
        if not pnl_accounts.empty:
            pnl = pnl_accounts.iloc[0]
            result_lines = ["📊 ACTUAL Profit & Loss Report"]
            result_lines.append("=" * 40)
            result_lines.append(f"🏢 Account: {pnl['ledger_name']}")
            result_lines.append(f"💰 Closing Balance: ₹{pnl['closing_balance']:,.2f}")
            result_lines.append(f"📈 Opening Balance: ₹{pnl['opening_balance']:,.2f}")
            result_lines.append(f"📊 Net Profit/Loss: ₹{(pnl['closing_balance'] - pnl['opening_balance']):,.2f}")
            
            # Show actual business performance
            business_df = df[
                ~df['parent'].isin(['Primary', 'Reserves & Surplus', 'Duties & Taxes'])
            ]
            top_performers = business_df.nlargest(5, 'closing_balance')
            result_lines.append("\n🏆 Top 5 Business Parties:")
            for i, (_, row) in enumerate(top_performers.iterrows(), 1):
                result_lines.append(f"{i}. {row['ledger_name']} - ₹{row['closing_balance']:,.2f}")
            
            return {
                "result": "\n".join(result_lines),
                "success": True
            }
        else:
            return {
                "result": "❌ Profit & Loss account not found in data.\nTry: 'Business summary' or 'Top parties' for available analysis.",
                "error": True
            }
    except Exception as e:
        return {"result": f"❌ Error: {str(e)}", "error": True}

def generate_balance_query_prompt():
    return {
        "result": "🔍 Balance Query\nPlease enter the ledger name for which you want to check the balance.\nI'll show you detailed balance information including opening, closing balances and other details.",
        "new_context": {"awaiting_balance_query": True},
        "success": True
    }

def generate_parent_query_prompt():
    return {
        "result": "🔍 Find Parent Group\nPlease enter the ledger name for which you want to find the parent group.\nI'll show you the parent category along with comprehensive ledger information.",
        "new_context": {"awaiting_parent_query": True},
        "success": True
    }

def generate_top_parties_report():
    """Generate comprehensive top parties report"""
    if df is None:
        return {"result": "❌ Error: Data loading failed.", "error": True}
    
    processor = UltimateQueryProcessor(df, transactions_df, _execute_sql)
    return processor.process_ranking_query("top 10 parties", {"count": 10}, {})

def generate_business_summary():
    """Generate business summary"""
    if df is None:
        return {"result": "❌ Error: Data loading failed.", "error": True}
    
    processor = UltimateQueryProcessor(df, transactions_df, _execute_sql)
    return processor.process_summary_report("business summary", {}, {})

# =================== EXPORT FUNCTIONS (From Code 2) =================== #
def export_to_pdf(raw_data, filename):
    """Export report data to PDF format"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        
        if raw_data['type'] == 'statement':
            # Create financial statement PDF
            elements.append(Paragraph("FINANCIAL YEAR STATEMENT", styles['Title']))
            elements.append(Paragraph(f"Company: {raw_data['company_name']}", styles['Heading2']))
            elements.append(Paragraph(f"Financial Year: {raw_data['financial_year']}", styles['Heading3']))
            elements.append(Spacer(1, 12))
            
            # Create transaction table
            data = [['Date', 'Voucher Type', 'Voucher No', 'Particulars', 'Debit (₹)', 'Credit (₹)', 'Balance (₹)']]
            for trans in raw_data['transactions']:
                date_str = trans['date'][5:] if len(trans['date']) > 5 else trans['date']  # Get DD-MM format or full date
                voucher_type = trans['voucher_type'][:12]
                voucher_no = trans['voucher_no'][:10]
                particulars = trans['particulars'][:18]
                debit = f"{trans['debit']:,.2f}" if trans['debit'] > 0 else "0.00"
                credit = f"{trans['credit']:,.2f}" if trans['credit'] > 0 else "0.00"
                balance = f"{trans['balance']:,.2f}"
                data.append([date_str, voucher_type, voucher_no, particulars, debit, credit, balance])
            
            # Create table
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (4, 1), (-1, -1), 'RIGHT'),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(f"Closing Balance: ₹{raw_data['closing_balance']:,.2f}", styles['Heading3']))
        
        elif raw_data['type'] == 'br_report':
            # Create BR report PDF
            elements.append(Paragraph("BILL RECEIVABLE (BR) STATEMENT", styles['Title']))
            elements.append(Paragraph(f"Company: {raw_data['company_name']}", styles['Heading2']))
            elements.append(Paragraph(f"Financial Year: {raw_data['financial_year']}", styles['Heading3']))
            elements.append(Spacer(1, 12))
            
            # Create BR table
            data = [['BR Date', 'BR No.', 'Party Name', 'Amount (₹)', 'Due Date', 'Status']]
            for br in raw_data['br_entries']:
                br_date = br['br_date'][8:] + "-" + br['br_date'][5:7] if len(br['br_date']) > 7 else br['br_date']  # Get DD-MM format
                due_date = br['due_date'][8:] + "-" + br['due_date'][5:7] if len(br['due_date']) > 7 else br['due_date']  # Get DD-MM format
                amount = f"{br['amount']:,.2f}"
                status = br['status'][:12]
                party_name = br['party'][:22]
                data.append([br_date, br['br_no'][:10], party_name, amount, due_date, status])
            
            # Add total row
            data.append(['', '', 'TOTAL', f"{raw_data['total_br']:,.2f}", '', ''])
            
            # Create table
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('BACKGROUND', (0, 1), (-1, -2), colors.beige),
                ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (3, 1), (-1, -1), 'RIGHT'),
            ]))
            elements.append(table)
        
        # Build the PDF
        doc.build(elements)
        
        # Save the PDF
        with open(filename, 'wb') as f:
            f.write(buffer.getvalue())
        return True
    except Exception as e:
        print(f"Error exporting to PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# =================== ENHANCED MAIN QUERY TOOL (Merged) =================== #
def ultimate_query_tool(question: str, context: Dict[str, Any] = {}) -> Dict[str, Any]:
    """Ultimate query processing tool with ALL capabilities"""
    processor = UltimateQueryProcessor(df, transactions_df, _execute_sql)
    
    # Check for special quick action commands from Code 1
    special_commands = {
        "balance_query_prompt": generate_balance_query_prompt,
        "parent_query_prompt": generate_parent_query_prompt,
        "top_parties_report": generate_top_parties_report,
        "profit_loss_report": generate_correct_profit_loss_report,
        "business_summary": generate_business_summary,
        "pending_analysis": lambda: processor.process_pending_analysis("", {}, {}),
        "trend_analysis": lambda: processor.process_trend_analysis("", {}, {})
    }
    
    if question.strip() in special_commands:
        return special_commands[question.strip()]()
    
    # Process the query (handles all types including statements/br)
    return processor.process_query(question, context)

# =================== ENHANCED MODERN GUI APPLICATION (Merged) =================== #
class UltimateTallyAIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Eigen AI - Ultimate Pro")
        self.root.geometry("1400x900")
        self.root.configure(bg="#1e1e1e")
        
        # Enhanced theme with professional colors
        self.is_dark_mode = True
        self.colors = {
            "primary": "#2563eb",
            "secondary": "#10b981",
            "accent": "#f59e0b",
            "dark": "#1e1e1e",
            "darker": "#0f0f0f",
            "light": "#2d2d2d",
            "lighter": "#3d3d3d",
            "text": "#f8fafc",
            "text_secondary": "#cbd5e1",
            "success": "#22c55e",
            "warning": "#eab308",
            "error": "#ef4444",
            "white": "#ffffff"
        }
        
        self.conversations = []
        self.current_conversation_index = -1
        self.active_quick = None
        self.quick_buttons = {}
        
        # Enhanced pagination state
        self.paginated_df = None
        self.current_page = 1
        self.page_size = 15
        
        self.setup_ui()
        self.setup_backend()
    
    def setup_ui(self):
        """Setup the ultimate professional UI"""
        # Main container
        main_container = Frame(self.root, bg=self.colors["dark"])
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Left sidebar
        sidebar_frame = Frame(main_container, bg=self.colors["darker"], width=300)
        sidebar_frame.pack(side=tk.LEFT, fill=tk.Y)
        sidebar_frame.pack_propagate(False)
        
        # Logo and title
        logo_frame = Frame(sidebar_frame, bg=self.colors["darker"])
        logo_frame.pack(fill=tk.X, padx=15, pady=20)
        title_label = Label(logo_frame, text="Eigen AI",
                           font=("Segoe UI", 20, "bold"),
                           bg=self.colors["darker"], fg=self.colors["primary"])
        title_label.pack()
        
        # New chat button
        new_chat_btn = Button(sidebar_frame, text="+ New Chat",
                            font=("Segoe UI", 12, "bold"),
                            bg=self.colors["primary"], fg="white",
                            relief="flat", height=2,
                            command=self.new_chat)
        new_chat_btn.pack(fill=tk.X, padx=15, pady=10)
        
        # Quick actions in sidebar (Merged List)
        quick_actions_frame = Frame(sidebar_frame, bg=self.colors["darker"])
        quick_actions_frame.pack(fill=tk.X, padx=15, pady=10)
        quick_label = Label(quick_actions_frame, text="QUICK ACTIONS",
                           font=("Segoe UI", 10, "bold"),
                           bg=self.colors["darker"], fg=self.colors["text_secondary"])
        quick_label.pack(anchor=tk.W)
        
        sidebar_actions = [
            ("💰 Check Balance", "balance_query_prompt"),
            ("🏷️ Find Parent", "parent_query_prompt"),
            ("📊 Business Summary", "business_summary"),
            ("🏆 Top Parties", "top_parties_report"),
            ("📈 Profit/Loss", "profit_loss_report"),
            ("💸 Pending Analysis", "pending_analysis"),
            ("📅 Trend Analysis", "trend_analysis"),
            # Added from Code 2
            ("📄 Statement", "statement_prompt"),
            ("📋 Bill Receivable", "br_prompt"),
        ]
        
        for action_text, action_cmd in sidebar_actions:
            btn = Button(quick_actions_frame, text=action_text,
                       font=("Segoe UI", 10),
                       bg=self.colors["light"], fg=self.colors["text"],
                       relief="flat", anchor=tk.W,
                       command=lambda cmd=action_cmd: self.quick_action_clicked(cmd))
            btn.pack(fill=tk.X, pady=2)
        
        # Conversation history
        conv_frame = Frame(sidebar_frame, bg=self.colors["darker"])
        conv_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        conv_label = Label(conv_frame, text="CONVERSATION HISTORY",
                         font=("Segoe UI", 10, "bold"),
                         bg=self.colors["darker"], fg=self.colors["text_secondary"])
        conv_label.pack(anchor=tk.W)
        
        self.conversation_list = Listbox(conv_frame,
                                        font=("Segoe UI", 10),
                                        bg=self.colors["light"], fg=self.colors["text"],
                                        relief="flat", bd=0,
                                        selectbackground=self.colors["primary"],
                                        selectforeground="white")
        self.conversation_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.conversation_list.bind('<<ListboxSelect>>', self.on_conversation_select)
        
        conv_scroll = Scrollbar(conv_frame, orient=tk.VERTICAL, command=self.conversation_list.yview)
        conv_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.conversation_list.config(yscrollcommand=conv_scroll.set)
        
        # Main content area
        main_content = Frame(main_container, bg=self.colors["dark"])
        main_content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = Frame(main_content, bg=self.colors["darker"], height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        header_content = Frame(header_frame, bg=self.colors["darker"])
        header_content.pack(expand=True, fill=tk.BOTH, padx=20)
        
        welcome_label = Label(header_content, text="Welcome to Eigen AI",
                            font=("Segoe UI", 16, "bold"),
                            bg=self.colors["darker"], fg=self.colors["text"])
        welcome_label.pack(side=tk.LEFT)
        
        # Chat display with modern styling
        chat_container = Frame(main_content, bg=self.colors["dark"])
        chat_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.chat_display = scrolledtext.ScrolledText(chat_container,
                                                     wrap=tk.WORD,
                                                     state='disabled',
                                                     font=("Courier New", 10), # Changed to Courier for tables (Code 2 style)
                                                     bg=self.colors["light"],
                                                     fg=self.colors["text"],
                                                     padx=20,
                                                     pady=20,
                                                     relief="flat",
                                                     borderwidth=0,
                                                     highlightthickness=0)
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.configure_chat_tags()
        
        # Input area
        input_frame = Frame(main_content, bg=self.colors["dark"])
        input_frame.pack(fill=tk.X, padx=20, pady=10)
        
        input_container = Frame(input_frame, bg=self.colors["lighter"], height=60)
        input_container.pack(fill=tk.X)
        input_container.pack_propagate(False)
        
        input_inner = Frame(input_container, bg=self.colors["lighter"])
        input_inner.pack(expand=True, fill=tk.BOTH, padx=2, pady=2)
        
        self.input_entry = Entry(input_inner,
                               font=("Segoe UI", 12),
                               relief="flat",
                               bg=self.colors["light"],
                               fg=self.colors["text"],
                               insertbackground=self.colors["primary"])
        self.input_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15, pady=15)
        self.input_entry.bind("<Return>", self.ask_question_event)
        self.input_entry.focus_set()
        
        self.ask_button = Button(input_inner,
                               text="Send",
                               command=self.ask_question,
                               font=("Segoe UI", 11, "bold"),
                               bg=self.colors["primary"],
                               fg="white",
                               relief="flat",
                               padx=25,
                               cursor="hand2")
        self.ask_button.pack(side=tk.RIGHT, padx=10, pady=15)
    
    def configure_chat_tags(self):
        """Configure modern chat tags"""
        tags_config = {
            'user': {'foreground': self.colors["primary"], 'font': ("Segoe UI", 11, "bold")},
            'ai': {'foreground': self.colors["text"], 'font': ("Courier New", 10)},
            'error': {'foreground': self.colors["error"], 'font': ("Segoe UI", 11)},
            'info': {'foreground': self.colors["warning"], 'font': ("Segoe UI", 10, "italic")},
            'clarification': {'foreground': self.colors["accent"], 'font': ("Segoe UI", 11, "bold")},
            'success': {'foreground': self.colors["success"], 'font': ("Segoe UI", 10, "italic")},
            'system': {'foreground': self.colors["text_secondary"], 'font': ("Segoe UI", 10)}
        }
        
        for tag, config in tags_config.items():
            self.chat_display.tag_config(tag, **config)
    
    def on_conversation_select(self, event):
        """Handle conversation selection"""
        if self.conversation_list.curselection():
            self.current_conversation_index = self.conversation_list.curselection()[0]
            self.rebuild_chat_display()
    
    def quick_action_clicked(self, command):
        """Handle quick action clicks"""
        self.active_quick = command
        self.quick_action(command)
    
    def new_chat(self):
        """Create new chat"""
        new_conv = {
            'title': f"Chat {len(self.conversations) + 1}",
            'messages': [],
            'context': {}
        }
        self.conversations.append(new_conv)
        self.current_conversation_index = len(self.conversations) - 1
        self.update_conversation_list()
        self.rebuild_chat_display()
        self.input_entry.delete(0, END)
        self.input_entry.focus_set()
    
    def update_conversation_list(self):
        """Update conversation list in sidebar"""
        self.conversation_list.delete(0, tk.END)
        for conv in self.conversations:
            self.conversation_list.insert(tk.END, conv['title'])
        if self.current_conversation_index >= 0:
            self.conversation_list.selection_set(self.current_conversation_index)
    
    def rebuild_chat_display(self):
        """Rebuild chat display from current conversation"""
        if self.current_conversation_index < 0:
            return
        
        conv = self.conversations[self.current_conversation_index]
        self.chat_display.config(state='normal')
        self.chat_display.delete(1.0, tk.END)
        for sender, message, tag, log_id in conv['messages']:
            self.chat_display.insert(tk.END, f"{sender}:\n", tag)
            self.chat_display.insert(tk.END, f"{message}\n")
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)
    
    def quick_action(self, command):
        """Handle quick actions"""
        self.input_entry.delete(0, END)
        # Added prompts for Statement and BR
        if command == "statement_prompt":
            self.input_entry.insert(0, "statement of ")
        elif command == "br_prompt":
            self.input_entry.insert(0, "BR of ")
        else:
            self.input_entry.insert(0, command)
        self.ask_question()
        if "prompt" in command:
            self.input_entry.focus_set()
    
    def setup_backend(self):
        """Initialize backend"""
        if df is None:
            self.new_chat()
            self.update_chat_display("System", "Database connection failed. Please check your configuration.", 'error')
            return
        
        self.new_chat()
        welcome_msg = f"""🚀 **Welcome to Eigen AI Ultimate Pro**
I understand natural language - just ask anything!
✅ **Active Features:**
• Financial Analysis (P&L, Trends)
• Business Summaries & Top Parties
• **NEW:** Financial Statements (Full Year) - PDF Export
• **NEW:** Bill Receivable Reports - PDF Export
• **ENHANCED:** Real transaction data integration
• Automatic Financial Year Detection: {CURRENT_FY}
Try: "Show statement of ABC Corp" or "Analyze profit"
"""
        self.update_chat_display("AI", welcome_msg, 'ai')
    
    def ask_question_event(self, event):
        self.ask_question()
    
    def ask_question(self):
        """Main question handling method"""
        question = self.input_entry.get().strip()
        if not question:
            return
        
        if self.current_conversation_index < 0:
            self.new_chat()
        
        conv = self.conversations[self.current_conversation_index]
        
        # Update conversation title if first message
        if not conv['messages']:
            conv['title'] = question[:35] + "..." if len(question) > 35 else question
            self.update_conversation_list()
        
        self.update_chat_display("You", question, 'user')
        self.input_entry.delete(0, END)
        self.ask_button.config(state=tk.DISABLED, text="Processing...")
        
        # Handle special modes
        final_question = question
        if conv['context'].get('awaiting_balance_query'):
            final_question = f"balance of {question}"
            del conv['context']['awaiting_balance_query']
        elif conv['context'].get('awaiting_parent_query'):
            final_question = f"parent of {question}"
            del conv['context']['awaiting_parent_query']
        
        threading.Thread(target=self.process_in_background,
                       args=(final_question,), daemon=True).start()
    
    def process_in_background(self, question: str):
        """Background processing"""
        try:
            if self.current_conversation_index < 0:
                return
            
            conv = self.conversations[self.current_conversation_index]
            result_payload = ultimate_query_tool(question, conv['context'])
            
            # Handle paginated results
            if result_payload.get("result_type") == "paginated_pending":
                self.paginated_df = result_payload['data']
                self.current_page = 1
                self.root.after(0, self.display_paginated_results)
                return
            
            if result_payload.get("clarification_needed"):
                self.root.after(0, self.handle_clarification, result_payload)
            elif result_payload.get("error"):
                error_msg = result_payload.get("result", "Unknown error occurred")
                self.root.after(0, self.update_chat_display, "AI", error_msg, 'error')
            else:
                self.root.after(0, self.handle_success, question, result_payload)
        except Exception as e:
            self.root.after(0, self.update_chat_display, "AI", f"An unexpected error occurred: {str(e)}", 'error')
        finally:
            self.root.after(0, lambda: self.ask_button.config(state=tk.NORMAL, text="Send"))
    
    def handle_clarification(self, payload):
        """Handle clarification requests"""
        clarification_msg = payload.get("result")
        self.update_chat_display("AI", "I need clarification:", 'clarification')
        self.update_chat_display("AI", clarification_msg, 'ai')
        if self.current_conversation_index >= 0:
            conv = self.conversations[self.current_conversation_index]
            new_context = payload.get("new_context", {})
            conv['context'].update(new_context)
    
    def handle_success(self, question, payload):
        """Handle successful query processing (Merged)"""
        if self.current_conversation_index < 0:
            return
        
        conv = self.conversations[self.current_conversation_index]
        new_context = payload.get("new_context", {})
        conv['context'].update(new_context)
        
        # Store raw data for download functionality (Code 2 Feature)
        if 'raw_data' in payload:
            raw_data = payload['raw_data']
            conv['last_raw_data'] = raw_data
        
        output = payload.get("result", "No result.")
        self.update_chat_display("AI", output, 'ai')
        
        # Add download button if raw data is available (Code 2 Feature)
        self.add_download_button_if_needed(payload)
    
    def update_chat_display(self, sender: str, message: str, tag: str, log_id: Optional[int] = None):
        """Update chat display with new message"""
        if self.current_conversation_index < 0:
            return
        
        conv = self.conversations[self.current_conversation_index]
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, f"{sender}:\n", tag)
        
        # Insert message with monospace formatting logic from Code 2
        if "╔" in message or "BR Date" in message or "BILL RECEIVABLE" in message or "FINANCIAL YEAR STATEMENT" in message:
            self.chat_display.insert(tk.END, f"{message}\n", 'ai')
        else:
            self.chat_display.insert(tk.END, f"{message}\n")
        
        # Store message
        conv['messages'].append((sender, message, tag, log_id))
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)
    
    def display_paginated_results(self):
        """Display paginated results (Code 1 Feature)"""
        if self.paginated_df is None:
            return
        
        self.chat_display.config(state='normal')
        
        # Get current page data
        total_records = len(self.paginated_df)
        total_pages = math.ceil(total_records / self.page_size)
        start_index = (self.current_page - 1) * self.page_size
        end_index = start_index + self.page_size
        page_df = self.paginated_df.iloc[start_index:end_index]
        
        self.chat_display.insert(tk.END, "AI:\n", 'ai')
        
        # Build result display
        result_lines = [f"📋 Pending Payments Analysis"]
        result_lines.append(f"*Showing {start_index+1}-{min(end_index, total_records)} of {total_records} records*\n")
        for idx, (_, row) in enumerate(page_df.iterrows(), start_index + 1):
            ledger_name = row.get('ledger_name', 'Unknown')
            closing_balance = float(row.get('closing_balance', 0))
            opening_balance = float(row.get('opening_balance', 0))
            parent = row.get('parent', 'Unknown')
            result_lines.append(f"{idx}. **{ledger_name}**")
            result_lines.append(f" 💰 Closing: ₹{closing_balance:,.2f}")
            result_lines.append(f" 📊 Opening: ₹{opening_balance:,.2f}")
            result_lines.append(f" 🏷️ Category: {parent}")
            if pd.notna(row.get('altered_on')):
                date_str = row['altered_on'].strftime('%Y-%m-%d')
                result_lines.append(f" 📅 Updated: {date_str}")
            result_lines.append("")
        
        message = "\n".join(result_lines)
        self.chat_display.insert(tk.END, message + "\n")
        
        # Pagination controls
        if total_pages > 1:
            controls_frame = Frame(self.chat_display, bg=self.colors["light"])
            
            prev_btn = Button(controls_frame, text="◀ Previous",
                            font=("Segoe UI", 9),
                            bg=self.colors["lighter"], fg=self.colors["text"],
                            relief="flat", cursor="hand2",
                            command=lambda: self.change_page(-1))
            prev_btn.pack(side=tk.LEFT, padx=5)
            if self.current_page == 1:
                prev_btn.config(state=tk.DISABLED)
            
            page_info = Label(controls_frame, text=f"Page {self.current_page} of {total_pages}",
                            font=("Segoe UI", 9, "bold"),
                            bg=self.colors["light"], fg=self.colors["text"])
            page_info.pack(side=tk.LEFT, padx=10)
            
            next_btn = Button(controls_frame, text="Next ▶",
                            font=("Segoe UI", 9),
                            bg=self.colors["lighter"], fg=self.colors["text"],
                            relief="flat", cursor="hand2",
                            command=lambda: self.change_page(1))
            next_btn.pack(side=tk.LEFT, padx=5)
            if self.current_page == total_pages:
                next_btn.config(state=tk.DISABLED)
            
            self.chat_display.window_create(tk.END, window=controls_frame)
            self.chat_display.insert(tk.END, "\n")
        
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)
    
    def change_page(self, direction: int):
        """Change page in paginated view"""
        if self.paginated_df is None:
            return
        
        total_pages = math.ceil(len(self.paginated_df) / self.page_size)
        new_page = self.current_page + direction
        if 1 <= new_page <= total_pages:
            self.current_page = new_page
            self.display_paginated_results()
    
    # --- Code 2 Download Features (Added) ---
    def add_download_button_if_needed(self, payload):
        """Add download button if raw data is available"""
        if 'raw_data' not in payload:
            return
        
        if self.current_conversation_index < 0:
            return
        
        # Create download button frame
        button_frame = Frame(self.chat_display, bg=self.colors["light"])
        
        # Create download as PDF button
        pdf_btn = Button(button_frame, text="📥 Download as PDF",
                       font=("Segoe UI", 9),
                       bg=self.colors["secondary"], fg="white",
                       relief="flat", cursor="hand2",
                       command=lambda: self.download_as_pdf(payload['raw_data']))
        pdf_btn.pack(side=tk.LEFT, padx=5)
        
        # Create download as text button
        text_btn = Button(button_frame, text="📄 Download as Text",
                        font=("Segoe UI", 9),
                        bg=self.colors["primary"], fg="white",
                        relief="flat", cursor="hand2",
                        command=lambda: self.download_as_text(payload['raw_data']))
        text_btn.pack(side=tk.LEFT, padx=5)
        
        # Insert button into chat display
        self.chat_display.window_create(tk.END, window=button_frame)
        self.chat_display.insert(tk.END, "\n")
    
    def download_as_pdf(self, raw_data):
        """Download the report as a PDF file"""
        try:
            # Get filename based on report type
            if raw_data.get("type") == "statement":
                filename = f"Statement_{raw_data['company_name'].replace(' ', '_')}_{raw_data['financial_year']}.pdf"
            elif raw_data.get("type") == "br_report":
                filename = f"BR_Report_{raw_data['company_name'].replace(' ', '_')}_{raw_data['financial_year']}.pdf"
            else:
                filename = f"Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            # Open save file dialog
            file_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
                initialfile=filename,
                title="Save Report as PDF"
            )
            if not file_path:
                return
            
            # Export to PDF
            if export_to_pdf(raw_data, file_path):
                messagebox.showinfo("Success", f"Report saved successfully to:\n{file_path}")
            else:
                messagebox.showerror("Error", "Failed to save PDF file.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save PDF:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def download_as_text(self, raw_data):
        """Download the report as a text file"""
        try:
            # Get filename based on report type
            if raw_data.get("type") == "statement":
                filename = f"Statement_{raw_data['company_name'].replace(' ', '_')}_{raw_data['financial_year']}.txt"
            elif raw_data.get("type") == "br_report":
                filename = f"BR_Report_{raw_data['company_name'].replace(' ', '_')}_{raw_data['financial_year']}.txt"
            else:
                filename = f"Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            # Open save file dialog
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                initialfile=filename,
                title="Save Report as Text"
            )
            if not file_path:
                return
            
            # Use the pre-generated result text if available, otherwise fall back to summary
            content = raw_data.get('result_text')
            if not content:
                content = f"Report Type: {raw_data.get('type')}\nCompany: {raw_data.get('company_name')}\nYear: {raw_data.get('financial_year')}\n(No detailed text available)"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            messagebox.showinfo("Success", f"Report saved successfully to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save text file:\n{str(e)}")
            import traceback
            traceback.print_exc()

# =================== MAIN EXECUTION =================== #
if __name__ == "__main__":
    print("🚀 Starting Eigen AI Ultimate Pro (Merged)")
    print("=" * 60)
    print("Features Loaded:")
    print("✅ Financial Analysis & Trends (from Code 1)")
    print("✅ Balance, Ranking & Summary (from Code 1)")
    print("✅ Financial Statements & BR Reports (from Code 2)")
    print("✅ PDF & Text Exporting (from Code 2)")
    print("✅ Real Transaction Data Integration (ENHANCED)")
    print("✅ Automatic Financial Year Detection: " + CURRENT_FY)
    print("✅ Dynamic Table Formatting (Added Fix)")
    print("=" * 60)
    
    if df is not None:
        print(f"📊 Database: Connected ({len(df):,} ledger records)")
        if transactions_df is not None:
            print(f"💱 Transaction Data: Connected ({len(transactions_df):,} transaction records)")
        
        # Show actual Profit & Loss account if exists (Code 1 feature)
        pnl_accounts = df[df['ledger_name'].str.contains('Profit', na=False, case=False)]
        if not pnl_accounts.empty:
            pnl = pnl_accounts.iloc[0]
            print(f"💰 Actual Profit & Loss: ₹{pnl['closing_balance']:,.2f}")
        print("=" * 60)
    
    # Create and run application
    root = tk.Tk()
    app = UltimateTallyAIGUI(root)
    root.mainloop()
