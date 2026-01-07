

import time
import tkinter as tk
from tkinter import ttk, messagebox
import requests
import pandas as pd
from io import StringIO
import re
import xml.etree.ElementTree as ET
import logging
from sqlalchemy import create_engine, exc, text
import datetime

# --- LIBRARY CHECK ---
try:
    import pandas as pd
    import requests
    from sqlalchemy import create_engine
except ImportError as e:
    print("CRITICAL ERROR: Libraries missing. Please run: pip install pandas sqlalchemy requests psycopg2-binary lxml")
    raise e

start = time.time()

class TallyLedgerViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Tally Voucher Viewer (Full History: 2000-2035)")
        self.root.geometry("1200x700")
        self.current_df = None
        self.auto_fetch_running = False
        self.is_first_run = True  # Track first run

        # Configure logging
        logging.basicConfig(level=logging.ERROR)
        
        # Database connection parameters
        self.db_config = {
            'host': '',
            'database': '',
            'user': '',
            'password': ''
        }
        
        # — Tally Connection Input —
        frame = tk.Frame(root)
        frame.pack(pady=10)

        tk.Label(frame, text="Tally Host:").grid(row=0, column=0, padx=5)
        self.host_entry = tk.Entry(frame, width=15)
        self.host_entry.insert(0, "192.168.1.150")
        self.host_entry.grid(row=0, column=1)

        tk.Label(frame, text="Port:").grid(row=0, column=2, padx=5)
        self.port_entry = tk.Entry(frame, width=8)
        self.port_entry.insert(0, "9000")
        self.port_entry.grid(row=0, column=3)

        fetch_btn = tk.Button(
            frame, text="Fetch All Data", command=self.load_data,
            bg="green", fg="white", font=('Arial', 12)
        )
        fetch_btn.grid(row=0, column=4, padx=10)

        save_btn = tk.Button(
            frame, text="Save to DB", command=self.save_to_database,
            bg="blue", fg="white", font=('Arial', 12)
        )
        save_btn.grid(row=0, column=5, padx=10)
        
        self.auto_fetch_btn = tk.Button(
            frame, text="Start Auto-Fetch", command=self.toggle_auto_fetch,
            bg="orange", fg="black", font=('Arial', 12)
        )
        self.auto_fetch_btn.grid(row=0, column=6, padx=10)

        # — Treeview Table Display —
        self.tree = ttk.Treeview(root)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side='right', fill='y')
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.count_label = tk.Label(root, text="Rows: 0", font=('Arial', 12), anchor='w')
        self.count_label.pack(fill=tk.X, padx=10)
        
        self.auto_fetch_status = tk.Label(
            root, text="Auto-Fetch: Stopped", font=('Arial', 10), fg="red", anchor='w'
        )
        self.auto_fetch_status.pack(fill=tk.X, padx=10, pady=(0, 5))

    def toggle_auto_fetch(self):
        self.auto_fetch_running = not self.auto_fetch_running
        if self.auto_fetch_running:
            self.auto_fetch_btn.config(text="Stop Auto-Fetch", bg="red")
            self.auto_fetch_status.config(text="Auto-Fetch: Running (60s interval)", fg="green")
            self.auto_fetch()
        else:
            self.auto_fetch_btn.config(text="Start Auto-Fetch", bg="orange")
            self.auto_fetch_status.config(text="Auto-Fetch: Stopped", fg="red")

    def auto_fetch(self):
        if self.auto_fetch_running:
            try:
                self.load_data()
            except Exception as e:
                logging.error(f"Auto-fetch error: {e}")
            finally:
                self.root.after(60000, self.auto_fetch)

    # --- XML CLEANING (Fixes Unbound Prefix & Invalid Chars) ---
    def clean_xml(self, xml_content: str) -> str:
        if not isinstance(xml_content, str):
            try:
                xml_content = str(xml_content, 'utf-8', errors='ignore')
            except:
                xml_content = str(xml_content)

        # 1. Remove Namespaces (e.g. <UDF:Value> -> <Value>)
        xml_content = re.sub(r'(</?)[a-zA-Z0-9_]+:', r'\1', xml_content)
        # 2. Remove invalid control characters
        xml_content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', xml_content)
        # 3. Fix Ampersands
        xml_content = re.sub(r'&(?!(amp|lt|gt|apos|quot);)', '&amp;', xml_content)

        return xml_content

    # --- SAFE HELPERS ---
    def get_xml_text(self, element, tag, default=""):
        if element is None:
            return default
        child = element.find(tag)
        if child is None:
            return default
        return str(child.text) if child.text is not None else default

    def safe_float(self, value):
        try:
            if value is None:
                return 0.0
            val_str = str(value).strip()
            if not val_str:
                return 0.0
            return float(val_str)
        except (ValueError, TypeError):
            return 0.0

    def fetch_ledger_data(self, url: str) -> pd.DataFrame:
        # --- FIXED REQUEST FOR ALL DATES (2000 to 2035) ---
        TALLY_REQUEST = """<ENVELOPE>
         <HEADER>
          <VERSION>1</VERSION>
          <TALLYREQUEST>Export Data</TALLYREQUEST>
          <TYPE>Collection</TYPE>
          <ID>Voucher Collection</ID>
         </HEADER>
         <BODY>
          <DESC>
           <STATICVARIABLES>
            <SVEXPORTFORMAT>$$SysName:XML</SVEXPORTFORMAT>
            <SVFROMDATE>20000401</SVFROMDATE>
            <SVTODATE>20350331</SVTODATE>
           </STATICVARIABLES>
           <TDL>
            <TDLMESSAGE>
             <COLLECTION NAME="Voucher Collection" ISMODIFY="No">
              <TYPE>Voucher</TYPE>
              <FETCH>Date, VoucherTypeName, VoucherNumber, PartyLedgerName, Reference</FETCH>
              <NATIVEMETHOD>AllLedgerEntries.List.LedgerName</NATIVEMETHOD>
              <NATIVEMETHOD>AllLedgerEntries.List.Amount</NATIVEMETHOD>
             </COLLECTION>
            </TDLMESSAGE>
           </TDL>
          </DESC>
         </BODY>
        </ENVELOPE>"""

        try:
            # Increased timeout to 120s as historical data is large
            resp = requests.post(
                url, 
                data=TALLY_REQUEST.encode('utf-8'),
                headers={'Content-Type': 'application/xml'},
                timeout=120
            )
            resp.raise_for_status()
            
            cleaned = self.clean_xml(resp.text)
            
            try:
                root = ET.fromstring(cleaned)
                transactions = []
                
                for voucher in root.findall('.//VOUCHER'):
                    date = self.get_xml_text(voucher, 'DATE')
                    v_type = self.get_xml_text(voucher, 'VOUCHERTYPENAME')
                    v_no = self.get_xml_text(voucher, 'VOUCHERNUMBER')
                    party_name = self.get_xml_text(voucher, 'PARTYLEDGERNAME', default="Unknown")
                    
                    # Case insensitive search for entries
                    entries = voucher.findall('.//ALLLEDGERENTRIES.LIST')
                    if not entries:
                         entries = voucher.findall('.//allledgerentries.list')

                    for entry in entries:
                        ledger_name = self.get_xml_text(entry, 'LEDGERNAME')
                        if not ledger_name:
                             ledger_name = self.get_xml_text(entry, 'ledgername')

                        amount_str = self.get_xml_text(entry, 'AMOUNT', default="0")
                        if amount_str == "0":
                            amount_str = self.get_xml_text(entry, 'amount', default="0")
                        
                        amount = self.safe_float(amount_str)

                        row = {
                            'date': date,
                            'voucher_type': v_type,
                            'voucher_no': v_no,
                            'party_name': party_name,
                            'particulars': ledger_name,
                            'amount': amount
                        }
                        transactions.append(row)
                
                df = pd.DataFrame(transactions)
                return df

            except ET.ParseError as parse_error:
                error_msg = f"XML Parsing Error: {str(parse_error)}"
                messagebox.showerror("XML Error", error_msg)
                return pd.DataFrame()
            
        except requests.RequestException as e:
            messagebox.showerror("Connection Error", f"Failed to connect to Tally:\n{e}")
            return pd.DataFrame()

    def clean_data(self, df):
        if df.empty:
            return df

        if 'date' in df.columns:
            # Format Date - convert to string format for database compatibility
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
            # Keep as datetime object, not date object
            df = df.dropna(subset=['date'])  # Remove rows with invalid dates

        if 'amount' in df.columns:
            # Create Debit/Credit Columns based on sign
            df['debit_amount'] = df['amount'].apply(lambda x: abs(float(x)) if x > 0 else 0.0)
            df['credit_amount'] = df['amount'].apply(lambda x: abs(float(x)) if x < 0 else 0.0)
            # Round to 2 decimal places
            df['debit_amount'] = df['debit_amount'].round(2)
            df['credit_amount'] = df['credit_amount'].round(2)
        else:
            df['debit_amount'] = 0.0
            df['credit_amount'] = 0.0
        
        if 'voucher_no' in df.columns:
            df['voucher_no'] = df['voucher_no'].astype(str)
        
        if 'voucher_type' in df.columns:
            df['voucher_type'] = df['voucher_type'].astype(str)
        
        if 'party_name' in df.columns:
            df['party_name'] = df['party_name'].astype(str)
        
        if 'particulars' in df.columns:
            df['particulars'] = df['particulars'].astype(str)
        
        # Fixed column names for database (lowercase, no spaces)
        display_cols = ['date', 'voucher_type', 'voucher_no', 'party_name', 'particulars', 'debit_amount', 'credit_amount']
        final_cols = [c for c in display_cols if c in df.columns]
        return df[final_cols]
    
    def drop_table_if_exists(self, engine):
        """Drop table if exists on first run"""
        try:
            with engine.connect() as conn:
                conn.execute(text("DROP TABLE IF EXISTS tally_transactions CASCADE"))
                conn.commit()
            print("Existing table dropped successfully")
        except Exception as e:
            print(f"Error dropping table: {e}")
    
    def load_data(self):
        host = self.host_entry.get().strip()
        port = self.port_entry.get().strip()
        
        if not host or not port:
             messagebox.showwarning("Input Error", "Please check Host IP and Port")
             return

        url = f"http://{host}:{port}"
        
        # Cursor fix for Linux (Wait -> Watch)
        try:
            self.root.config(cursor="watch")
            self.root.update()
            df = self.fetch_ledger_data(url)
        except tk.TclError:
            pass 
        finally:
            try:
                self.root.config(cursor="")
            except:
                pass
        
        if df.empty:
            self.count_label.config(text="Rows: 0 (No Data Found)")
            for i in self.tree.get_children():
                self.tree.delete(i)
            self.current_df = None
            return

        df = self.clean_data(df)
        self.current_df = df.copy()

        self.tree.delete(*self.tree.get_children())
        
        # Display columns with proper names (for UI only)
        display_columns = {
            'date': 'Date',
            'voucher_type': 'Voucher Type',
            'voucher_no': 'Voucher No',
            'party_name': 'Party Name',
            'particulars': 'Particulars',
            'debit_amount': 'Debit Amount',
            'credit_amount': 'Credit Amount'
        }
        
        self.tree["columns"] = list(df.columns)
        self.tree["show"] = "headings"
        
        for col in df.columns:
            display_name = display_columns.get(col, col)
            self.tree.heading(col, text=display_name, anchor=tk.W)
            if col in ['particulars', 'party_name']:
                width = 200
            elif col == 'date':
                width = 100
            else:
                width = 120
            self.tree.column(col, width=width, minwidth=100, stretch=tk.YES)

        for _, row in df.iterrows():
            self.tree.insert("", "end", values=list(row))

        self.count_label.config(text=f"Rows: {len(df)}")
        self.save_to_database()

    def save_to_database(self):
        if self.current_df is None or self.current_df.empty:
            return

        try:
            conn_str = (
                f"postgresql://{self.db_config['user']}:{self.db_config['password']}"
                f"@{self.db_config['host']}/{self.db_config['database']}"
            )
            engine = create_engine(conn_str)
            
            # Drop table on first run
            if self.is_first_run:
                self.drop_table_if_exists(engine)
                self.is_first_run = False
            
            # Prepare data for SQL - ensure proper types
            df_to_save = self.current_df.copy()
            
            # Convert date to proper format
            if 'date' in df_to_save.columns:
                df_to_save['date'] = pd.to_datetime(df_to_save['date']).dt.date
            
            # Ensure numeric columns are float
            for col in ['debit_amount', 'credit_amount']:
                if col in df_to_save.columns:
                    df_to_save[col] = pd.to_numeric(df_to_save[col], errors='coerce').fillna(0.0)
            
            # Save with clean column names (no spaces, lowercase)
            df_to_save.to_sql(
                'tally_transactions',
                engine,
                if_exists='replace',
                index=False,
                method='multi'
            )
            
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            self.count_label.config(
                text=f"Rows: {len(self.current_df)} - Saved to DB at {current_time}"
            )
            
        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to save:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TallyLedgerViewer(root)
    root.mainloop()
    print("Execution Time:", time.time()-start)
