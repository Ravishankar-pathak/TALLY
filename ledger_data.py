

import time
import tkinter as tk
from tkinter import ttk, messagebox
import requests
import pandas as pd
from io import StringIO
import re
import xml.etree.ElementTree as ET
import logging
from sqlalchemy import create_engine, exc
import datetime


start=time.time()
class TallyLedgerViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Tally Ledger Viewer")
        self.root.geometry("1100x650")
        self.current_df = None  # Store fetched data
        self.auto_fetch_running = False  # Auto-fetch status

        # Configure logging
        logging.basicConfig(level=logging.ERROR)
        
        # Database connection parameters
        self.db_config = {
            'host': 'localhost',
            'database': 'tall_ydata',
            'user': 'postgres',
            'password': 'mypassword'
        }
        
        # — Tally Connection Input —
        frame = tk.Frame(root)
        frame.pack(pady=10)

        tk.Label(frame, text="Tally Host:").grid(row=0, column=0, padx=5)
        self.host_entry = tk.Entry(frame, width=15)
        self.host_entry.insert(0, "192.168.1.150")
        self.host_entry.grid(row=0, column=1)  # FIXED

        tk.Label(frame, text="Port:").grid(row=0, column=2, padx=5)
        self.port_entry = tk.Entry(frame, width=8)
        self.port_entry.insert(0, "9000")
        self.port_entry.grid(row=0, column=3)  # FIXED

        fetch_btn = tk.Button(
            frame,
            text="Fetch Ledger Data",
            command=self.load_data,
            bg="green",
            fg="white",
            font=('Arial', 12)
        )
        fetch_btn.grid(row=0, column=4, padx=10)

        save_btn = tk.Button(
            frame,
            text="Save to Database",
            command=self.save_to_database,
            bg="blue",
            fg="white",
            font=('Arial', 12)
        )
        save_btn.grid(row=0, column=5, padx=10)
        
        # Auto-fetch toggle button
        self.auto_fetch_btn = tk.Button(
            frame,
            text="Start Auto-Fetch",
            command=self.toggle_auto_fetch,
            bg="orange",
            fg="black",
            font=('Arial', 12)
        )
        self.auto_fetch_btn.grid(row=0, column=6, padx=10)

        # — Treeview Table Display —
        self.tree = ttk.Treeview(root)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Configure scrollbar
        scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side='right', fill='y')
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.count_label = tk.Label(root, text="Rows: 0", font=('Arial', 12), anchor='w')
        self.count_label.pack(fill=tk.X, padx=10)
        
        # Start auto-fetch status
        self.auto_fetch_status = tk.Label(
            root, 
            text="Auto-Fetch: Stopped",
            font=('Arial', 10),
            fg="red",
            anchor='w'
        )
        self.auto_fetch_status.pack(fill=tk.X, padx=10, pady=(0, 5))

    def toggle_auto_fetch(self):
        """Toggle auto-fetch on/off"""
        self.auto_fetch_running = not self.auto_fetch_running
        
        if self.auto_fetch_running:
            self.auto_fetch_btn.config(text="Stop Auto-Fetch", bg="red")
            self.auto_fetch_status.config(text="Auto-Fetch: Running (60s interval)", fg="green")
            self.auto_fetch()  # Start the auto-fetch cycle
        else:
            self.auto_fetch_btn.config(text="Start Auto-Fetch", bg="orange")
            self.auto_fetch_status.config(text="Auto-Fetch: Stopped", fg="red")

    def auto_fetch(self):
        """Automatically fetch data every minute"""
        if self.auto_fetch_running:
            try:
                self.load_data()
            except Exception as e:
                logging.error(f"Auto-fetch error: {e}")
            finally:
                # Schedule next run in 60 seconds
                self.root.after(60000, self.auto_fetch)

    def clean_xml(self, xml_text: str) -> str:
        """Enhanced XML cleaning to handle invalid characters and character references"""
        # Remove invalid character references
        cleaned = re.sub(r'&#(?:[0-9]+|[xX][0-9a-fA-F]+);?', '', xml_text)
        
        # Remove control characters except whitespace
        return ''.join(
            c for c in cleaned
            if ord(c) >= 32 or c in {'\t', '\n', '\r'}
        )

    def fetch_ledger_data(self, url: str) -> pd.DataFrame:
        TALLY_REQUEST = """
        <ENVELOPE>
         <HEADER>
          <VERSION>1</VERSION>
          <TALLYREQUEST>Export Data</TALLYREQUEST>
          <TYPE>Collection</TYPE>
          <ID>Ledger Collection</ID>
         </HEADER>
         <BODY>
          <DESC>
           <STATICVARIABLES>
            <SVEXPORTFORMAT>$$SysName:XML</SVEXPORTFORMAT>
           </STATICVARIABLES>
           <TDL>
            <TDLMESSAGE>
             <COLLECTION NAME="Ledger Collection" ISMODIFY="No">
              <TYPE>Ledger</TYPE>
              <FETCH>NAME, PARENT, OPENINGBALANCE, CLOSINGBALANCE, ALTEREDON</FETCH>
             </COLLECTION>
            </TDLMESSAGE>
           </TDL>
          </DESC>
         </BODY>
        </ENVELOPE>
        """

        try:
            resp = requests.post(
                url, 
                data=TALLY_REQUEST.encode('utf-8'),
                headers={'Content-Type': 'application/xml'},
                timeout=15
            )
            resp.raise_for_status()
            
            # Enhanced cleaning pipeline
            cleaned = self.clean_xml(resp.text)
            
            # Try parsing with pandas first
            try:
                df = pd.read_xml(StringIO(cleaned), xpath=".//LEDGER")
            except Exception as pd_error:
                # Fallback to manual parsing
                try:
                    root = ET.fromstring(cleaned)
                    ledgers = []
                    for ledger in root.findall('.//LEDGER'):
                        data = {}
                        for elem in ledger:
                            tag = elem.tag
                            if tag in ['NAME', 'PARENT', 'OPENINGBALANCE', 
                                      'CLOSINGBALANCE', 'ALTEREDON']:
                                data[tag.lower()] = elem.text
                        ledgers.append(data)
                    
                    df = pd.DataFrame(ledgers)
                except ET.ParseError as parse_error:
                    # Detailed error logging
                    error_line = parse_error.position[0] if hasattr(parse_error, 'position') else 'unknown'
                    error_msg = f"XML Parse Error (line {error_line}): {str(parse_error)}"
                    messagebox.showerror("XML Error", error_msg)
                    return pd.DataFrame()
            
            # If we got a DataFrame, standardize column names
            if not df.empty:
                # Create consistent column names regardless of source
                column_mapping = {
                    'NAME': 'ledger_name',
                    'name': 'ledger_name',
                    'PARENT': 'parent',
                    'parent': 'parent',
                    'OPENINGBALANCE': 'opening_balance',
                    'openingbalance': 'opening_balance',
                    'CLOSINGBALANCE': 'closing_balance',
                    'closingbalance': 'closing_balance',
                    'ALTEREDON': 'altered_on',
                    'alteredon': 'altered_on'
                }
                
                # Rename columns using the mapping
                df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
                
                # Ensure we have all expected columns
                expected_columns = ['ledger_name', 'parent', 'opening_balance', 'closing_balance', 'altered_on']
                for col in expected_columns:
                    if col not in df.columns:
                        df[col] = None  
                
                return df[expected_columns]
            
            return df
            
        except requests.RequestException as e:
            messagebox.showerror("Connection Error", f"Failed to connect to Tally:\n{e}")
            return pd.DataFrame()
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error:\n{str(e)}")
            return pd.DataFrame()

    def clean_data(self, df):
        """Clean and transform data before saving"""
        # Convert numeric columns
        for col in ['opening_balance', 'closing_balance']:
            # Handle empty strings and non-numeric values
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # Convert date column
        if 'altered_on' in df.columns:
            # Extract date part (YYYYMMDD) from strings like "20230119.0"
            df['altered_on'] = df['altered_on'].astype(str).str.split('.').str[0]
            
            # Convert to datetime and then to date
            df['altered_on'] = pd.to_datetime(
                df['altered_on'], 
                format='%Y%m%d', 
                errors='coerce'
            ).dt.date
        
        return df
    
    def load_data(self):
        host = self.host_entry.get().strip()
        port = self.port_entry.get().strip()
        if not host or not port:
            messagebox.showwarning("Input Error", "Please enter both Host and Port.")
            return

        url = f"http://{host}:{port}"
        df = self.fetch_ledger_data(url)
        if df.empty:
            self.count_label.config(text="Rows: 0")
            # Clear existing data
            for i in self.tree.get_children():
                self.tree.delete(i)
            self.current_df = None
            return

        # Clean data
        df = self.clean_data(df)
        self.current_df = df.copy()

        # Clear old data and configure columns
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = list(df.columns)
        self.tree["show"] = "headings"
        
        # Configure columns
        for col in df.columns:
            self.tree.heading(col, text=col, anchor=tk.W)
            self.tree.column(col, width=150, minwidth=100, stretch=tk.YES)

        # Insert new data
        for _, row in df.iterrows():
            self.tree.insert("", "end", values=list(row))

        self.count_label.config(text=f"Rows: {len(df)}")
        
        # Auto-save after fetching
        self.save_to_database()

    def save_to_database(self):
        if self.current_df is None or self.current_df.empty:
            messagebox.showwarning("No Data", "No data to save. Fetch data first.")
            return

        try:
            # Create database connection string
            conn_str = (
                f"postgresql://{self.db_config['user']}:{self.db_config['password']}"
                f"@{self.db_config['host']}/{self.db_config['database']}"
            )
            
            # Create SQLAlchemy engine
            engine = create_engine(conn_str)
            
            # Save to PostgreSQL
            self.current_df.to_sql(
                'tally_data',
                engine,
                if_exists='replace',  # Overwrite existing data
                index=False,
                method='multi'  # Faster insertion
            )
            
            # Get current time for status message
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            self.count_label.config(
                text=f"Rows: {len(self.current_df)} - Saved to database at {current_time}"
            )
            
        except exc.SQLAlchemyError as e:
            messagebox.showerror("Database Error", f"Failed to save data to database:\n{str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error while saving:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = TallyLedgerViewer(root)
    root.mainloop()




print("Execution Time:", time.time()-start)

