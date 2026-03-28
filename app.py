import streamlit as st
import requests
import base64
import pandas as pd
import re
import io

import os
from dotenv import load_dotenv

load_dotenv()

import io
import zipfile

# Helper to create ZIP of images
def create_zip(image_dict, prefix="plot"):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for col, b64 in image_dict.items():
            img_bytes = base64.b64decode(b64)
            zip_file.writestr(f"{prefix}_{col}.png", img_bytes)
    return zip_buffer.getvalue()

# Helper to create PDF from text (A4 + Support for long lines)
def create_pdf(text_content):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    import io
    
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica", 10)
    
    y = height - 50
    margin = 50
    max_width = width - (2 * margin)
    
    # Process text line by line
    for raw_line in text_content.split('\n'):
        # Basic line wrapping for extremely long AI responses
        chunk_size = 95 
        chunks = [raw_line[i:i+chunk_size] for i in range(0, len(raw_line), chunk_size)]
        
        for line in chunks:
            if y < 50:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - 50
            
            # Sanitize characters
            safe_line = "".join(i if ord(i) < 256 else ' ' for i in line)
            c.drawString(margin, y, safe_line)
            y -= 14
        
    c.save()
    return buffer.getvalue()

# Rule 3: Use environment variables for Production Ready code
API_URL = os.getenv("API_URL", "https://eda-wizard-1.onrender.com")

st.set_page_config(page_title="AI Powered EDA Wizard", layout="wide", initial_sidebar_state="collapsed")

# Inject Custom CSS for Good Background
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
        color: #1e293b;
    }
    .big-title {
        font-size: 4rem !important;
        font-weight: 800;
        text-align: center;
        background: -webkit-linear-gradient(#0f172a, #0284c7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-top: 2rem;
    }
    .sub-quote {
        font-size: 1.5rem;
        text-align: center;
        font-style: italic;
        color: #475569;
        margin-bottom: 3rem;
    }
    .success-title {
        font-size: 2.5rem;
        text-align: center;
        color: #16a34a;
        margin-top: 2rem;
        margin-bottom: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "eda_data" not in st.session_state:
    st.session_state.eda_data = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "page" not in st.session_state:
    st.session_state.page = "auth"
if "selected_type" not in st.session_state:
    st.session_state.selected_type = None
if "ml_ready" not in st.session_state:
    st.session_state.ml_ready = False
if "ml_choice" not in st.session_state:
    st.session_state.ml_choice = None
if "ml_extra_cols" not in st.session_state:
    st.session_state.ml_extra_cols = []

def validate_password(password):
    # password must contain numers and alphabets
    if not re.search(r'[A-Za-z]', password) or not re.search(r'[0-9]', password):
        return False
    return True

def login(username, password):
    try:
        resp = requests.post(f"{API_URL}/auth/login", data={"username": username, "password": password})
        if resp.status_code == 200:
            st.session_state.token = resp.json()["access_token"]
            st.session_state.username = username
            st.session_state.page = "success"
            st.rerun()
        else:
            try:
                msg = resp.json().get("detail", "Login Failed")
            except:
                msg = "Server is not responding yet. Please wait a moment."
            st.error(msg)
    except Exception as e:
        st.error(f"Connection Error: {e}. If developing locally, ensure Docker is running completely.")

def register(username, password):
    try:
        resp = requests.post(f"{API_URL}/auth/register", data={"username": username, "password": password})
        if resp.status_code == 200:
            st.success("Registration Successful! You can now log in with username and password.")
        else:
            try:
                msg = resp.json().get("detail", "Registration Failed")
            except:
                msg = resp.text
            st.error(msg)
    except Exception as e:
        st.error(f"Connection Error: {e}")

# Router
if st.session_state.page == "auth":
    st.markdown('<p class="big-title">Well Come To AI Powered EDA Wizard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-quote">"Transform raw data into Machine Learning Ready Intelligence"</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs(["Login", "Signup"])
        
        with tab1:
            st.subheader("Please enter your name and password")
            l_user = st.text_input("Name", key="l_user")
            l_pass = st.text_input("Password", type="password", key="l_pass")
            if st.button("Login", use_container_width=True):
                login(l_user, l_pass)

        with tab2:
            st.subheader("Signup to AI Powered EDA Wizard")
            r_user = st.text_input("Name", key="r_user")
            r_pass = st.text_input("Password", type="password", key="r_pass")
            r_pass_conf = st.text_input("Confirm Password", type="password", key="r_pass_conf")
            
            if st.button("Register", use_container_width=True):
                if r_pass != r_pass_conf:
                    st.error("Passwords do not match!")
                elif not validate_password(r_pass):
                    st.error("Password must contain numbers and alphabets!")
                else:
                    register(r_user, r_pass)

elif st.session_state.page == "success":
    st.markdown('<p class="success-title">Successfully you logined into AI Powered EDA Wizard 🎉</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Start", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()

elif st.session_state.page == "dashboard":
    st.sidebar.title("AI EDA Wizard")
    
    with st.sidebar.expander("👤 My Profile", expanded=True):
        st.write(f"**UserID:** {st.session_state.get('username', 'Admin')}")
        if st.button("Logout", use_container_width=True):
            st.session_state.token = None
            st.session_state.username = None
            st.session_state.eda_data = None
            st.session_state.selected_type = None
            st.session_state.chat_history = []
            st.session_state.page = "auth"
            st.rerun()

    if not st.session_state.eda_data:
        st.subheader("📤 Upload your Dataset (CSV Only)")
        st.info("Currently, this Wizard supports high-performance analysis only for CSV files.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        
        if uploaded_file:
            if st.button("🚀 Analyze Dataset", type="primary", use_container_width=True):
                # Simulated UX Progress
                progress_bar = st.progress(5, "Initiating analytical engine...")
                status_text = st.empty()
                
                try:
                    import time
                    status_text.info("🛠 Phase 1: Cleaning & Imputing Missing Values...")
                    progress_bar.progress(15)
                    
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "multipart/form-data")}
                    headers = {"Authorization": f"Bearer {st.session_state.token}"}
                    
                    # API Request
                    resp = requests.post(f"{API_URL}/eda/upload", headers=headers, files=files)
                    
                    if resp.status_code == 200:
                        status_text.info("📈 Phase 2: Analyzing Feature Distributions & Trends...")
                        progress_bar.progress(45)
                        time.sleep(0.5)
                        
                        status_text.info("🧠 Phase 3: AI Report Generation & Feature Selection...")
                        progress_bar.progress(75)
                        time.sleep(0.5)
                        
                        status_text.success("✨ Analysis Completed Successfully!")
                        progress_bar.progress(100)
                        
                        st.session_state.eda_data = resp.json()
                        st.session_state.chat_history = []
                        st.rerun()
                    else:
                        st.error(f"Error: {resp.text}")
                except Exception as e:
                    st.error(f"Communication error: {e}")
    data = st.session_state.eda_data
    if data:
        content_col, nav_col = st.columns([4, 1])
        
        with nav_col:
            st.markdown("### Navigation")
            selected_section = st.radio(
                "Select Section:",
                [
                    "Start Analysis",
                    "Categorical Univariate Analysis",
                    "Numerical Analysis",
                    "Bivariate Analysis",
                    "Multivariate Analysis",
                    "Feature Engineering",
                    "ML Recommendation & Report",
                    "AI Assistant"
                ]
            )
            
        with content_col:
            if selected_section == "Start Analysis":
                st.header("Part 1: Data Preprocessing")
                
                eda_rep = data["eda_report"]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Auto-Detected Target Feature", eda_rep.get("target_column", "Unknown"))
                with col2:
                    shape = eda_rep.get("final_shape", [0, 0])
                    st.metric("Final Dataset Shape", f"{shape[0]} Rows, {shape[1]} Columns")
                    
                st.subheader("Data Cleaning Log")
                for step in eda_rep.get("clean_steps", []):
                    st.success(step)
                    
                st.subheader("Missing Value Treatments")
                missing_df = pd.DataFrame(eda_rep.get("missing_treatment", []))
                if not missing_df.empty:
                    st.dataframe(missing_df, use_container_width=True)

            elif selected_section == "Categorical Univariate Analysis":
                st.header("Part 2: Categorical Univariate Analysis")
                cat_data = data["categorical_analysis"]

                # Tables
                st.subheader("📋 Frequency Tables")
                freq_tables = cat_data.get("freq_tables", {})
                for col, freq_dict in freq_tables.items():
                    st.markdown(f"**{col}**")
                    st.dataframe(pd.DataFrame(list(freq_dict.items()), columns=[col, "Count"]), use_container_width=True)

                # Charts
                st.subheader("📊 Bar Charts")
                bar_charts = cat_data.get("bar_charts", {})
                for col, b64 in bar_charts.items():
                    st.image(base64.b64decode(b64), caption=f"Bar: {col}", width=400)

                st.subheader("🥧 Pie Charts")
                pie_charts = cat_data.get("pie_charts", {})
                for col, b64 in pie_charts.items():
                    st.image(base64.b64decode(b64), caption=f"Pie: {col}", width=350)

                # Subplots
                if cat_data.get("bar_subplot"):
                    st.subheader("📊 Combined Bar Subplots")
                    st.image(base64.b64decode(cat_data["bar_subplot"]), width=800)
                if cat_data.get("pie_subplot"):
                    st.subheader("🥧 Combined Pie Subplots")
                    st.image(base64.b64decode(cat_data["pie_subplot"]), width=800)

                # AI Report
                st.subheader("🤖 AI Report")
                cat_report = cat_data.get("ai_report", "No AI report generated.")
                st.info(cat_report)

                # Downloads
                st.markdown("---")
                st.header("📥 MASTER EXPORTS")
                st.markdown("---")
                dl1, dl2, dl3, dl4 = st.columns(4)
                with dl1: st.download_button("📂 Download ALL Bar Charts (ZIP)", 
                    data=create_zip(bar_charts, "bar_chart"), 
                    file_name="all_bar_charts.zip", mime="application/zip", use_container_width=True)
                with dl2: st.download_button("📂 Download ALL Pie Charts (ZIP)", 
                    data=create_zip(pie_charts, "pie_chart"), 
                    file_name="all_pie_charts.zip", mime="application/zip", use_container_width=True)
                with dl3: st.download_button("🖼️ Download Subplots (PNG)", 
                    data=base64.b64decode(cat_data["bar_subplot"]) if cat_data.get("bar_subplot") else b"", 
                    file_name="categorical_subplots.png", use_container_width=True)
                with dl4: 
                    st.download_button("📄 Download AI Report (PDF)", 
                        data=create_pdf(cat_report), 
                        file_name="categorical_ai_report.pdf", mime="application/pdf", use_container_width=True)

            elif selected_section == "Numerical Analysis":
                st.header("Part 3: Numerical Analysis")
                num_data = data["numerical_analysis"]

                st.subheader("📋 Descriptive Statistics")
                st.dataframe(pd.DataFrame(num_data.get("describe_table", [])), use_container_width=True)

                st.subheader("📉 Skewness & Outliers (BEFORE)")
                st.dataframe(pd.DataFrame(num_data.get("skew_before", [])), use_container_width=True)

                st.subheader("📊 Histograms & KDE (BEFORE)")
                hists_before = num_data.get("histograms_before", {})
                for col, b64 in hists_before.items():
                    st.image(base64.b64decode(b64), caption=f"Hist: {col}", use_container_width=True)

                if num_data.get("box_before"):
                    st.subheader("📦 Boxplots (BEFORE)")
                    st.image(base64.b64decode(num_data["box_before"]), use_container_width=True)

                st.markdown("---")
                st.header("🔧 DATA CLEANING & TREATMENT")
                st.markdown("---")
                for log in num_data.get("treatment_logs", []):
                    st.success(log)

                st.subheader("📊 Histograms & KDE (AFTER TREATMENT)")
                hists_after = num_data.get("histograms_after", {})
                for col, b64 in hists_after.items():
                    st.image(base64.b64decode(b64), caption=f"Hist (After): {col}", use_container_width=True)

                st.subheader("📉 New Skewness (AFTER TREATMENT)")
                st.dataframe(pd.DataFrame(num_data.get("skew_after", [])), use_container_width=True)

                if num_data.get("box_after"):
                    st.subheader("📦 Boxplots (AFTER TREATMENT)")
                    st.image(base64.b64decode(num_data["box_after"]), use_container_width=True)

                # AI Report
                st.subheader("🤖 AI Report")
                num_report = num_data.get("ai_report", "No report generated.")
                st.info(num_report)

                # Downloads
                st.markdown("---")
                st.header("📥 DOWNLOAD OPTIONS")
                st.markdown("---")
                dl1, dl2, dl3 = st.columns(3)
                with dl1: st.download_button("1 → Download Histograms", 
                    data=base64.b64decode(num_data["hist_subplot_after"]) if num_data.get("hist_subplot_after") else b"", 
                    file_name="numerical_histograms.png", use_container_width=True)
                with dl2: st.download_button("2 → Download Boxplots", 
                    data=base64.b64decode(num_data["box_after"]) if num_data.get("box_after") else b"", 
                    file_name="numerical_boxplots.png", use_container_width=True)
                with dl3: st.download_button("3 → Download AI Report (PDF)", 
                    data=create_pdf(num_report), 
                    file_name="numerical_ai_report.pdf", mime="application/pdf", use_container_width=True)

            elif selected_section == "Bivariate Analysis":
                st.header("Part 4: Bivariate Relationships")
                biv_data = data["bivariate_analysis"]
                
                # 1. Numerical vs Numerical
                st.subheader("📊 Numerical vs Numerical Analysis")
                if biv_data.get("heatmap"):
                    st.markdown("#### Correlation Matrix Table")
                    # We can't easily extract the correlation table from the backend if it didn't return it
                    # But the backend returns heatmap. Let's show the strength logs.
                    st.image(base64.b64decode(biv_data["heatmap"]), caption="Correlation Heatmap", use_container_width=True)
                    
                    st.markdown("#### Relationship Strength")
                    for log in biv_data.get("relationship_logs", []):
                        st.write(log)
                
                # 2. Categorical vs Target
                st.markdown("---")
                st.subheader("📋 Categorical vs Target Analysis")
                cat_target = biv_data.get("cat_target_results", [])
                for res in cat_target:
                    st.markdown(f"#### {res['column']} vs {data['eda_report'].get('target_column')}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Crosstab Table**")
                        st.dataframe(pd.DataFrame(res["table"]), use_container_width=True)
                    with col2:
                        st.image(base64.b64decode(res["chart"]), use_container_width=True)

                # 3. Numerical vs Target
                st.markdown("---")
                st.subheader("📈 Numerical vs Target Analysis")
                num_target = biv_data.get("num_target_results", [])
                for res in num_target:
                    st.markdown(f"#### {res['column']} vs {data['eda_report'].get('target_column')}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Mean Values by Target Category**")
                        st.dataframe(pd.DataFrame(res["table"]), use_container_width=True)
                    with col2:
                        st.image(base64.b64decode(res["chart"]), use_container_width=True)

                # 4. AI Relationship Report
                st.markdown("---")
                st.header("🤖 AI Relationship Report")
                st.info(biv_data.get("ai_report", "No AI Relationship report generated."))

                # 5. Download Options
                st.markdown("---")
                st.header("📥 DOWNLOAD OPTIONS")
                st.markdown("---")
                dl1, dl2 = st.columns(2)
                with dl1:
                    biv_report = biv_data.get("ai_report", "No report generated.")
                    st.download_button("1 → Download AI Relationship Report (PDF)", 
                        data=create_pdf(biv_report), 
                        file_name="bivariate_ai_report.pdf", mime="application/pdf", use_container_width=True)
                with dl2:
                    st.info("Use Sidebar Navigation to move to the next part.")
                
            elif selected_section == "Multivariate Analysis":
                st.header("Part 5: Multivariate Analysis")
                multi_data = data["multivariate_analysis"]
                
                # 1. Correlation Matrix
                st.subheader("📊 Correlation Matrix (Numerical Columns)")
                st.dataframe(pd.DataFrame(multi_data.get("corr_table", [])), use_container_width=True)
                
                # 2. Heatmap
                st.subheader("📊 Correlation Heatmap")
                if multi_data.get("heatmap"):
                    st.image(base64.b64decode(multi_data["heatmap"]), use_container_width=True)
                
                # 3. Pairplot
                st.subheader("📊 PairPlot Analysis")
                if multi_data.get("pairplot"):
                    st.image(base64.b64decode(multi_data["pairplot"]), caption="Numerical Pairplot with Hue (Target)", use_container_width=True)
                
                # 4. AI Report
                st.markdown("---")
                st.header("🤖 AI MULTIVARIATE REPORT")
                st.info(multi_data.get("ai_report", "No AI Multivariate report generated."))

                # 5. Download Options
                st.markdown("---")
                st.header("📥 DOWNLOAD OPTIONS")
                st.markdown("---")
                dl1, dl2 = st.columns(2)
                with dl1:
                    multi_report = multi_data.get("ai_report", "No report generated.")
                    st.download_button("1 → Download AI Multivariate Report (PDF)", 
                        data=create_pdf(multi_report), 
                        file_name="multivariate_ai_report.pdf", mime="application/pdf", use_container_width=True)
                with dl2:
                    st.info("Navigate through all sections in the sidebar.")

            elif selected_section == "Feature Engineering":
                st.markdown("`===================================`")
                st.markdown("`FEATURE ENGINEERING`")
                st.markdown("`===================================`")
                st.write("")

                # Backend results
                fe_data = data["feature_engineering"]
                target_col = data["eda_report"].get("target_column", "Target")

                # 1 ENCODING SUMMARY
                st.write("Encoding Summary")
                st.write("")
                enc_report = fe_data.get("encoding_report", [])
                if enc_report:
                    enc_df = pd.DataFrame(enc_report)
                    enc_df.columns = ["Column Name", "Column Type", "Encoding Used"]
                    st.dataframe(enc_df, use_container_width=True)
                
                # 2 ENCODING COMPLETED
                st.write("Encoding Completed")
                st.write("")

                # 3 FEATURE IMPORTANCE
                st.write("Feature Importance Table")
                st.write("")
                fi_list = fe_data.get("feature_importance", [])
                if fi_list:
                    fi_df = pd.DataFrame(fi_list)
                    if len(fi_df.columns) == 4:
                        fi_df.columns = ["Column Name", "Importance Score", "Importance %", "Important"]
                    st.dataframe(fi_df, use_container_width=True)

                # 4 FEATURE IMPORTANCE BAR CHART
                if fe_data.get("feature_chart"):
                    st.image(base64.b64decode(fe_data["feature_chart"]), use_container_width=True)

                # 4.1 AI ASSISTANT EXPLANATION
                st.write("")
                st.write("AI Feature Engineering Assistant")
                st.info(fe_data.get("ai_report", "Generating AI explanation..."))
                st.write("")

                # 5 IMPORTANT FEATURES
                important_cols = fe_data.get("selected_columns", [])
                st.write("")
                st.write("Important Columns For ML Model")
                st.write("")
                for col in important_cols:
                    if col != target_col:
                        st.write(f"- {col}")

                # 6 USER DECISION
                st.write("")
                st.write("Do you want to use these features for ML?")
                st.write("")
                st.write("1 → OK Generate")
                st.write("2 → Add More Columns")
                
                choice_col1, choice_col2 = st.columns(2)
                with choice_col1:
                    if st.button("1 → OK Generate", use_container_width=True):
                        st.session_state.ml_choice = "1"
                        st.session_state.ml_ready = True
                with choice_col2:
                    if st.button("2 → Add More Columns", use_container_width=True):
                        st.session_state.ml_choice = "2"
                        st.session_state.ml_ready = False

                # 7 ADD MORE COLUMNS OPTION
                if st.session_state.get("ml_choice") == "2":
                    all_cols = fe_data.get("all_columns", [])
                    remaining = [c for c in all_cols if c not in important_cols and c != target_col]
                    st.write("")
                    st.write("Remaining Columns")
                    added_cols = st.multiselect("Select option:", remaining)
                    if st.button("Enough Columns Selected", use_container_width=True):
                        st.session_state.ml_extra_cols = added_cols
                        st.session_state.ml_ready = True
                        st.session_state.ml_choice = "1"

                # FINAL FEATURE LIST & DATASET
                if st.session_state.get("ml_ready", False):
                    st.write("")
                    st.write("Final Features Used For ML")
                    st.write("")
                    final_features = list(dict.fromkeys(important_cols + st.session_state.get("ml_extra_cols", [])))
                    if target_col in final_features: final_features.remove(target_col)
                    for col in final_features:
                        st.write(f"- {col}")
                    
                    st.write("")
                    st.markdown("`===================================`")
                    st.markdown("`FINAL DATASET FOR MACHINE LEARNING`")
                    st.markdown("`===================================`")
                    st.write("")
                    
                    # Compute Final Dataframe for Display
                    full_b64 = fe_data.get("all_dataset_b64", "")
                    if full_b64:
                        all_csv = base64.b64decode(full_b64).decode('utf-8')
                        all_df = pd.read_csv(io.StringIO(all_csv))
                        final_dataset = all_df[final_features + [target_col]]
                        
                        st.dataframe(final_dataset.head(10), use_container_width=True)
                        st.write(f"{len(final_dataset)} rows × {len(final_dataset.columns)} columns")

                        st.write("")
                        st.write("Do you want to download this dataset?")
                        st.write("")
                        st.write("1 → Download Dataset")
                        st.write("2 → Skip")
                        
                        dl_col1, dl_col2 = st.columns(2)
                        with dl_col1:
                            final_buffer = io.StringIO()
                            final_dataset.to_csv(final_buffer, index=False)
                            st.download_button("1 → Download Dataset", 
                                data=final_buffer.getvalue(), 
                                file_name="final_ml_dataset.csv", 
                                mime="text/csv", use_container_width=True)
                        with dl_col2:
                            if st.button("2 → Skip", use_container_width=True):
                                st.warning("Download skipped")

            elif selected_section == "ML Recommendation & Report":
                
                ml_data = data["ml_recommendation"]
                target_col = data["eda_report"].get("target_column", "Target")

                st.write(f"Dataset Type : {ml_data['ml_type']}")
                st.write(f"Problem Type : {ml_data['task']}")
                st.write("")
                st.write("Suggested Algorithms")
                for algo in ml_data["suggested_algorithms"]:
                    st.write(f"- {algo}")

                st.write("")
                st.markdown("`===================================`")
                st.markdown("` REPORT PREVIEW `")
                st.markdown("`===================================`")
                st.write("")

                # 3 CREATE REPORT TEXT
                report_text = f"""EDA WIZARD REPORT\n==============================\nDataset Information\n------------------------------\nRows : {data['eda_report']['final_shape'][0]}\nColumns : {data['eda_report']['final_shape'][1]}\n\nEDA Steps Completed\n------------------------------\n- Missing Value Analysis\n- Outlier Detection\n- Univariate Analysis\n- Bivariate Analysis\n- Encoding\n- Feature Engineering\n\nEncoding Summary\n------------------------------\n"""
                for row in data["feature_engineering"]["encoding_report"]:
                    report_text += f"{row[0]} : {row[2]}\n"
                
                report_text += "\nImportant Features\n------------------------------\n"
                for col in data["feature_engineering"]["selected_columns"]:
                    report_text += f"{col}\n"
                
                report_text += f"\nMachine Learning Recommendation\n------------------------------\nDataset Type : {ml_data['ml_type']}\nProblem Type : {ml_data['task']}\n\nSuggested Algorithms\n------------------------------\n"
                for algo in ml_data["suggested_algorithms"]:
                    report_text += f"{algo}\n"

                st.text_area("Final Report Content", report_text, height=300)

                # 3 DOWNLOAD PDF
                st.write("")
                st.write("Do you want to download the full EDA report?")
                st.write("")
                st.write("1 - Download Final Report (PDF)")
                
                pdf_col1, pdf_col2 = st.columns(2)
                with pdf_col1:
                    st.download_button("1 - Download Final Report (PDF)", 
                        data=create_pdf(report_text), 
                        file_name="EDA_Wizard_Final_Report.pdf", 
                        mime="application/pdf", use_container_width=True)
                with pdf_col2:
                    st.info("Complete! You can explore other sections in the sidebar.")

            elif selected_section == "AI Assistant":
                st.markdown("`===================================`")
                st.markdown("`EDA AI ASSISTANT`")
                st.markdown("`===================================`")
                st.write("")
                
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                
                if prompt_input := st.chat_input("Ask question about your dataset"):
                    st.session_state.chat_history.append({"role": "user", "content": prompt_input})
                    with st.chat_message("user"): st.markdown(prompt_input)
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            headers = {"Authorization": f"Bearer {st.session_state.token}"}
                            try:
                                resp = requests.post(f"{API_URL}/eda/chat", headers=headers, json={"question": prompt_input, "eda_summary": data["eda_summary"]})
                                if resp.status_code == 200:
                                    ans = resp.json()["answer"]
                                    st.markdown(ans)
                                    st.session_state.chat_history.append({"role": "assistant", "content": ans})
                                else: st.error("Error communicating with AI.")
                            except Exception as e: st.error(f"Error: {e}")

            elif selected_section == "Project Logs":
                st.markdown("`===================================`")
                st.markdown("`PROJECT AUDIT LOGS (SQLITE)`")
                st.markdown("`===================================`")
                st.write("")
                
                headers = {"Authorization": f"Bearer {st.session_state.token}"}
                try:
                    resp = requests.get(f"{API_URL}/logs", headers=headers)
                    if resp.status_code == 200:
                        logs = resp.json()
                        if logs:
                            log_df = pd.DataFrame(logs)
                            log_df = log_df[["timestamp", "filename", "action"]]
                            st.table(log_df)
                        else:
                            st.info("No project logs found for your account.")
                    else:
                        st.error("Failed to fetch logs from database.")
                except Exception as e:
                    st.error(f"Database error: {e}")

