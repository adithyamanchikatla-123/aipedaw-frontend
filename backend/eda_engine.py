import pandas as pd
import numpy as np
import io
import sqlite3
import os
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from backend.groq_integration import MODEL, CHAT_MODEL

def generate_base64_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_str

def analyze_and_clean_data(df: pd.DataFrame, drop_high_null: bool = False):
    report = {
        "original_shape": list(df.shape),
        "clean_steps": []
    }
    
    # 1. Clean Column Names
    df.columns = df.columns.str.strip()

    # 2. Duplicate Row Check
    duplicates = int(df.duplicated().sum())
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        report["clean_steps"].append(f"Dropped {duplicates} duplicate rows")
    else:
        report["clean_steps"].append("No duplicate rows found")
    
    # 3. Duplicate Column Check
    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if len(duplicate_columns) > 0:
        df = df.loc[:, ~df.columns.duplicated()]
        report["clean_steps"].append(f"Dropped {len(duplicate_columns)} duplicate columns: {duplicate_columns}")
    else:
        report["clean_steps"].append("No duplicate columns found")

    # 4. Detect ID Columns
    id_cols = []
    for col in df.columns:
        col_data = df[col].astype(str)
        if col_data.str.contains('[A-Za-z]').any() and col_data.str.contains('[0-9]').any():
            if df[col].nunique() == df.shape[0]:
                id_cols.append(col)
    if id_cols:
        df.drop(columns=id_cols, inplace=True)
        report["clean_steps"].append(f"Dropped ID columns: {id_cols}")

    # 5. Column Type Detection (Optimized Batch AI Prompt)
    numerical_cols = []
    categorical_cols = []
    nominal_cols = []
    ordinal_cols = []
    
    col_info = []
    for col in df.columns:
        if df[col].dtype != "object":
            if set(df[col].dropna().unique()).issubset({0,1}):
                categorical_cols.append(col)
                nominal_cols.append(col)
            else:
                numerical_cols.append(col)
        else:
            categorical_cols.append(col)
            col_info.append({"name": col, "sample": df[col].dropna().unique().tolist()[:10]})

    if col_info:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        prompt = f"""
Analyze these columns and classify them as NOMINAL or ORDINAL.
Data: {col_info}
Rules:
1. ORDINAL: Values have logical order (e.g. Low < Medium < High).
2. NOMINAL: Values have no order (e.g. Gender, City).
Return ONLY a JSON list of strings in the exact same order as input: ["NOMINAL", "ORDINAL", ...]
"""
        try:
            res = client.chat.completions.create(
                messages=[{"role":"user","content":prompt}],
                model=MODEL,
                temperature=0
            )
            import json
            import re
            ans_text = res.choices[0].message.content
            # Extract list from response
            match = re.search(r"\[.*\]", ans_text.replace("\n", ""), re.DOTALL)
            if match:
                ans_list = json.loads(match.group(0))
                for i, col_obj in enumerate(col_info):
                    if i < len(ans_list) and "ordinal" in ans_list[i].lower():
                        ordinal_cols.append(col_obj["name"])
                    else:
                        nominal_cols.append(col_obj["name"])
            else:
                for col_obj in col_info: nominal_cols.append(col_obj["name"])
        except Exception as e:
            print(f"Batch detection error: {e}")
            for col_obj in col_info: nominal_cols.append(col_obj["name"])

    # 6. Target Column Detection

    # 6. Target Column Detection
    target_prompt = f"Dataset Columns:{list(df.columns)}\nWhich column is most likely the TARGET variable for prediction?\nReturn ONLY the column name."
    try:
        res = client.chat.completions.create(
            messages=[{"role":"user","content":target_prompt}],
            model=MODEL,
            temperature=0
        )
        target_col = res.choices[0].message.content.strip().split()[-1]
    except:
        target_col = df.columns[-1]

    # Clean target col from string matching noise
    target_col = target_col.replace('`',MODEL).replace('*',MODEL).strip()

    # Remove target col from groups
    for group in [numerical_cols, categorical_cols, nominal_cols, ordinal_cols]:
        if target_col in group:
            group.remove(target_col)

    report["target_column"] = target_col
    report["column_types"] = {
        "Numerical": numerical_cols,
        "Categorical": categorical_cols,
        "Nominal": nominal_cols,
        "Ordinal": ordinal_cols
    }

    # 7. Missing Value Handling
    missing_report = []
    for col in df.columns.tolist():
        if col not in df.columns: continue  # Skip if already dropped
        null_count = int(df[col].isnull().sum())
        null_percent = (null_count/len(df))*100
        method = "No Missing Values"
        
        if null_count > 0:
            if null_percent > 20:
                if drop_high_null:
                    df.drop(columns=[col], inplace=True)
                    method = "Column Dropped"
                else:
                    null_mask = df[col].isnull()
                    vals = df[col].dropna()
                    if len(vals) > 0:
                        probs = vals.value_counts(normalize=True)
                        df.loc[null_mask, col] = np.random.choice(probs.index, size=null_mask.sum(), p=probs.values)
                        method = "Probabilistic Imputation"
                    else:
                        df.drop(columns=[col], inplace=True)
                        method = "Column Dropped (all null)"
            elif null_percent >= 5:
                null_mask = df[col].isnull()
                vals = df[col].dropna()
                if len(vals) > 0:
                    probs = vals.value_counts(normalize=True)
                    df.loc[null_mask, col] = np.random.choice(probs.index, size=null_mask.sum(), p=probs.values)
                    method = "Probabilistic Imputation"
            else:
                if col in numerical_cols:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    method = "Median Imputation"
                else:
                    modes = df[col].mode()
                    if not modes.empty:
                        df[col] = df[col].fillna(modes[0])
                        method = "Mode Imputation"
                    else:
                        df.drop(columns=[col], inplace=True)
                        method = "Column Dropped (no mode)"
                    
        missing_report.append({
            "column_name": col,
            "null_count": null_count,
            "null_percentage": round(null_percent, 2),
            "imputation_method": method
        })

    report["missing_treatment"] = missing_report
    report["final_shape"] = list(df.shape)
    
    return report, df

def generate_base64_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return image_base64

def run_categorical_univariate(df: pd.DataFrame, categorical_cols: list):
    freq_tables = {}
    bar_charts = {}
    pie_charts = {}
    summary_text = ""
    
    # PERFORMANCE LIMIT: Process only top 10 categorical columns for visuals
    plot_cols = categorical_cols[:10]
    
    if not categorical_cols:
        return {
            "freq_tables": {}, "ai_report": "No categorical columns found.",
            "bar_charts": {}, "pie_charts": {}, "bar_subplot": "", "pie_subplot": ""
        }

    for col in categorical_cols:
        counts = df[col].value_counts()
        counts_dict = {str(k): int(v) for k, v in counts.items()}
        freq_tables[col] = counts_dict
        if col in plot_cols:
            summary_text += f"\nColumn: {col} (Top 10 Categories):\n"
            for val, count in list(counts_dict.items())[:10]:
                summary_text += f"{val} : {count}\n"

    # AI Report
    api_key = os.getenv("GROQ_API_KEY")
    ai_report = "Analysis is currently unavailable (AI Failure)."
    if api_key:
        try:
            client = Groq(api_key=api_key)
            prompt = f"Analyze these categorical distributions and give a brief summary in SIMPLE ENGLISH. Focus on trends and imbalance:\n{summary_text}"
            chat = client.chat.completions.create(
                model=MODEL,
                messages=[{"role":"user", "content": prompt}],
                temperature=0.7
            )
            ai_report = chat.choices[0].message.content
        except Exception as e:
            print(f"CATEGORICAL AI ERROR: {str(e)}")
            ai_report = "AI Report failed: Please check data volume or API key."
    else:
        ai_report = "Groq API key missing in environment."

    # Plots (Limited to 10)
    for col in plot_cols:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        df[col].value_counts().plot(kind="bar", ax=ax1, color='cornflowerblue')
        bar_charts[col] = generate_base64_img(fig1)
        
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        df[col].value_counts().plot(kind="pie", ax=ax2)
        pie_charts[col] = generate_base64_img(fig2)

    # Subplots (Limited to 10)
    cols_per_row = 3
    num_plots = len(plot_cols)
    rows = (num_plots + cols_per_row - 1) // cols_per_row
    
    bar_subplot_b64 = ""
    if num_plots > 0:
        fig_bar, axs_bar = plt.subplots(rows, cols_per_row, figsize=(4*cols_per_row, 3*rows), squeeze=False)
        axs_bar = axs_bar.flatten()
        for i, col in enumerate(plot_cols):
            df[col].value_counts().plot(kind="bar", ax=axs_bar[i])
        for j in range(i+1, len(axs_bar)): axs_bar[j].axis("off")
        bar_subplot_b64 = generate_base64_img(fig_bar)

    pie_subplot_b64 = ""
    if num_plots > 0:
        fig_pie, axs_pie = plt.subplots(rows, cols_per_row, figsize=(4*cols_per_row, 3*rows), squeeze=False)
        axs_pie = axs_pie.flatten()
        for i, col in enumerate(plot_cols):
            df[col].value_counts().plot(kind="pie", ax=axs_pie[i])
        for j in range(i+1, len(axs_pie)): axs_pie[j].axis("off")
        pie_subplot_b64 = generate_base64_img(fig_pie)

    return {
        "freq_tables": freq_tables, "ai_report": ai_report,
        "bar_charts": bar_charts, "pie_charts": pie_charts,
        "bar_subplot": bar_subplot_b64, "pie_subplot": pie_subplot_b64
    }

def run_numerical_univariate(df: pd.DataFrame, numerical_cols: list):
    # Performance Limit
    plot_cols = numerical_cols[:10]
    
    # 1. Describe table
    describe_df = df[numerical_cols].describe().T
    describe_dict = describe_df.round(3).reset_index().rename(columns={"index": "Column"}).replace({np.nan: None}).to_dict('records')
    
    # 2. Skewness + Outlier Table Before
    skew_data = []
    for col in numerical_cols:
        data = df[col].dropna()
        if len(data) == 0: continue
        skew_val = data.skew()
        if pd.isna(skew_val): skew_val = 0.0
            
        Q1, Q3 = np.percentile(data, [25, 75])
        IQR = Q3 - Q1
        outliers = int(((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum())
        
        skew_data.append({
            "column_name": col, "skewness_value": round(skew_val, 3), "outlier_count": outliers
        })
        
    # 3. Histograms Before (Limited to 10)
    histograms_before = {}
    for col in plot_cols:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="cornflowerblue")
        histograms_before[col] = generate_base64_img(fig)

    # 4. Boxplot Before (Limited to 10)
    box_before_b64 = ""
    if plot_cols:
        cols_per_row = 3
        rows = (len(plot_cols) + cols_per_row - 1) // cols_per_row
        fig_box, axs_box = plt.subplots(rows, cols_per_row, figsize=(6*cols_per_row, 4*rows), squeeze=False)
        axs_box = axs_box.flatten()
        for i, col in enumerate(plot_cols):
            axs_box[i].boxplot(df[col].dropna())
            axs_box[i].set_title(col)
        for j in range(i+1, len(axs_box)): axs_box[j].axis("off")
        box_before_b64 = generate_base64_img(fig_box)

    # 5. Treatment (Imputation + Outlier/Skewness)
    treatment_logs = []
    for col in numerical_cols:
        # Handle Missing Values (Imputation)
        null_count = df[col].isnull().sum()
        if null_count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            treatment_logs.append(f"Filled {null_count} missing values in {col} with median ({round(median_val, 2)})")
            
        # Handle Skewness and Outliers
        skew_val = df[col].skew()
        if pd.isna(skew_val): continue
        if abs(skew_val) > 1:
            treatment_logs.append(f"Treated skewness in {col} (Skew: {round(skew_val, 2)})")
            # Log Transformation for high skew, clipping for moderate
            if abs(skew_val) > 3:
                # Add a small constant to avoid log(0)
                df[col] = np.log1p(df[col].clip(lower=0))
            else:
                Q1, Q3 = np.percentile(df[col], [25, 75])
                IQR = Q3 - Q1
                df[col] = np.clip(df[col], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    # 6. Final Outputs (Limited to 10)
    after_skew = [{"column_name": c, "skewness_after": round(df[c].skew(), 3)} for c in plot_cols]
    histograms_after = {}
    for col in plot_cols:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="green")
        histograms_after[col] = generate_base64_img(fig)

    box_after_b64 = ""
    if plot_cols:
        fig_box2, axs_box2 = plt.subplots(rows, cols_per_row, figsize=(6*cols_per_row, 4*rows), squeeze=False)
        axs_box2 = axs_box2.flatten()
        for i, col in enumerate(plot_cols):
            axs_box2[i].boxplot(df[col].dropna()); axs_box2[i].set_title(col)
        for j in range(i+1, len(axs_box2)): axs_box2[j].axis("off")
        box_after_b64 = generate_base64_img(fig_box2)

    # 7. AI REPORT GENERATION
    api_key = os.getenv("GROQ_API_KEY")
    ai_report = "AI Report currently unavailable (Numerical Analysis)."
    if api_key:
        try:
            client = Groq(api_key=api_key)
            prompt = f"""
            Analyze these descriptive statistics and skewness levels for numerical columns:
            Describe Table: {describe_dict[:10]} (Top 10)
            Skewness Before: {skew_data[:10]}
            Skewness After: {after_skew[:10]}
            
            Summarize:
            1. Distribution of data (Normal vs Skewed).
            2. Outlier impact.
            3. Effectiveness of the data treatment performed.
            Focus on insights useful for Machine Learning.
            """
            chat = client.chat.completions.create(
                model=MODEL,
                messages=[{"role":"user","content":prompt}],
                temperature=0.7
            )
            ai_report = chat.choices[0].message.content
        except Exception as e:
            ai_report = f"AI Report failed: {str(e)}"

    return {
        "describe_table": describe_dict, "skew_before": skew_data,
        "histograms_before": histograms_before, "box_before": box_before_b64,
        "treatment_logs": treatment_logs, "skew_after": after_skew,
        "histograms_after": histograms_after, "box_after": box_after_b64,
        "ai_report": ai_report
    }

def run_bivariate_analysis(df: pd.DataFrame, numerical_cols: list, categorical_cols: list, target_col: str):
    # Check if target is in dataframe
    if target_col not in df.columns:
        return {"error": "Target column not found"}

    # 1. Numerical vs Numerical
    relation_summary = ""
    heatmap_b64 = ""
    relationship_logs = []
    
    if len(numerical_cols) >= 2:
        correlation_matrix = df[numerical_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax
        )
        ax.set_title("Correlation Heatmap")
        fig.tight_layout()
        heatmap_b64 = generate_base64_img(fig)
        
        for i in range(len(numerical_cols)):
            for j in range(i+1, len(numerical_cols)):
                col1 = numerical_cols[i]
                col2 = numerical_cols[j]
                corr_val = correlation_matrix.loc[col1, col2]
                if pd.isna(corr_val):
                    continue
                if abs(corr_val) > 0.7:
                    relation = "Strong Relationship"
                elif abs(corr_val) > 0.4:
                    relation = "Moderate Relationship"
                else:
                    relation = "Weak Relationship"
                log = f"{col1} vs {col2} â†’ {round(corr_val,2)} ({relation})"
                relationship_logs.append(log)
                relation_summary += f"{col1} vs {col2} : {round(corr_val,2)} ({relation})\n"

    # 2. Categorical vs Target
    cat_target_summary = ""
    cat_target_results = []
    
    for col in categorical_cols:
        if col == target_col:
            continue
        try:
            table = pd.crosstab(df[col], df[target_col])
            
            # Crosstab Table for JSON
            table_dict = table.reset_index().replace({np.nan: None}).to_dict('records')
            
            fig, ax = plt.subplots(figsize=(6, 4))
            table.plot(kind="bar", stacked=True, ax=ax, cmap="Set2")
            ax.set_title(f"{col} vs {target_col}")
            ax.set_ylabel("Count")
            fig.tight_layout()
            chart_b64 = generate_base64_img(fig)
            
            cat_target_results.append({
                "column": col,
                "table": table_dict,
                "chart": chart_b64
            })
            
            cat_target_summary += f"\n{col} vs {target_col}\n"
            cat_target_summary += table.to_string()
        except:
            pass

    # 3. Numerical vs Target
    num_target_summary = ""
    num_target_results = []
    
    for col in numerical_cols:
        if col == target_col:
            continue
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=df[target_col], y=df[col], ax=ax, palette="Set3")
            ax.set_title(f"{col} vs {target_col}")
            fig.tight_layout()
            chart_b64 = generate_base64_img(fig)
            
            means = df.groupby(target_col)[col].mean()
            table_dict = means.to_frame("Mean Value").reset_index().replace({np.nan: None}).to_dict('records')
            
            num_target_results.append({
                "column": col,
                "table": table_dict,
                "chart": chart_b64
            })
            
            num_target_summary += f"\n{col} vs {target_col}\n"
            num_target_summary += means.to_frame("Mean Value").to_string()
        except:
            pass

    # 4. AI REPORT GENERATION
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and api_key != "your_groq_api_key_here":
        client = Groq(api_key=api_key)
        ai_prompt = f"""
    You are a professional data scientist. Analyze these relationships and generate a report.
    Format your response exactly as follows:
    1. Numerical vs Numerical Summary
    2. Categorical vs Target Summary
    3. Numerical vs Target Summary
    4. **Suggested Important Features for Machine Learning:** (Provide a numbered list)

    Correlation Data:
    {relation_summary}
    Categorical vs Target:
    {cat_target_summary}
    Numerical vs Target:
    {num_target_summary}
    
    Explain everything in simple English. 
    At the end, explicitly list the features that appear most important.
    """
        ai_report = "Unable to generate Bivariate AI Report."
        try:
            chat = client.chat.completions.create(
                model=MODEL,
                messages=[{"role":"user","content":ai_prompt}],
                temperature=0.7
            )
            ai_report = chat.choices[0].message.content
        except Exception as e:
            ai_report = f"Could not generate report: {str(e)}"
    else:
        ai_report = "Groq API key not configured properly."

    return {
        "heatmap": heatmap_b64,
        "relationship_logs": relationship_logs,
        "cat_target_results": cat_target_results,
        "num_target_results": num_target_results,
        "ai_report": ai_report
    }

def run_multivariate_analysis(df: pd.DataFrame, numerical_cols: list, target_col: str):
    # 1. Correlation Matrix Heatmap
    heatmap_b64 = ""
    relation_summary = ""
    if len(numerical_cols) >= 2:
        corr_matrix = df[numerical_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax
        )
        ax.set_title("Correlation Heatmap")
        fig.tight_layout()
        heatmap_b64 = generate_base64_img(fig)
        
        for i in range(len(numerical_cols)):
            for j in range(i+1, len(numerical_cols)):
                col1 = numerical_cols[i]
                col2 = numerical_cols[j]
                corr_val = corr_matrix.loc[col1, col2]
                if not pd.isna(corr_val):
                    relation_summary += f"{col1} and {col2} correlation is {round(corr_val,2)}\n"

    # 2. Pairplot (OPTIMIZED: Only Top 6 Correlation Columns + Max 1000 Rows)
    pairplot_b64 = ""
    if target_col in df.columns:
        # Sort numerical columns by correlation with target (Only if Target is numeric)
        try:
            temp_df = df[numerical_cols + [target_col]]
            corr_matrix_full = temp_df.corr()
            if target_col in corr_matrix_full.columns:
                target_corr = corr_matrix_full[target_col].abs().sort_values(ascending=False)
                top_correlated = target_corr.index[1:7].tolist() # Top 6 excluding target itself
            else:
                top_correlated = numerical_cols[:6] # Fallback to first 6 numerical
        except Exception:
            top_correlated = numerical_cols[:6]
        
        plot_cols = top_correlated + [target_col]
        plot_cols = list(dict.fromkeys(plot_cols))
        
        # Sample for speed
        clean_df = df[plot_cols].dropna().sample(n=min(len(df), 1000), random_state=42)
        
        if len(clean_df) > 0:
            try:
                g = sns.pairplot(clean_df, hue=target_col, diag_kind='kde')
                pairplot_b64 = generate_base64_img(g.figure)
            except Exception:
                pass

    # AI MULTIVARIATE Insight
    api_key = os.getenv("GROQ_API_KEY")
    ai_report = "Groq API key not configured properly."
    if api_key and api_key != "your_groq_api_key_here":
        client = Groq(api_key=api_key)
        ai_prompt = f"""
    Explain the multivariate relationships and the pairplot analysis in simple English.
    Focus on:
    1. Strong correlations between numerical columns.
    2. How the distribution looks in the pairplot (clusters, separation by target).
    3. Key multivariate patterns that could help a machine learning model.
    4. Important insights for building the model.

    Relationships Data:
    {relation_summary}
    """
        try:
            chat = client.chat.completions.create(
                model=MODEL,
                messages=[{"role":"user","content":ai_prompt}],
                temperature=0.7
            )
            ai_report = chat.choices[0].message.content
        except Exception as e:
            ai_report = f"Could not generate report: {str(e)}"

    return {
        "corr_table": df[numerical_cols].corr().round(3).reset_index().replace({np.nan: None}).to_dict('records') if len(numerical_cols) >= 2 else [],
        "heatmap": heatmap_b64 if heatmap_b64 else "",
        "pairplot": pairplot_b64 if pairplot_b64 else "",
        "ai_report": ai_report
    }

def run_feature_engineering_and_selection(df: pd.DataFrame, nominal_cols: list, ordinal_cols: list, target_col: str):
    if target_col not in df.columns:
        return {"error": "Target column not found"}
        
    # 1. ENCODING LOGIC (Exactly as requested by user)
    dataset = df.copy()
    encoding_report = []

    # Detect categorical columns if not provided
    for col in nominal_cols:
        if col in dataset.columns:
            if dataset[col].nunique() == 2:
                encoding_report.append([col, "Binary Nominal", "Label Encoding"])
            else:
                encoding_report.append([col, "Nominal", "Label Encoding"])
    for col in ordinal_cols:
        if col in dataset.columns:
            encoding_report.append([col, "Ordinal", "Ordinal Encoding"])

    # 2. APPLY ENCODING (Snippet logic)
    for col in nominal_cols:
        if col in dataset.columns:
            le = LabelEncoder()
            dataset[col] = le.fit_transform(dataset[col].astype(str))
    
    if len(ordinal_cols) > 0:
        oe = OrdinalEncoder()
        cols_to_encode = [c for c in ordinal_cols if c in dataset.columns]
        if cols_to_encode:
            dataset[cols_to_encode] = oe.fit_transform(dataset[cols_to_encode].astype(str))
            
    if target_col in dataset.columns and (dataset[target_col].dtype == "object" or not pd.api.types.is_numeric_dtype(dataset[target_col])):
        le = LabelEncoder()
        dataset[target_col] = le.fit_transform(dataset[target_col].astype(str))
        # Note: We don't add target to encoding_report here as it's handled in importance

    # 3. FEATURE IMPORTANCE (Ensuring X consists only of numbers)
    X = dataset.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.median())
    
    y = dataset[target_col].fillna(0)
    
    model = RandomForestClassifier(random_state=42)
    feature_b64 = ""
    feature_importance_list = []
    
    try:
        model.fit(X, y)
        importance = model.feature_importances_
        feature_df = pd.DataFrame({
            "Column Name": X.columns,
            "Importance Score": importance
        })
        feature_df["Importance %"] = (feature_df["Importance Score"] * 100).round(2)
        feature_df["Important"] = feature_df["Importance %"].apply(lambda x: "Yes" if x > 5 else "No")
        feature_df = feature_df.sort_values(by="Importance %", ascending=False)
        
        feature_importance_list = feature_df.replace({np.nan: None}).values.tolist()
        
        # 4. CHART
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(feature_df["Column Name"], feature_df["Importance %"])
        ax.set_xticklabels(feature_df["Column Name"], rotation=45, ha='right')
        ax.set_ylabel("Importance %")
        ax.set_title("Feature Importance")
        fig.tight_layout()
        feature_b64 = generate_base64_img(fig)
        
        important_cols = feature_df[feature_df["Important"] == "Yes"]["Column Name"].tolist()
    except Exception as e:
        important_cols = list(X.columns)
        feature_importance_list = [["Error",0,0,"No"]]

    # Return full state for frontend to handle choices
    csv_buffer = io.StringIO()
    dataset.to_csv(csv_buffer, index=False)
    all_dataset_b64 = base64.b64encode(csv_buffer.getvalue().encode('utf-8')).decode('utf-8')

    return {
        "encoding_report": encoding_report,
        "feature_importance": feature_importance_list,
        "feature_chart": feature_b64,
        "selected_columns": important_cols,
        "all_dataset_b64": all_dataset_b64,
        "all_columns": list(X.columns)
    }

def run_ml_recommendation(dataset: pd.DataFrame, target_col: str):
    # 1 DETECT ML TYPE
    if target_col is None or target_col == "":
        ml_type = "Unsupervised Learning"
        task = "Clustering"
        algorithms = ["K-Means Clustering", "DBSCAN", "Hierarchical Clustering", "Gaussian Mixture Model"]
    else:
        ml_type = "Supervised Learning"
        if target_col not in dataset.columns:
            return {"ml_type": "Unknown", "task": "Unknown", "suggested_algorithms": []}
            
        if dataset[target_col].nunique() <= 10:
            task = "Classification"
            algorithms = ["Logistic Regression", "Random Forest Classifier", "Support Vector Machine", "XGBoost", "K-Nearest Neighbors"]
        else:
            task = "Regression"
            algorithms = ["Linear Regression", "Random Forest Regressor", "Gradient Boosting Regressor", "XGBoost Regressor", "SVR"]

    return {
        "ml_type": ml_type,
        "task": task,
        "suggested_algorithms": algorithms
    }
