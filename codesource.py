import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import BytesIO
from scipy import stats
from datetime import datetime
import base64
from jinja2 import Template

# Set Streamlit page config
st.set_page_config(page_title="Data Analysis Pro", layout="wide")

st.title("🔍 Data Analysis Pro - Automated Insights & Cleaning")
st.markdown("""
**Upload your dataset for automatic cleaning, comprehensive analysis, and statistical insights**
- Automatic data cleaning
- Detailed data profiling
- Smart visualization
- Statistical relationship analysis
""")

def styled_header(text):
    st.markdown(f"""
    <div style="
        background: #2b5876;
        color: white;
        padding: 12px;
        border-radius: 8px;
        margin: 15px 0;
        font-size: 20px;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
    {text}
    </div>
    """, unsafe_allow_html=True)

def clean_data(df):
    """Automatically clean the dataset"""
    original_shape = df.shape
    
    # Convert various missing value representations to NaN
    missing_values = ['', ' ', 'NA', 'N/A', 'NaN', 'nan', 'null', 'NULL', '-']
    df = df.replace(missing_values, np.nan)
    
    # Remove empty columns (>90% missing)
    cols_before = df.shape[1]
    df = df.loc[:, df.isnull().mean() < 0.9]
    cols_removed = cols_before - df.shape[1]
    
    # Remove rows with any remaining missing values
    df = df.dropna(how='any')
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Convert to best possible dtypes
    df = df.convert_dtypes()
    
    return df, original_shape, cols_removed

def display_file_info(uploaded_file, df):
    """Show comprehensive file metadata"""
    file_info = {
        "File Name": uploaded_file.name,
        "File Size": f"{uploaded_file.size/1024:.2f} KB",
        "File Type": uploaded_file.type,
        "Total Columns": df.shape[1],
        "Total Rows": df.shape[0],
        "Missing Values": df.isnull().sum().sum(),
        "Duplicate Rows": df.duplicated().sum(),
        "Data Types": df.dtypes.value_counts().to_dict()
    }
    
    st.subheader("📄 File Metadata")
    cols = st.columns(4)
    cols[0].metric("File Name", file_info["File Name"])
    cols[1].metric("File Size", file_info["File Size"])
    cols[2].metric("Total Columns", file_info["Total Columns"])
    cols[3].metric("Total Rows", file_info["Total Rows"])
    
    with st.expander("View Detailed Data Profile"):
        st.write("**Column Data Types:**")
        st.dataframe(pd.DataFrame(file_info["Data Types"].items(), 
                    columns=["Data Type", "Count"]), height=200)
        
        st.write("**Missing Values Breakdown:**")
        missing = df.isnull().sum().to_frame("Missing Count")
        st.dataframe(missing, height=400)

def create_downloadable_report(df, cleaned_df, original_shape, cols_removed, visualizations, insights):
    """Generate HTML report with all analysis content"""
    report_template = Template('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { color: #2b5876; border-bottom: 2px solid #2b5876; padding-bottom: 10px; }
            .section { margin: 30px 0; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f0f2f6; }
            .plot { margin: 40px 0; }
            .insight-box { background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 20px 0; }
        </style>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <h1 class="header">Data Analysis Report</h1>
        <div class="section">
            <h2>Dataset Overview</h2>
            <p><strong>Original Dimensions:</strong> {{ original_rows }} rows × {{ original_cols }} columns</p>
            <p><strong>Cleaned Dimensions:</strong> {{ cleaned_rows }} rows × {{ cleaned_cols }} columns</p>
            <p><strong>Removed Columns:</strong> {{ cols_removed }}</p>
        </div>

        <div class="section">
            <h2>Statistical Summary</h2>
            {{ stats_table }}
        </div>

        <div class="section">
            <h2>Correlation Analysis</h2>
            {{ correlation_plot }}
            <div class="insight-box">
                <h3>Key Correlation Insights</h3>
                <ul>
                    {% for insight in corr_insights %}
                    <li>{{ insight }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>

        <div class="section">
            <h2>Visual Analysis</h2>
            {% for plot in visualizations %}
            <div class="plot">
                {{ plot }}
            </div>
            {% endfor %}
        </div>

        <div class="section">
            <h2>Key Insights</h2>
            <ul>
                {% for insight in insights %}
                <li>{{ insight }}</li>
                {% endfor %}
            </ul>
        </div>
    </body>
    </html>
    ''')

    # Prepare statistics table and correlation insights
    stats_table = cleaned_df.describe().T.to_html()
    corr_insights = insights.get('correlation', [])
    other_insights = insights.get('general', [])

    html_content = report_template.render(
        original_rows=original_shape[0],
        original_cols=original_shape[1],
        cleaned_rows=cleaned_df.shape[0],
        cleaned_cols=cleaned_df.shape[1],
        cols_removed=cols_removed,
        stats_table=stats_table,
        correlation_plot=visualizations.get('correlation', ''),
        corr_insights=corr_insights,
        visualizations=visualizations.get('general', []),
        insights=other_insights
    )

    return html_content

# File uploader
uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx", "json", "txt"])

if uploaded_file:
    try:
        # Read file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            df = pd.read_csv(uploaded_file, delimiter='\t')
        
        display_file_info(uploaded_file, df)
        
        # Clean data
        styled_header("🧹 Data Cleaning Report")
        df_clean, original_shape, cols_removed = clean_data(df)
        
        cols = st.columns(3)
        cols[0].metric("Original Rows", original_shape[0], 
                      delta=original_shape[0]-df_clean.shape[0])
        cols[1].metric("Original Columns", original_shape[1], 
                      delta=original_shape[1]-df_clean.shape[1])
        cols[2].metric("Removed Columns", cols_removed)
        
        with st.expander("View Cleaned Data"):
            st.dataframe(df_clean, height=600)
        
        # Statistical Analysis
        styled_header("📈 Statistical Overview")
        numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
        all_insights = {'general': [], 'correlation': []}
        vis_dict = {'general': [], 'correlation': []}

        if numeric_cols:
            st.subheader("Numerical Columns Summary")
            stats_df = df_clean[numeric_cols].describe().T
            stats_df['skewness'] = df_clean[numeric_cols].skew()
            stats_df['kurtosis'] = df_clean[numeric_cols].kurtosis()
            st.dataframe(stats_df.style.format("{:.2f}").background_gradient(cmap='Blues'),
                        height=600)

            # Enhanced Correlation Analysis
            styled_header("🔗 Correlation Matrix")
            corr_matrix = df_clean[numeric_cols].corr()
            
            # Create correlation plot
            fig_corr = px.imshow(corr_matrix,
                                title="Feature Correlation Matrix",
                                labels=dict(x="Features", y="Features", color="Correlation"),
                                x=corr_matrix.columns,
                                y=corr_matrix.columns,
                                color_continuous_scale='Viridis',
                                text_auto=".2f",
                                zmin=-1, 
                                zmax=1)
            
            fig_corr.update_traces(textfont=dict(color='black', size=12))
            fig_corr.update_xaxes(tickangle=45, tickfont=dict(size=12))
            fig_corr.update_yaxes(tickfont=dict(size=12))
            fig_corr.update_layout(width=800, height=800,
                                coloraxis_colorbar=dict(title="Correlation",
                                                        thickness=20,
                                                        tickvals=[-1, -0.5, 0, 0.5, 1]))
            st.plotly_chart(fig_corr)

            # Calculate correlation insights
            corr_values = corr_matrix.mask(np.triu(np.ones(corr_matrix.shape)).astype(bool)).stack()
            if not corr_values.empty:
                max_corr = corr_values.max()
                min_corr = corr_values.min()
                max_pair = corr_values.idxmax()
                min_pair = corr_values.idxmin()

                corr_insights = [
                    f"Maximum correlation ({max_corr:.2f}) between {max_pair[0]} and {max_pair[1]}",
                    f"Minimum correlation ({min_corr:.2f}) between {min_pair[0]} and {min_pair[1]}",
                    "Correlation interpretation:",
                    "• 1.0: Perfect positive correlation",
                    "• 0.8-1.0: Very strong positive",
                    "• 0.6-0.8: Strong positive",
                    "• 0.4-0.6: Moderate positive",
                    "• 0.2-0.4: Weak positive",
                    "• 0.0-0.2: Very weak/no correlation",
                    "• Negative values indicate inverse relationships"
                ]
                
                all_insights['correlation'] = corr_insights
                vis_dict['correlation'] = [fig_corr.to_html(full_html=False, include_plotlyjs='cdn')]

                # Display correlation insights
                styled_header("🔍 Correlation Insights")
                st.markdown("\n".join(corr_insights))

        # Visualization & Insights Section
        styled_header("📊 Interactive Analysis")
        chart_type = st.selectbox("Select Visualization Type", 
                                ["Dot Plot", "Line Chart", "Bar Chart", 
                                 "Histogram", "Box Plot", "Pie Chart"])
        
        all_cols = df_clean.columns.tolist()
        numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df_clean.select_dtypes(exclude=np.number).columns.tolist()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            x_axis = st.selectbox("X-Axis", all_cols)
        with col2:
            if chart_type in ["Dot Plot", "Line Chart", "Bar Chart"]:
                y_axis = st.selectbox("Y-Axis", numeric_cols)
            elif chart_type == "Pie Chart":
                y_axis = st.selectbox("Values", numeric_cols)
            else:
                y_axis = None
        with col3:
            show_values = st.checkbox("Show Data Values") if chart_type != "Pie Chart" else None
        
        # Generate visualization
        fig = None
        insights = []
        if chart_type == "Dot Plot":
            try:
                fig = px.scatter(df_clean, x=x_axis, y=y_axis, trendline="ols",
                                title=f"{y_axis} vs {x_axis} Relationship",
                                labels={x_axis: x_axis, y_axis: y_axis})
            except ImportError:
                fig = px.scatter(df_clean, x=x_axis, y=y_axis,
                                title=f"{y_axis} vs {x_axis} Relationship",
                                labels={x_axis: x_axis, y_axis: y_axis})
                st.warning("Trendline disabled - Install statsmodels: `pip install statsmodels`")
        elif chart_type == "Line Chart":
            fig = px.line(df_clean, x=x_axis, y=y_axis, 
                         title=f"{y_axis} Trend over {x_axis}")
        elif chart_type == "Bar Chart":
            fig = px.bar(df_clean, x=x_axis, y=y_axis, 
                        title=f"{y_axis} by {x_axis}")
        elif chart_type == "Histogram":
            fig = px.histogram(df_clean, x=x_axis, 
                               title=f"Distribution of {x_axis}")
        elif chart_type == "Box Plot":
            fig = px.box(df_clean, x=x_axis, 
                        title=f"{x_axis} Distribution")
        elif chart_type == "Pie Chart":
            fig = px.pie(df_clean, names=x_axis, values=y_axis, 
                         title=f"Proportion of {x_axis}")
        
        # Add data values if requested
        if fig and show_values:
            if chart_type in ["Bar Chart", "Line Chart"]:
                fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
            elif chart_type == "Dot Plot":
                fig.update_traces(textposition='top center')
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            # Generate insights
            styled_header("🔍 Statistical Insights")
            insights = []
            
            # X-axis analysis
            insights.append(f"**{x_axis} Analysis**")
            if df_clean[x_axis].dtype in [np.number]:
                insights.append(f"- Range: {df_clean[x_axis].min():.2f} to {df_clean[x_axis].max():.2f}")
                insights.append(f"- Mean: {df_clean[x_axis].mean():.2f}")
                insights.append(f"- Median: {df_clean[x_axis].median():.2f}")
            else:
                insights.append(f"- Unique Categories: {df_clean[x_axis].nunique()}")
                insights.append(f"- Most Common: {df_clean[x_axis].mode()[0]} ({df_clean[x_axis].value_counts().max()} counts)")
            
            if y_axis:
                # Y-axis analysis
                insights.append(f"\n**{y_axis} Analysis**")
                insights.append(f"- Range: {df_clean[y_axis].min():.2f} to {df_clean[y_axis].max():.2f}")
                insights.append(f"- Mean: {df_clean[y_axis].mean():.2f}")
                insights.append(f"- Median: {df_clean[y_axis].median():.2f}")
                
                # Relationship analysis
                if chart_type in ["Dot Plot", "Line Chart", "Bar Chart"]:
                    corr = df_clean[x_axis].corr(df_clean[y_axis]) if df_clean[x_axis].dtype in [np.number] else None
                    if corr is not None:
                        insights.append(f"\n**Relationship Analysis**")
                        insights.append(f"- Correlation Coefficient: {corr:.2f}")
                        strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
                        direction = "positive" if corr > 0 else "negative"
                        insights.append(f"- The relationship shows a {strength} {direction} correlation")
            
            # Display and collect insights
            st.markdown("\n".join(insights))
            all_insights['general'].extend(insights)
            vis_dict['general'].append(fig.to_html(full_html=False, include_plotlyjs='cdn'))

        # Download Report Section
        styled_header("📥 Download Full Report")
        if st.button("Generate Full Analysis Report"):
            with st.spinner("Compiling report..."):
                # Create and save report
                report_html = create_downloadable_report(
                    df, df_clean, original_shape, cols_removed, 
                    vis_dict, all_insights
                )
                
                # Create download link
                b64 = base64.b64encode(report_html.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="analysis_report_{datetime.now().strftime("%Y%m%d_%H%M")}.html">Download Full Report</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success("Report generated! Click the download link above")

    except Exception as e:
        if "No module named 'statsmodels'" in str(e):
            st.error("Install required package: `pip install statsmodels`")
        else:
            st.error(f"Error processing file: {str(e)}")