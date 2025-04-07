#  Data Analysis Tool – Streamlit EDA App

An interactive, web-based tool for performing automated **Exploratory Data Analysis (EDA)**. Built with **Streamlit**, this application allows users to upload datasets and receive powerful insights through **data cleaning**, **statistical summaries**, **correlation analysis**, and **interactive visualizations**. Users can also generate and download a detailed HTML report.


# Features

- Upload support for `.csv`, `.xlsx`, `.json`, `.txt`
- Automated data cleaning**:
  - Handles various missing value types
  - Drops low-information columns
  - Removes duplicates
- Statistical overview** with skewness, kurtosis, and descriptive stats
- Correlation matrix** with visual heatmap and analysis
- Interactive visualizations** using Plotly:
  - Scatter (Dot) Plot with trendlines
  - Line Chart
  - Bar Chart
  - Histogram
  - Box Plot
  - Pie Chart
- Insight generation** for variables and relationships
- HTML report generation** using Jinja2


# Tech Stack

- Python 3
- Streamlit – frontend & interactivity
- Pandas
- NumPy – data processing
- Matplotlib
- Seaborn – traditional plots
- Plotly Express – interactive visualization
- SciPy – statistical calculations
- Jinja2 – templated report generation
- Base64 – file export utility

