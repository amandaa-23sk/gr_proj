import os
import re
import base64
import subprocess
from pathlib import Path
from markdown_it import MarkdownIt

# Determine paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "pdf_pre" else SCRIPT_DIR

# Locate markdown source
md_candidates = [
    SCRIPT_DIR / "MD_Reports" / "pv_panel_eda_report.md",
    SCRIPT_DIR / "pv_panel_eda_report.md",
    PROJECT_ROOT / "pdf_pre" / "MD_Reports" / "pv_panel_eda_report.md"
]
md_path = next((p for p in md_candidates if p.exists()), None)
if not md_path:
    raise FileNotFoundError(f"Could not find pv_panel_eda_report.md in: {md_candidates}")

print(f"Reading markdown from: {md_path}")
with open(md_path, "r", encoding="utf-8") as f:
    md_content = f.read()

# Function to convert image paths to Base64 data URIs
def img_to_base64(match):
    alt_text = match.group(1)
    img_rel = match.group(2).lstrip("./\\/")
    
    img_candidates = [
        PROJECT_ROOT / img_rel,
        PROJECT_ROOT / "eda_outputs" / Path(img_rel).name,
        PROJECT_ROOT / "eda_clean_outputs" / Path(img_rel).name,
        SCRIPT_DIR / img_rel
    ]
    local_path = next((p for p in img_candidates if p.exists()), None)
    
    if local_path and local_path.exists():
        with open(local_path, "rb") as img_f:
            b64_data = base64.b64encode(img_f.read()).decode("utf-8")
            return f'<div class="img-container"><img src="data:image/png;base64,{b64_data}" alt="{alt_text}" /><p class="img-caption">{alt_text}</p></div>'
    return match.group(0)

md_content_b64 = re.sub(r'!\[(.*?)\]\((.*?)\)', img_to_base64, md_content)

# Handle Mermaid diagram representation
mermaid_replacement = """
<div class="mermaid-box">
  <div class="flow-step"><strong>Raw Dataset:</strong> PV_Diagnostic_Final_Labels.csv (226,175 rows)</div>
  <div class="flow-arrow">&darr;</div>
  <div class="flow-grid">
    <div class="flow-card"><strong>Panel 1 Stream</strong><br>I1, V1, P1, T1, Irr, Label_S1 &rarr; <code>panel_1_data.csv</code></div>
    <div class="flow-card"><strong>Panel 2 Stream</strong><br>I2, V2, P2, T2, Irr, Label_S2 &rarr; <code>panel_2_data.csv</code></div>
  </div>
  <div class="flow-arrow">&darr;</div>
  <div class="flow-card diff-card"><strong>Inter-Panel Differential Diagnostics</strong><br>&Delta;I = I1 - I2, &Delta;V = V1 - V2, &Delta;P = P1 - P2, &Delta;T = T1 - T2</div>
</div>
"""
md_content_b64 = re.sub(r'```mermaid[\s\S]*?```', mermaid_replacement, md_content_b64)

# Handle GitHub alerts
def format_alerts(text):
    text = re.sub(r'> \[!WARNING\]\s*\n> (.*)', r'<div class="alert alert-warning"><strong>⚠️ WARNING:</strong> \1</div>', text)
    text = re.sub(r'> \[!TIP\]\s*\n> (.*)', r'<div class="alert alert-tip"><strong>💡 TIP:</strong> \1</div>', text)
    text = re.sub(r'> \[!NOTE\]\s*\n> (.*)', r'<div class="alert alert-note"><strong>ℹ️ NOTE:</strong> \1</div>', text)
    return text

md_content_b64 = format_alerts(md_content_b64)

# Parse Markdown to HTML
md = MarkdownIt('commonmark', {'html': True}).enable('table')
html_body = md.render(md_content_b64)

# Construct Full HTML with Premium Styling
full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Solar PV Fault Diagnostics Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  
  @page {{
    size: A4;
    margin: 18mm 16mm 18mm 16mm;
    @bottom-right {{
      content: counter(page);
    }}
  }}
  
  * {{
    box-sizing: border-box;
  }}
  
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1e293b;
    background-color: #ffffff;
    line-height: 1.6;
    font-size: 13px;
    padding: 0;
    margin: 0;
  }}
  
  h1 {{
    font-size: 22px;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 2px solid #3b82f6;
    padding-bottom: 8px;
    margin-top: 0;
    margin-bottom: 14px;
  }}
  
  h2 {{
    font-size: 16px;
    font-weight: 600;
    color: #1e3a8a;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 5px;
    margin-top: 22px;
    margin-bottom: 10px;
    page-break-after: avoid;
  }}
  
  h3 {{
    font-size: 14px;
    font-weight: 600;
    color: #334155;
    margin-top: 14px;
    margin-bottom: 6px;
    page-break-after: avoid;
  }}
  
  p, li {{
    color: #334155;
    font-size: 12.5px;
  }}
  
  hr {{
    border: 0;
    height: 1px;
    background: #e2e8f0;
    margin: 18px 0;
  }}
  
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 16px 0;
    font-size: 11.5px;
    page-break-inside: avoid;
  }}
  
  th, td {{
    padding: 7px 10px;
    border: 1px solid #cbd5e1;
    text-align: left;
  }}
  
  th {{
    background-color: #f1f5f9;
    color: #0f172a;
    font-weight: 600;
  }}
  
  tr:nth-child(even) {{
    background-color: #f8fafc;
  }}
  
  .img-container {{
    text-align: center;
    margin: 14px 0;
    page-break-inside: avoid;
  }}
  
  .img-container img {{
    max-width: 95%;
    height: auto;
    border-radius: 6px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    border: 1px solid #e2e8f0;
  }}
  
  .img-caption {{
    font-size: 11px;
    color: #64748b;
    margin-top: 4px;
    font-style: italic;
  }}
  
  .alert {{
    padding: 10px 14px;
    border-radius: 6px;
    margin: 12px 0;
    font-size: 12px;
    page-break-inside: avoid;
  }}
  
  .alert-warning {{
    background-color: #fffbeb;
    border-left: 4px solid #f59e0b;
    color: #92400e;
  }}
  
  .alert-tip {{
    background-color: #f0fdf4;
    border-left: 4px solid #22c55e;
    color: #166534;
  }}
  
  .alert-note {{
    background-color: #eff6ff;
    border-left: 4px solid #3b82f6;
    color: #1e40af;
  }}
  
  .mermaid-box {{
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 12px;
    margin: 12px 0;
    text-align: center;
    page-break-inside: avoid;
  }}
  
  .flow-step {{
    display: inline-block;
    background: #e2e8f0;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 12px;
  }}
  
  .flow-arrow {{
    font-size: 16px;
    color: #64748b;
    margin: 4px 0;
  }}
  
  .flow-grid {{
    display: flex;
    justify-content: center;
    gap: 12px;
  }}
  
  .flow-card {{
    background: #ffffff;
    border: 1px solid #94a3b8;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 11.5px;
    flex: 1;
    max-width: 45%;
  }}
  
  .diff-card {{
    background: #fdf2f8;
    border-color: #f472b6;
    max-width: 92%;
    margin: 0 auto;
  }}
  
  code {{
    background: #f1f5f9;
    color: #0f172a;
    padding: 2px 4px;
    border-radius: 4px;
    font-size: 11px;
    font-family: Consolas, Monaco, monospace;
  }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

# Output paths
html_dir = SCRIPT_DIR / "HTML_Reports"
html_dir.mkdir(parents=True, exist_ok=True)
html_file = html_dir / "pv_panel_eda_report.html"
with open(html_file, "w", encoding="utf-8") as f:
    f.write(full_html)
print(f"Generated HTML report at: {html_file}")

pdf_dir = PROJECT_ROOT / "PDF_Reports"
pdf_dir.mkdir(parents=True, exist_ok=True)
pdf_file = pdf_dir / "pv_panel_eda_report.pdf"

# Find browser executable
browser_candidates = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe")
]
browser_exe = next((b for b in browser_candidates if os.path.exists(b)), None)

if browser_exe:
    abs_html = str(html_file.resolve())
    abs_pdf = str(pdf_file.resolve())
    cmd = [
        browser_exe,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={abs_pdf}",
        f"file:///{abs_html.replace(os.sep, '/')}"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if pdf_file.exists():
        print(f"SUCCESS: Generated PDF at {pdf_file} (Size: {pdf_file.stat().st_size:,} bytes)")
    else:
        print(f"Failed to generate PDF. Error: {res.stderr}")
else:
    print("Warning: No headless browser found for PDF conversion.")
