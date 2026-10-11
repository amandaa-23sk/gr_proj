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
    SCRIPT_DIR / "MD_Reports" / "clean_dataset_eda_report.md",
    SCRIPT_DIR / "clean_dataset_eda_report.md",
    PROJECT_ROOT / "pdf_pre" / "MD_Reports" / "clean_dataset_eda_report.md"
]
md_path = next((p for p in md_candidates if p.exists()), None)
if not md_path:
    raise FileNotFoundError(f"Could not find clean_dataset_eda_report.md in: {md_candidates}")

print(f"Reading markdown from: {md_path}")
with open(md_path, "r", encoding="utf-8") as f:
    md_content = f.read()

# Replace images with base64 data URIs
def img_to_base64(match):
    alt_text = match.group(1)
    img_rel = match.group(2).lstrip("./\\/")
    
    img_candidates = [
        PROJECT_ROOT / img_rel,
        PROJECT_ROOT / "eda_clean_outputs" / Path(img_rel).name,
        PROJECT_ROOT / "eda_outputs" / Path(img_rel).name,
        SCRIPT_DIR / img_rel
    ]
    local_path = next((p for p in img_candidates if p.exists()), None)
    
    if local_path and local_path.exists():
        with open(local_path, "rb") as img_f:
            b64_data = base64.b64encode(img_f.read()).decode("utf-8")
            return f'<div class="img-container"><img src="data:image/png;base64,{b64_data}" alt="{alt_text}" /><p class="img-caption">{alt_text}</p></div>'
    return match.group(0)

md_content_b64 = re.sub(r'!\[(.*?)\]\((.*?)\)', img_to_base64, md_content)

# Parse Markdown to HTML
md = MarkdownIt('commonmark', {'html': True}).enable('table')
html_body = md.render(md_content_b64)

# Construct Full HTML
full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Clean PV Dataset EDA Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  
  @page {{
    size: A4;
    margin: 16mm 14mm 16mm 14mm;
  }}
  
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #1e293b;
    background-color: #ffffff;
    line-height: 1.5;
    font-size: 12.5px;
    margin: 0;
    padding: 0;
  }}
  
  h1 {{
    font-size: 20px;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 2px solid #10b981;
    padding-bottom: 6px;
    margin-top: 0;
    margin-bottom: 12px;
  }}
  
  h2 {{
    font-size: 15px;
    font-weight: 600;
    color: #047857;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 4px;
    margin-top: 18px;
    margin-bottom: 8px;
    page-break-after: avoid;
  }}
  
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0 14px 0;
    font-size: 11px;
    page-break-inside: avoid;
  }}
  
  th, td {{
    padding: 6px 8px;
    border: 1px solid #cbd5e1;
    text-align: left;
  }}
  
  th {{
    background-color: #ecfdf5;
    color: #065f46;
    font-weight: 600;
  }}
  
  tr:nth-child(even) {{
    background-color: #f8fafc;
  }}
  
  .img-container {{
    text-align: center;
    margin: 12px 0;
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
html_file = html_dir / "clean_dataset_eda_report.html"
with open(html_file, "w", encoding="utf-8") as f:
    f.write(full_html)
print(f"Generated HTML report at: {html_file}")

pdf_dir = PROJECT_ROOT / "PDF_Reports"
pdf_dir.mkdir(parents=True, exist_ok=True)
pdf_file = pdf_dir / "clean_dataset_eda_report.pdf"

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
    print("Warning: No headless browser (Chrome/Edge) found for PDF conversion.")
