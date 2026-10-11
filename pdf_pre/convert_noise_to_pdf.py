import os
import re
import subprocess
from pathlib import Path
from markdown_it import MarkdownIt

# Determine paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "pdf_pre" else SCRIPT_DIR

# Locate markdown source
md_candidates = [
    SCRIPT_DIR / "MD_Reports" / "pv_power_noise_analysis.md",
    SCRIPT_DIR / "pv_power_noise_analysis.md",
    PROJECT_ROOT / "pdf_pre" / "MD_Reports" / "pv_power_noise_analysis.md"
]
md_path = next((p for p in md_candidates if p.exists()), None)
if not md_path:
    raise FileNotFoundError(f"Could not find pv_power_noise_analysis.md in: {md_candidates}")

print(f"Reading markdown from: {md_path}")
with open(md_path, "r", encoding="utf-8") as f:
    md_content = f.read()

# Handle GitHub alerts
def format_alerts(text):
    text = re.sub(r'> \[!CAUTION\]\s*\n> (.*)', r'<div class="alert alert-caution"><strong>⚠️ CAUTION:</strong> \1</div>', text)
    text = re.sub(r'> \[!WARNING\]\s*\n> (.*)', r'<div class="alert alert-warning"><strong>⚠️ WARNING:</strong> \1</div>', text)
    text = re.sub(r'> \[!TIP\]\s*\n> (.*)', r'<div class="alert alert-tip"><strong>💡 TIP:</strong> \1</div>', text)
    return text

md_content = format_alerts(md_content)

# Parse Markdown to HTML
md = MarkdownIt('commonmark', {'html': True}).enable('table')
html_body = md.render(md_content)

# Construct Full HTML
full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PV Noise & Data Quality Analysis Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  
  @page {{
    size: A4;
    margin: 16mm 14mm 16mm 14mm;
  }}
  
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1e293b;
    background-color: #ffffff;
    line-height: 1.6;
    font-size: 12.5px;
    padding: 0;
    margin: 0;
  }}
  
  h1 {{
    font-size: 20px;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 2px solid #ef4444;
    padding-bottom: 6px;
    margin-top: 0;
    margin-bottom: 12px;
  }}
  
  h2 {{
    font-size: 15px;
    font-weight: 600;
    color: #b91c1c;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 4px;
    margin-top: 20px;
    margin-bottom: 8px;
    page-break-after: avoid;
  }}
  
  h3 {{
    font-size: 13.5px;
    font-weight: 600;
    color: #334155;
    margin-top: 14px;
    margin-bottom: 6px;
    page-break-after: avoid;
  }}
  
  p, li {{
    color: #334155;
    font-size: 12px;
  }}
  
  hr {{
    border: 0;
    height: 1px;
    background: #e2e8f0;
    margin: 16px 0;
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
    background-color: #f1f5f9;
    color: #0f172a;
    font-weight: 600;
  }}
  
  tr:nth-child(even) {{
    background-color: #f8fafc;
  }}
  
  .alert {{
    padding: 10px 14px;
    border-radius: 6px;
    margin: 12px 0;
    font-size: 11.5px;
    page-break-inside: avoid;
  }}
  
  .alert-caution {{
    background-color: #fef2f2;
    border-left: 4px solid #ef4444;
    color: #991b1b;
  }}
  
  code {{
    background: #f1f5f9;
    color: #0f172a;
    padding: 2px 4px;
    border-radius: 4px;
    font-size: 11px;
    font-family: Consolas, Monaco, monospace;
  }}
  
  pre {{
    background: #0f172a;
    color: #f8fafc;
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 11px;
    font-family: Consolas, Monaco, monospace;
    overflow-x: auto;
    page-break-inside: avoid;
  }}
  
  pre code {{
    background: transparent;
    color: inherit;
    padding: 0;
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
html_file = html_dir / "pv_power_noise_analysis.html"
with open(html_file, "w", encoding="utf-8") as f:
    f.write(full_html)
print(f"Generated HTML report at: {html_file}")

pdf_dir = PROJECT_ROOT / "PDF_Reports"
pdf_dir.mkdir(parents=True, exist_ok=True)
pdf_file = pdf_dir / "pv_power_noise_analysis.pdf"

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
