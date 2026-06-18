import urllib.request, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

NEW_IDS = [
    ("1PD7PZ4pPVRHcHVelXQxLpn5M1HJrxwsmRrBjBDNqL8s", "New Doc 1"),
    ("1j8qSKShjGkVAr-CbeHhOffphRLmQZgB4RXOAdWzjJ34", "New Doc 2"),
    ("1DpmNY24lBGnJKbUWiTSWcCEMXVctAHJUBvRER8oubYU", "New Doc 3"),
    ("17uIMIfaCAiazIK53pHY33GH_JldF_24-qLr_MtyFE-c",  "New Doc 4"),
    ("1kYSafBd9ii4Whck6D1hNm9p6XXJGQOpf995cJ6pleiE",  "New Doc 5"),
    ("1UDKznWf3Yro2Rw8H8sHHMmHwTxu78Lg8MhX1EPoFeck",  "New Doc 6"),
    ("1ACSuOBTeFa6vcJuPP3RJRTrhCZRbA2L06Fq3M72KCF8",  "New Doc 7"),
]

for doc_id, label in NEW_IDS:
    url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            text = r.read().decode("utf-8", errors="replace").lstrip("﻿").strip()
        print(f"=== {label} | {doc_id} | {len(text)} chars ===")
        print(text[:800])
        print()
    except Exception as e:
        print(f"=== {label} | {doc_id} | FAILED: {e} ===")
