"""
campus_contacts.py — Zewail City office contacts and topic-based routing.
Used by the human escalation pathway (Feature 5.3).
"""
from __future__ import annotations

# ── Contact registry ───────────────────────────────────────────────────────────
CONTACTS: dict[str, dict] = {
    "academic": {
        "name":     "Dr. Ahmed Sayed Abdelsamea",
        "role":     "Director of Academic Advising Unit",
        "dept":     "Math Department",
        "email":    "aabdelsamea@zewailcity.edu.eg",
        "location": "Academic Building, Zewail City of Science and Technology",
        "icon":     "🎓",
        "topics":   "Course planning, graduation requirements, prerequisite overrides, "
                    "academic policies, program advising, academic standing, course load.",
    },
    "it": {
        "name":     "IT Support",
        "role":     "Information Technology Helpdesk",
        "dept":     "",
        "email":    "it-support@zewailcity.edu.eg",
        "location": "",
        "icon":     "💻",
        "topics":   "Portal access, account blocks, VPN, LTS, password reset, "
                    "system errors, internet, Blackboard/Moodle, software.",
    },
    "career": {
        "name":     "Career Advising & Talent Support (CATS)",
        "role":     "Career Services",
        "dept":     "",
        "email":    "cats@zewailcity.edu.eg",
        "location": "",
        "icon":     "💼",
        "topics":   "Internships, job placements, CV/resume review, career guidance, "
                    "interview preparation, hiring, employment opportunities.",
    },
    "financial": {
        "name":     "Student Accounts & Financial Support",
        "role":     "Financial Aid & Student Accounts",
        "dept":     "",
        "email":    "finsupport@zewailcity.edu.eg",
        "location": "",
        "icon":     "💳",
        "topics":   "Scholarship applications, financial holds, tuition, "
                    "student account balance, financial aid, grants, fee waivers.",
    },
    "finance_office": {
        "name":     "Finance Office",
        "role":     "University Finance Department",
        "dept":     "",
        "email":    "all-finance@zewailcity.edu.eg",
        "location": "",
        "icon":     "🏦",
        "topics":   "Payments, invoices, financial transactions, receipts, billing.",
    },
    "library": {
        "name":     "Zewail City Library",
        "role":     "Library & Information Services",
        "dept":     "",
        "email":    "all-library@zewailcity.edu.eg",
        "location": "",
        "icon":     "📚",
        "topics":   "Books, journals, research databases, course material borrowing, "
                    "study rooms, research resources.",
    },
}

# ── Routing rules: (keywords, contact_key) ordered by specificity ──────────────
_ROUTING_RULES: list[tuple[list[str], str]] = [
    (
        ["portal", "vpn", "login", "access", "account", "blocked", "block", "password",
         "lts", "learning technology", "blackboard", "moodle", "wifi", "internet",
         "can't log", "cannot log", "locked out", "system error", "it helpdesk",
         "it support", "technical issue", "student email", "outlook"],
        "it",
    ),
    (
        ["career", "internship", "job", "cv", "resume", "placement", "interview",
         "hiring", "cats", "employment", "company visit", "graduate job", "work",
         "job fair", "linkedin", "talent"],
        "career",
    ),
    (
        ["scholarship", "financial aid", "tuition", "fee waiver", "financial hold",
         "finsupport", "financial support", "grant", "fund", "bursary",
         "balance", "student account", "hold on account", "fees", "pay tuition"],
        "financial",
    ),
    (
        ["invoice", "receipt", "transaction", "finance office", "payment",
         "billing", "all-finance"],
        "finance_office",
    ),
    (
        ["library", "book", "journal", "research resource", "article", "database",
         "borrow", "reserve", "reading material", "all-library", "librarian"],
        "library",
    ),
]


def route_contact(question: str, intent: str = "general") -> str:
    """
    Return the contact key most relevant to the question.
    Defaults to 'academic' (Academic Advising).
    """
    q = question.lower()
    for keywords, key in _ROUTING_RULES:
        if any(kw in q for kw in keywords):
            return key
    return "academic"


def get_contact(key: str) -> dict:
    """Return the contact dict for the given key, falling back to academic."""
    return CONTACTS.get(key, CONTACTS["academic"])
