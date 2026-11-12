from pathlib import Path
import pymupdf
from docx import Document

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)

def create_pdf(file_name, pages):
    pdf = pymupdf.open()

    for heading, content in pages:
        page = pdf.new_page()
        page.insert_text(
            (50, 50),
            heading,
            fontsize=16
        )
        page.insert_textbox(
            pymupdf.Rect(50, 85, 550, 780),
            content,
            fontsize=11
        )

    pdf.save(DATA_DIR / file_name)
    pdf.close()

def create_docx(file_name, pages):
    document = Document()

    for index, (heading, content) in enumerate(pages):
        document.add_heading(heading, level=1)
        document.add_paragraph(content)

        if index < len(pages) - 1:
            document.add_page_break()

    document.save(DATA_DIR / file_name)

policy_pages = [
    (
        "Company Policy - Introduction",
        """The company maintains policies to ensure consistent
        business operations and employee accountability.

        All employees must follow approved company policies.
        Department managers are responsible for communicating
        policy changes to their teams.

        Policy documents must be reviewed regularly.
        Employees should contact their manager when clarification
        is required."""
    ),
    (
        "Company Policy - Leave Management",
        """Employees must submit leave requests through the
        approved leave management system.

        Planned leave should be requested in advance.
        Emergency leave must be reported to the reporting manager
        as soon as possible.

        Managers are responsible for reviewing leave requests
        based on business requirements and available leave balance."""
    ),
    (
        "Company Policy - Code of Conduct",
        """Employees must maintain professional behaviour
        in the workplace.

        Confidential company information must not be shared
        with unauthorized persons.

        Conflicts of interest must be reported to management.
        Employees must protect company property and information."""
    )
]

security_pages = [
    (
        "Security Manual - Introduction",
        """Information security protects company data,
        systems and customer information.

        Every employee is responsible for following
        security guidelines.

        Access to company systems must be provided only
        to authorized users."""
    ),
    (
        "Security Manual - Password Security",
        """Employees must use strong and unique passwords.

        Passwords must not be shared with colleagues.
        Multi-factor authentication should be enabled
        wherever available.

        Suspected password compromise must be reported
        immediately to the IT security team."""
    ),
    (
        "Security Manual - Data Protection",
        """Customer information must be handled carefully.

        Sensitive files must be stored in approved locations.
        Data must not be transferred using unauthorized devices.

        Any suspected data breach must be reported
        to the information security team."""
    )
]

employee_pages = [
    (
        "Employee Manual - Onboarding",
        """New employees must complete the onboarding process.

        The onboarding process includes identity verification,
        document submission and workplace orientation.

        HR provides information about company policies,
        employee benefits and reporting structures."""
    ),
    (
        "Employee Manual - Attendance",
        """Employees must record their attendance
        using the approved attendance system.

        Late arrivals and attendance corrections
        must be reported to the reporting manager.

        Managers should review attendance records regularly."""
    ),
    (
        "Employee Manual - Training",
        """Employees should participate in mandatory training.

        Training programs help employees improve
        professional knowledge and technical skills.

        Training completion records must be maintained
        by the HR department."""
    )
]

compliance_pages = [
    (
        "Compliance Manual - Introduction",
        """Compliance ensures that business activities
        follow applicable laws and regulations.

        Employees must understand the compliance
        requirements relevant to their responsibilities.

        Compliance concerns must be reported
        through approved reporting channels."""
    ),
    (
        "Compliance Manual - Customer Verification",
        """Customer identity must be verified
        before providing regulated financial services.

        Employees must collect and validate
        required customer identification documents.

        Incomplete verification must be escalated
        according to company procedures."""
    ),
    (
        "Compliance Manual - Record Keeping",
        """Business records must be accurate,
        complete and maintained securely.

        Customer records and transaction documents
        must be retained according to applicable rules.

        Unauthorized alteration or deletion
        of official records is prohibited."""
    )
]

banking_text = """BANKING POLICY - INTRODUCTION

Banks must follow approved operational procedures.
Employees must protect customer information and
maintain accurate transaction records.

BANKING POLICY - CUSTOMER SERVICE

Customer requests must be handled professionally.
Complaints should be registered and tracked until
resolution. Customers must receive clear information
about banking products and services.

BANKING POLICY - TRANSACTION MONITORING

Transactions must be monitored according to
applicable banking regulations. Suspicious activity
must be escalated to the compliance department.
"""

create_pdf("policy.pdf", policy_pages)
create_pdf("security_manual.pdf", security_pages)

create_docx("employee_manual.docx", employee_pages)
create_docx("compliance_manual.docx", compliance_pages)

(DATA_DIR / "banking_policy.txt").write_text(
    banking_text,
    encoding="utf-8"
)

print("Sample documents created")

print("\nCreated documents:")

for file_path in sorted(DATA_DIR.iterdir()):
    if file_path.is_file():
        print(file_path.name)