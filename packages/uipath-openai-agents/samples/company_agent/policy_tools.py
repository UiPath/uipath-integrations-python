"""Policy department tools for company policies and compliance."""

from agents import function_tool


@function_tool
async def get_company_policy(policy_type: str) -> str:
    """Get detailed company policy information.

    Args:
        policy_type: Type of policy (remote_work, expense, code_of_conduct, pto, security, travel, etc.)

    Returns:
        Comprehensive policy information
    """
    # Mock implementation
    policies = {
        "remote_work": """REMOTE WORK POLICY (Rev. 2024.1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ELIGIBILITY:
  • Available to all full-time employees
  • Must have manager approval
  • Role must be suitable for remote work

SCHEDULE:
  • Up to 3 days per week remote
  • Core hours: 10:00 AM - 3:00 PM (local time)
  • Must be available for meetings/collaboration

REQUIREMENTS:
  ✓ Secure home office setup
  ✓ Reliable high-speed internet (minimum 25 Mbps)
  ✓ Company-provided equipment only
  ✓ VPN connection for all work activities

EXPECTATIONS:
  • Maintain same productivity standards
  • Respond to communications promptly
  • Attend all required meetings
  • Available during core hours

EQUIPMENT:
  Company provides: Laptop, monitor, headset
  Employee provides: Desk, chair, internet

Questions? Contact: remote-work@company.com
Policy effective: January 1, 2024""",
        "expense": """EXPENSE REIMBURSEMENT POLICY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUBMISSION REQUIREMENTS:
  • Submit within 30 days of expense date
  • Include itemized receipts (>$25)
  • Use company expense system
  • Manager approval required

DAILY LIMITS:
  Meals:
    Breakfast: $15
    Lunch: $25
    Dinner: $50
    (No alcohol reimbursement)

  Transportation:
    Mileage: $0.67/mile (IRS rate)
    Parking: Actual cost (receipt required)
    Ride-share: Up to $75/trip

TRAVEL:
  Pre-approval Required (>$500):
    • Flight bookings
    • Hotel reservations (>3 nights)
    • Rental cars

  Guidelines:
    • Book economy class only
    • Hotels: up to $200/night
    • No first-class upgrades

PROHIBITED:
  ✗ Personal entertainment
  ✗ Minibar charges
  ✗ Luxury upgrades
  ✗ Non-business guests

Submit expenses: expenses.company.com
Policy Number: EXP-2024.1""",
        "code_of_conduct": """CODE OF CONDUCT & ETHICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORE PRINCIPLES:
  1. Act with Integrity
  2. Respect Everyone
  3. Protect Company Assets
  4. Maintain Confidentiality
  5. Avoid Conflicts of Interest

PROFESSIONAL BEHAVIOR:
  ✓ Treat all colleagues with respect
  ✓ Maintain professional communication
  ✓ Dress appropriately for workplace
  ✓ Be punctual and reliable

PROHIBITED CONDUCT:
  ✗ Harassment or discrimination
  ✗ Violence or threats
  ✗ Substance abuse at work
  ✗ Theft or fraud
  ✗ Disclosure of confidential information

DIVERSITY & INCLUSION:
  We are committed to a workplace free from
  discrimination based on:
    • Race, color, religion
    • Gender, sexual orientation
    • Age, disability
    • National origin

REPORTING VIOLATIONS:
  • Report to: HR or ethics@company.com
  • Anonymous hotline: 1-800-ETHICS-1
  • No retaliation for good-faith reports
  • Investigation within 10 business days

Violations may result in disciplinary action
up to and including termination.

Acknowledgment required annually.
Last Updated: January 2024""",
        "pto": """PAID TIME OFF (PTO) POLICY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACCRUAL:
  • 20 days per year (full-time)
  • 1.67 days accrued per month
  • Pro-rated for mid-year hires
  • Begins accruing on start date

USAGE:
  • Request via HR system
  • Manager approval required
  • Minimum 2 weeks notice (>3 days)
  • 48 hours notice for single days

CARRYOVER:
  • Maximum: 5 days to next year
  • Use-it-or-lose-it after 12 months
  • No cash payout for unused PTO

BLACKOUT PERIODS:
  Limited PTO during:
    • End of quarter (last 2 weeks)
    • Major product launches
    • Peak business seasons
  (Exceptions for emergencies)

SICK LEAVE:
  • Separate bank: 10 days/year
  • No advance notice required
  • Doctor's note (>3 consecutive days)
  • Does not carry over

HOLIDAYS:
  11 Paid Company Holidays:
    New Year's Day, Memorial Day,
    Independence Day, Labor Day,
    Thanksgiving (2 days), Christmas (2 days),
    + 3 Floating Holidays

Questions? pto@company.com
Policy: HR-PTO-2024.1""",
        "security": """INFORMATION SECURITY POLICY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA CLASSIFICATION:
  🔴 Confidential: Customer data, financials
  🟡 Internal: Company documents, projects
  🟢 Public: Marketing materials, press releases

PASSWORD REQUIREMENTS:
  • Minimum 12 characters
  • Mix of upper/lower/numbers/symbols
  • Change every 90 days
  • No password reuse
  • Enable MFA (Multi-Factor Auth)

DEVICE SECURITY:
  ✓ Use company-provided devices only
  ✓ Enable full disk encryption
  ✓ Auto-lock after 5 minutes
  ✓ Keep software updated
  ✓ Install company antivirus

DATA HANDLING:
  • No confidential data on personal devices
  • Use encrypted file sharing only
  • No public WiFi for sensitive work
  • VPN required for remote access
  • Shred sensitive documents

EMAIL SECURITY:
  ⚠️ Be alert for phishing attempts
  • Verify sender before clicking links
  • Don't share credentials via email
  • Report suspicious emails: security@company.com

INCIDENT REPORTING:
  Report immediately:
    • Lost/stolen devices
    • Suspected breaches
    • Phishing attempts
    • Unauthorized access
  Contact: security@company.com | 1-800-SEC-HELP

Violations may result in termination and
legal action. Training required annually.

Policy: SEC-2024.1 | Effective: Jan 1, 2024""",
        "travel": """BUSINESS TRAVEL POLICY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRE-APPROVAL REQUIRED:
  • All travel must be pre-approved
  • Submit request 2 weeks in advance
  • Use corporate travel portal
  • Business justification required

AIRFARE:
  • Book economy class only
  • Choose lowest logical fare
  • Direct flights when possible
  • No first/business class (unless >6 hours)
  • Use company travel portal

LODGING:
  • Up to $200/night in major cities
  • Up to $150/night in other locations
  • Book through company portal
  • Stay at preferred hotels (discounts)

GROUND TRANSPORTATION:
  • Rental cars: Compact/mid-size only
  • Decline extra insurance (covered)
  • Ride-share: Acceptable for short trips
  • Public transit: Encouraged when practical

MEALS (per day):
  • Breakfast: $15
  • Lunch: $25
  • Dinner: $50
  • No alcohol reimbursement

EXPENSE SUBMISSION:
  • Submit within 30 days of return
  • Attach all receipts
  • Itemize daily expenses
  • Manager approval required

CANCELLATION:
  • Cancel unused reservations immediately
  • Company not responsible for personal
    expenses from trip extensions

Contact: travel@company.com
Policy: TRV-2024.1""",
    }

    policy_content = policies.get(policy_type.lower())
    if policy_content:
        return policy_content
    else:
        return f"""Policy Not Found: '{policy_type}'
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Available policies:
  • remote_work - Remote work guidelines
  • expense - Expense reimbursement
  • code_of_conduct - Ethics and behavior
  • pto - Paid time off and holidays
  • security - Information security
  • travel - Business travel guidelines

For other policies, contact:
  📧 policy@company.com
  📞 1-800-POLICY-1
  🌐 intranet.company.com/policies"""


@function_tool
async def check_compliance_status(
    policy_area: str, department: str = "Company-wide"
) -> str:
    """Check compliance status for a policy area.

    Args:
        policy_area: Area to check (data_security, safety, training, hr_compliance, etc.)
        department: Specific department or "Company-wide"

    Returns:
        Detailed compliance status and requirements
    """
    # Mock implementation
    import random

    compliance_pct = random.randint(85, 100)
    status = "COMPLIANT" if compliance_pct >= 90 else "NEEDS ATTENTION"
    dept_code = department[:3].upper()

    if compliance_pct < 95:
        actions_text = """REQUIRED ACTIONS:
  • Complete outstanding training modules
  • Update security protocols
  • Review and sign policy updates"""
    else:
        actions_text = (
            "No immediate actions required. Maintain current compliance level."
        )

    return f"""Compliance Status Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Area: {policy_area.upper()}
Scope: {department}
Status: {status} ({compliance_pct}%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUDIT HISTORY:
  Last Audit: January 15, 2024
  Auditor: External Compliance Firm
  Findings: {random.randint(0, 3)} minor issues
  Remediation: {"Complete" if compliance_pct >= 95 else "In Progress"}

CERTIFICATIONS:
  {"✓" if compliance_pct >= 95 else "⚠"} ISO 27001 (Information Security)
  {"✓" if compliance_pct >= 95 else "⚠"} SOC 2 Type II
  {"✓" if compliance_pct >= 90 else "⚠"} GDPR Compliance
  {"✓" if compliance_pct >= 90 else "⚠"} OSHA Safety Standards

UPCOMING REQUIREMENTS:
  • Next Audit: July 2024
  • Annual Training: Due March 31, 2024
  • Policy Review: Quarterly
  • Certification Renewal: Q3 2024

{actions_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Questions? compliance@company.com
Report: COMP-{dept_code}-2024-01"""


@function_tool
async def request_policy_clarification(
    policy_name: str, specific_question: str, employee_id: str
) -> str:
    """Request clarification on a specific policy question.

    Args:
        policy_name: Name of the policy
        specific_question: Specific question about the policy
        employee_id: Employee ID requesting clarification

    Returns:
        Clarification request confirmation
    """
    # Mock implementation
    import random

    ticket_id = f"POL-{random.randint(10000, 99999)}"

    return f"""Policy Clarification Request Submitted
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ticket ID: {ticket_id}
Employee: {employee_id}
Policy: {policy_name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR QUESTION:
{specific_question}

RESPONSE TIMELINE:
  Priority: Normal
  Expected Response: 1-2 business days
  Assigned To: Policy Team

You will receive a response via email from:
policy-support@company.com

ADDITIONAL RESOURCES:
  📚 Policy Portal: intranet.company.com/policies
  📧 Email: policy@company.com
  📞 Policy Hotline: 1-800-POLICY-1

Reference your ticket ID ({ticket_id}) in any
follow-up communications."""


@function_tool
async def search_policy_documents(keyword: str) -> str:
    """Search company policy documents by keyword.

    Args:
        keyword: Search term (e.g., "vacation", "laptop", "expense")

    Returns:
        Relevant policy documents and sections
    """
    # Mock implementation
    results = {
        "vacation": [
            "PTO Policy (Section 2.1)",
            "Holiday Schedule 2024",
            "Leave Request Process",
        ],
        "laptop": [
            "IT Equipment Policy",
            "Security Policy (Device Management)",
            "Remote Work Policy",
        ],
        "expense": [
            "Expense Reimbursement Policy",
            "Travel Policy",
            "Corporate Card Guidelines",
        ],
        "remote": [
            "Remote Work Policy",
            "Home Office Setup Guidelines",
            "VPN Usage Policy",
        ],
        "travel": [
            "Business Travel Policy",
            "Expense Reimbursement",
            "Travel Booking Procedures",
        ],
    }

    found = []
    for key, policies in results.items():
        if keyword.lower() in key:
            found.extend(policies)

    if not found:
        found = [
            "General Employee Handbook",
            "Contact policy@company.com for specific information",
        ]

    return (
        f"""Policy Search Results: "{keyword}"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Found {len(found)} relevant document(s):

"""
        + "\n".join([f"  {i + 1}. {doc}" for i, doc in enumerate(found)])
        + """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Access full documents:
  🌐 intranet.company.com/policies
  📱 Mobile App: Company Policies

Need help? policy@company.com"""
    )
