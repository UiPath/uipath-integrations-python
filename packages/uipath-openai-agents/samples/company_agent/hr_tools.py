"""HR department tools for employee management."""

from agents import function_tool
from employees import resolve_employee_id


@function_tool
async def check_pto_balance(employee_identifier: str) -> str:
    """Check remaining PTO (Paid Time Off) balance for an employee.

    Args:
        employee_identifier: The employee name or ID to look up

    Returns:
        Detailed PTO balance information
    """
    # Resolve name to ID
    employee_id, employee_name = resolve_employee_id(employee_identifier)

    # Mock implementation - simulate different balances
    import hashlib

    hash_val = int(hashlib.md5(employee_id.encode()).hexdigest(), 16)
    remaining_days = (hash_val % 15) + 5  # 5-20 days
    used_days = 20 - remaining_days
    accrual_rate = 1.67

    return f"""PTO Balance for {employee_name} ({employee_id}):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Annual Allowance: 20 days
Used to Date: {used_days} days
REMAINING BALANCE: {remaining_days} days
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accrual Rate: {accrual_rate} days/month
Next Accrual Date: 1st of next month
Carryover Limit: 5 days max
Status: Active"""


@function_tool
async def submit_leave_request(
    employee_identifier: str, leave_type: str, start_date: str, end_date: str, days: int
) -> str:
    """Submit a leave request for an employee and start the approval process.

    Args:
        employee_identifier: The employee name or ID
        leave_type: Type of leave (vacation, sick, personal, unpaid)
        start_date: Start date of leave (format: YYYY-MM-DD)
        end_date: End date of leave (format: YYYY-MM-DD)
        days: Number of days requested

    Returns:
        Leave request confirmation with tracking details
    """
    # Resolve name to ID
    employee_id, employee_name = resolve_employee_id(employee_identifier)

    # Mock implementation
    import random

    request_id = f"LR-2024-{random.randint(10000, 99999)}"

    return f"""✓ Leave Request Submitted Successfully!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Request ID: {request_id}
Employee: {employee_name} ({employee_id})
Leave Type: {leave_type.upper()}
Start Date: {start_date}
End Date: {end_date}
Total Days: {days}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: PENDING APPROVAL
Submitted: 2024-01-28 10:30 AM
Approver: Manager - Sarah Johnson
Expected Response: Within 2 business days
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Next Steps:
1. Your manager will be notified via email
2. You'll receive approval status within 48 hours
3. Track status in HR Portal with Request ID

Note: This is an automated confirmation. Your leave is NOT approved until you receive manager approval."""


@function_tool
async def check_leave_request_status(request_id: str) -> str:
    """Check the status of a submitted leave request.

    Args:
        request_id: The leave request ID (format: LR-2024-XXXXX)

    Returns:
        Current status of the leave request
    """
    # Mock implementation
    import random

    statuses = [
        ("APPROVED", "Your leave request has been approved by your manager."),
        ("PENDING", "Your leave request is awaiting manager review."),
        ("UNDER_REVIEW", "Your manager is currently reviewing your request."),
    ]
    status, message = random.choice(statuses)

    return f"""Leave Request Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Request ID: {request_id}
Status: {status}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{message}
Last Updated: 2024-01-28 2:15 PM
Reviewed By: Sarah Johnson (Manager)"""


@function_tool
async def check_employee_benefits(employee_identifier: str) -> str:
    """Check comprehensive employee benefits information.

    Args:
        employee_identifier: The employee name or ID to look up

    Returns:
        Detailed employee benefits information
    """
    # Resolve name to ID
    employee_id, employee_name = resolve_employee_id(employee_identifier)

    # Mock implementation
    return f"""Employee Benefits Summary - {employee_name} ({employee_id})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HEALTH & WELLNESS:
  • Medical: Blue Cross Premium Plan
    - Deductible: $500 individual / $1,000 family
    - Coverage: 90% after deductible
  • Dental: Delta Dental PPO
  • Vision: VSP Vision Care
  • Mental Health: 24/7 Employee Assistance Program

RETIREMENT:
  • 401(k): 5% company match (fully vested)
  • Current Contribution: 6% of salary
  • Roth 401(k) available

TIME OFF:
  • PTO: 20 days/year (accrued monthly)
  • Sick Leave: 10 days/year
  • Holidays: 11 paid company holidays
  • Parental Leave: 12 weeks paid

ADDITIONAL BENEFITS:
  • Life Insurance: 2x annual salary (company paid)
  • Disability: Short & Long-term coverage
  • Professional Development: $2,000/year
  • Gym Membership: $50/month reimbursement
  • Remote Work: Up to 3 days/week"""


@function_tool
async def get_salary_info(employee_identifier: str) -> str:
    """Get salary and compensation information for an employee.

    Args:
        employee_identifier: The employee name or ID

    Returns:
        Detailed salary information
    """
    # Resolve name to ID
    employee_id, employee_name = resolve_employee_id(employee_identifier)

    # Mock implementation
    import hashlib

    hash_val = int(hashlib.md5(employee_id.encode()).hexdigest(), 16)
    base_salary = 75000 + (hash_val % 40000)

    return f"""Compensation Details - {employee_name} ({employee_id})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BASE SALARY:
  Annual: ${base_salary:,}
  Monthly: ${base_salary // 12:,}
  Bi-weekly: ${base_salary // 26:,}

COMPENSATION HISTORY:
  Last Increase: 3.5% (January 2024)
  Previous Salary: ${int(base_salary / 1.035):,}
  Years at Current Level: 1.2 years

UPCOMING REVIEWS:
  Next Review: June 2024
  Eligible for: Merit increase (2-5%)
  Performance Rating: Meets Expectations

TOTAL COMPENSATION:
  Base Salary: ${base_salary:,}
  401k Match (5%): ${int(base_salary * 0.05):,}
  Benefits Value: ~$15,000
  TOTAL: ~${base_salary + int(base_salary * 0.05) + 15000:,}/year"""


@function_tool
async def schedule_hr_meeting(
    employee_identifier: str, topic: str, preferred_date: str
) -> str:
    """Schedule a meeting with HR department.

    Args:
        employee_identifier: The employee name or ID
        topic: Topic of discussion (benefits, performance, concerns, etc.)
        preferred_date: Preferred date (format: YYYY-MM-DD)

    Returns:
        Meeting confirmation
    """
    # Resolve name to ID
    employee_id, employee_name = resolve_employee_id(employee_identifier)

    # Mock implementation
    import random

    meeting_id = f"HR-MTG-{random.randint(1000, 9999)}"

    return f"""HR Meeting Scheduled
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Meeting ID: {meeting_id}
Employee: {employee_name} ({employee_id})
Topic: {topic}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Scheduled: {preferred_date} at 2:00 PM
Duration: 30 minutes
Location: HR Office / Zoom (link will be sent)
HR Representative: Jennifer Martinez
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Calendar invite sent to your email.
Please bring any relevant documents."""
