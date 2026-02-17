"""Procurement department tools for purchasing and vendor management."""

from agents import function_tool


@function_tool
async def check_budget_availability(
    department: str, amount: float, category: str = "General"
) -> str:
    """Check if budget is available for a department and spending category.

    Args:
        department: Department name (Engineering, Marketing, Sales, etc.)
        amount: Amount to check in USD
        category: Budget category (Equipment, Software, Travel, Supplies, etc.)

    Returns:
        Detailed budget availability status
    """
    # Mock implementation
    import hashlib

    hash_val = int(hashlib.md5(department.encode()).hexdigest(), 16)
    available = amount * (1.2 + (hash_val % 100) / 100)
    allocated = available * 1.5
    spent = allocated - available
    percent_used = (spent / allocated) * 100

    status = "APPROVED" if available >= amount else "INSUFFICIENT FUNDS"

    return f"""Budget Analysis - {department}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Category: {category}
Requested Amount: ${amount:,.2f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUDGET SUMMARY:
  Total Allocated (FY2024): ${allocated:,.2f}
  Spent to Date: ${spent:,.2f}
  Remaining Balance: ${available:,.2f}
  Budget Utilized: {percent_used:.1f}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STATUS: {status}
{"✓ Sufficient funds available for this purchase." if status == "APPROVED" else "✗ Insufficient funds. Request budget reallocation."}

Approval Required: {"Manager only" if amount < 5000 else "Director + Finance"}"""


@function_tool
async def get_vendor_information(vendor_name: str) -> str:
    """Get information about an approved vendor.

    Args:
        vendor_name: Name of the vendor

    Returns:
        Vendor details and rating
    """
    # Mock implementation
    import random

    rating = round(3.5 + random.random() * 1.5, 1)

    return f"""Vendor Profile - {vendor_name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: APPROVED VENDOR ✓
Vendor ID: VEN-{abs(hash(vendor_name)) % 10000:04d}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RATING: {"⭐" * int(rating)} ({rating}/5.0)
Total Orders: {random.randint(50, 500)}
On-Time Delivery: {random.randint(85, 98)}%
Quality Score: {random.randint(80, 95)}/100

PAYMENT TERMS:
  Standard Terms: Net 30
  Early Payment Discount: 2% (Net 10)
  Accepted Methods: Wire, ACH, Check

CONTACT:
  Account Manager: {random.choice(["John Smith", "Mary Johnson", "David Chen"])}
  Phone: 1-800-VENDOR-{random.randint(1, 9)}
  Email: orders@{vendor_name.lower().replace(" ", "")}.com

CONTRACT STATUS:
  Master Agreement: Active
  Expires: Dec 31, 2024
  Discount Tier: Gold (15% off standard)"""


@function_tool
async def create_purchase_order(
    item: str,
    quantity: int,
    unit_price: float,
    vendor: str,
    department: str,
    justification: str,
) -> str:
    """Create a detailed purchase order and initiate approval workflow.

    Args:
        item: Item description
        quantity: Quantity to order
        unit_price: Price per unit in USD
        vendor: Vendor name
        department: Requesting department
        justification: Business justification for purchase

    Returns:
        Purchase order confirmation with approval workflow details
    """
    # Mock implementation
    import random

    po_number = f"PO-2024-{random.randint(10000, 99999)}"
    total_amount = quantity * unit_price
    tax = total_amount * 0.08
    shipping = 50 if total_amount < 1000 else 0
    grand_total = total_amount + tax + shipping

    approval_level = (
        "Manager"
        if grand_total < 5000
        else "Director"
        if grand_total < 25000
        else "VP + Finance"
    )

    return f"""✓ Purchase Order Created Successfully!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PO NUMBER: {po_number}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VENDOR: {vendor}
Department: {department}
Requested By: Current User
Date: 2024-01-28

ITEMS:
  Description: {item}
  Quantity: {quantity}
  Unit Price: ${unit_price:,.2f}
  Subtotal: ${total_amount:,.2f}

COST BREAKDOWN:
  Subtotal: ${total_amount:,.2f}
  Tax (8%): ${tax:,.2f}
  Shipping: ${shipping:,.2f}
  ━━━━━━━━━━━━━━━━━━━━━━
  GRAND TOTAL: ${grand_total:,.2f}

JUSTIFICATION:
  {justification}

APPROVAL WORKFLOW:
  Status: PENDING APPROVAL
  Required Approver: {approval_level}
  Expected Approval: 1-3 business days
  Auto-notification sent to approvers

NEXT STEPS:
1. Approval notification sent
2. Upon approval, PO sent to vendor
3. Track shipment with PO number
4. Goods Receipt in 7-14 business days

Reference: {po_number}"""


@function_tool
async def track_order_status(po_number: str) -> str:
    """Track the detailed status of a purchase order.

    Args:
        po_number: Purchase order number (format: PO-2024-XXXXX)

    Returns:
        Comprehensive order tracking information
    """
    # Mock implementation
    import random

    statuses = [
        (
            "PENDING_APPROVAL",
            "Purchase order awaiting management approval",
            "N/A",
            "N/A",
        ),
        (
            "APPROVED",
            "Order approved, sent to vendor",
            "TRK-2024-" + str(random.randint(10000, 99999)),
            "5-7 business days",
        ),
        (
            "IN_TRANSIT",
            "Order shipped and in transit",
            "TRK-2024-" + str(random.randint(10000, 99999)),
            "2-3 business days",
        ),
        (
            "DELIVERED",
            "Order delivered successfully",
            "TRK-2024-" + str(random.randint(10000, 99999)),
            "Delivered on 2024-01-25",
        ),
    ]
    status, description, tracking, eta = random.choice(statuses)

    return f"""Order Tracking - {po_number}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current Status: {status}
{description}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIMELINE:
  ✓ Order Created: 2024-01-20 10:30 AM
  {"✓" if status != "PENDING_APPROVAL" else "○"} Approved: 2024-01-21 2:15 PM
  {"✓" if status in ["IN_TRANSIT", "DELIVERED"] else "○"} Shipped: 2024-01-23 9:00 AM
  {"✓" if status == "DELIVERED" else "○"} Delivered: {eta}

SHIPMENT DETAILS:
  Carrier: FedEx Express
  Tracking Number: {tracking}
  Estimated Delivery: {eta}
  Shipping Method: Ground

VENDOR CONTACT:
  Need to modify order? Contact vendor within 24hrs of PO creation
  Vendor Support: 1-800-VENDOR-1"""


@function_tool
async def request_budget_reallocation(
    from_category: str,
    to_category: str,
    amount: float,
    justification: str,
    department: str,
) -> str:
    """Request to move budget between categories within a department.

    Args:
        from_category: Source budget category
        to_category: Destination budget category
        amount: Amount to reallocate in USD
        justification: Business justification
        department: Department name

    Returns:
        Budget reallocation request confirmation
    """
    # Mock implementation
    import random

    request_id = f"BR-2024-{random.randint(1000, 9999)}"

    return f"""Budget Reallocation Request Submitted
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Request ID: {request_id}
Department: {department}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REALLOCATION DETAILS:
  From Category: {from_category}
  To Category: {to_category}
  Amount: ${amount:,.2f}

JUSTIFICATION:
  {justification}

APPROVAL WORKFLOW:
  Status: PENDING REVIEW
  Required Approver: Finance Director
  Expected Response: 2-3 business days
  Priority: Normal

You will be notified via email once reviewed.
Track status with Request ID: {request_id}"""


@function_tool
async def search_preferred_vendors(category: str) -> str:
    """Search for preferred vendors by product/service category.

    Args:
        category: Product or service category (Electronics, Software, Office Supplies, etc.)

    Returns:
        List of approved preferred vendors
    """
    # Mock implementation
    vendors_db = {
        "Electronics": ["Dell Technologies", "HP Inc", "Lenovo", "Apple Business"],
        "Software": ["Microsoft", "Adobe", "Salesforce", "Atlassian"],
        "Office Supplies": ["Staples Business", "Office Depot", "Amazon Business"],
        "Furniture": ["Herman Miller", "Steelcase", "IKEA Business"],
    }

    vendor_list = vendors_db.get(
        category, ["Contact Procurement for vendor recommendations"]
    )

    return (
        f"""Preferred Vendors - {category}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Approved vendor list for company purchases:

"""
        + "\n".join(
            [f"  {i + 1}. {v} (Contract on file)" for i, v in enumerate(vendor_list)]
        )
        + """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Benefits of using preferred vendors:
  ✓ Pre-negotiated pricing (10-20% discount)
  ✓ Expedited approval process
  ✓ Established payment terms
  ✓ Dedicated account manager
  ✓ Priority support

Need a vendor not listed? Submit vendor approval request to procurement@company.com"""
    )
