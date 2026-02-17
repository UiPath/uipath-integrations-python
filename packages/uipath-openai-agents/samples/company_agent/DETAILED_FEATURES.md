# Detailed Company Agent Features

## Overview

This is an enhanced multi-agent system with detailed tools for HR, Procurement, and Policy departments. Each agent has comprehensive functionality with realistic workflows and detailed responses.

---

## HR Agent - Human Resources Specialist

### Capabilities

The HR agent handles all employee-related queries with 6 specialized tools:

#### 1. **PTO Balance Checking** (`check_pto_balance`)
- Shows annual allowance (20 days)
- Displays days used and remaining
- Shows accrual rate (1.67 days/month)
- Provides carryover information (5-day max)

**Example Output:**
```
PTO Balance for Employee EMP12345:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Annual Allowance: 20 days
Used to Date: 8 days
REMAINING BALANCE: 12 days
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accrual Rate: 1.67 days/month
Next Accrual Date: 1st of next month
```

#### 2. **Leave Request Submission** (`submit_leave_request`)
- Generates unique request ID
- Starts approval workflow
- Provides expected timeline
- Shows manager information
- Returns detailed confirmation

**Example Output:**
```
✓ Leave Request Submitted Successfully!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Request ID: LR-2024-45789
Employee: EMP12345
Leave Type: VACATION
Start Date: 2024-02-15
End Date: 2024-02-20
Total Days: 5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: PENDING APPROVAL
Approver: Manager - Sarah Johnson
Expected Response: Within 2 business days
```

#### 3. **Leave Request Status Tracking** (`check_leave_request_status`)
- Track approval status
- See reviewer information
- Get last update timestamp

#### 4. **Comprehensive Benefits Information** (`check_employee_benefits`)
Returns detailed breakdown of:
- Health & Wellness (Medical, Dental, Vision, Mental Health)
- Retirement (401k with 5% match)
- Time Off (PTO, Sick Leave, Holidays)
- Additional Benefits (Life Insurance, Professional Development, etc.)

#### 5. **Salary & Compensation Details** (`get_salary_info`)
- Annual, monthly, and bi-weekly salary
- Compensation history
- Upcoming review dates
- Total compensation package value

#### 6. **HR Meeting Scheduling** (`schedule_hr_meeting`)
- Schedule meetings with HR representatives
- Get meeting confirmation with Zoom links
- Specify topic and preferred date

### HR Agent Workflow

**For Leave Requests, the agent follows this process:**
1. Check PTO balance first
2. Show remaining days to employee
3. If sufficient balance, submit the request
4. Provide confirmation with request ID
5. Remind about pending manager approval

---

## Procurement Agent - Purchasing Specialist

### Capabilities

The Procurement agent manages all purchasing activities with 6 specialized tools:

#### 1. **Budget Availability Check** (`check_budget_availability`)
- Shows total allocated budget for fiscal year
- Displays spent amount and remaining balance
- Calculates budget utilization percentage
- Indicates approval requirements based on amount

**Example Output:**
```
Budget Analysis - Engineering
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Category: Equipment
Requested Amount: $60,000.00
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUDGET SUMMARY:
  Total Allocated (FY2024): $90,000.00
  Spent to Date: $18,000.00
  Remaining Balance: $72,000.00
  Budget Utilized: 20.0%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STATUS: APPROVED
✓ Sufficient funds available for this purchase.

Approval Required: Director + Finance
```

#### 2. **Vendor Information** (`get_vendor_information`)
- Vendor rating (out of 5 stars)
- Total orders and on-time delivery percentage
- Payment terms and discounts
- Account manager contact
- Contract status and expiration

#### 3. **Purchase Order Creation** (`create_purchase_order`)
Creates detailed PO with:
- Unique PO number
- Item details with cost breakdown
- Tax and shipping calculations
- Grand total
- Approval workflow information
- Expected timeline

**Parameters:**
- Item description
- Quantity
- Unit price
- Vendor name
- Department
- Business justification

#### 4. **Order Tracking** (`track_order_status`)
- Current order status (Pending/Approved/In Transit/Delivered)
- Timeline with checkpoints
- Carrier and tracking number
- Estimated delivery date
- Vendor contact information

#### 5. **Budget Reallocation** (`request_budget_reallocation`)
- Move funds between budget categories
- Submit justification
- Track request ID
- Get approval timeline

#### 6. **Preferred Vendor Search** (`search_preferred_vendors`)
- Find approved vendors by category
- See pre-negotiated discounts (10-20% off)
- Get benefits of using preferred vendors
- Categories: Electronics, Software, Office Supplies, Furniture

### Procurement Agent Workflow

**For Purchase Orders, the agent follows this process:**
1. Understand purchase requirements
2. Check budget availability FIRST
3. Get vendor information if needed
4. Create detailed purchase order
5. Provide PO number and approval workflow
6. Remind about approval requirements based on amount

**Approval Thresholds:**
- < $5,000: Manager approval only
- $5,000 - $25,000: Director approval
- > $25,000: VP + Finance approval

---

## Policy Agent - Company Policy & Compliance Specialist

### Capabilities

The Policy agent provides policy information and compliance status with 4 specialized tools:

#### 1. **Company Policy Retrieval** (`get_company_policy`)

**Available Policies:**
- **remote_work**: Remote work guidelines (up to 3 days/week, core hours, requirements)
- **expense**: Expense reimbursement (meal limits, travel pre-approval, submission)
- **code_of_conduct**: Ethics and professional behavior standards
- **pto**: PTO accrual, usage, carryover, and blackout periods
- **security**: Information security (passwords, device security, data handling)
- **travel**: Business travel booking guidelines and limits

Each policy includes:
- Detailed requirements and guidelines
- Specific limits and thresholds
- Submission/approval processes
- Contact information
- Policy number and effective date

#### 2. **Compliance Status Checking** (`check_compliance_status`)
- Department or company-wide compliance
- Compliance percentage
- Audit history and findings
- Certification status (ISO 27001, SOC 2, GDPR, OSHA)
- Upcoming requirements
- Required actions if needed

#### 3. **Policy Clarification Requests** (`request_policy_clarification`)
- Submit specific policy questions
- Get ticket ID for tracking
- Expected response timeline (1-2 business days)
- Assigned to policy team

#### 4. **Policy Document Search** (`search_policy_documents`)
- Search by keyword
- Find relevant policy sections
- Get links to full documents
- Access via intranet or mobile app

---

## Testing Examples

### Test HR Agent
```bash
uv run uipath run agent --input-file input_hr.json
```

**Input:**
```json
{
  "message": "I want to take a vacation from February 15 to February 20, 2024. Can you check if I have enough PTO days available and submit the request? My employee ID is EMP12345."
}
```

**Expected Flow:**
1. Agent hands off to `hr_agent`
2. Checks PTO balance → Shows remaining days
3. Submits leave request → Returns confirmation with request ID
4. Provides next steps and timeline

---

### Test Procurement Agent
```bash
uv run uipath run agent --input-file input_procurement.json
```

**Input:**
```json
{
  "message": "I need to order 50 Dell XPS laptops at $1,200 each for the Engineering department. This is for our new hire onboarding program. Can you check if we have budget in the Equipment category and create a purchase order?"
}
```

**Expected Flow:**
1. Agent hands off to `procurement_agent`
2. Checks budget for Engineering/Equipment → Shows $60,000 request is approved
3. Gets vendor information for Dell
4. Creates purchase order → Returns PO number and details
5. Indicates Director + Finance approval required

---

### Test Policy Agent
```bash
uv run uipath run agent --input-file input_policy.json
```

**Input:**
```json
{
  "message": "What is the company's remote work policy? I'd like to know how many days I can work from home."
}
```

**Expected Flow:**
1. Agent hands off to `policy_agent`
2. Retrieves remote_work policy
3. Returns formatted policy with:
   - Up to 3 days/week eligibility
   - Core hours requirement (10 AM - 3 PM)
   - Equipment and security requirements
   - Contact information

---

## Key Implementation Features

### 1. **Realistic Mock Data**
- Hash-based consistent data generation
- Different values for different employees/departments
- Realistic ranges and variations

### 2. **Professional Formatting**
- Unicode box drawing characters for clean tables
- Structured sections with clear headers
- Status indicators (✓, ✗, ⚠️, ○)
- Color-coded information

### 3. **Comprehensive Information**
- Policy numbers and effective dates
- Contact information and support channels
- Reference IDs for tracking
- Timeline expectations

### 4. **Workflow Logic**
- Sequential tool calls (check first, then act)
- Validation and approval requirements
- Status tracking and updates
- Next steps guidance

### 5. **Scalability**
- Easy to add new tools
- Easy to add new agents
- Easy to modify policies
- Template-based responses

---

## Benefits of This Implementation

1. **Employee Self-Service**: Employees can get instant answers without waiting for human agents
2. **Consistent Information**: Same policy information provided to everyone
3. **Audit Trail**: All requests generate tracking IDs
4. **Process Compliance**: Built-in approval workflows
5. **Time Savings**: Automated budget checks, status tracking, policy lookups

---

## Future Enhancements

Potential additions for production:
- Integration with actual HRIS systems (Workday, BambooHR)
- Real-time PTO balance from database
- Integration with procurement systems (SAP, Oracle)
- Real vendor APIs for pricing and availability
- Document management system integration
- Email notifications for approvals
- Mobile app integration
- Analytics and reporting dashboard

---

## Technical Notes

### Agent Configuration
All agents use `gpt-4o-mini` model for cost efficiency while maintaining quality responses.

### Tool Organization
Tools are organized by department with clear naming conventions:
- HR tools: `check_*`, `submit_*`, `schedule_*`, `get_*`
- Procurement tools: `check_*`, `create_*`, `track_*`, `request_*`, `search_*`
- Policy tools: `get_*`, `check_*`, `request_*`, `search_*`

### Error Handling
All tools include mock implementations that always succeed for demo purposes. Production implementations should include:
- Database connection error handling
- API timeout handling
- Validation error messages
- Permission checks
- Rate limiting

---

For questions or issues, refer to the main README or contact the development team.
