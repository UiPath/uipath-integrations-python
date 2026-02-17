# Sample Input Files

This folder contains sample input JSON files for testing the Company Multi-Agent System.

## Test Files

### Current Company Agent System

| File | Purpose | Agent | Description |
|------|---------|-------|-------------|
| `input_hr.json` | HR Leave Request | hr_agent | Request 5 days vacation (Feb 15-20) |
| `input_procurement.json` | Procurement Order | procurement_agent | Order 50 Dell laptops for Engineering |
| `input_policy.json` | Policy/Leave Request | hr_agent | Request 3 days vacation in March |
| `input_march_complete.json` | Complete Leave Flow | hr_agent | Full leave request (March 11-13) |

### Legacy/Old Test Files

| File | Purpose | Notes |
|------|---------|-------|
| `input.json` | Original test | Legacy weather query |
| `input_weather.json` | Weather test | From original weather agent |
| `input_knowledge.json` | Knowledge test | From original knowledge agent |
| `input_march_leave.json` | March leave | Alternative March request |

---

## Usage

### Run with Sample Input

```bash
# From the basic/ directory
uv run uipath run agent --input-file sampleinputs/input_hr.json
uv run uipath run agent --input-file sampleinputs/input_procurement.json
uv run uipath run agent --input-file sampleinputs/input_policy.json
```

---

## File Contents

### input_hr.json
```json
{
  "message": "I want to take a vacation from February 15 to February 20, 2024. Can you check if I have enough PTO days available and submit the request? My employee ID is EMP12345."
}
```
**Tests:** HR agent → PTO balance check → Leave request submission

---

### input_procurement.json
```json
{
  "message": "I need to order 50 Dell XPS laptops at $1,200 each for the Engineering department. This is for our new hire onboarding program. Can you check if we have budget in the Equipment category and create a purchase order?"
}
```
**Tests:** Procurement agent → Budget check → Vendor info → PO creation

---

### input_policy.json
```json
{
  "message": "Hi, I'm Eusebiu Jecan and I need to request 3 days off from March 11-13, 2024 for a family vacation. Can you check my PTO balance and submit the leave request?"
}
```
**Tests:** Orchestrator routing (policy→hr) → Name resolution → PTO check → Leave request

---

### input_march_complete.json
```json
{
  "message": "Hi, I'm Eusebiu Jecan and I need to submit a vacation leave request for March 11-13, 2024 (3 days). Please check my balance and submit the request for me."
}
```
**Tests:** Complete workflow with name-based lookup and direct submission

---

## Creating New Test Inputs

Template for new test files:

```json
{
  "message": "Your test message here"
}
```

### Tips:
- Include employee name for HR requests (e.g., "I'm John Smith")
- Specify dates for leave requests (format: Month DD-DD, YYYY)
- Include quantities and prices for procurement
- Mention specific policies by name (remote_work, expense, pto, etc.)

---

## Expected Behaviors

### HR Requests
1. Orchestrator routes to `hr_agent`
2. Agent checks PTO balance first
3. Validates sufficient days available
4. Submits request if approved
5. Returns Request ID for tracking

### Procurement Requests
1. Orchestrator routes to `procurement_agent`
2. Agent checks budget availability
3. Gets vendor information if needed
4. Creates purchase order
5. Returns PO number and approval requirements

### Policy Requests
1. Orchestrator routes to `policy_agent`
2. Agent retrieves policy details
3. Formats and presents policy
4. Offers clarification if needed

---

## Test All Agents

```bash
# Quick test all three departments
for file in input_hr.json input_procurement.json input_policy.json; do
  echo "Testing $file..."
  uv run uipath run agent --input-file sampleinputs/$file
  echo "---"
done
```

---

## Troubleshooting

### File Not Found Error
```bash
# Make sure you're in the basic/ directory
cd samples/basic

# Or use absolute path
uv run uipath run agent --input-file /full/path/to/sampleinputs/input_hr.json
```

### Agent Not Routing Correctly
- Check that your message clearly indicates the department (HR/Procurement/Policy)
- Review orchestrator instructions in main.py
- Test with the sample files first to verify system is working

---

Last Updated: 2024-01-28
