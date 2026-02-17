# Company Multi-Agent System with Handoff

This is a multi-agent system that demonstrates the handoff pattern using UiPath OpenAI Agents. The system uses an orchestrator agent to route employee requests to specialized department agents.

## Architecture

### Agent Hierarchy

```
orchestrator_agent (Main Entry Point)
    ├── hr_agent (Human Resources)
    ├── procurement_agent (Purchasing & Orders)
    └── policy_agent (Company Policies & Compliance)
```

### Flow Diagram

See `agent.mermaid` for the visual flow diagram showing how requests are routed between agents.

## Agents

### 1. Orchestrator Agent
**Role:** Routes incoming requests to the appropriate department specialist

**Routing Logic:**
- HR matters (benefits, leave, salary, PTO) → `hr_agent`
- Procurement matters (purchases, orders, budgets) → `procurement_agent`
- Policy matters (company policies, compliance) → `policy_agent`

### 2. HR Agent
**Specialization:** Human Resources and Employee Services

**Tools:**
- `check_employee_benefits(employee_id)` - View employee benefits
- `submit_leave_request(employee_id, leave_type, days)` - Submit PTO requests
- `get_salary_info(employee_id)` - Access salary information

**Example Queries:**
- "I need to submit a leave request for 5 days"
- "What are my employee benefits?"
- "When is my next salary review?"

### 3. Procurement Agent
**Specialization:** Purchasing and Budget Management

**Tools:**
- `create_purchase_order(item, quantity, vendor)` - Create POs
- `check_budget_availability(department, amount)` - Verify budgets
- `track_order_status(po_number)` - Track shipments

**Example Queries:**
- "I need to order 50 laptops from Dell"
- "Check if Engineering has budget for $10,000"
- "Track purchase order PO-2024-0050"

### 4. Policy Agent
**Specialization:** Company Policies and Compliance

**Tools:**
- `get_company_policy(policy_type)` - Retrieve policy information
- `check_compliance_status(policy_area)` - Check compliance status

**Example Queries:**
- "What is the remote work policy?"
- "Show me the expense reimbursement policy"
- "Check compliance status for data security"

## Testing

### Test Files Included

1. **HR Test:** `input_hr.json`
   ```json
   {
     "message": "I need to submit a leave request for 5 days of vacation. My employee ID is EMP12345."
   }
   ```

2. **Procurement Test:** `input_procurement.json`
   ```json
   {
     "message": "I need to order 50 laptops from Dell for the Engineering department."
   }
   ```

3. **Policy Test:** `input_policy.json`
   ```json
   {
     "message": "What is the company's remote work policy?"
   }
   ```

### Running Tests

```bash
# Test HR agent
uv run uipath run agent --input-file input_hr.json

# Test Procurement agent
uv run uipath run agent --input-file input_procurement.json

# Test Policy agent
uv run uipath run agent --input-file input_policy.json
```

## How It Works

### Handoff Pattern

The system uses the **handoffs** parameter in the Agent configuration:

```python
orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions="Route requests to appropriate specialists...",
    handoffs=[hr_agent, procurement_agent, policy_agent],
)
```

### Key Features

1. **Automatic Routing:** The orchestrator analyzes the request and hands off to the right specialist
2. **Specialized Tools:** Each agent has domain-specific tools
3. **Context Preservation:** Conversation context is maintained during handoffs
4. **Return Flow:** Specialists return control to orchestrator after completion

### Execution Flow

1. User sends request → `orchestrator_agent`
2. Orchestrator analyzes request type
3. Orchestrator hands off to specialist (e.g., `hr_agent`)
4. Specialist uses its tools to fulfill request
5. Specialist returns result to orchestrator
6. Orchestrator delivers final response

## Customization

### Adding New Agents

1. Create a new specialist agent with its tools:
   ```python
   finance_agent = Agent(
       name="finance_agent",
       instructions="Handle financial queries...",
       tools=[your_finance_tools],
   )
   ```

2. Add to orchestrator's handoffs:
   ```python
   orchestrator_agent = Agent(
       handoffs=[hr_agent, procurement_agent, policy_agent, finance_agent],
   )
   ```

3. Update orchestrator instructions to include routing logic

### Adding New Tools

1. Define a function tool:
   ```python
   @function_tool
   async def your_tool(param: str) -> str:
       """Tool description."""
       # Implementation
       return result
   ```

2. Add to the appropriate agent's tools list

## Benefits of This Pattern

- **Separation of Concerns:** Each agent handles one domain
- **Scalability:** Easy to add new departments/agents
- **Maintainability:** Tools and logic are isolated per domain
- **Flexibility:** Orchestrator can route based on complex logic
- **Testability:** Each agent can be tested independently

## Production Considerations

For production use, replace mock implementations with:
- Real database queries
- API integrations
- Authentication/authorization checks
- Audit logging
- Error handling and fallbacks

## Related Documentation

- See `@.agent/REQUIRED_STRUCTURE.md` for agent structure patterns
- See `@.agent/SDK_REFERENCE.md` for UiPath SDK methods
- See parent samples for other handoff patterns (triage-agent, message_filter)
