"""Employee database and resolution utilities."""

# Employee name to ID mapping (mock employee database)
EMPLOYEE_DATABASE = {
    "eusebiu jecan": "EMP67890",
    "sarah martinez": "EMP12345",
    "john smith": "EMP54321",
    "emily chen": "EMP98765",
    "michael brown": "EMP11111",
}


def resolve_employee_id(identifier: str) -> tuple[str, str]:
    """Resolve employee name or ID to standardized employee ID and name.

    Args:
        identifier: Employee name or ID

    Returns:
        Tuple of (employee_id, employee_name)
    """
    identifier_lower = identifier.lower().strip()

    # Check if it's a name in our database
    if identifier_lower in EMPLOYEE_DATABASE:
        emp_id = EMPLOYEE_DATABASE[identifier_lower]
        emp_name = identifier.title()
        return emp_id, emp_name

    # Check if it's an employee ID
    if identifier.startswith("EMP"):
        # Reverse lookup for name
        for name, emp_id in EMPLOYEE_DATABASE.items():
            if emp_id == identifier:
                return identifier, name.title()
        return identifier, "Employee"

    # Default: treat as ID
    return identifier, "Employee"
