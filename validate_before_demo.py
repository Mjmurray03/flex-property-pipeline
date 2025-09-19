def validate_demo_readiness():
    """Pre-demo validation checklist"""

    print("DEMO READINESS CHECKLIST")
    print("="*40)

    checks = []

    # Check 1: File exists
    import os
    file_exists = os.path.exists(r'C:\flex-property-pipeline\data\raw\Full Property Export.xlsx')
    checks.append(("Excel file exists", file_exists))

    # Check 2: Can import modules
    try:
        from processors.private_property_analyzer import PrivatePropertyAnalyzer
        can_import = True
    except:
        can_import = False
    checks.append(("Can import analyzer", can_import))

    # Check 3: MongoDB connection (optional)
    try:
        from utils.mongodb_client import get_db_manager
        db = get_db_manager()
        db_connected = True
    except:
        db_connected = False
    checks.append(("MongoDB connected", db_connected))

    # Check 4: Output directory writable
    try:
        test_file = r'C:\flex-property-pipeline\demo\output\test.txt'
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        can_write = True
    except:
        can_write = False
    checks.append(("Can write output", can_write))

    # Display results
    for check, passed in checks:
        status = "Pass" if passed else "Fail"
        print(f"{status} {check}")

    if all(c[1] for c in checks):
        print("\nREADY FOR DEMO!")
    else:
        print("\nFix issues before demo")

    return all(c[1] for c in checks)

if __name__ == "__main__":
    validate_demo_readiness()