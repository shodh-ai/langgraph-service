import sys
import pkg_resources

print("Python path:")
for path in sys.path:
    print(f"  {path}")

print("\nInstalled packages:")
for pkg in pkg_resources.working_set:
    print(f"  {pkg.project_name} {pkg.version}")

print("\nTrying to import mem0ai:")
try:
    import mem0ai
    print(f"mem0ai imported successfully")
    print(f"mem0ai path: {mem0ai.__file__}")
    print(f"mem0ai contents: {dir(mem0ai)}")
except ImportError as e:
    print(f"Import error: {e}")

# Try alternative import names
for name in ["mem0", "mem0_ai", "mem0-ai", "mem0.ai", "mem0ai.memory", "mem0ai.Memory"]:
    try:
        print(f"\nTrying to import {name}:")
        if "." in name:
            parts = name.split(".")
            module = __import__(parts[0])
            for part in parts[1:]:
                module = getattr(module, part)
        else:
            module = __import__(name)
        print(f"{name} imported successfully")
        print(f"{name} path: {getattr(module, '__file__', 'No __file__ attribute')}")
        print(f"{name} contents: {dir(module)}")
    except (ImportError, AttributeError) as e:
        print(f"Error: {e}")

# Try to find the Memory class specifically
print("\nLooking for Memory class in installed packages:")
for pkg in pkg_resources.working_set:
    pkg_name = pkg.project_name
    if "mem" in pkg_name.lower():
        print(f"  Checking {pkg_name}...")
        try:
            module = __import__(pkg_name)
            if hasattr(module, "Memory"):
                print(f"  Found Memory class in {pkg_name}")
                print(f"  {pkg_name}.Memory: {module.Memory}")
            else:
                print(f"  No Memory class in {pkg_name} top level")
                # Try to look in submodules
                for item_name in dir(module):
                    if not item_name.startswith("_"):
                        try:
                            item = getattr(module, item_name)
                            if hasattr(item, "Memory"):
                                print(f"  Found Memory class in {pkg_name}.{item_name}")
                                print(f"  {pkg_name}.{item_name}.Memory: {item.Memory}")
                        except:
                            pass
        except Exception as e:
            print(f"  Error checking {pkg_name}: {e}")
