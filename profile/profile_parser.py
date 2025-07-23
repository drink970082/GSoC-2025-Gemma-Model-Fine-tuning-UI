import sys

logfile = "import_profile.log"
with open(logfile) as f:
    lines = [l for l in f if "import time:" in l]


# Find slowest line
def is_data_line(l):
    try:
        int(l.split("|")[1])
        return True
    except Exception:
        return False


data_lines = [l for l in lines if is_data_line(l)]
# Sort by cumulative time (column 2) and get top 5
top_5 = sorted(data_lines, key=lambda l: int(l.split("|")[1]), reverse=True)[:5]

print("Top 5 slowest imports:")
for i, line in enumerate(top_5, 1):
    print(f"{i}. {line.strip()}")

print("\n" + "=" * 50)

# For each of the top 5, show only the top parent
for i, slowest in enumerate(top_5, 1):
    print(f"\n{i}. Top parent for: {slowest.strip()}")
    slowest_idx = lines.index(slowest)
    slowest_indent = len(slowest) - len(slowest.lstrip(" "))

    # Walk up to find the very top parent
    top_parent = None
    for j in range(slowest_idx, -1, -1):
        line = lines[j]
        indent = len(line) - len(line.lstrip(" "))
        if indent <= slowest_indent and line != slowest:  # Changed from < to <=
            top_parent = line.strip()
            break  # Stop at the first (topmost) parent

    if top_parent:
        print(f"  Root: {top_parent}")
    else:
        print("  No parent found (root import)")
