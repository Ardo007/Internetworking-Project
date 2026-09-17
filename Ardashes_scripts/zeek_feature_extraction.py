import json
import sys


def read_dns_log(file_path):
    records = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)
            records.append(record)

    return records


if __name__ == "__main__":
    dns_log = sys.argv[1]

    records = read_dns_log(dns_log)

    print(f"Loaded {len(records)} DNS records")

    for record in records[:5]:
        print(record.get("query"))