import json
import math
import os
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from py_abac import PDP, Policy, AccessRequest
from py_abac.storage.memory import MemoryStorage
import pandas as pd
from pathlib import Path

# Cache for quick log value lookup (avoids computing -log repeatedly)
LOG_CACHE = np.array([-math.log(i / 1000) if i else 25.0 for i in range(1001)], dtype=np.float32)

def default_counter():
    return [0, 0]

def extract_policy_resource_attributes(policy_dicts):
    """
    Extract resource attributes used in policy conditions.
    """
    used_attrs = set()
    for pol in policy_dicts:
        resource_rules = pol.get("rules", {}).get("resource", {})
        for attr in resource_rules:
            if attr.startswith("$."):
                used_attrs.add(attr[2:])
    return used_attrs

def convert_numpy_types(obj):
    """
    Convert NumPy float types into native Python floats for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    return obj

def compute_policy_matrix(policies, policy_attrs):
    """
    Builds a matrix mapping object types and actions to resource attributes used in policies.
    """
    policy_matrix = defaultdict(lambda: defaultdict(set))
    for p in policies:
        try:
            rules = p.rules
            obj_type = rules.resource.get("$.obj_type")
            if obj_type and hasattr(obj_type, 'value'):
                obj_type = obj_type.value

            method_rule = rules.action.get("$.method")
            actions = method_rule.values if hasattr(method_rule, 'values') else [method_rule.value]

            resource_attrs = [
                attr_path[2:] for attr_path, rule in rules.resource.items()
                if attr_path.startswith("$.") and attr_path[2:] in policy_attrs
            ]

            print(f"Policy UID: {p.uid}")
            print(f"→ obj_type: {obj_type}")
            print(f"→ actions: {actions}")
            print(f"→ resource attributes used in policy: {resource_attrs}")

            for action in actions:
                if obj_type and action:
                    policy_matrix[obj_type][action].update(resource_attrs)

        except Exception as e:
            print(f"Error parsing policy {p.uid}: {e}")
    return policy_matrix

def compute_object_sensitivity(access_requests, storage, policy_attrs):
    """
    Evaluates attribute-level and object-level sensitivity using ABAC policy decisions.
    """
    policies = list(storage.get_all(20, 0))
    pdp = PDP(storage)
    policy_matrix = compute_policy_matrix(policies, policy_attrs)

    def process_requests(batch):
        stats = defaultdict(default_counter)
        allowed_count = 0
        denied_count = 0

        for req in batch:
            res = req["resource"]["attributes"]
            action = req["action"]["attributes"]["method"]
            res_type = res.get("obj_type")
            obj_id = req["resource"]["id"]
            attrs = policy_matrix.get(res_type, {}).get(action, set())

            if not attrs:
                continue

            relevant_attrs = [a for a in attrs if res.get(a) is not None]
            if not relevant_attrs:
                print(f"Skipping object {obj_id} → no relevant attributes with values for action {action}")
                continue

            allowed = pdp.is_allowed(AccessRequest.from_json(req))
            for attr in relevant_attrs:
                stats[(obj_id, attr)][0] += 1
                if allowed:
                    stats[(obj_id, attr)][1] += 1
                    allowed_count += 1
                else:
                    denied_count += 1

        return stats, allowed_count, denied_count

    obj_stats = defaultdict(default_counter)
    num_requests = len(access_requests)
    batch_size = max(500, num_requests // os.cpu_count())
    batches = [access_requests[i:i + batch_size] for i in range(0, num_requests, batch_size)]

    print(f"Processing {num_requests} requests in {len(batches)} batches")
    allowed_total, denied_total = 0, 0

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_requests, batch) for batch in batches]
        for future in as_completed(futures):
            try:
                result, batch_allowed, batch_denied = future.result()
                allowed_total += batch_allowed
                denied_total += batch_denied
                for k, v in result.items():
                    obj_stats[k][0] += v[0]
                    obj_stats[k][1] += v[1]
            except Exception as e:
                print(f"Batch failed: {e}")

    print(f"\nAccess Evaluation Summary:")
    print(f"Allowed requests: {allowed_total}")
    print(f"Denied requests: {denied_total}")

    # Compute per-object sensitivity using RMS of -log(success rate)
    grouped = defaultdict(list)
    for (obj_id, attr), (total, allowed) in obj_stats.items():
        p = allowed / total if total else 0.0
        idx = min(max(int(p * 1000), 0), 1000)
        grouped[obj_id].append((attr, LOG_CACHE[idx]))

    sensitivity_data = []
    for obj_id, attr_scores in grouped.items():
        rms = math.sqrt(sum(s**2 for _, s in attr_scores) / len(attr_scores))
        sensitivity_data.append({
            "object_id": obj_id,
            "sensitivity": rms,
            "attributes": {a: s for a, s in attr_scores}
        })

    all_rms = [s["sensitivity"] for s in sensitivity_data]
    threshold = math.sqrt(sum(v**2 for v in all_rms) / len(all_rms)) if all_rms else 0.0
    avg_sens = sum(all_rms) / len(all_rms) if all_rms else 0.0

    sensitive_attrs = {
        attr
        for entry in sensitivity_data
        for attr, score in entry["attributes"].items()
        if score >= threshold
    }

    print(f"Done. Threshold: {threshold:.4f}, Avg: {avg_sens:.4f}")
    return threshold, avg_sens, sensitivity_data, sensitive_attrs, obj_stats

def main():
    ROOT = Path(__file__).resolve().parents[1]
    POLICY_PATH = ROOT / "request_builder" / "policy.json"
    REQUESTS_PATH = ROOT / "datasets" / "processed" / "access_requests.json"
    OUTDIR = ROOT / "datasets" / "processed" / "sensitivity_results"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    with open(POLICY_PATH, encoding="utf-8") as f:
        policy_data = json.load(f)["policies"]

    storage = MemoryStorage()
    for p in policy_data:
        storage.add(Policy.from_json(p))

    policy_attrs = extract_policy_resource_attributes(policy_data)
    print("Extracted Policy Attributes:")
    print(policy_attrs)

    with open(REQUESTS_PATH, encoding="utf-8") as f:
        access_requests = json.load(f)

    threshold, avg_sens, all_sensitivities, sensitive_attrs, obj_stats = compute_object_sensitivity(
        access_requests, storage, policy_attrs
    )

    with open(OUTDIR / "final_sensitivity.json", "w") as f:
        json.dump(convert_numpy_types({
            "avg_sensitivity": avg_sens,
            "threshold": threshold,
            "sensitive_attributes": list(sensitive_attrs),
            "detailed_sensitivities": all_sensitivities
        }), f, indent=2)

    attr_values = defaultdict(list)
    for obj in all_sensitivities:
        for attr, score in obj["attributes"].items():
            attr_values[attr].append(score)

    print("\nAttribute Sensitivity: Mean & Standard Deviation")
    for attr, scores in attr_values.items():
        print(f"- {attr}: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")

    flat_rows = []
    for entry in all_sensitivities:
        obj_id = entry["object_id"]
        sens = entry["sensitivity"]
        for attr, score in entry["attributes"].items():
            flat_rows.append({
                "object_id": obj_id,
                "attribute": attr,
                "sensitivity": float(score),
                "object_sensitivity": float(sens)
            })

    df_flat = pd.DataFrame(flat_rows)
    df_summary = df_flat.groupby("attribute")["sensitivity"].agg(["mean", "std", "count"]).reset_index()

    df_flat.to_csv(OUTDIR / "sensitivity_flat.csv", index=False)
    df_summary.to_csv(OUTDIR / "sensitivity_attribute_summary.csv", index=False)

    denial_summary = []
    for (obj_id, attr), (total, allowed) in obj_stats.items():
        if total > 0:
            denial_summary.append({
                "attribute": attr,
                "total": total,
                "allowed": allowed,
                "denied": total - allowed,
                "denial_rate": round(1 - (allowed / total), 4)
            })

    denial_df = pd.DataFrame(denial_summary)
    denial_df = denial_df.groupby("attribute").agg({
        "total": "sum",
        "allowed": "sum",
        "denied": "sum",
        "denial_rate": "mean"
    }).reset_index()

    denial_df.to_csv(OUTDIR / "denial_rate_summary.csv", index=False)

if __name__ == "__main__":
    main()