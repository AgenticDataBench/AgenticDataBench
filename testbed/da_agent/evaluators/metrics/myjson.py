"""
JSON Metrics Module - JSON comparison metric for evaluation.

This module compares JSON outputs:
- Structural comparison of JSON objects
- Key-value matching with type checking
- Handles nested structures
"""

import json
from typing import Any, Dict
import logging
import re


def truncate_errors(errors_list, max_count=50, max_str_len=5000, mode='min'):
    """
    Truncate list of error messages to max_count and max_str_len characters.

    Args:
        errors_list: List of tuples (error_score, error_message)
        max_count: Maximum number of errors to keep
        max_str_len: Maximum string length for combined message
        mode: 'min' for evaluate_field (0=worst, 1=best),
              'max' for evaluate_field_normalized (0=best, 1=worst)

    Returns:
        Tuple of (error_score, truncated_combined_msg)
    """
    if not errors_list:
        if mode == 'min':
            return 1.0, ""
        else:
            return 0.0, ""

    # Sort by error_score and take max_count
    sorted_errors = sorted(errors_list, key=lambda x: (x[0], x[1]))[:max_count]

    # Get error_score based on mode
    if mode == 'min':
        error_score = min(es for es, _ in sorted_errors)
    else:
        error_score = max(es for es, _ in sorted_errors)

    # Combine messages
    messages = [msg for _, msg in sorted_errors]
    combined = "; ".join(messages)

    # Truncate if too long
    if len(combined) > max_str_len:
        combined = combined[:max_str_len] + "..."

    return error_score, combined


def normalize_key(key: str) -> str:
    """
    Normalize a key for fuzzy matching:
    - Convert to lowercase
    - Replace underscores, spaces, and hyphens with empty string
    - Remove common filler words (e.g., 'the', 'a', 'an')

    Examples:
        "User_Name" -> "username"
        "user name" -> "username"
        "user-name" -> "username"
        "UserName" -> "username"
        "the_user_name" -> "username"
    """
    if not isinstance(key, str):
        key = str(key)
    # Convert to lowercase
    normalized = key.lower()
    # Remove common filler words (at word boundaries)
    filler_words = ['the', 'a', 'an', 'of', 'in', 'for', 'to', 'and', 'or']
    for word in filler_words:
        normalized = re.sub(rf'\b{word}\b', '', normalized)
    # Remove underscores, spaces, hyphens
    normalized = normalized.replace('_', '').replace(' ', '').replace('-', '')
    return normalized


def calculate_similarity(s1: str, s2: str) -> float:
    """
    Calculate similarity between two strings based on character set overlap.
    Returns a value between 0.0 and 1.0.
    """
    if not s1 or not s2:
        return 0.0
    set1 = set(s1.lower())
    set2 = set(s2.lower())
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def find_matching_key(target_key: str, available_keys: list, strict_first: bool = True, fuzzy_threshold: float = 0.6) -> str | None:
    """
    Find the best matching key from available_keys for target_key.

    Args:
        target_key: The key we're looking for
        available_keys: List of keys to search in
        strict_first: If True, try exact match first before fuzzy matching
        fuzzy_threshold: Minimum similarity threshold for fuzzy matching (default 0.6 = 60%)

    Returns:
        The matching key from available_keys, or None if no match found
    """
    if strict_first and target_key in available_keys:
        return target_key

    # Try normalized exact match
    target_normalized = normalize_key(target_key)
    for avail_key in available_keys:
        avail_normalized = normalize_key(avail_key)
        if target_normalized in avail_normalized or avail_normalized in target_normalized:
            return avail_key

    # Try fuzzy matching with similarity threshold
    best_match = None
    best_similarity = fuzzy_threshold  # Only accept matches above threshold
    for avail_key in available_keys:
        similarity = calculate_similarity(target_key, avail_key)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = avail_key

    return best_match


def count_nested_fields(threshold: dict) -> int:
    """
    Count the total number of fields in a nested threshold configuration.

    Args:
        threshold: A threshold dict (possibly nested)

    Returns:
        Total count of non-dict threshold values (leaf nodes)
    """
    count = 0
    for v in threshold.values():
        if isinstance(v, dict):
            count += count_nested_fields(v)
        else:
            count += 1
    return count


def compare_json(
    output_file_name: str = 'output.json',
    gold_file_name: str = 'result.json',
    thresholds: dict[str, Any] | None = None,
    matched_keys: list[str] | None = None
) -> Dict[str, Any]:
    """
    Compare JSON files with hierarchical threshold configuration.

    Args:
        output_file_name: Path to the output/predicted JSON file
        gold_file_name: Path to the gold/expected JSON file
        thresholds: Hierarchical threshold configuration defining tolerance rules for each field
            - float: Relative error tolerance (|x'-x|/max(|x|, 1e-12))
            - list[2]: Range [min, max] for allowed values
            - None: Default rule (1% relative error for numbers, strict equality for strings)
            - dict: Nested configuration for nested JSON structures

    Returns:
        dict: {'score': float, 'errors': list[str]}
            - score: average score across all specified fields (0.0 to 1.0)
            - errors: detailed error messages including key, expected vs actual values, and failure reason

    Example:
        thresholds = {
            "TripAdvisor": {
                "mean": 0.01,      # 1% relative error allowed
                "std": [0, 1]      # range [0, 1] allowed
            },
            "ANOVA": {
                "f_statistic": [40, 42],  # range [40, 42] allowed
                "p_value": None    # default rule
            }
        }

    matched_keys:
        List of key names used to match JSON objects when comparing lists.
        If provided and both output_obj and gold_obj are lists, objects are matched by these keys
        and compared regardless of order.

    Note: Fields not specified in thresholds are ignored and not scored.
    """
    if thresholds is None:
        thresholds = {}

    # Load JSON files
    try:
        with open(output_file_name, 'r') as f:
            output_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return {
            'score': 0.0,
            'errors': [f"Error: output file '{output_file_name}' not found or invalid JSON"],
            'meaning': f"Failed to load output file: '{output_file_name}'",
            'output_data': None,
            'gold_data': None
        }

    try:
        with open(gold_file_name, 'r') as f:
            gold_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return {
            'score': 0.0,
            'errors': [f"Error: gold file '{gold_file_name}' not found or invalid JSON"],
            'meaning': f"Failed to load gold file: '{gold_file_name}'",
            'output_data': None,
            'gold_data': None
        }

    # Collect scores for all specified fields
    scores = []
    errors = []  # Will store tuples of (error_score, error_message)
                 # error_score: lower score = more severe error (0 is worst, 1 is perfect)

    def evaluate_field(output_val: Any, gold_val: Any, threshold: Any, field_path: str = "", quiet: bool = False) -> tuple[float, tuple[float, str]]:
        """
        Evaluate a single field based on its threshold configuration.

        Args:
            quiet: If True, suppress warning logs (used during matching process)

        Returns:
            tuple: (score, (error_score, error_message))
            score: 1.0 if the value is within tolerance, 0.0 otherwise
            error_score: inverse of severity (0.0 = worst error, 1.0 = no error)
            error_message: detailed error description if score is 0, empty string otherwise
        """
        # Helper for logging warning
        def warn(msg):
            if not quiet:
                logging.warning(msg)
        # Handle None threshold - use default rules
        if threshold is None:
            if gold_val is None and output_val is None:
                return 1.0, (1.0, "")
            if gold_val is None:
                warn(f"{field_path}: gold value is null, output={output_val}")
                return 0.0, (0.0, f"{field_path}: gold value is null, output={output_val}")
            if isinstance(gold_val, (int, float)):
                # Default: 1% relative error for numbers
                if output_val is None:
                    warn(f"{field_path}: output value is null, expected number {gold_val}")
                    return 0.0, (0.0, f"{field_path}: output value is null, expected number {gold_val}")
                if not isinstance(output_val, (int, float)):
                    warn(f"{field_path}: type mismatch - expected number {gold_val}, got {type(output_val).__name__} '{output_val}'")
                    return 0.0, (0.0, f"{field_path}: type mismatch - expected number {gold_val}, got {type(output_val).__name__} '{output_val}'")
                rel_error = abs(output_val - gold_val) / max(abs(gold_val), 1e-12)
                if rel_error <= 0.01:
                    return 1.0, (1.0, "")
                return 0.0, (1.0 - min(rel_error, 1.0), f"{field_path}: FAIL - relative error {rel_error:.4f} exceeds 1% tolerance (expected {gold_val}, got {output_val}, difference {abs(output_val - gold_val):.6f})")
            elif isinstance(gold_val, str):
                # Default: strict equality for strings (case-insensitive, whitespace-normalized)
                if not isinstance(output_val, str):
                    output_val = str(output_val)
                output_norm = output_val.lower().strip()
                gold_norm = gold_val.lower().strip()
                if output_norm == gold_norm:
                    return 1.0, (1.0, "")
                # warn(f"{field_path}: string mismatch - expected '{gold_norm}', got '{output_norm}'")
                return 0.0, (0.0, f"{field_path}: FAIL - string mismatch (expected '{gold_norm}', got '{output_norm}', original output: '{output_val}')")
            elif isinstance(gold_val, list):
                if not isinstance(output_val, list):
                    warn(f"{field_path}: type mismatch - expected list, got {type(output_val).__name__}")
                    return 0.0, (0.0, f"{field_path}: type mismatch - expected list, got {type(output_val).__name__}")
                if len(output_val) != len(gold_val):
                    warn(f"{field_path}: list length mismatch - expected {len(gold_val)}, got {len(output_val)}")
                    return 0.0, (0.0, f"{field_path}: list length mismatch - expected {len(gold_val)}, got {len(output_val)}")
                # Unordered matching: find best match for each gold element
                element_scores = []
                element_errors = []
                output_used = [False] * len(output_val)  # Track which output elements have been matched
                for i, g in enumerate(gold_val):
                    best_score = -1.0
                    best_error = (1.0, "")
                    best_idx = -1
                    for j, o in enumerate(output_val):
                        if output_used[j]:
                            continue
                        s, (es, em) = evaluate_field(o, g, None, f"{field_path}[{i}]", quiet=True)
                        if s > best_score:
                            best_score = s
                            best_error = (es, em)
                            best_idx = j
                    if best_idx >= 0:
                        output_used[best_idx] = True
                        element_scores.append(best_score)
                        if best_error[1]:
                            element_errors.append(best_error)
                    else:
                        # No match found (shouldn't happen if lengths match)
                        warn(f"{field_path}[{i}]: no matching element found output")
                        element_scores.append(0.0)
                        element_errors.append((0.0, f"{field_path}[{i}]: no matching element found in output"))
                avg = sum(element_scores) / len(element_scores) if element_scores else 0.0
                min_es, combined_msg = truncate_errors(element_errors)
                # Only warn if there were actual mismatches
                return avg, (min_es, combined_msg)
            elif isinstance(gold_val, dict):
                if not isinstance(output_val, dict):
                    warn(f"{field_path}: type mismatch - expected dict, got {type(output_val).__name__}")
                    return 0.0, (0.0, f"{field_path}: type mismatch - expected dict, got {type(output_val).__name__}")
                element_scores = []
                element_errors = []
                output_keys = list(output_val.keys())
                gold_keys = list(gold_val.keys())
                for key, g in gold_val.items():
                    matched_key = find_matching_key(key, output_keys)
                    if matched_key is None:
                        warn(f"{field_path}.{key}: key not found in output dict")
                        element_scores.append(0.0)
                        element_errors.append((0.0, f"{field_path}.{key}: key not found in output dict"))
                    else:
                        o = output_val[matched_key]
                        s, (es, em) = evaluate_field(o, g, None, f"{field_path}.{key}")
                        element_scores.append(s)
                        if s < 1.0:
                            element_errors.append((es, em))
                avg = sum(element_scores) / len(element_scores) if element_scores else 0.0
                min_es, combined_msg = truncate_errors(element_errors)
                return avg, (min_es, combined_msg)
            else:
                # For other types, use strict equality
                warn(f"{field_path}: no specific threshold provided for type {type(gold_val).__name__}, using strict equality")
                if output_val == gold_val:
                    return 1.0, (1.0, "")
                return 0.0, (0.0, f"{field_path}: FAIL - value mismatch (expected {gold_val}, got {output_val})")

        # Handle float threshold - relative error
        if isinstance(threshold, (int, float)):
            if threshold < 0 or threshold > 1:
                warn(f"{field_path}: invalid threshold value {threshold}, must be between 0 and 1 for relative error tolerance")
            if gold_val is None and output_val is None:
                return 1.0, (1.0, "")
            if gold_val is None or output_val is None:
                warn(f"{field_path}: null mismatch - gold={gold_val}, output={output_val}")
                return 0.0, (0.0, f"{field_path}: null mismatch - gold={gold_val}, output={output_val}")
            # Handle list values - apply threshold recursively to each element
            if isinstance(gold_val, list):
                if not isinstance(output_val, list):
                    warn(f"{field_path}: type mismatch - expected list, got {type(output_val).__name__}")
                    return 0.0, (0.0, f"{field_path}: type mismatch - expected list, got {type(output_val).__name__}")
                if len(output_val) != len(gold_val):
                    warn(f"{field_path}: list length mismatch - expected {len(gold_val)}, got {len(output_val)}")
                    return 0.0, (0.0, f"{field_path}: list length mismatch - expected {len(gold_val)}, got {len(output_val)}")
                # Unordered matching: find best match for each gold element
                element_errors = []
                output_used = [False] * len(output_val)  # Track which output elements have been matched
                for i, g in enumerate(gold_val):
                    best_score = -1.0
                    best_error = (1.0, "")
                    best_idx = -1
                    for j, o in enumerate(output_val):
                        if output_used[j]:
                            continue
                        s, (es, em) = evaluate_field(o, g, threshold, f"{field_path}[{i}]", quiet=True)
                        if s > best_score:
                            best_score = s
                            best_error = (es, em)
                            best_idx = j
                    if best_idx >= 0:
                        output_used[best_idx] = True
                        if best_score < 1.0:
                            element_errors.append(best_error)
                    else:
                        warn(f"{field_path}[{i}]: no matching element found in output")
                        element_errors.append((0.0, f"{field_path}[{i}]: no matching element found in output"))
                if not element_errors:
                    return 1.0, (1.0, "")
                min_es, combined_msg = truncate_errors(element_errors)
                return 0.0, (min_es, combined_msg)
            # Handle dict values - apply threshold recursively to each element using fuzzy key matching
            if isinstance(gold_val, dict):
                if not isinstance(output_val, dict):
                    warn(f"{field_path}: type mismatch - expected dict, got {type(output_val).__name__}")
                    return 0.0, (0.0, f"{field_path}: type mismatch - expected dict, got {type(output_val).__name__}")
                element_scores = []
                element_errors = []
                output_keys = list(output_val.keys())
                gold_keys = list(gold_val.keys())
                for key, g in gold_val.items():
                    matched_key = find_matching_key(key, output_keys)
                    if matched_key is None:
                        warn(f"{field_path}.{key}: key not found in output dict")
                        element_scores.append(0.0)
                        element_errors.append((0.0, f"{field_path}.{key}: key not found in output dict"))
                    else:
                        o = output_val[matched_key]
                        s, (es, em) = evaluate_field(o, g, threshold, f"{field_path}.{key}")
                        element_scores.append(s)
                        if s < 1.0:
                            element_errors.append((es, em))
                avg = sum(element_scores) / len(element_scores) if element_scores else 0.0
                min_es, combined_msg = truncate_errors(element_errors)
                return avg, (min_es, combined_msg)
            if not isinstance(output_val, (int, float)) or not isinstance(gold_val, (int, float)):
                warn(f"{field_path}: type mismatch - output: {type(output_val).__name__} '{output_val}', gold: {type(gold_val).__name__} '{gold_val}'")
                return 0.0, (0.0, f"{field_path}: type mismatch - output: {type(output_val).__name__} '{output_val}', gold: {type(gold_val).__name__} '{gold_val}'")
            rel_error = abs(output_val - gold_val) / max(abs(gold_val), 1e-12)
            if rel_error <= threshold:
                return 1.0, (1.0, "")
            return 0.0, (1.0 - min(rel_error, 1.0), f"{field_path}: FAIL - relative error {rel_error:.4f} exceeds {threshold:.2%} tolerance (expected {gold_val}, got {output_val}, difference {abs(output_val - gold_val):.6f})")

        # Handle list threshold - range [min, max]
        if isinstance(threshold, list) and len(threshold) == 2:
            if not isinstance(output_val, (int, float)):
                warn(f"{field_path}: type mismatch - expected number in range {threshold}, got {type(output_val).__name__} '{output_val}'")
                return 0.0, (0.0, f"{field_path}: type mismatch - expected number in range {threshold}, got {type(output_val).__name__} '{output_val}'")
            min_val, max_val = threshold
            if min_val <= output_val <= max_val:
                return 1.0, (1.0, "")
            # Calculate how far from the range (lower error_score = farther from range)
            if output_val < min_val:
                distance = (min_val - output_val) / abs(min_val) if min_val != 0 else 1.0
                direction = "below"
            else:  # output_val > max_val
                distance = (output_val - max_val) / abs(max_val) if max_val != 0 else 1.0
                direction = "above"
            error_score = max(0.0, 1.0 - min(distance, 2.0))  # Cap distance at 2.0 to avoid negative scores
            return 0.0, (error_score, f"{field_path}: FAIL - value {output_val} {direction} allowed range [{min_val}, {max_val}] (difference: {abs(output_val - (min_val if output_val < min_val else max_val)):.6f})")

        # Handle dict threshold - nested structure
        if isinstance(threshold, dict):
            if not isinstance(output_val, dict) or not isinstance(gold_val, dict):
                warn(f"{field_path}: type mismatch - expected dict, got {type(output_val).__name__}")
                return 0.0, (0.0, f"{field_path}: type mismatch - expected dict, got {type(output_val).__name__}")

            nested_scores = []
            nested_errors = []
            output_keys = list(output_val.keys())
            gold_keys = list(gold_val.keys())
            for key, nested_threshold in threshold.items():
                nested_path = f"{field_path}.{key}" if field_path else key
                # Use fuzzy matching to find the key in gold and output
                matched_gold_key = find_matching_key(key, gold_keys)
                matched_output_key = find_matching_key(key, output_keys)

                if matched_gold_key is not None:
                    gold_val_for_key = gold_val[matched_gold_key]
                    if matched_output_key is not None:
                        output_val_for_key = output_val[matched_output_key]
                        nested_score, (nested_error_score, nested_error) = evaluate_field(output_val_for_key, gold_val_for_key, nested_threshold, nested_path)
                        nested_scores.append(nested_score)
                        if nested_error:
                            nested_errors.append((nested_error_score, nested_error))
                    else:
                        warn(f"{nested_path}: key not found in output (tried '{key}', available keys: {output_keys}), scoring 0 for this field")
                        nested_scores.append(0.0)
                        nested_errors.append((0.0, f"{nested_path}: key not found in output (tried '{key}')"))

            avg_score = sum(nested_scores) / len(nested_scores) if nested_scores else 0.0
            # Aggregate error_score as min of nested error_scores (worst error)
            min_error_score, error_msg = truncate_errors(nested_errors)
            return avg_score, (min_error_score, error_msg)

        # Unknown threshold type
        warn(f"{field_path}: unknown threshold type {type(threshold).__name__}")
        return 0.0, (0.0, f"{field_path}: unknown threshold type {type(threshold).__name__}")

    def traverse_thresholds(output_obj: Any, gold_obj: Any, threshold_obj: dict, obj_path: str = ""):
        """
        Recursively traverse threshold configuration and evaluate fields.
        """
        # Check for type mismatch between output and gold
        if isinstance(gold_obj, list) and not isinstance(output_obj, list):
            logging.warning(f"{obj_path}: type mismatch - gold is list but output is {type(output_obj).__name__}, cannot compare")
            # Count fields same as when both are lists: for each gold item, count threshold fields
            fields_per_item = count_nested_fields(threshold_obj) if threshold_obj else 0
            total_fields = len(gold_obj) * fields_per_item
            if total_fields > 0:
                scores.extend([0.0] * total_fields)
                errors.append((0.0, f"{obj_path}: type mismatch - gold is list with {len(gold_obj)} items but output is {type(output_obj).__name__}, expected list format"))
            return
        if isinstance(output_obj, list) and not isinstance(gold_obj, list):
            logging.warning(f"{obj_path}: type mismatch - output is list but gold is {type(gold_obj).__name__}, cannot compare")
            # Count fields same as when both are dicts: count threshold fields
            total_fields = count_nested_fields(threshold_obj) if threshold_obj else 0
            if total_fields > 0:
                scores.extend([0.0] * total_fields)
                errors.append((0.0, f"{obj_path}: type mismatch - output is list but gold is {type(gold_obj).__name__}, cannot match items"))
            return

        # If both are lists and matched_keys provided, do unordered comparison
        if isinstance(output_obj, list) and isinstance(gold_obj, list):
            if not matched_keys:
                logging.warning(f"{obj_path}: both output and gold are lists but no matched_keys provided, cannot compare these lists")
                return
            # Throttle repeated warnings by pattern (strip item details)
            _warn_pattern_counts = {}
            _WARN_LIMIT = 5
            def throttled_warn(msg):
                # Extract pattern: keep prefix before item details
                pattern = msg.split('{')[0] if '{' in msg else msg
                _warn_pattern_counts[pattern] = _warn_pattern_counts.get(pattern, 0) + 1
                count = _warn_pattern_counts[pattern]
                if count <= _WARN_LIMIT:
                    logging.warning(msg)
                elif count == _WARN_LIMIT + 1:
                    logging.warning(f"{pattern}{{...}} (more suppressed)")

            # Track which output items have been matched
            output_used = [False] * len(output_obj)
            # Match items by matched_keys
            matched_output = []
            unmatched_items = []  # Track unmatched items with their severity
            matched_item_idx = 0  # Counter for matched items
            for gold_item in gold_obj:
                if not isinstance(gold_item, dict):
                    continue

                # Build identifier for this gold item using matched_keys
                gold_identifier = ", ".join([f"{mk}={gold_item.get(mk)}" for mk in matched_keys if mk in gold_item])
                item_label = f"[item#{matched_item_idx} ({gold_identifier})]" if gold_identifier else f"[item#{matched_item_idx}]"

                # Find matching output item
                matched_item = None
                matched_idx = -1
                for idx, output_item in enumerate(output_obj):
                    if output_used[idx]:
                        continue  # Skip already matched output items
                    if not isinstance(output_item, dict):
                        continue

                    # Check if all matched_keys match
                    is_match = True
                    for mk in matched_keys:
                        if mk not in gold_item:
                            throttled_warn(f"{obj_path}: matched key '{mk}' not found in gold item {gold_item}, cannot match this item")
                            is_match = False
                            break
                        if mk not in output_item:
                            throttled_warn(f"{obj_path}: matched key '{mk}' not found in output item {output_item}, cannot match this item")
                            is_match = False
                            break
                        if gold_item[mk] != output_item[mk]:
                            is_match = False
                            break

                    if is_match:
                        matched_item = output_item
                        matched_idx = idx
                        break

                if matched_item is None:
                    # No match found, score 0 for all fields in this item
                    # Store item_label for later error reporting
                    unmatched_items.append((0.0, gold_item, item_label))  # error_score 0.0 = completely unmatched
                    for key, threshold in threshold_obj.items():
                        if isinstance(threshold, dict):
                            # Nested threshold: count nested fields
                            field_count = count_nested_fields(threshold)
                            scores.extend([0.0] * field_count)
                        else:
                            # Simple threshold: one score
                            scores.append(0.0)
                else:
                    output_used[matched_idx] = True  # Mark this output item as used
                    matched_output.append(matched_item)
                    matched_item_idx += 1
                    # Traverse the matched item
                    matched_item_keys = list(matched_item.keys())
                    gold_item_keys = list(gold_item.keys())
                    for key, threshold in threshold_obj.items():
                        field_path = f"{obj_path}{item_label}" if obj_path else f"matched{item_label}"
                        matched_key_output = find_matching_key(key, matched_item_keys)
                        matched_key_gold = find_matching_key(key, gold_item_keys)
                        if isinstance(threshold, dict):
                            if matched_key_output is not None and matched_key_gold is not None:
                                traverse_thresholds(matched_item[matched_key_output], gold_item[matched_key_gold], threshold, f"{field_path}.{key}")
                            else:
                                if matched_key_output is None:
                                    logging.warning(f"{field_path}.{key}: key not found in matched output item (tried '{key}', available keys: {matched_item_keys}), scoring 0 for all its fields")
                                if matched_key_gold is None:
                                    logging.warning(f"{field_path}.{key}: key not found in gold item (tried '{key}', available keys: {gold_item_keys}), scoring 0 for all its fields")
                                field_count = count_nested_fields(threshold)
                                scores.extend([0.0] * field_count)
                                errors.append((0.0, f"{field_path}.{key}: key not found"))
                        else:
                            if matched_key_gold is not None:
                                gold_val = gold_item[matched_key_gold]
                                if matched_key_output is not None:
                                    output_val = matched_item[matched_key_output]
                                    score, (error_score, error) = evaluate_field(output_val, gold_val, threshold, f"{field_path}.{key}")
                                    scores.append(score)
                                    if error:
                                        errors.append((error_score, error))
                                else:
                                    logging.warning(f"{field_path}.{key}: key not found in matched output item (tried '{key}', available keys: {matched_item_keys}), scoring 0")
                                    scores.append(0.0)
                                    errors.append((0.0, f"{field_path}.{key}: key not found in output"))
                            else:
                                logging.warning(f"{field_path}.{key}: key not found in gold item (tried '{key}', available keys: {gold_item_keys}), scoring 0")
                                scores.append(0.0)
                                errors.append((0.0, f"{field_path}.{key}: key not found in gold"))

            # Sample errors for unmatched items (max 10, sorted by item_label for consistency)
            if unmatched_items:
                if len(unmatched_items) == len(gold_obj):
                    logging.warning(f"{obj_path}: no items matched, scoring 0 for entire list")
                sample_count = min(10, len(unmatched_items))
                sampled_items = sorted(unmatched_items[:sample_count], key=lambda x: x[2])
                for _, _, item_label in sampled_items:
                    errors.append((0.0, f"{obj_path}{item_label}: item not found in output"))
                if len(unmatched_items) > sample_count:
                    errors.append((0.0, f"{obj_path}: ... and {len(unmatched_items) - sample_count} more items not found"))

            return

        # Normal traversal for dicts
        output_keys = list(output_obj.keys()) if isinstance(output_obj, dict) else []
        gold_keys = list(gold_obj.keys()) if isinstance(gold_obj, dict) else []
        for key, threshold in threshold_obj.items():
            field_path = f"{obj_path}.{key}" if obj_path else key
            matched_key_output = find_matching_key(key, output_keys)
            matched_key_gold = find_matching_key(key, gold_keys)
            # For nested dict thresholds, recurse into the nested structure
            if isinstance(threshold, dict):
                if isinstance(output_obj, dict) and isinstance(gold_obj, dict):
                    if matched_key_output is not None and matched_key_gold is not None:
                        traverse_thresholds(output_obj[matched_key_output], gold_obj[matched_key_gold], threshold, field_path)
                    else:
                        # Missing nested object - score 0 for all its fields
                        logging.warning(f"{field_path}: missing nested object in output or gold, scoring 0 for all its fields")
                        field_count = count_nested_fields(threshold)
                        scores.extend([0.0] * field_count)
                        errors.append((0.0, f"{field_path}: key not found"))
            else:
                # Simple threshold - evaluate single field
                if isinstance(gold_obj, dict) and matched_key_gold is not None:
                    gold_val = gold_obj[matched_key_gold]
                    if isinstance(output_obj, dict) and matched_key_output is not None:
                        output_val = output_obj[matched_key_output]
                        score, (error_score, error) = evaluate_field(output_val, gold_val, threshold, field_path)
                        scores.append(score)
                        if error:
                            errors.append((error_score, error))
                    else:
                        logging.warning(f"{field_path}: key not found in output (tried '{key}', available keys: {output_keys}), scoring 0")
                        # Field missing in output
                        scores.append(0.0)
                        errors.append((0.0, f"{field_path}: key not found in output"))
                else:
                    # Key not found in gold (or gold_obj is not a dict)
                    logging.warning(f"{field_path}: key not found in gold (tried '{key}', available keys: {gold_keys}), scoring 0")
                    scores.append(0.0)
                    errors.append((0.0, f"{field_path}: key not found in gold"))

    # Start traversal from root for compare_json
    traverse_thresholds(output_data, gold_data, thresholds)

    # Limit total errors to max 50, sorted by error_score (lowest/most severe first)
    max_total_errors = 50
    if len(errors) > max_total_errors:
        # Sort by error_score ascending (0.0 = worst, 1.0 = best), then alphabetically
        errors_sorted = sorted(errors, key=lambda x: (x[0], x[1]))
        top_errors = errors_sorted[:max_total_errors]
        remaining = len(errors) - max_total_errors
        # Convert to just messages
        errors = [msg for _, msg in top_errors]
        errors.append(f"... and {remaining} more errors")
    else:
        # Convert to just messages
        errors = [msg for _, msg in errors]

    # Return average score, errors, and meaning
    avg_score = sum(scores) / len(scores) if scores else 0.0
    meaning = f"Comparing JSON files: output='{output_file_name}', gold='{gold_file_name}'"

    # Check if comparing lists with matched_keys
    if matched_keys:
        meaning += f" (list comparison with keys: {matched_keys})"

    meaning += ", evaluating "

    # Explain thresholds
    def describe_threshold(thresh):
        if thresh is None:
            return "None (default: 1% rel error for numbers, strict equality for strings)"
        elif isinstance(thresh, (int, float)):
            return f"{thresh} (relative error tolerance)"
        elif isinstance(thresh, list) and len(thresh) == 2:
            return f"{thresh} (allowed range [{thresh[0]}, {thresh[1]}])"
        elif isinstance(thresh, dict):
            nested = []
            for k, v in thresh.items():
                nested.append(f"{k}:{describe_threshold(v)}")
            return "{" + ", ".join(nested) + "}"
        else:
            return str(thresh)

    threshold_desc = []
    for key, thresh in thresholds.items():
        threshold_desc.append(f"{key}={describe_threshold(thresh)}")

    meaning += f"{len(scores)} fields with thresholds: {'; '.join(threshold_desc)}"

    result = {'score': avg_score, 'errors': errors, 'meaning': meaning}
    # Include data if file sizes are small (<1000 characters when formatted as JSON)
    try:
        output_json_str = json.dumps(output_data, indent=2)
        gold_json_str = json.dumps(gold_data, indent=2)

        if len(output_json_str) < 1000:
            result['output_data'] = output_data
        else:
            result['output_data'] = output_json_str[:1000] + f"\n... ({len(output_json_str)} characters total)"

        if len(gold_json_str) < 1000:
            result['gold_data'] = gold_data
        else:
            result['gold_data'] = gold_json_str[:1000] + f"\n... ({len(gold_json_str)} characters total)"
    except Exception:
        pass  # If serialization fails, just skip adding data

    return result

def compare_json_normalized(
    output_file_name: str = 'output.json',
    gold_file_name: str | None = None,
    thresholds: dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Compare JSON files with hierarchical threshold configuration, using normalized scoring for range thresholds.

    Similar to compare_json, but for range thresholds [bound1, bound2], applies linear normalization
    instead of binary pass/fail:
        - If bound1 < bound2: higher is better, score = (value - bound1) / (bound2 - bound1)
        - If bound1 > bound2: lower is better, score = (bound1 - value) / (bound1 - bound2)
        - Score is clipped to [0, 1]

    Args:
        output_file_name: Path to the output/predicted JSON file
        gold_file_name: Path to the gold/expected JSON file (optional, only needed for float/None thresholds)
        thresholds: Hierarchical threshold configuration defining tolerance rules for each field
            - float: Relative error tolerance (binary pass/fail, |x'-x|/max(|x|, 1e-12)), requires gold_file_name
            - list[2]: Range [bound1, bound2] for normalized linear scoring (does not require gold_file_name)
                - bound1 < bound2: higher is better
                - bound1 > bound2: lower is better
            - None: Default rule (binary: 1% relative error for numbers, strict equality for strings), requires gold_file_name
            - dict: Nested configuration for nested JSON structures

    Returns:
        dict: {'score': float, 'errors': list[str]}
            - score: average score across all specified fields (0.0 to 1.0)
            - errors: detailed error messages including key, expected vs actual values, and failure reason

    Example:
        thresholds = {
            "AUC": [0.5, 1.0],          # higher is better, normalized
            "PCVR_deviation": [1.0, 0.0], # lower is better, normalized
            "samples": None              # binary pass/fail with default rule
        }

    Note: Fields not specified in thresholds are ignored and not scored.
    """
    if thresholds is None:
        thresholds = {}

    # Load output JSON file
    try:
        with open(output_file_name, 'r') as f:
            output_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return {
            'score': 0.0,
            'errors': [f"Error: output file '{output_file_name}' not found or invalid JSON"],
            'meaning': f"Failed to load output file: '{output_file_name}'",
            'output_data': None,
            'gold_data': None
        }

    # Load gold JSON file (optional, only needed for float/None thresholds)
    gold_data = None
    if gold_file_name is not None:
        try:
            with open(gold_file_name, 'r') as f:
                gold_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            return {
                'score': 0.0,
                'errors': [f"Error: gold file '{gold_file_name}' not found or invalid JSON"],
                'meaning': f"Failed to load gold file: '{gold_file_name}'",
                'output_data': None,
                'gold_data': None
            }

    # Collect scores for all specified fields
    scores = []
    errors = []  # Will store tuples of (severity, error_message)

    def evaluate_field_normalized(output_val: Any, gold_val: Any, threshold: Any, field_path: str = "") -> tuple[float, tuple[float, str]]:
        """
        Evaluate a single field based on its threshold configuration.
        For range thresholds, returns normalized score instead of binary.

        Returns:
            tuple: (score, (severity, error_message))
            score: float score in range [0.0, 1.0]
            severity: error severity (0.0 to 1.0, where score < 1.0, higher is worse)
            error_message: detailed error description if score < 1.0, empty string otherwise
        """
        # Handle None threshold - use default rules (binary)
        if threshold is None:
            if gold_val is None:
                # Cannot evaluate without gold value
                logging.warning(f"{field_path}: cannot evaluate relative error without gold value")
                return 0.0, (1.0, f"{field_path}: cannot evaluate without gold value")
            if isinstance(gold_val, (int, float)):
                # Default: 1% relative error for numbers
                if not isinstance(output_val, (int, float)):
                    logging.warning(f"{field_path}: type mismatch - expected number {gold_val}, got {type(output_val).__name__} '{output_val}'")
                    return 0.0, (1.0, f"{field_path}: type mismatch - expected number {gold_val}, got {type(output_val).__name__} '{output_val}'")
                rel_error = abs(output_val - gold_val) / max(abs(gold_val), 1e-12)
                # Add small epsilon to handle floating point precision issues
                if rel_error <= 0.01 + 1e-9:
                    return 1.0, (0.0, "")
                return 0.0, (min(rel_error, 1.0), f"{field_path}: FAIL - relative error {rel_error:.4f} exceeds 1% tolerance (expected {gold_val}, got {output_val}, difference {abs(output_val - gold_val):.6f})")
            elif isinstance(gold_val, str):
                # Default: strict equality for strings (case-insensitive, whitespace-normalized)
                if not isinstance(output_val, str):
                    output_val = str(output_val)
                output_norm = output_val.lower().strip()
                gold_norm = gold_val.lower().strip()
                if output_norm == gold_norm:
                    return 1.0, (0.0, "")
                # logging.warning(f"{field_path}: string mismatch - expected '{gold_norm}', got '{output_norm}'")
                return 0.0, (1.0, f"{field_path}: FAIL - string mismatch (expected '{gold_norm}', got '{output_norm}', original output: '{output_val}')")
            elif isinstance(gold_val, list):
                if not isinstance(output_val, list):
                    logging.warning(f"{field_path}: type mismatch - expected list, got {type(output_val).__name__}")
                    return 0.0, (1.0, f"{field_path}: type mismatch - expected list, got {type(output_val).__name__}")
                if len(output_val) != len(gold_val):
                    logging.warning(f"{field_path}: list length mismatch - expected {len(gold_val)}, got {len(output_val)}")
                    return 0.0, (1.0, f"{field_path}: list length mismatch - expected {len(gold_val)}, got {len(output_val)}")
                # Unordered matching: find best match for each gold element
                element_scores = []
                element_errors = []
                output_used = [False] * len(output_val)  # Track which output elements have been matched
                for i, g in enumerate(gold_val):
                    best_score = -1.0
                    best_error = (1.0, "")
                    best_idx = -1
                    for j, o in enumerate(output_val):
                        if output_used[j]:
                            continue
                        s, (es, em) = evaluate_field_normalized(o, g, None, f"{field_path}[{i}]")
                        if s > best_score:
                            best_score = s
                            best_error = (es, em)
                            best_idx = j
                    if best_idx >= 0:
                        output_used[best_idx] = True
                        element_scores.append(best_score)
                        if best_error[1]:
                            element_errors.append(best_error)
                    else:
                        logging.warning(f"{field_path}[{i}]: no matching element found in output")
                        element_scores.append(0.0)
                        element_errors.append((1.0, f"{field_path}[{i}]: no matching element found in output"))
                avg = sum(element_scores) / len(element_scores) if element_scores else 0.0
                max_severity, combined_msg = truncate_errors(element_errors, mode='max')
                return avg, (max_severity, combined_msg)
            else:
                # For other types, use strict equality
                logging.warning(f"{field_path}: no specific threshold provided for type {type(gold_val).__name__}, using strict equality")
                if output_val == gold_val:
                    return 1.0, (0.0, "")
                return 0.0, (1.0, f"{field_path}: value mismatch - expected {gold_val}, got {output_val}")

        # Handle float threshold - relative error (binary)
        if isinstance(threshold, (int, float)):
            if threshold < 0 or threshold > 1:
                logging.warning(f"{field_path}: invalid threshold value {threshold}, must be between 0 and 1 for relative error tolerance")
            if gold_val is None:
                # Cannot evaluate without gold value
                logging.warning(f"{field_path}: cannot evaluate relative error without gold value")
                return 0.0, (1.0, f"{field_path}: cannot evaluate without gold value")
            # Handle list values - apply threshold recursively to each element
            if isinstance(gold_val, list):
                if not isinstance(output_val, list):
                    logging.warning(f"{field_path}: type mismatch - expected list, got {type(output_val).__name__}")
                    return 0.0, (1.0, f"{field_path}: type mismatch - expected list, got {type(output_val).__name__}")
                if len(output_val) != len(gold_val):
                    logging.warning(f"{field_path}: list length mismatch - expected {len(gold_val)}, got {len(output_val)}")
                    return 0.0, (1.0, f"{field_path}: list length mismatch - expected {len(gold_val)}, got {len(output_val)}")
                # Unordered matching: find best match for each gold element
                element_errors = []
                output_used = [False] * len(output_val)  # Track which output elements have been matched
                for i, g in enumerate(gold_val):
                    best_score = -1.0
                    best_error = (1.0, "")
                    best_idx = -1
                    for j, o in enumerate(output_val):
                        if output_used[j]:
                            continue
                        s, (es, em) = evaluate_field_normalized(o, g, threshold, f"{field_path}[{i}]")
                        if s > best_score:
                            best_score = s
                            best_error = (es, em)
                            best_idx = j
                    if best_idx >= 0:
                        output_used[best_idx] = True
                        if best_score < 1.0:
                            element_errors.append(best_error)
                    else:
                        logging.warning(f"{field_path}[{i}]: no matching element found in output")
                        element_errors.append((1.0, f"{field_path}[{i}]: no matching element found in output"))
                if not element_errors:
                    return 1.0, (0.0, "")
                max_severity, combined_msg = truncate_errors(element_errors, mode='max')
                return 0.0, (max_severity, combined_msg)
            # Handle dict values - apply threshold recursively to each element using fuzzy key matching
            if isinstance(gold_val, dict):
                if not isinstance(output_val, dict):
                    logging.warning(f"{field_path}: type mismatch - expected dict, got {type(output_val).__name__}")
                    return 0.0, (1.0, f"{field_path}: type mismatch - expected dict, got {type(output_val).__name__}")
                element_scores = []
                element_errors = []
                output_keys = list(output_val.keys())
                gold_keys = list(gold_val.keys())
                for key, g in gold_val.items():
                    matched_key = find_matching_key(key, output_keys)
                    if matched_key is None:
                        logging.warning(f"{field_path}.{key}: key not found in output dict")
                        element_scores.append(0.0)
                        element_errors.append((1.0, f"{field_path}.{key}: key not found in output dict"))
                    else:
                        o = output_val[matched_key]
                        s, (es, em) = evaluate_field_normalized(o, g, threshold, f"{field_path}.{key}")
                        element_scores.append(s)
                        if s < 1.0:
                            element_errors.append((es, em))
                avg = sum(element_scores) / len(element_scores) if element_scores else 0.0
                max_severity, combined_msg = truncate_errors(element_errors, mode='max')
                return avg, (max_severity, combined_msg)
            if not isinstance(output_val, (int, float)) or not isinstance(gold_val, (int, float)):
                logging.warning(f"{field_path}: type mismatch - expected number, got {type(output_val).__name__} '{output_val}'")
                return 0.0, (1.0, f"{field_path}: type mismatch - expected number, got {type(output_val).__name__} '{output_val}'")
            rel_error = abs(output_val - gold_val) / max(abs(gold_val), 1e-12)
            # Add small epsilon to handle floating point precision issues
            if rel_error <= threshold + 1e-9:
                return 1.0, (0.0, "")
            return 0.0, (min(rel_error, 1.0), f"{field_path}: FAIL - relative error {rel_error:.4f} exceeds {threshold:.2%} tolerance (expected {gold_val}, got {output_val}, difference {abs(output_val - gold_val):.6f})")

        # Handle list threshold - range [bound1, bound2] with normalization
        # Note: This does NOT require gold_val
        if isinstance(threshold, list) and len(threshold) == 2:
            if not isinstance(output_val, (int, float)):
                logging.warning(f"{field_path}: type mismatch - expected number in range {threshold}, got {type(output_val).__name__} '{output_val}'")
                return 0.0, (1.0, f"{field_path}: type mismatch - expected number in range {threshold}, got {type(output_val).__name__} '{output_val}'")
            bound1, bound2 = threshold
            if bound1 == bound2:
                # Edge case: if bounds are equal, check if value matches exactly
                if output_val == bound1:
                    return 1.0, (0.0, "")
                severity = 1.0
                return 0.0, (severity, f"{field_path}: FAIL - value {output_val} != target {bound1}")
            if bound1 < bound2:
                # Higher is better: score = (value - bound1) / (bound2 - bound1)
                score = (output_val - bound1) / (bound2 - bound1)
                if score >= 1.0:
                    return 1.0, (0.0, "")
                # Severity = 1 - score (higher severity when score is lower)
                severity = 1.0 - min(max(score, 0.0), 1.0)
                error = f"{field_path}: FAIL - value {output_val} below target range [{bound1}, {bound2}], needs higher value (normalized score {score:.4f})"
            else:
                # Lower is better: score = (bound1 - value) / (bound1 - bound2)
                score = (bound1 - output_val) / (bound1 - bound2)
                if score >= 1.0:
                    return 1.0, (0.0, "")
                severity = 1.0 - min(max(score, 0.0), 1.0)
                error = f"{field_path}: FAIL - value {output_val} above target range [{bound2}, {bound1}], needs lower value (normalized score {score:.4f})"
            # Clip to [0, 1]
            clipped_score = max(0.0, min(1.0, score))
            return clipped_score, (severity, error)

        # Handle dict threshold - nested structure
        if isinstance(threshold, dict):
            if not isinstance(output_val, dict):
                logging.warning(f"{field_path}: type mismatch - expected dict, got {type(output_val).__name__}")
                return 0.0, (1.0, f"{field_path}: type mismatch - expected dict, got {type(output_val).__name__}")
            if gold_val is not None and not isinstance(gold_val, dict):
                logging.warning(f"{field_path}: type mismatch - expected dict for gold value, got {type(gold_val).__name__}")
                return 0.0, (1.0, f"{field_path}: type mismatch - expected dict, got {type(gold_val).__name__}")

            nested_scores = []
            nested_errors = []
            output_keys = list(output_val.keys())
            gold_keys = list(gold_val.keys()) if gold_val else []
            for key, nested_threshold in threshold.items():
                nested_path = f"{field_path}.{key}" if field_path else key
                matched_output_key = find_matching_key(key, output_keys)
                matched_gold_key = find_matching_key(key, gold_keys) if gold_val else None

                if matched_output_key is not None:
                    nested_gold = gold_val.get(matched_gold_key) if matched_gold_key and gold_val else None
                    nested_score, (nested_error_score, nested_error) = evaluate_field_normalized(output_val[matched_output_key], nested_gold, nested_threshold, nested_path)
                    nested_scores.append(nested_score)
                    if nested_error:
                        nested_errors.append((nested_error_score, nested_error))
                else:
                    logging.warning(f"{nested_path}: key not found in output (tried '{key}', available keys: {output_keys}), scoring 0 for all its fields")
                    # Key missing in output
                    nested_scores.append(0.0)
                    nested_errors.append((1.0, f"{nested_path}: key not found in output"))

            avg_score = sum(nested_scores) / len(nested_scores) if nested_scores else 0.0
            # Aggregate error_score as max of nested error_scores (worst error in normalized mode)
            max_error_score, error_msg = truncate_errors(nested_errors, mode='max')
            return avg_score, (max_error_score, error_msg)

        # Unknown threshold type
        logging.warning(f"{field_path}: unknown threshold type {type(threshold).__name__}")
        return 0.0, (1.0, f"{field_path}: unknown threshold type {type(threshold).__name__}")

    def traverse_thresholds_normalized(output_obj: Any, gold_obj: Any, threshold_obj: dict, obj_path: str = ""):
        """
        Recursively traverse threshold configuration and evaluate fields with normalization.
        """
        output_keys = list(output_obj.keys()) if isinstance(output_obj, dict) else []
        gold_keys = list(gold_obj.keys()) if isinstance(gold_obj, dict) and gold_obj else []
        for key, threshold in threshold_obj.items():
            field_path = f"{obj_path}.{key}" if obj_path else key
            matched_output_key = find_matching_key(key, output_keys)
            matched_gold_key = find_matching_key(key, gold_keys)
            # For nested dict thresholds, recurse into the nested structure
            if isinstance(threshold, dict):
                if isinstance(output_obj, dict):
                    if matched_output_key is not None:
                        nested_gold = gold_obj.get(matched_gold_key) if matched_gold_key and gold_obj else None
                        traverse_thresholds_normalized(output_obj[matched_output_key], nested_gold, threshold, field_path)
                    else:
                        # Missing nested object - score 0 for all its fields
                        logging.warning(f"{field_path}: missing nested object in output (tried '{key}', available keys: {output_keys}), scoring 0 for all its fields")
                        field_count = count_nested_fields(threshold)
                        scores.extend([0.0] * field_count)
                        errors.append((1.0, f"{field_path}: key not found"))
            else:
                # Simple threshold - evaluate single field
                gold_val = None
                if matched_gold_key is not None and gold_obj is not None:
                    gold_val = gold_obj[matched_gold_key]

                if matched_output_key is not None and isinstance(output_obj, dict):
                    output_val = output_obj[matched_output_key]
                    score, (error_score, error) = evaluate_field_normalized(output_val, gold_val, threshold, field_path)
                    scores.append(score)
                    if error:
                        errors.append((error_score, error))
                else:
                    # Field missing in output
                    logging.warning(f"{field_path}: key not found in output (tried '{key}', available keys: {output_keys}), scoring 0")
                    scores.append(0.0)
                    errors.append((1.0, f"{field_path}: key not found in output"))

    # Start traversal from root
    traverse_thresholds_normalized(output_data, gold_data, thresholds)

    # Limit total errors to max 50, sorted by error_score (lowest/most severe first)
    max_total_errors = 50
    if len(errors) > max_total_errors:
        # Sort by error_score ascending (0.0 = worst, 1.0 = best), then alphabetically
        errors_sorted = sorted(errors, key=lambda x: (x[0], x[1]))
        top_errors = errors_sorted[:max_total_errors]
        remaining = len(errors) - max_total_errors
        # Convert to just messages
        errors = [msg for _, msg in top_errors]
        errors.append(f"... and {remaining} more errors")
    else:
        # Convert to just messages
        errors = [msg for _, msg in errors]

    # Return average score, errors, and meaning
    avg_score = sum(scores) / len(scores) if scores else 0.0
    meaning = f"Comparing JSON files with normalized scoring: output='{output_file_name}'"
    if gold_file_name:
        meaning += f", gold='{gold_file_name}'"
    meaning += ", evaluating "

    # Explain thresholds
    def describe_threshold(thresh):
        if thresh is None:
            return "None (default: 1% rel error for numbers, strict equality for strings)"
        elif isinstance(thresh, (int, float)):
            return f"{thresh} (relative error tolerance)"
        elif isinstance(thresh, list) and len(thresh) == 2:
            if thresh[0] < thresh[1]:
                return f"{thresh} (higher better, range [{thresh[0]}, {thresh[1]}])"
            else:
                return f"{thresh} (lower better, range [{thresh[1]}, {thresh[0]}])"
        elif isinstance(thresh, dict):
            nested = []
            for k, v in thresh.items():
                nested.append(f"{k}:{describe_threshold(v)}")
            return "{" + ", ".join(nested) + "}"
        else:
            return str(thresh)

    threshold_desc = []
    for key, thresh in thresholds.items():
        threshold_desc.append(f"{key}={describe_threshold(thresh)}")

    meaning += f"{len(scores)} fields with thresholds: {'; '.join(threshold_desc)}"

    # Include data if file sizes are small (<1000 characters when formatted as JSON)
    result = {'score': avg_score, 'errors': errors, 'meaning': meaning}
    try:
        output_json_str = json.dumps(output_data, indent=2)
        if len(output_json_str) < 1000:
            result['output_data'] = output_data
        else:
            result['output_data'] = output_json_str[:1000] + f"\n... ({len(output_json_str)} characters total)"

        if gold_file_name and gold_data is not None:
            gold_json_str = json.dumps(gold_data, indent=2)
            if len(gold_json_str) < 1000:
                result['gold_data'] = gold_data
            else:
                result['gold_data'] = gold_json_str[:1000] + f"\n... ({len(gold_json_str)} characters total)"
    except Exception:
        pass  # If serialization fails, just skip adding data

    return result