"""
Table Metrics Module - CSV and SQLite comparison metrics for evaluation.

This module compares tabular data:
- CSV file comparison with column matching and tolerance support
- SQLite database comparison by converting to CSVs
- Handles multiple potential gold answers
- Supports specified columns and ignore_order options

Reference: https://github.com/yiyihum/da-code/tree/main/da_agent/evaluators/metrics/table.py
"""

import logging
import os.path
# import operator
from typing import Any
from typing import Dict, List
import math
import random
from typing import Dict, List
import pandas as pd
import sqlite3

def compare_csv(output_file_name: str, gold_file_name, **options) -> Dict[str, Any]:
    """
    @args:
        output_file_name(str): the pred csv file
        gold_file_name(str|List[str]): the gold csv file or csv files, maybe multiple potential answers, not there are two answers
        option(dict): the configuration dictionary
            - specified_columns(List|List[List]): the column name that should be used to compare the two csv files
            - ignore_order(bool|List(bool)): whether to ignore the order of the rows
            - thresholds(dict): column index to tolerance mapping, e.g., {0: None, 1: 0.02}
                               None means default tolerance (0.01), otherwise use the specified value
                               This allows different tolerance for different columns (e.g., derived statistics)
    @return:
        dict: {'score': float, 'errors': list[str]}
            - score: the similarity score (0.0 to 1.0)
            - errors: detailed error messages including missing columns
    """
    assert options.keys() <= {'specified_columns', 'ignore_order', 'thresholds'}, f"Unexpected options: {options.keys()}"
    if isinstance(gold_file_name, List):
        specified_columns = options.get('specified_columns', [[]]*len(gold_file_name))
        # score_rule = options.get('score_rule', ['divide']*len(gold_file_name))
        ignore_order = options.get('ignore_order', [True]*len(gold_file_name))
        # total_scores = options.get('total_scores', [1]*len(gold_file_name))
    elif isinstance(gold_file_name, str):
        specified_columns = [options.get('specified_columns', [])]
        # score_rule = [options.get('score_rule', 'divide')]
        ignore_order = [options.get('ignore_order', True)]
        # total_scores = [options.get('total_scores', 1)]
        gold_file_name = [gold_file_name]
    thresholds = options.get('thresholds', {})
    default_abs_tol = 1e-2  # default absolute tolerance

    def resolve_threshold_keys(thresholds_dict, gold_df):
        """
        Convert threshold keys from column names to column indices.
        Args:
            thresholds_dict: dict with keys as column indices (int) or column names (str)
            gold_df: gold DataFrame to lookup column indices
        Returns:
            dict: thresholds with keys converted to column indices
        """
        resolved = {}
        for key, value in thresholds_dict.items():
            if isinstance(key, int):
                resolved[key] = value
            elif isinstance(key, str):
                # Column name: find index in gold_df
                if key in gold_df.columns:
                    resolved[gold_df.columns.get_loc(key)] = value
                else:
                    logging.warning(f"Threshold column '{key}' not found in gold file, available columns: {list(gold_df.columns)}")
            else:
                logging.warning(f"Invalid threshold key type: {type(key)}, expected int (column index) or str (column name)")
        return resolved

    def get_tolerance_info(col_idx, thresholds_dict):
        """
        Get tolerance info for a specific column index.
        Returns: (tolerance_value, is_relative)
        - If column not in thresholds: (default_abs_tol, False) - use absolute tolerance
        - If column in thresholds with None: (0.01, True) - use relative tolerance with default 0.01
        - If column in thresholds with value: (value, True) - use relative tolerance with specified value
        """
        if col_idx in thresholds_dict:
            tol = thresholds_dict[col_idx]
            if tol is None:
                return (0.01, True)  # relative tolerance, default 0.01
            return (tol, True)  # relative tolerance, specified value
        return (default_abs_tol, False)  # absolute tolerance, default 0.01

    def normalize_value(val, tol, is_relative=False):
        """Normalize value for hashing - round floats for tolerance, lowercase strings"""
        if pd.isna(val):
            return "__NA__"
        elif isinstance(val, (int, float)):
            if is_relative:
                # Relative tolerance: snap to a "shared logarithmic grid" so that two
                # numbers within relative error tol land in the same bucket (mirroring
                # the shared linear grid used for absolute tolerance).
                # We must NOT snap using each number's own grid width abs(val)*tol —
                # that gives every number its own bucket, the grid can never be shared,
                # and relative tolerance degrades to "floats must be exactly equal".
                if val == 0 or tol <= 0:
                    return ("rel", 0.0 if val == 0 else float(val))
                sign = 1 if val > 0 else -1
                # Same bucket => |log|a| - log|b|| < log1p(tol) => relative error < tol
                # (Conservative: a shared bucket is always within tolerance, never
                # mistaking two numbers beyond tolerance as equal.)
                return ("rel", sign, round(math.log(abs(val)) / math.log1p(tol)))
            actual_tol = tol
            return round(val / actual_tol) * actual_tol
        elif isinstance(val, str):
            return val.lower().strip()
        return val

    def vector_to_hashable(v, tol, is_relative=False, do_sort=False):
        """Convert vector to hashable tuple for fast comparison"""
        normalized = [normalize_value(x, tol, is_relative) for x in v]
        if do_sort:
            normalized = sorted(normalized, key=lambda x: (x == "__NA__", str(x)))
        return tuple(normalized)

    def vectors_match(v1, v2, col_idx, ignore_order_=False, thresholds_dict={}):
        tol, is_relative = get_tolerance_info(col_idx, thresholds_dict)
        if ignore_order_:
            # For ignore_order mode, check if all values in v1 exist in v2
            # Build normalized sets
            set_v1 = set(normalize_value(x, tol, is_relative) for x in v1)
            set_v2 = set(normalize_value(x, tol, is_relative) for x in v2)
            # All elements in v1 must be in v2
            return set_v1.issubset(set_v2)
        else:
            # Exact match mode: lengths and order must match
            if len(v1) != len(v2):
                return False
            for a, b in zip(v1, v2):
                if pd.isna(a) and pd.isna(b):
                    continue
                elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    if is_relative:
                        # 相对误差
                        if abs(a) < 1e-12 and abs(b) < 1e-12:
                            continue  # 两个都接近0，认为匹配
                        if abs(a) < 1e-12 or abs(b) < 1e-12:
                            return False  # 一个为0一个不为0
                        rel_error = abs(a - b) / max(abs(a), abs(b))
                        if rel_error > tol:
                            return False
                    else:
                        # 绝对误差
                        if not math.isclose(float(a), float(b), abs_tol=tol):
                            return False
                elif isinstance(a, str) and isinstance(b, str):
                    # Case-insensitive string comparison
                    if a.lower().strip() != b.lower().strip():
                        return False
                elif a != b:
                    return False
            return True

    def csv_score(pred, gold, specified_columns_=[], score_rule_='divide', ignore_order_=False, total_scores_=1, thresholds_={}):
        """
        Compare CSV files and return detailed score, errors, and meaning.
        Returns:
            tuple: (score, errors, meaning)
            errors: list of strings, each containing column name, match score, and unmatched values
            meaning: string describing what the comparison did with the specified parameters
        """
        # Build mapping from filtered column index to original column index
        if specified_columns_ != []:
            gold_cols = gold.loc[:, specified_columns_]
            # Map filtered index to original index
            original_col_indices = [gold.columns.get_loc(col) for col in specified_columns_]
        else:
            gold_cols = gold
            # No filtering, so indices are the same
            original_col_indices = list(range(len(gold.columns)))
        pred_cols = pred

        t_gold_list = gold_cols.transpose().values.tolist()
        t_pred_list = pred_cols.transpose().values.tolist()

        max_elements = 10000
        if t_gold_list and len(t_gold_list[0]) > max_elements:
            t_gold_list = [col[:max_elements] for col in t_gold_list]
        if not ignore_order_:
            if t_pred_list and len(t_pred_list[0]) > max_elements:
                t_pred_list = [col[:max_elements] for col in t_pred_list]

        # Pre-compute hashes for pred columns for O(1) lookup
        pred_hashes = {}
        for j, pred_col in enumerate(t_pred_list):
            tol, is_rel = get_tolerance_info(j, thresholds_)
            h = vector_to_hashable(pred_col, tol, is_relative=is_rel, do_sort=ignore_order_)
            if h not in pred_hashes:
                pred_hashes[h] = j

        errors = []
        if score_rule_ == "all":
            pre_score = total_scores_
            for i, gold_col in enumerate(t_gold_list):
                orig_idx = original_col_indices[i]
                tol, is_rel = get_tolerance_info(orig_idx, thresholds_)
                gold_hash = vector_to_hashable(gold_col, tol, is_relative=is_rel, do_sort=ignore_order_)
                # Fast hash lookup first
                if gold_hash in pred_hashes:
                    continue
                # Fallback to exact comparison if hash miss (due to tolerance)
                found = False
                for pred_col in t_pred_list:
                    if vectors_match(gold_col, pred_col, orig_idx, ignore_order_=ignore_order_, thresholds_dict=thresholds_):
                        found = True
                        break
                if not found:
                    pre_score = 0
                    # Get column info
                    col_name = specified_columns_[i] if specified_columns_ else f"Column index {i}"
                    # Get unmatched values using same normalization as vectors_match
                    pred_normalized_sets = []
                    for pred_col in t_pred_list:
                        normalized = [normalize_value(x, tol, is_relative=is_rel) for x in pred_col]
                        pred_normalized_sets.append(set(normalized))

                    # Shuffle gold_col and find unmatched values (stop when count or string length limit reached)
                    shuffled_gold = gold_col.copy()
                    random.shuffle(shuffled_gold)
                    unmatched = []
                    max_count = 50  # Maximum number of unmatched values
                    max_str_len = 5000  # Maximum total string length for sample
                    for val in shuffled_gold:
                        val_norm = normalize_value(val, tol, is_relative=is_rel)
                        found_val = any(val_norm in pred_set for pred_set in pred_normalized_sets)
                        if not found_val:
                            unmatched.append(val)
                            # Stop if count limit reached
                            if len(unmatched) >= max_count:
                                break
                            # Stop if string length limit reached
                            sample_str = ", ".join([str(v) for v in unmatched])
                            if len(sample_str) >= max_str_len:
                                break

                    if unmatched:
                        if len(unmatched) == len(gold_col) and all(isinstance(x, str) for x in gold_col):
                            logging.warning(f"Column '{col_name}': match score 0.0/1.0, all values unmatched")
                        full_str = ", ".join([str(v) for v in unmatched])
                        if len(full_str) > max_str_len:
                            sample_str = full_str[:max_str_len] + "..."
                        else:
                            sample_str = full_str
                        error_msg = f"Column '{col_name}' does not match: expected values {sample_str} but none found in prediction (score_rule='all' requires all columns to match)"
                    else:
                        error_msg = f"Column '{col_name}' does not match: values are completely different from expected (score_rule='all' requires all columns to match)"
                    errors.append(error_msg)
        elif score_rule_ == "divide":
            matches = 0
            total = len(t_gold_list) if t_gold_list else 1

            for i, gold_col in enumerate(t_gold_list):
                orig_idx = original_col_indices[i]
                tol, is_rel = get_tolerance_info(orig_idx, thresholds_)
                gold_hash = vector_to_hashable(gold_col, tol, is_relative=is_rel, do_sort=ignore_order_)
                # Fast hash lookup first
                if gold_hash in pred_hashes:
                    matches += total_scores_
                    continue
                # Fallback to exact comparison
                found = False
                for pred_col in t_pred_list:
                    if vectors_match(gold_col, pred_col, orig_idx, ignore_order_=ignore_order_, thresholds_dict=thresholds_):
                        found = True
                        matches += total_scores_
                        break
                if not found:
                    # Get column info
                    col_name = specified_columns_[i] if specified_columns_ else f"Column index {i}"
                    # Get unmatched values using same normalization as vectors_match
                    pred_normalized_sets = []
                    for pred_col in t_pred_list:
                        normalized = [normalize_value(x, tol, is_relative=is_rel) for x in pred_col]
                        pred_normalized_sets.append(set(normalized))

                    # Shuffle gold_col and find unmatched values (stop when count or string length limit reached)
                    shuffled_gold = gold_col.copy()
                    random.shuffle(shuffled_gold)
                    unmatched = []
                    max_count = 50  # Maximum number of unmatched values
                    max_str_len = 5000  # Maximum total string length for sample
                    for val in shuffled_gold:
                        val_norm = normalize_value(val, tol, is_relative=is_rel)
                        found_val = any(val_norm in pred_set for pred_set in pred_normalized_sets)
                        if not found_val:
                            unmatched.append(val)
                            # Stop if count limit reached
                            if len(unmatched) >= max_count:
                                break
                            # Stop if string length limit reached
                            sample_str = ", ".join([str(v) for v in unmatched])
                            if len(sample_str) >= max_str_len:
                                break

                    if unmatched:
                        if len(unmatched) == len(gold_col) and all(isinstance(x, str) for x in gold_col):
                            logging.warning(f"{col_name}: no values matched, scoring 0 for entire column")
                        full_str = ", ".join([str(v) for v in unmatched])
                        if len(full_str) > max_str_len:
                            sample_str = full_str[:max_str_len] + "..."
                        else:
                            sample_str = full_str
                        error_msg = f"Column '{col_name}' does not match: expected values {sample_str} but none found in prediction (score_rule='divide' scores each column independently)"
                    else:
                        error_msg = f"Column '{col_name}' does not match: values are completely different from expected (score_rule='divide' scores each column independently)"
                    errors.append(error_msg)

            pre_score = matches / total

        # Build meaning description
        if specified_columns_:
            meaning = f"Compared {len(t_gold_list)} specified columns: {', '.join(specified_columns_)}"
        else:
            meaning = f"Compared all {len(t_gold_list)} columns in the CSV files"

        if ignore_order_:
            meaning += " (ignoring row order)"

        if score_rule_ == "all":
            meaning += " with score rule 'all' (all columns must match for full score)"
        else:
            meaning += f" with score rule 'divide' (score = matched columns / total columns)"

        meaning += f". Score: {pre_score:.2f}/{total_scores_}"

        return pre_score, errors, meaning

    output = []
    output_errors = []
    output_meanings = []
    output_data = None
    gold_data = None

    if not os.path.exists(output_file_name):
        return {
            'score': 0,
            'errors': ['Output file does not exist'],
            'meaning': f"Failed to compare CSV files: output file '{os.path.basename(output_file_name)}' does not exist",
            'output_data': None,
            'gold_data': None
        }

    # Fast row count using wc -l (much faster than iterating in Python)
    import subprocess
    try:
        output_total_rows = int(subprocess.run(['wc', '-l', output_file_name], capture_output=True, text=True).stdout.split()[0]) - 1
    except Exception:
        output_total_rows = sum(1 for _ in open(output_file_name)) - 1

    max_gold_rows = 0
    for gold_file in gold_file_name:
        if os.path.exists(gold_file):
            try:
                gold_rows = int(subprocess.run(['wc', '-l', gold_file], capture_output=True, text=True).stdout.split()[0]) - 1
            except Exception:
                gold_rows = sum(1 for _ in open(gold_file)) - 1
            max_gold_rows = max(max_gold_rows, gold_rows)

    # If output has fewer rows than gold, score 0 immediately
    if output_total_rows < max_gold_rows:
        return {
            'score': 0,
            'errors': [f'Row count mismatch: output has {output_total_rows} rows, gold has {max_gold_rows} rows (output too short)'],
            'meaning': f"Failed: output ({output_total_rows} rows) has fewer rows than gold ({max_gold_rows} rows)",
            'output_data': None,
            'gold_data': None
        }

    # Read output file (limit rows when ignore_order=False, matching csv_score logic)
    all_ignore_order_false = all(not io for io in ignore_order)
    try:
        df1 = pd.read_csv(output_file_name, low_memory=False,
                           nrows=10000 if (all_ignore_order_false and output_total_rows > 10000) else None)
        if df1.empty:
            return {
                'score': 0,
                'errors': ['Output file is empty'],
                'meaning': f"Failed to compare CSV files: output file '{os.path.basename(output_file_name)}' is empty",
                'output_data': None,
                'gold_data': None
            }
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logging.warning(f"Failed to read result CSV {output_file_name}: {e}")
        return {
            'score': 0,
            'errors': [f'Failed to read result CSV: {e}'],
            'meaning': f"Failed to compare CSV files: error reading output file '{os.path.basename(output_file_name)}': {e}",
            'output_data': None,
            'gold_data': None
        }

    for i in range(len(gold_file_name)):
        try:
            df2 = pd.read_csv(gold_file_name[i], low_memory=False, nrows=10000)
            if df2.empty:
                output.append(0)
                output_errors.append(['Gold file is empty'])
                output_meanings.append(f"Gold file '{os.path.basename(gold_file_name[i])}' is empty")
                continue
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            logging.warning(f"Failed to read expected CSV {gold_file_name[i]}: {e}")
            output.append(0)
            output_errors.append([f'Failed to read gold CSV: {e}'])
            output_meanings.append(f"Error reading gold file '{os.path.basename(gold_file_name[i])}': {e}")
            continue

        # Save output_data and gold_data (incremental to_string to avoid full conversion on large files)
        for head_n in [10, 100, 1000, None]:
            output_str = df1.head(head_n).to_string() if head_n else df1.to_string()
            if len(output_str) >= 1000 or head_n is None:
                break

        if len(output_str) < 1000:
            output_data = output_str
        else:
            output_data = output_str[:1000] + f"\n..."

        for head_n in [10, 100, 1000, None]:
            gold_str = df2.head(head_n).to_string() if head_n else df2.to_string()
            if len(gold_str) >= 1000 or head_n is None:
                break

        if len(gold_str) < 1000:
            gold_data = gold_str
        else:
            gold_data = gold_str[:1000] + f"\n..."

        # Resolve threshold column names to indices using gold file columns
        resolved_thresholds = resolve_threshold_keys(thresholds, df2)

        pre_score, errors, meaning = csv_score(df1, df2, specified_columns_=specified_columns[i], ignore_order_=ignore_order[i], thresholds_=resolved_thresholds)
        output.append(pre_score)
        output_errors.append(errors)
        output_meanings.append(meaning)

    max_idx = output.index(max(output)) if output else 0
    best_meaning = output_meanings[max_idx] if output else f"Failed to compare: no valid gold files"
    return {
        'score': max(output) if output else 0,
        'errors': output_errors[max_idx] if output else [],
        'meaning': best_meaning,
        'output_data': output_data,
        'gold_data': gold_data
    }


def compare_sqlite(output_file_name: str, gold_file_name: str, **options) -> Dict[str, Any]:
    """
    @args:
        output_file_name(str): the pred database
        gold_file_name(str|List[str]): the gold database or database files, maybe multiple potential answers
        option(dict): the configuration dictionary
            - specified_schema(dict): the specified schema for the tables, e.g. {"table1": ["col1", "col2"], "table2": ["colA", "colB"]}
                                     if a table's value is None, compare all columns of that table
            - ignore_order(bool|List(bool)): whether to ignore the order of the rows (can be dict per table)
    @return:
        dict: {'score': float, 'errors': list[str], 'meaning': str}
            - score: the similarity score (0.0 to 1.0)
            - errors: detailed error messages including missing tables or columns
            - meaning: description of what was compared
    """
    assert options.keys() <= {'specified_schema', 'ignore_order'}, f"Unexpected options: {options.keys()}"
    if isinstance(gold_file_name, List):
        specified_schema = options.get('specified_schema', [{}]*len(gold_file_name))
    elif isinstance(gold_file_name, str):
        specified_schema = [options.get('specified_schema', {})]
        gold_file_name = [gold_file_name]

    def convert_to_csvs(db_path, schema, max_rows=10000):
        """ Convert specified tables in a SQLite database to CSV files and return their paths. """
        csv_dir = os.path.dirname(db_path)
        csv_paths = []
        table_names = []
        conn = sqlite3.connect(db_path)
        try:
            for table_name in schema:
                column_names = schema[table_name]
                if column_names is None:
                    query = f"SELECT * FROM {table_name} LIMIT {max_rows}"
                else:
                    cols_str = ", ".join([f'"{col}"' for col in column_names])
                    query = f"SELECT {cols_str} FROM {table_name} LIMIT {max_rows}"
                df = pd.read_sql_query(query, conn)
                csv_path = os.path.join(csv_dir, f"_{table_name}.csv")
                df.to_csv(csv_path, index=False)
                csv_paths.append(csv_path)
                table_names.append(table_name)
        except Exception as e:
            logging.warning(f"Error converting table {table_name} from {db_path}: {e}")
        finally:
            conn.close()
        return csv_paths, table_names

    def get_table_names(db_path):
        if not os.path.exists(db_path):
            logging.warning(f"Database file does not exist: {db_path}")
            return []
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()
        return [table[0] for table in tables]

    def get_table_schema(db_path, table_name):
        """Get the schema (column names) of a table"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        conn.close()
        return [col[1] for col in columns]

    output_data = None
    gold_data = None

    # Check if output file exists
    if not os.path.exists(output_file_name):
        return {
            'score': 0,
            'errors': ['Output database file does not exist'],
            'meaning': f"Failed to compare SQLite databases: output file '{os.path.basename(output_file_name)}' does not exist",
            'output_data': None,
            'gold_data': None
        }
    output_tables = get_table_names(output_file_name)
    if not output_tables:
        return {
            'score': 0,
            'errors': ['Output database is empty or invalid'],
            'meaning': f"Failed to compare SQLite databases: output database '{os.path.basename(output_file_name)}' is empty or invalid",
            'output_data': None,
            'gold_data': None
        }

    # Compare with each gold database
    output_scores = []
    output_errors = []
    output_meanings = []

    for i in range(len(gold_file_name)):
        gold_db = gold_file_name[i]

        if not os.path.exists(gold_db):
            logging.warning(f"Gold database file does not exist: {gold_db}")
            output_scores.append(0)
            output_errors.append([f'Gold database file does not exist: {os.path.basename(gold_db)}'])
            output_meanings.append(f"Gold database '{os.path.basename(gold_db)}' does not exist")
            continue

        # Get table names from gold database
        gold_tables = get_table_names(gold_db)
        if not gold_tables:
            logging.warning(f"Gold database is empty or invalid: {gold_db}")
            output_scores.append(0)
            output_errors.append([f'Gold database is empty or invalid: {os.path.basename(gold_db)}'])
            output_meanings.append(f"Gold database '{os.path.basename(gold_db)}' is empty or invalid")
            continue

        # Get tables to compare
        if specified_schema[i]:
            # Use specified tables from schema
            tables_to_compare = list(specified_schema[i].keys())
            # Check if all specified tables exist in gold database
            missing_gold_tables = [t for t in tables_to_compare if t not in gold_tables]
            if missing_gold_tables:
                logging.warning(f"Specified tables not found in gold database: {missing_gold_tables}")
                output_scores.append(0)
                output_errors.append([f"Specified tables not found in gold database: {', '.join(missing_gold_tables)}"])
                output_meanings.append(f"Failed: specified tables {missing_gold_tables} not in gold database")
                continue

            # Check if all specified tables exist in output database
            missing_output_tables = [t for t in tables_to_compare if t not in output_tables]
            if missing_output_tables:
                logging.warning(f"Specified tables not found in output database: {missing_output_tables}")
                output_scores.append(0)
                output_errors.append([f"Specified tables not found in output database: {', '.join(missing_output_tables)}"])
                output_meanings.append(f"Failed: specified tables {missing_output_tables} not in output database")
                continue
        else:
            # Compare all tables that exist in both databases
            tables_to_compare = sorted(list(set(output_tables) & set(gold_tables)))
            missing_gold_tables = sorted(list(set(gold_tables) - set(output_tables)))
            missing_output_tables = sorted(list(set(output_tables) - set(gold_tables)))

            if missing_gold_tables:
                logging.warning(f"Tables missing in output database: {missing_gold_tables}")

            if missing_output_tables:
                logging.info(f"Extra tables in output database: {missing_output_tables}")

        if not tables_to_compare:
            output_scores.append(0)
            output_errors.append(['No common tables to compare'])
            output_meanings.append("Failed: no common tables between databases")
            continue

        # Build schema for conversion
        schema_to_use = {}
        for table_name in tables_to_compare:
            if specified_schema[i] and table_name in specified_schema[i]:
                schema_to_use[table_name] = specified_schema[i][table_name]
            else:
                # Compare all columns
                schema_to_use[table_name] = None

        # Convert tables to CSV files
        gold_csvs, gold_table_names = convert_to_csvs(gold_db, schema_to_use)
        pred_csvs, pred_table_names = convert_to_csvs(output_file_name, schema_to_use)

        if len(pred_csvs) != len(gold_csvs) or len(pred_csvs) == 0:
            output_scores.append(0)
            output_errors.append([f"Failed to convert tables to CSV: pred={len(pred_csvs)}, gold={len(gold_csvs)}"])
            output_meanings.append(f"Failed to convert tables to CSV for comparison")
            # Clean up temp CSVs
            for p in gold_csvs + pred_csvs:
                if os.path.exists(p):
                    os.remove(p)
            continue

        # Compare each table
        table_scores = []
        table_errors = []

        for j, table_name in enumerate(tables_to_compare):
            # Get specified columns for this table
            if specified_schema[i] and table_name in specified_schema[i]:
                specified_cols = specified_schema[i][table_name]
                if specified_cols is None:
                    # Compare all columns - pass empty list to compare_csv to compare all
                    specified_cols = []
            else:
                specified_cols = []

            csv_result = compare_csv(
                pred_csvs[j],
                gold_csvs[j],
                specified_columns=specified_cols,
                ignore_order=False
                # score_rule='divide'
            )

            table_scores.append(csv_result['score'])
            if csv_result['errors']:
                table_errors.extend([f"Table '{table_name}': {err}" for err in csv_result['errors']])

            # Check schema differences when comparing all columns
            if specified_cols == []:
                gold_schema = get_table_schema(gold_db, table_name)
                pred_schema = get_table_schema(output_file_name, table_name)

                # Find missing or extra columns
                missing_cols = set(gold_schema) - set(pred_schema)
                extra_cols = set(pred_schema) - set(gold_schema)

                if missing_cols:
                    table_errors.append(f"Table '{table_name}': missing columns in output: {', '.join(missing_cols)}")
                    logging.warning(f"Table '{table_name}': missing columns {missing_cols}")

                if extra_cols:
                    logging.info(f"Table '{table_name}': extra columns in output: {', '.join(extra_cols)}")

        if table_scores:
            overall_score = (sum(table_scores) / len(table_scores))
            meaning = f"Compared {len(table_scores)} tables with score rule 'divide' (score = average of table scores)"
        else:
            overall_score = 0
            meaning = "No tables compared"

        # Add table count to meaning
        meaning += f". Tables: {', '.join(tables_to_compare)}"

        # Add schema info to meaning
        if specified_schema[i]:
            meaning += f" (with specified schema: {specified_schema[i]})"
        else:
            meaning += f" (comparing all common tables)"

        # Add row count info
        try:
            gold_rows = sum(len(pd.read_csv(csv)) for csv in gold_csvs)
            pred_rows = sum(len(pd.read_csv(csv)) for csv in pred_csvs)
            meaning += f". Total rows - gold: {gold_rows}, output: {pred_rows}"
        except:
            pass

        output_scores.append(overall_score)
        output_errors.append(table_errors)
        output_meanings.append(meaning)

        # Clean up temp CSV files
        for p in gold_csvs + pred_csvs:
            if os.path.exists(p):
                os.remove(p)

    if not output_scores:
        return {
            'score': 0,
            'errors': ['No valid gold databases to compare'],
            'meaning': 'Failed: no valid gold databases',
            'output_data': None,
            'gold_data': None
        }

    max_idx = output_scores.index(max(output_scores))
    best_meaning = output_meanings[max_idx]

    return {
        'score': max(output_scores),
        'errors': output_errors[max_idx],
        'meaning': best_meaning,
        'output_data': output_data,
        'gold_data': gold_data
    }
