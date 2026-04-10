import json
import glob
import os
import sys
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
SAMPLE_PATHS_FILE = os.path.join(os.path.dirname(BASE_DIR), "skill_cluster", "data", "sample_paths.jsonl")
ANNOTATE_RESULTS_DIR = os.path.join(
    BASE_DIR, "annotate_0309_result3 (revised by Zhong)"
)

AUX_ACCEPTED_EVENTS = {
    "aux_step_accepted",
    "aux_step_regenerated_accepted",
    "deferred_retry_accepted",
}


# ---------------------------------------------------------------------------
#  File collection helpers (scan OUTPUT_DIR and its immediate subdirectories)
# ---------------------------------------------------------------------------

def _collect_eval_case_files(base_dir=OUTPUT_DIR):
    """Collect all eval_cases JSONL files from *base_dir* and its immediate subdirectories."""
    files = []
    for pattern_template in ["eval_cases_(*.jsonl"]:
        files.extend(glob.glob(os.path.join(base_dir, pattern_template)))
        files.extend(glob.glob(os.path.join(base_dir, "*", pattern_template)))
    return sorted(set(files))


def _collect_trace_files(base_dir=OUTPUT_DIR):
    """Collect all synthesis_trace JSONL files from *base_dir* and its immediate subdirectories."""
    files = []
    for scan_dir in [base_dir] + sorted(glob.glob(os.path.join(base_dir, "*"))):
        if not os.path.isdir(scan_dir):
            continue
        for fname in os.listdir(scan_dir):
            if fname.startswith("synthesis_trace") and fname.endswith(".jsonl"):
                files.append(os.path.join(scan_dir, fname))
    return sorted(set(files))


# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------

def count_code_lines_from_str(code_str):
    count = 0
    in_docstring = False
    for line in code_str.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('"""') or line.startswith("'''"):
            if not in_docstring:
                in_docstring = True
            elif in_docstring:
                in_docstring = False
            continue
        if in_docstring:
            continue
        if line.startswith('#'):
            continue
        count += 1
    return count


# ---------------------------------------------------------------------------
#  1. Count skills from eval_cases
# ---------------------------------------------------------------------------

def count_skills():
    files = _collect_eval_case_files()

    if not files:
        print("No eval_cases files found")
        return

    all_skills = set()
    skill_counter = Counter()
    domain_skills = {}
    domain_case_counts = {}
    total_cases = 0

    for fpath in sorted(files):
        fname = os.path.basename(fpath)
        domain = fname.replace("eval_cases_(", "").replace(").jsonl", "")

        if domain not in domain_skills:
            domain_skills[domain] = set()
            domain_case_counts[domain] = 0

        case_count = 0
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                skills = record.get("skills", [])
                for s in skills:
                    all_skills.add(s)
                    skill_counter[s] += 1
                    domain_skills[domain].add(s)
                case_count += 1

        domain_case_counts[domain] += case_count
        total_cases += case_count

    for domain in sorted(domain_skills):
        print(f"[{domain}] {domain_case_counts[domain]} cases, {len(domain_skills[domain])} unique skills")

    print(f"\n{'='*60}")
    print(f"Scanned {len(files)} domains, {total_cases} cases")
    print(f"Total unique skills: {len(all_skills)}")

    print(f"\n{'='*60}")
    print("Skills sorted by frequency:")
    for skill, count in skill_counter.most_common():
        print(f"  {count:>5}x  {skill}")


# ---------------------------------------------------------------------------
#  2. Check referenced_examples coverage in synthesis_trace
# ---------------------------------------------------------------------------

def _extract_final_steps(trace: list) -> list:
    """Extract steps that made it into the final pipeline from a trace.

    For the main step, only the last successful event is kept (no
    double-counting between main_step_revalidated and main_step_regenerated).
    """
    aux_steps = []
    main_step = None

    for evt in trace:
        event_name = evt.get("event", "")

        if event_name in AUX_ACCEPTED_EVENTS:
            aux_steps.append({
                "event": event_name,
                "skill": evt.get("skill", "?"),
                "referenced_examples": evt.get("referenced_examples", []),
            })

        elif event_name == "main_step_revalidated" and evt.get("applicable"):
            main_step = {
                "event": event_name,
                "skill": evt.get("skill", "(main)"),
                "referenced_examples": evt.get("referenced_examples", []),
            }

        elif event_name == "main_step_regenerated" and evt.get("success"):
            main_step = {
                "event": event_name,
                "skill": evt.get("skill", "(main)"),
                "referenced_examples": evt.get("referenced_examples", []),
            }

    result = list(aux_steps)
    if main_step:
        result.append(main_step)
    return result


def analyze_trace_file(filepath: str) -> dict:
    """Analyze a single domain's synthesis_trace JSONL file.

    Only successful attempts are considered.
    """
    attempts = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [warn] skipping malformed JSON at line {line_no}")
                continue

            outcome = entry.get("outcome", "unknown")
            if outcome != "success":
                continue

            trace = entry.get("step_generation", {}).get("trace", [])
            if not trace:
                continue

            final_steps = _extract_final_steps(trace)

            details = []
            for s in final_steps:
                details.append({
                    **s,
                    "has_examples": bool(s["referenced_examples"]),
                })

            steps_total = len(details)
            steps_with = sum(1 for d in details if d["has_examples"])
            steps_without = steps_total - steps_with

            attempts.append({
                "attempt": entry.get("attempt_number") or entry.get("task_id", line_no),
                "outcome": outcome,
                "steps_total": steps_total,
                "steps_with_examples": steps_with,
                "steps_without_examples": steps_without,
                "details": details,
            })

    total_steps = sum(a["steps_total"] for a in attempts)
    total_with = sum(a["steps_with_examples"] for a in attempts)
    total_without = sum(a["steps_without_examples"] for a in attempts)

    return {
        "attempts": attempts,
        "total_steps": total_steps,
        "total_with_examples": total_with,
        "total_without_examples": total_without,
        "empty_ratio": total_without / total_steps if total_steps else 0.0,
        "num_successful_attempts": len(attempts),
    }


def check_referenced_examples(verbose=False):
    trace_file_paths = _collect_trace_files()

    if not trace_file_paths:
        print(f"No synthesis_trace files found in {OUTPUT_DIR} (including subdirectories)")
        return

    print("=" * 80)
    print("Referenced Examples Coverage Report")
    print("=" * 80)

    domain_stats: dict = {}

    for filepath in trace_file_paths:
        tf = os.path.basename(filepath)
        domain = (tf.replace("synthesis_trace_(", "")
                    .replace("synthesis_trace", "")
                    .replace(").jsonl", "")
                    .replace(".jsonl", "")
                    .strip())
        if not domain:
            domain = "(global)"

        stats = analyze_trace_file(filepath)

        if domain not in domain_stats:
            domain_stats[domain] = {
                "total_steps": 0,
                "total_with_examples": 0,
                "total_without_examples": 0,
                "num_successful_attempts": 0,
                "all_attempts": [],
            }
        ds = domain_stats[domain]
        ds["total_steps"] += stats["total_steps"]
        ds["total_with_examples"] += stats["total_with_examples"]
        ds["total_without_examples"] += stats["total_without_examples"]
        ds["num_successful_attempts"] += stats["num_successful_attempts"]
        ds["all_attempts"].extend(stats["attempts"])

    summary_rows = []
    for domain in sorted(domain_stats):
        ds = domain_stats[domain]
        empty_ratio = ds["total_without_examples"] / ds["total_steps"] if ds["total_steps"] else 0.0
        summary_rows.append({
            "domain": domain,
            "total_steps": ds["total_steps"],
            "total_with_examples": ds["total_with_examples"],
            "total_without_examples": ds["total_without_examples"],
            "empty_ratio": empty_ratio,
            "num_successful_attempts": ds["num_successful_attempts"],
        })

        print(f"\n{'─' * 70}")
        print(f"Domain: {domain}")
        print(f"  Successful cases:         {ds['num_successful_attempts']}")
        print(f"  Final pipeline steps:     {ds['total_steps']}")
        print(f"  Steps WITH examples:      {ds['total_with_examples']}")
        print(f"  Steps WITHOUT examples:   {ds['total_without_examples']}")
        print(f"  Empty ratio:              {empty_ratio:.1%}")

        if verbose:
            for att in ds["all_attempts"]:
                if att["steps_total"] == 0:
                    continue
                print(f"\n  Attempt {att['attempt']} ({att['outcome']})  "
                      f"— {att['steps_with_examples']}/{att['steps_total']} have examples")
                for d in att["details"]:
                    flag = "✓" if d["has_examples"] else "✗"
                    refs = (", ".join(d["referenced_examples"])
                            if d["referenced_examples"] else "(none)")
                    print(f"    {flag} [{d['event']}] {d['skill']}  →  {refs}")

    print(f"\n{'=' * 80}")
    print("Summary Across All Domains")
    print(f"{'=' * 80}")

    grand_total = sum(r["total_steps"] for r in summary_rows)
    grand_with = sum(r["total_with_examples"] for r in summary_rows)
    grand_without = sum(r["total_without_examples"] for r in summary_rows)
    grand_ratio = grand_without / grand_total if grand_total else 0.0
    grand_cases = sum(r["num_successful_attempts"] for r in summary_rows)

    header = (f"{'Domain':<20} {'Cases':>6} {'Steps':>6} "
              f"{'With':>6} {'Without':>8} {'Empty%':>8}")
    print(header)
    print("─" * len(header))
    for r in summary_rows:
        print(f"{r['domain']:<20} {r['num_successful_attempts']:>6} "
              f"{r['total_steps']:>6} {r['total_with_examples']:>6} "
              f"{r['total_without_examples']:>8} {r['empty_ratio']:>7.1%}")
    print("─" * len(header))
    print(f"{'TOTAL':<20} {grand_cases:>6} {grand_total:>6} "
          f"{grand_with:>6} {grand_without:>8} {grand_ratio:>7.1%}")


# ---------------------------------------------------------------------------
#  3. Dataset-level statistics
# ---------------------------------------------------------------------------

def dataset_statistics():
    examples = set()
    skills = set()
    sum_skill_num = 0
    sum_step_num = 0
    sum_solution_lines = 0
    file_nums = []

    if os.path.isfile(SAMPLE_PATHS_FILE):
        with open(SAMPLE_PATHS_FILE, 'r') as fin:
            for line in fin:
                obj = json.loads(line)
                examples.update([e['example_id'] for e in obj['evidence']])
    else:
        print(f"[warn] sample_paths file not found: {SAMPLE_PATHS_FILE}")

    file_sizes = {}
    if os.path.isdir(DATASETS_DIR):
        for dir_name in os.listdir(DATASETS_DIR):
            dir_path = os.path.join(DATASETS_DIR, dir_name)
            if os.path.isdir(dir_path):
                for root, dirs, files in os.walk(dir_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_sizes[file_path] = os.path.getsize(file_path)
    else:
        print(f"[warn] datasets directory not found: {DATASETS_DIR}")

    for filepath in _collect_eval_case_files():
        with open(filepath, 'r') as fin:
            for line in fin:
                obj = json.loads(line)
                skills.update(obj['skills'])
                sum_skill_num += len(obj['skills'])
                sum_step_num += len(obj['pipeline'])
                for step in obj['pipeline']:
                    sum_solution_lines += count_code_lines_from_str(
                        step['code_snippet']
                    )
                cur_file_num = 0
                cur_file_size = 0
                domain = obj['domain']
                data_path = None
                for data_source in obj['data_sources']:
                    data_path = os.path.join(
                        DATASETS_DIR, domain, data_source
                    )
                    if data_path in file_sizes:
                        cur_file_num += 1
                        cur_file_size += file_sizes[data_path]
                    else:
                        print(f"Not found: {data_path}")
                file_nums.append((data_path, cur_file_num, cur_file_size))

    num = len(file_nums)
    if num == 0:
        print("No eval_cases data found.")
        return

    print(f"\n{'=' * 60}")
    print("Dataset Statistics")
    print(f"{'=' * 60}")
    print(f"Number of unique examples: {len(examples)}")
    print(f"Number of unique skills: {len(skills)}")
    print(f"Number of evaluation cases: {num}")
    print(f"Average number of skills per case: {sum_skill_num / num:.2f}")
    print(f"Average number of steps per case: {sum_step_num / num:.2f}")
    print(f"Average number of code lines per case: {sum_solution_lines / num:.2f}")
    print(f"Min data sources per instance: {min(file_nums, key=lambda x: x[1])}")
    print(f"Max data sources per instance: {max(file_nums, key=lambda x: x[1])}")
    print(f"Average data sources per instance: "
          f"{sum(n for _, n, _ in file_nums) / num:.2f}")
    print(f"Average data size (bytes) per instance: "
          f"{sum(s for _, _, s in file_nums) / num:.2f}")


# ---------------------------------------------------------------------------
#  4. Annotate results skill statistics
# ---------------------------------------------------------------------------

def _collect_annotate_question_files(base_dir=ANNOTATE_RESULTS_DIR):
    """Collect question JSON files from the annotate results directory.

    Walks every sub-directory, picks up JSON files that contain a top-level
    ``skills`` key, and skips ``output.json`` / ``output*.json`` files.
    """
    question_files = []
    if not os.path.isdir(base_dir):
        return question_files
    for root, _dirs, files in os.walk(base_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            if fname.startswith("output"):
                continue
            question_files.append(os.path.join(root, fname))
    return sorted(question_files)


def count_annotate_skills(base_dir=ANNOTATE_RESULTS_DIR):
    """Count unique skills across all annotated evaluation cases."""
    question_files = _collect_annotate_question_files(base_dir)

    if not question_files:
        print(f"No evaluation case JSON files found (directory: {base_dir})")
        return

    all_skills = set()
    all_llm_skills = set()
    skill_counter = Counter()
    llm_skill_counter = Counter()
    domain_skills: dict[str, set] = {}
    domain_llm_skills: dict[str, set] = {}
    domain_case_counts: dict[str, int] = {}
    total_cases = 0

    for fpath in question_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                record = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        skills = record.get("skills")
        if skills is None:
            continue

        rel = os.path.relpath(fpath, base_dir)
        domain = rel.split(os.sep)[0] if os.sep in rel else "(unknown)"

        if domain not in domain_skills:
            domain_skills[domain] = set()
            domain_llm_skills[domain] = set()
            domain_case_counts[domain] = 0

        for s in skills:
            all_skills.add(s)
            skill_counter[s] += 1
            domain_skills[domain].add(s)

        for s in record.get("llm_skills", []):
            all_llm_skills.add(s)
            llm_skill_counter[s] += 1
            domain_llm_skills[domain].add(s)

        domain_case_counts[domain] += 1
        total_cases += 1

    # --- Per-domain summary ---
    for domain in sorted(domain_skills):
        print(
            f"[{domain}] {domain_case_counts[domain]} cases, "
            f"{len(domain_skills[domain])} unique skills, "
            f"{len(domain_llm_skills[domain])} unique llm_skills"
        )

    # --- Totals ---
    print(f"\n{'='*60}")
    print(f"Scanned {len(domain_skills)} domains, {total_cases} cases")
    print(f"Total unique skills: {len(all_skills)}")

    # --- Skill frequency (skills) ---
    print(f"\n{'='*60}")
    print(f"Skills sorted by frequency (total {len(all_skills)}):")
    for skill, count in skill_counter.most_common():
        print(f"  {count:>5}x  {skill}")



# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    if "--annotate_results" in sys.argv:
        print("\n" + "▶" * 30 + " Annotate Results Skill Stats " + "◀" * 30)
        count_annotate_skills()
        return

    print("\n" + "▶" * 30 + " [1/3] Count Skills " + "◀" * 30)
    count_skills()

    print("\n" + "▶" * 30 + " [2/3] Referenced Examples " + "◀" * 30)
    check_referenced_examples(verbose=verbose)

    print("\n" + "▶" * 30 + " [3/3] Dataset Statistics " + "◀" * 30)
    dataset_statistics()


if __name__ == "__main__":
    main()
