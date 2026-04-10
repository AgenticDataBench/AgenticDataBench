#!/usr/bin/env python3
"""
Test script for AutoProfiler file parsing capabilities

Usage:
    python test_profiler.py
    python test_profiler.py --domain agriculture --detailed true
    python test_profiler.py <file or directory path>

Examples:
    python test_profiler.py data.csv
    python test_profiler.py ./test_datasets/
    python test_profiler.py file1.csv file2.json file3.xlsx

Options:
    --detailed       true/false, whether to use detailed files_summary
    --domain         Profile a specific domain under datasets/
"""

import sys
import os
import argparse
import json
from typing import List

# Add module path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auto_profiler import AutoProfiler
from dataset_loader import build_files_summary, _profile_to_file_info


def print_separator(title: str = "", char: str = "=", width: int = 70):
    """Print a separator line"""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(char * width)


def parse_bool_arg(value: str) -> bool:
    """Parse a CLI bool argument (true/false)."""
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false")


def get_available_domains(datasets_dir: str) -> List[str]:
    """Return all available domain names under datasets directory."""
    if not os.path.isdir(datasets_dir):
        return []
    return sorted(
        d for d in os.listdir(datasets_dir)
        if os.path.isdir(os.path.join(datasets_dir, d)) and not d.startswith(".")
    )


def build_domain_report(
    profiler: AutoProfiler,
    domain_name: str,
    domain_dir: str,
    detailed: bool = False
) -> str:
    """Build a text report for one domain directory."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"Domain: {domain_name}")
    lines.append("=" * 70)

    profiles = profiler.profile_directory(domain_dir)
    total_count = len(profiles)
    success_profiles = [p for p in profiles if not p.error]
    error_profiles = [p for p in profiles if p.error]

    lines.append(f"Total supported files profiled: {total_count}")
    lines.append(f"Success: {len(success_profiles)}")
    lines.append(f"Errors: {len(error_profiles)}")

    type_counts = {}
    for p in profiles:
        type_counts[p.file_type] = type_counts.get(p.file_type, 0) + 1
    lines.append(f"Type distribution: {type_counts}")

    lines.append("\n[PROFILED FILES]")
    if not profiles:
        lines.append("No supported files found.")
    else:
        for p in profiles:
            status = "ERROR" if p.error else "OK"
            row_info = f", rows={p.row_count}" if p.row_count is not None else ""
            lines.append(f"- [{status}] {p.path} (type={p.file_type}{row_info})")
            if p.error:
                lines.append(f"  error: {p.error}")
            elif detailed:
                if p.content_summary:
                    lines.append(f"  summary: {p.content_summary}")
                if p.read_params:
                    lines.append(f"  read_params: {p.read_params}")

    files_info = []
    for p in success_profiles:
        files_info.extend(_profile_to_file_info(p))

    lines.append("\n" + "-" * 70)
    lines.append("build_files_summary() output")
    lines.append("-" * 70)
    if files_info:
        lines.append(build_files_summary(files_info, detailed=detailed))
    else:
        lines.append("No successful profiles to summarize.")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def write_reports_for_all_domains(
    profiler: AutoProfiler,
    datasets_dir: str,
    output_dir: str,
    detailed: bool = False
) -> List[str]:
    """Generate one profiler output file for each domain under datasets/."""
    domains = get_available_domains(datasets_dir)
    os.makedirs(output_dir, exist_ok=True)

    generated_files = []
    for domain in domains:
        domain_dir = os.path.join(datasets_dir, domain)
        report = build_domain_report(profiler, domain, domain_dir, detailed=detailed)
        output_path = os.path.join(output_dir, f"profiler_output ({domain}).txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        generated_files.append(output_path)

    return generated_files


def print_file_profile(profile, show_raw: bool = True):
    """
    Print profile details for a single file
    """
    print(f"\n[FILE] {profile.path}")
    print(f"   Type: {profile.file_type}")
    
    if profile.error:
        print(f"   [ERROR] {profile.error}")
        return
    
    print(f"   Row count: {profile.row_count}")
    print(f"   Summary: {profile.content_summary}")
    
    # Display different info based on file type
    s = profile.structure
    
    if profile.file_type in ("csv", "excel"):
        print(f"   Columns: {s.get('columns', [])}")
        print(f"   Numeric columns: {s.get('numeric_columns', [])}")
        print(f"   Categorical columns: {s.get('categorical_columns', [])}")
        if s.get('missing_values'):
            missing = {k: v for k, v in s['missing_values'].items() if v > 0}
            if missing:
                print(f"   Missing values: {missing}")
        
        # Excel specific
        if profile.file_type == "excel":
            sheets = s.get('sheets', [])
            print(f"   Sheet count: {len(sheets)}")
            for sheet in sheets:
                semi = "[SEMI-STRUCTURED]" if sheet.get('is_semi_structured') else ""
                header = sheet.get('detected_header_row', 0)
                print(f"      - {sheet['name']}: {sheet.get('num_rows', '?')} rows "
                      f"(header={header}) {semi}")
                if sheet.get('metadata_summary'):
                    print(f"        Metadata: {sheet['metadata_summary']}")
    
    elif profile.file_type in ("json", "yaml"):
        print(f"   Top-level type: {s.get('top_level_type', 'unknown')}")
        print(f"   Keys: {s.get('keys', s.get('top_level_keys', []))[:10]}")  # Show at most 10
        if s.get('schema'):
            print(f"   Schema: (generated)")
    
    elif profile.file_type == "sqlite":
        tables = s.get('tables', [])
        print(f"   Table count: {len(tables)}")
        for t in tables:
            cols = [c['name'] for c in t.get('columns', [])]
            print(f"      - {t['name']}: {t.get('row_count', '?')} rows, columns={cols}")
        fks = s.get('foreign_keys', [])
        if fks:
            print(f"   Foreign keys: {fks}")
    
    elif profile.file_type in ("html", "markdown", "tex"):
        print(f"   Headings: {s.get('headings', [])[:5]}")
        print(f"   Table count: {s.get('tables_count', 0)}")
        if s.get('elements'):
            print(f"   Element stats: {s['elements']}")
    
    elif profile.file_type == "text":
        print(f"   Encoding: {s.get('encoding', 'unknown')}")
        fmt = s.get('detected_format', 'not detected')
        print(f"   Detected format: {fmt}")
        if s.get('structure_description'):
            print(f"   Structure description: {s['structure_description']}")
        if s.get('processing_suggestion'):
            print(f"   Processing suggestion: {s['processing_suggestion']}")
    
    elif profile.file_type == "log":
        print(f"   Timestamp format: {s.get('timestamp_format', 'not detected')}")
        print(f"   Log levels: {s.get('log_levels', {})}")
    
    elif profile.file_type == "numpy":
        print(f"   Shape: {s.get('shape')}")
        print(f"   Dtype: {s.get('dtype')}")
        if s.get('stats'):
            print(f"   Stats: {s['stats']}")
    
    elif profile.file_type == "netcdf":
        print(f"   Dimensions: {s.get('dimensions', {})}")
        print(f"   Variable count: {len(s.get('variables', []))}")
    
    elif profile.file_type == "geo":
        layers = s.get('layers', [])
        print(f"   Layer count: {len(layers)}")
        for layer in layers:
            print(f"      - {layer.get('name')}: {layer.get('feature_count')} features, "
                  f"type={layer.get('geometry_type')}")
    
    # Show read_params
    if profile.read_params:
        print(f"   Read params: {profile.read_params}")
    
    # Show sample (truncated)
    if profile.sample:
        sample_preview = profile.sample[:500]
        if len(profile.sample) > 500:
            sample_preview += "\n... (truncated)"
        print(f"\n   [SAMPLE]")
        for line in sample_preview.split('\n')[:15]:
            print(f"      {line}")
    
    # Optional: show raw structure
    if show_raw:
        print(f"\n   [RAW STRUCTURE] (JSON):")
        try:
            # Filter out overly long fields
            s_filtered = {k: v for k, v in s.items() 
                         if not isinstance(v, str) or len(v) < 500}
            print(f"      {json.dumps(s_filtered, ensure_ascii=False, indent=6)[:1500]}")
        except:
            print(f"      (cannot serialize)")


def test_single_file(profiler: AutoProfiler, file_path: str, detailed: bool = False):
    """Test a single file"""
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return None
    
    print_separator(f"Profiling: {os.path.basename(file_path)}")
    
    profile = profiler.profile_file(file_path)
    print_file_profile(profile, show_raw=True)
    
    # Convert to files_info format and show build_files_summary output
    print_separator("build_files_summary() output", char="-")
    file_infos = _profile_to_file_info(profile)
    summary = build_files_summary(file_infos, detailed=detailed)
    print(summary)
    
    return profile


def test_directory(profiler: AutoProfiler, dir_path: str, detailed: bool = False):
    """Test an entire directory"""
    if not os.path.isdir(dir_path):
        print(f"[ERROR] Directory not found: {dir_path}")
        return []
    
    print_separator(f"Profiling directory: {dir_path}")
    
    profiles = profiler.profile_directory(dir_path)
    
    print(f"\nFound {len(profiles)} supported file(s):")
    
    # Group by type and count
    type_counts = {}
    for p in profiles:
        t = p.file_type
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"   Type distribution: {type_counts}")
    
    # Show profile for each file
    for profile in profiles:
        print_file_profile(profile, show_raw=False)
    
    # Show full build_files_summary output
    print_separator("build_files_summary() full output", char="-")
    files_info = []
    for p in profiles:
        if not p.error:
            files_info.extend(_profile_to_file_info(p))
    summary = build_files_summary(files_info, detailed=detailed)
    print(summary)
    
    return profiles


def try_import_real_llm():
    """
    Try to import a real LLM client.
    Modify this based on your project configuration.
    """
    try:
        sys.path.append('..')
        from utils.llm_client import QwenClient
        # Add configuration here if needed
        return QwenClient()
    except ImportError:
        pass
    except Exception:
        pass
    
    try:
        sys.path.append('..')
        from utils.llm_client import LLMClient
        return LLMClient()
    except ImportError:
        pass
    except Exception:
        pass
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Test AutoProfiler file profiling and domain-level reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_profiler.py
    python test_profiler.py --domain agriculture
    python test_profiler.py --domain ecommerce --detailed true
    python test_profiler.py data.csv
    python test_profiler.py file1.csv file2.json --detailed false
        """
    )
    parser.add_argument("paths", nargs="*", help="File or directory paths to test")
    parser.add_argument("--detailed", type=parse_bool_arg, default=True, metavar="{true,false}",
                       help="Whether to use detailed mode for build_files_summary (true/false, default: true)")
    parser.add_argument("--domain", type=str, default=None,
                       help="Domain name under datasets/ to profile")
    parser.add_argument("--output-dir", type=str, default="profiler_output",
                       help="Directory to save default per-domain output files")
    
    args = parser.parse_args()
    
    # Match AutoProfiler runtime behavior:
    # provide real llm_client if available, and let AutoProfiler itself decide
    # whether a specific file actually needs LLM-enhanced logic.
    llm_client = try_import_real_llm()
    if llm_client:
        print("[OK] Loaded real LLM client")
    else:
        print("[INFO] No real LLM client found; AutoProfiler will use rule-based path")
    
    # Create profiler
    profiler = AutoProfiler(llm_client=llm_client)
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    
    print_separator("AutoProfiler Test")
    print(f"LLM enhancement: {'enabled' if llm_client else 'disabled'}")
    print(f"Detailed mode: {'yes' if args.detailed else 'no'}")

    # Mode 1: explicit domain profiling
    if args.domain:
        domain_dir = os.path.join(datasets_dir, args.domain)
        if not os.path.isdir(domain_dir):
            available = get_available_domains(datasets_dir)
            print(f"[ERROR] Domain not found: {args.domain}")
            print(f"Available domains ({len(available)}): {', '.join(available)}")
            sys.exit(1)

        report = build_domain_report(
            profiler=profiler,
            domain_name=args.domain,
            domain_dir=domain_dir,
            detailed=args.detailed
        )
        print(report)
        return

    # Mode 2: backward compatibility for direct path profiling
    if args.paths:
        all_profiles = []
        for path in args.paths:
            path = os.path.abspath(path)

            if os.path.isdir(path):
                profiles = test_directory(profiler, path, detailed=args.detailed)
                all_profiles.extend(profiles)
            elif os.path.isfile(path):
                profile = test_single_file(profiler, path, detailed=args.detailed)
                if profile:
                    all_profiles.append(profile)
            else:
                print(f"[ERROR] Path not found: {path}")

        print_separator("Test complete")
        success = len([p for p in all_profiles if not p.error])
        errors = len([p for p in all_profiles if p.error])
        print(f"[OK] Success: {success} file(s)")
        if errors:
            print(f"[FAIL] Errors: {errors} file(s)")
        return

    # Mode 3: default run with no args -> generate one file per domain
    domains = get_available_domains(datasets_dir)
    generated_files = write_reports_for_all_domains(
        profiler=profiler,
        datasets_dir=datasets_dir,
        output_dir=os.path.abspath(args.output_dir),
        detailed=args.detailed
    )
    print_separator("Domain reports generated")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print(f"Domain count: {len(domains)}")
    print(f"Generated files: {len(generated_files)}")
    for f in generated_files:
        print(f" - {f}")


if __name__ == "__main__":
    main()
