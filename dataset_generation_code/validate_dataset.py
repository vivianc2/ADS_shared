#!/usr/bin/env python3
"""
Validate generated Bayesian network dataset for quality and balance.

Usage:
    python validate_dataset.py <dataset_directory>

Example:
    python validate_dataset.py out_bn_2_10_2
"""

import json
import glob
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Any


def load_worlds(directory: str) -> List[Dict[str, Any]]:
    """Load all world JSON files from directory."""
    pattern = os.path.join(directory, "world_*.json")
    files = sorted(glob.glob(pattern))

    worlds = []
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                data['_filepath'] = filepath
                worlds.append(data)
        except Exception as e:
            print(f"⚠️  Warning: Failed to load {filepath}: {e}")

    return worlds


def validate_structure(worlds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate basic structure of world files."""
    issues = []
    stats = {
        'total_files': len(worlds),
        'missing_keys': defaultdict(list),
        'invalid_questions': [],
    }

    required_keys = ['meta', 'story', 'variables', 'edges', 'cpds', 'questions', 'non_intervenable_variables']

    for world in worlds:
        filepath = world.get('_filepath', 'unknown')
        basename = os.path.basename(filepath)

        # Check required keys
        for key in required_keys:
            if key not in world:
                stats['missing_keys'][key].append(basename)

        # Validate questions structure
        questions = world.get('questions', [])
        for q in questions:
            if not isinstance(q.get('answer'), (str, list)):
                stats['invalid_questions'].append({
                    'file': basename,
                    'question_id': q.get('id', '?'),
                    'issue': f"Invalid answer type: {type(q.get('answer'))}",
                })

    return stats


def classify_relational(question_text: str, answer: str):
    """
    Map (question phrasing, yes/no answer) to 'dependent' / 'independent'.

    Two framings appear in the dataset:
      - Independent-framed ("statistically independent", "independent of"):
            Yes = independent, No = dependent
      - Dependent-framed ("statistical dependence", "statistically dependent",
        "change the probability distribution"):
            Yes = dependent, No = independent

    Returns None if the phrasing doesn't match either framing or the answer
    isn't Yes/No.
    """
    if answer not in ('Yes', 'No'):
        return None

    q = question_text.lower()
    indep_framed = ('independent of' in q) or ('statistically independent' in q)
    dep_framed = (
        'statistical dependence' in q
        or 'statistically dependent' in q
        or 'change the probability distribution' in q
    )

    if indep_framed and not dep_framed:
        return 'independent' if answer == 'Yes' else 'dependent'
    if dep_framed and not indep_framed:
        return 'dependent' if answer == 'Yes' else 'independent'
    return None


def analyze_question_counts(worlds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Count face-value (Yes/No/list) and relational (dep/indep) answers per type."""
    face_value = defaultdict(lambda: Counter())
    relational = defaultdict(lambda: Counter())
    totals = Counter()

    for world in worlds:
        for q in world.get('questions', []):
            qt = q.get('question_type', 'unknown')
            ans = q.get('answer', None)
            qtext = q.get('question', '') or ''

            totals[qt] += 1

            if isinstance(ans, list):
                face_value[qt]['[list]'] += 1
            elif isinstance(ans, str):
                face_value[qt][ans] += 1
            else:
                face_value[qt][f'[{type(ans).__name__}]'] += 1

            rel = classify_relational(qtext, ans if isinstance(ans, str) else '')
            if rel is not None:
                relational[qt][rel] += 1

    report = {}
    for qt in sorted(totals.keys()):
        report[qt] = {
            'total': totals[qt],
            'face_value': dict(face_value[qt]),
            'relational': dict(relational[qt]),
        }
    return report


def analyze_by_n_nodes(worlds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze question counts by network size."""
    by_n = defaultdict(lambda: {'files': 0, 'questions': 0, 'questions_per_file': []})

    for world in worlds:
        n_nodes = world.get('meta', {}).get('n_nodes', '?')
        n_questions = len(world.get('questions', []))

        by_n[n_nodes]['files'] += 1
        by_n[n_nodes]['questions'] += n_questions
        by_n[n_nodes]['questions_per_file'].append(n_questions)

    # Compute averages
    summary = {}
    for n, data in sorted(by_n.items()):
        if data['questions_per_file']:
            avg = sum(data['questions_per_file']) / len(data['questions_per_file'])
            expected = max(1, n // 10) if isinstance(n, int) else '?'
        else:
            avg = 0
            expected = '?'

        summary[n] = {
            'files': data['files'],
            'total_questions': data['questions'],
            'avg_questions_per_file': avg,
            'expected_per_file': expected,
            'matches_formula': (abs(avg - expected) < 0.1) if isinstance(expected, (int, float)) else None,
        }

    return summary


def check_golden_answers(worlds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check for potential issues with golden answers."""
    issues = []

    for world in worlds:
        filepath = world.get('_filepath', 'unknown')
        basename = os.path.basename(filepath)

        for q in world.get('questions', []):
            qid = q.get('id', '?')
            qt = q.get('question_type', 'unknown')
            ans = q.get('answer')

            # Check for suspicious patterns
            if qt in ['chain_marginal', 'fork_marginal', 'v_structure_conditional']:
                # These types should now be balanced (not always "No")
                pass  # Will be caught by balance check

            # Check for missing/invalid answers
            if ans is None:
                issues.append({
                    'file': basename,
                    'question_id': qid,
                    'type': qt,
                    'issue': 'Missing answer',
                })
            elif isinstance(ans, str) and ans not in ['Yes', 'No']:
                # Non-list answers should be Yes/No
                if qt not in ['root_nodes', 'leaf_nodes', 'markov_blanket', 'ancestors', 'descendants']:
                    issues.append({
                        'file': basename,
                        'question_id': qid,
                        'type': qt,
                        'issue': f'Unexpected answer: {ans}',
                    })

    return issues


def print_report(
    structure_stats: Dict,
    counts_report: Dict,
    n_nodes_summary: Dict,
    answer_issues: List[Dict],
):
    """Print validation report with face-value and relational counts."""

    print("=" * 80)
    print("DATASET VALIDATION REPORT")
    print("=" * 80)
    print()

    # 1. File structure
    print("1. FILE STRUCTURE")
    print("-" * 80)
    print(f"Total files: {structure_stats['total_files']}")

    if structure_stats['missing_keys']:
        print("\n⚠️  Files with missing keys:")
        for key, files in structure_stats['missing_keys'].items():
            print(f"  {key}: {len(files)} files")
            for f in files[:3]:
                print(f"    - {f}")
            if len(files) > 3:
                print(f"    ... and {len(files) - 3} more")
    else:
        print("✓ All files have required keys")

    if structure_stats['invalid_questions']:
        print(f"\n⚠️  Invalid questions found: {len(structure_stats['invalid_questions'])}")
        for issue in structure_stats['invalid_questions'][:5]:
            print(f"  {issue}")
    else:
        print("✓ All questions have valid structure")

    print()

    # 2. Question counts by type (face-value + relational)
    print("2. QUESTION COUNTS BY TYPE")
    print("-" * 80)

    total_questions = sum(info['total'] for info in counts_report.values())
    print(f"Total questions across all files: {total_questions}\n")

    # Aggregate totals across all types
    fv_total = Counter()
    rel_total = Counter()

    for qt, info in counts_report.items():
        for k, v in info['face_value'].items():
            fv_total[k] += v
        for k, v in info['relational'].items():
            rel_total[k] += v

    for qt in sorted(counts_report.keys()):
        info = counts_report[qt]
        fv_str = ', '.join(f"{k}={v}" for k, v in sorted(info['face_value'].items()))
        print(f"  {qt:30s} total={info['total']:3d}  face-value: {fv_str}")
        if info['relational']:
            rel_str = ', '.join(f"{k}={v}" for k, v in sorted(info['relational'].items()))
            print(f"  {'':30s}            relational: {rel_str}")

    print()
    print("  " + "-" * 76)
    fv_total_str = ', '.join(f"{k}={v}" for k, v in sorted(fv_total.items()))
    print(f"  {'GRAND TOTAL':30s} total={total_questions:3d}  face-value: {fv_total_str}")
    if rel_total:
        rel_total_str = ', '.join(f"{k}={v}" for k, v in sorted(rel_total.items()))
        print(f"  {'':30s}            relational: {rel_total_str}")
    print()

    # 3. Questions by network size
    print("3. QUESTIONS BY NETWORK SIZE")
    print("-" * 80)

    expected_total = 0
    actual_total = 0

    for n, summary in sorted(n_nodes_summary.items()):
        status = "✓" if summary.get('matches_formula') else "⚠️ "
        n_str = str(n)
        print(f"  n={n_str:>2s}: {summary['files']:2d} files, "
              f"{summary['total_questions']:3d} total questions, "
              f"avg={summary['avg_questions_per_file']:.1f}/file "
              f"(expected: {summary['expected_per_file']}) {status}")

        actual_total += summary['total_questions']
        if isinstance(summary['expected_per_file'], (int, float)):
            expected_total += summary['files'] * summary['expected_per_file']

    print(f"\nTotal: {actual_total} questions (expected: {expected_total})")
    print()

    # 4. Golden answer issues
    print("4. GOLDEN ANSWER VALIDATION")
    print("-" * 80)

    if answer_issues:
        print(f"⚠️  Found {len(answer_issues)} issues:")
        for issue in answer_issues[:10]:
            print(f"  {issue}")
        if len(answer_issues) > 10:
            print(f"  ... and {len(answer_issues) - 10} more")
    else:
        print("✓ No obvious issues with golden answers")

    print()
    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_dataset.py <dataset_directory>")
        print("Example: python validate_dataset.py out_bn_2_10_2")
        sys.exit(1)

    directory = sys.argv[1]

    if not os.path.isdir(directory):
        print(f"Error: Directory not found: {directory}")
        sys.exit(1)

    print(f"Loading dataset from: {directory}")
    worlds = load_worlds(directory)

    if not worlds:
        print(f"Error: No world files found in {directory}")
        sys.exit(1)

    print(f"Loaded {len(worlds)} world files\n")

    # Run validations
    structure_stats = validate_structure(worlds)
    counts_report = analyze_question_counts(worlds)
    n_nodes_summary = analyze_by_n_nodes(worlds)
    answer_issues = check_golden_answers(worlds)

    # Print report
    print_report(structure_stats, counts_report, n_nodes_summary, answer_issues)


if __name__ == "__main__":
    main()
