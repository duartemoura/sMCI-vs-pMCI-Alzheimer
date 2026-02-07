#!/usr/bin/env python3
import pandas as pd

print("=== Analyzing idaSearch_11_19_2025(1).csv for AD and CN ===\n")

# Load idaSearch file
ida_file = "idaSearch_11_19_2025(1).csv"
print(f"Loading {ida_file}...")
ida_search = pd.read_csv(ida_file)

# Get unique Subject IDs
subject_col = 'Subject ID' if 'Subject ID' in ida_search.columns else ida_search.columns[0]
ida_subject_ids = set(ida_search[subject_col].unique())
print(f"Total unique subjects in idaSearch: {len(ida_subject_ids)}")

# Load ADNIMERGE
adni_file = "ADNIMERGE_18Nov2025.csv"
print(f"Loading {adni_file}...")
adnimerge = pd.read_csv(adni_file, low_memory=False)
adnimerge_ptids = set(adnimerge['PTID'].unique())

# Find subjects in both
found_subjects = ida_subject_ids & adnimerge_ptids
print(f"Subjects found in ADNIMERGE: {len(found_subjects)} ({len(found_subjects)/len(ida_subject_ids)*100:.1f}%)\n")

# Get baseline diagnoses for all subjects found in ADNIMERGE
baseline_data = adnimerge[adnimerge['VISCODE'] == 'bl']
baseline_subjects = baseline_data[baseline_data['PTID'].isin(found_subjects)]

# Count by baseline diagnosis
diagnosis_counts = baseline_subjects['DX_bl'].value_counts()
print("=== Baseline Diagnoses in ADNIMERGE ===")
for dx, count in diagnosis_counts.items():
    pct = count / len(baseline_subjects) * 100 if len(baseline_subjects) > 0 else 0
    print(f"{dx}: {count} ({pct:.1f}%)")

# Specifically get AD and CN
ad_subjects = set(baseline_subjects[baseline_subjects['DX_bl'] == 'AD']['PTID'].unique())
cn_subjects = set(baseline_subjects[baseline_subjects['DX_bl'] == 'CN']['PTID'].unique())

# Also check for "Dementia" as AD baseline
dementia_baseline = set(baseline_subjects[baseline_subjects['DX_bl'] == 'Dementia']['PTID'].unique())
ad_subjects = ad_subjects | dementia_baseline

print(f"\n=== Summary for MCI Study ===")
print(f"AD subjects: {len(ad_subjects)} ({len(ad_subjects)/len(ida_subject_ids)*100:.1f}% of total idaSearch)")
print(f"CN subjects: {len(cn_subjects)} ({len(cn_subjects)/len(ida_subject_ids)*100:.1f}% of total idaSearch)")

# Get MCI counts too
mci_types = ['EMCI', 'LMCI', 'MCI']
mci_subjects = set(baseline_subjects[baseline_subjects['DX_bl'].isin(mci_types)]['PTID'].unique())
print(f"MCI subjects: {len(mci_subjects)} ({len(mci_subjects)/len(ida_subject_ids)*100:.1f}% of total idaSearch)")

# Check if any subjects are missing baseline diagnosis
subjects_with_baseline = set(baseline_subjects['PTID'].unique())
subjects_without_baseline = found_subjects - subjects_with_baseline
if subjects_without_baseline:
    print(f"\nSubjects in ADNIMERGE but missing baseline: {len(subjects_without_baseline)}")

print(f"\n=== Total Breakdown ===")
print(f"Total in idaSearch: {len(ida_subject_ids)}")
print(f"  → Found in ADNIMERGE: {len(found_subjects)} ({len(found_subjects)/len(ida_subject_ids)*100:.1f}%)")
print(f"    • AD: {len(ad_subjects)}")
print(f"    • CN: {len(cn_subjects)}")
print(f"    • MCI: {len(mci_subjects)}")
print(f"  → NOT in ADNIMERGE: {len(ida_subject_ids) - len(found_subjects)} ({100 - len(found_subjects)/len(ida_subject_ids)*100:.1f}%)")

