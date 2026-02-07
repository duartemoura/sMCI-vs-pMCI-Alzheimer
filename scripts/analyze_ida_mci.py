#!/usr/bin/env python3
import pandas as pd

print("=== Analyzing idaSearch_1_16_2026(1).csv for MCI Study ===\n")

# Load idaSearch file
ida_file = "./csvs/idaSearch_1_16_2026(1).csv"
print(f"Loading {ida_file}...")
ida_search = pd.read_csv(ida_file)

# Get unique Subject IDs
subject_col = 'Subject ID' if 'Subject ID' in ida_search.columns else ida_search.columns[0]
ida_subject_ids = set(ida_search[subject_col].unique())
print(f"Total unique subjects in idaSearch: {len(ida_subject_ids)}")

# Check for subjects with multiple scans in idaSearch
subject_counts = ida_search[subject_col].value_counts()
subjects_with_multiple = subject_counts[subject_counts > 1]
if len(subjects_with_multiple) > 0:
    print(f"Subjects with multiple scans in idaSearch: {len(subjects_with_multiple)}")
    print("  (Note: Only earliest scan per subject will be used in analysis)")

# Load ADNIMERGE
adni_file = "./csvs/ADNIMERGE_18Nov2025.csv"
print(f"Loading {adni_file}...")
adnimerge = pd.read_csv(adni_file, low_memory=False)
adnimerge_ptids = set(adnimerge['PTID'].unique())

# Find subjects in both
found_subjects = ida_subject_ids & adnimerge_ptids
print(f"Subjects found in ADNIMERGE: {len(found_subjects)} ({len(found_subjects)/len(ida_subject_ids)*100:.1f}%)")

# Filter for MCI baseline subjects
mci_types = ['EMCI', 'LMCI', 'MCI']
mci_baseline = adnimerge[adnimerge['DX_bl'].isin(mci_types)]
mci_ptids = set(mci_baseline['PTID'].unique())

# Find idaSearch subjects with MCI baseline
ida_mci_subjects = found_subjects & mci_ptids
print(f"\nSubjects from idaSearch with MCI baseline: {len(ida_mci_subjects)}")

# For each subject, identify the earliest scan based on IMAGEUID in ADNIMERGE
# (This matches the logic in the notebooks where earliest image ID is selected)
print("\nIdentifying earliest scan per subject from ADNIMERGE...")
earliest_scans = {}
for ptid in ida_mci_subjects:
    subject_scans = adnimerge[adnimerge['PTID'] == ptid].copy()
    # Filter to only scans that exist (have IMAGEUID)
    subject_scans = subject_scans[subject_scans['IMAGEUID'].notna()]
    
    if len(subject_scans) > 0:
        # Sort by IMAGEUID (lower = earlier) and EXAMDATE as tiebreaker
        subject_scans = subject_scans.sort_values(['IMAGEUID', 'EXAMDATE'])
        earliest_imageuid = subject_scans.iloc[0]['IMAGEUID']
        earliest_scans[ptid] = earliest_imageuid
        
        # Check if subject has multiple scans
        if len(subject_scans) > 1:
            print(f"  {ptid}: {len(subject_scans)} scans found, using earliest (IMAGEUID: {earliest_imageuid})")

print(f"Earliest scans identified for {len(earliest_scans)} subjects")

# Now check conversion status (similar to the function)
usable_pmci = []
usable_smci = []
excluded = {
    "no_followup": [],
    "reverted_cn": [],
    "short_stable": [],
    "inconsistent": []
}

print("\nAnalyzing conversion status...")
print("(Note: Analysis assumes earliest scan per subject is used, matching notebook logic)")
for ptid in ida_mci_subjects:
    subject_data = mci_baseline[mci_baseline['PTID'] == ptid].sort_values('EXAMDATE')
    follow_ups = subject_data[subject_data['VISCODE'] != 'bl']
    follow_up_dx = follow_ups['DX'].dropna().values
    
    if len(follow_up_dx) == 0:
        excluded["no_followup"].append(ptid)
        continue
    
    # Check for Dementia conversion
    if 'Dementia' in follow_up_dx:
        usable_pmci.append(ptid)
        continue
    
    # Check for CN reversion
    if 'CN' in follow_up_dx:
        excluded["reverted_cn"].append(ptid)
        continue
    
    # Check for stable MCI
    valid_mci_dx = {'EMCI', 'LMCI', 'MCI'}
    is_consistent_mci = all(dx in valid_mci_dx for dx in follow_up_dx)
    
    if is_consistent_mci:
        dates = pd.to_datetime(subject_data['EXAMDATE'])
        duration_days = (dates.max() - dates.min()).days
        if duration_days >= 540:  # 18 months
            usable_smci.append(ptid)
        else:
            excluded["short_stable"].append(ptid)
    else:
        excluded["inconsistent"].append(ptid)

print("\n=== MCI Conversion Study Results ===")
print(f"Total usable subjects: {len(usable_pmci) + len(usable_smci)}")
print(f"  - pMCI (progressive): {len(usable_pmci)}")
print(f"  - sMCI (stable ≥18mo): {len(usable_smci)}")
print("\nExcluded subjects:")
print(f"  - No follow-up data: {len(excluded['no_followup'])}")
print(f"  - Reverted to CN: {len(excluded['reverted_cn'])}")
print(f"  - Stable but <18mo: {len(excluded['short_stable'])}")
print(f"  - Inconsistent diagnoses: {len(excluded['inconsistent'])}")

if len(usable_pmci) + len(usable_smci) > 0:
    conversion_rate = len(usable_pmci) / (len(usable_pmci) + len(usable_smci)) * 100
    print(f"\nConversion rate: {conversion_rate:.1f}%")

print("\n=== Summary ===")
print(f"From {len(ida_subject_ids)} total subjects in idaSearch:")
print(f"  → {len(found_subjects)} found in ADNIMERGE ({len(found_subjects)/len(ida_subject_ids)*100:.1f}%)")
print(f"  → {len(ida_mci_subjects)} have MCI baseline ({len(ida_mci_subjects)/len(ida_subject_ids)*100:.1f}%)")
print(f"  → {len(usable_pmci) + len(usable_smci)} usable for MCI conversion study ({len(usable_pmci) + len(usable_smci)/len(ida_subject_ids)*100:.1f}%)")

