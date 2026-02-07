#!/usr/bin/env python3
"""
Simple script to count data samples (images) and subjects in pickle files 
within train, test, and validation tensor directories.

Note: Subject IDs are only available in k-fold datasets (kfold_*/), not in 
the basic AD/CN tensor datasets (tensors_fdg/, tensors_tau/).
"""

import os
import pickle
from pathlib import Path
import numpy as np

def analyze_pickle_data(pickle_path):
    """Analyze a pickle file to count samples, subjects, and label distribution."""
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        samples = 0
        labels = None
        label_counts = {}
        subjects = set()
        
        # Handle different data structures
        if isinstance(data, dict):
            # Look for common keys that contain the actual data
            if 'data' in data:
                samples = len(data['data'])
            elif 'images' in data:
                # Handle torch tensors
                images_data = data['images']
                if hasattr(images_data, 'shape'):
                    samples = images_data.shape[0]
                elif hasattr(images_data, '__len__'):
                    samples = len(images_data)
            elif 'X' in data:
                samples = len(data['X'])
            else:
                # If no common key, count all non-meta keys
                samples = sum(len(v) if hasattr(v, '__len__') and not isinstance(v, str) else 1 
                             for k, v in data.items() if not k.startswith('_'))
            
            # Look for labels
            if 'labels' in data:
                labels = data['labels']
            elif 'y' in data:
                labels = data['y']
            elif 'targets' in data:
                labels = data['targets']
            
            # Look for subject IDs
            if 'subject_ids' in data:
                subject_data = data['subject_ids']
                if isinstance(subject_data, np.ndarray):
                    subjects = set(subject_data.tolist())
                elif hasattr(subject_data, '__iter__'):
                    subjects = set(subject_data)
                else:
                    subjects = {subject_data}
            elif 'subjects' in data:
                subjects = set(data['subjects'])
            elif 'patient_ids' in data:
                subjects = set(data['patient_ids'])
            elif 'ids' in data:
                subjects = set(data['ids'])
        elif isinstance(data, (list, tuple)):
            samples = len(data)
        elif hasattr(data, 'shape'):
            # If it's a numpy array or tensor
            samples = data.shape[0]
        else:
            samples = 1
        
        # Count labels if available
        if labels is not None:
            # Handle PyTorch tensors
            if hasattr(labels, 'numpy'):
                try:
                    labels_np = labels.numpy()
                except:
                    # If numpy() fails, try to get shape at least
                    if hasattr(labels, 'shape'):
                        label_counts = {"Unknown": labels.shape[0] if len(labels.shape) > 0 else 1}
                    else:
                        label_counts = {"Unknown": len(labels) if hasattr(labels, '__len__') else 1}
                    labels_np = None
                
                if labels_np is not None:
                    if isinstance(labels_np, np.ndarray) and labels_np.size > 0:
                        unique_labels, counts = np.unique(labels_np, return_counts=True)
                        label_counts = dict(zip(unique_labels, counts))
            elif hasattr(labels, 'detach'):
                try:
                    labels_np = labels.detach().cpu().numpy()
                    unique_labels, counts = np.unique(labels_np, return_counts=True)
                    label_counts = dict(zip(unique_labels, counts))
                except:
                    label_counts = {"Unknown": len(labels) if hasattr(labels, '__len__') else 1}
            elif isinstance(labels, (list, tuple, np.ndarray)):
                unique_labels, counts = np.unique(labels, return_counts=True)
                label_counts = dict(zip(unique_labels, counts))
            else:
                label_counts = {str(labels): 1}
        
        return samples, len(subjects), label_counts
    except Exception as e:
        print(f"    Error reading {pickle_path}: {e}")
        return 0, 0, {}

def analyze_directory(directory_path):
    """Analyze all pickle files in a directory."""
    if not os.path.exists(directory_path):
        return 0, 0, {}, []
    
    total_samples = 0
    total_subjects = 0
    total_label_counts = {}
    file_details = []
    
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isfile(item_path) and item.endswith('.pkl'):
            samples, subjects, label_counts = analyze_pickle_data(item_path)
            total_samples += samples
            total_subjects += subjects
            file_details.append((item, samples, subjects, label_counts))
            
            # Aggregate label counts
            for label, count in label_counts.items():
                total_label_counts[label] = total_label_counts.get(label, 0) + count
    
    return total_samples, total_subjects, total_label_counts, file_details

def format_label_counts(label_counts):
    """Format label counts for display."""
    if not label_counts:
        return "No label info"
    
    # Convert numeric labels to meaningful names
    formatted_labels = {}
    for label, count in label_counts.items():
        if label == 0:
            formatted_labels['CN/sMCI'] = count
        elif label == 1:
            formatted_labels['AD/pMCI'] = count
        elif str(label).lower() == 'cn' or str(label).lower() == 'smci':
            formatted_labels['CN/sMCI'] = formatted_labels.get('CN/sMCI', 0) + count
        elif str(label).lower() == 'ad' or str(label).lower() == 'pmci':
            formatted_labels['AD/pMCI'] = formatted_labels.get('AD/pMCI', 0) + count
        else:
            formatted_labels[str(label)] = count
    
    return ", ".join([f"{label}: {count:,}" for label, count in sorted(formatted_labels.items())])

def main():
    # Base data directory
    data_dir = Path("data/processed")
    
    # All directories to check (including kfold datasets which have subject info)
    tensor_dirs = [
        "tensors_fdg",
        "tensors_fdg_3", 
        "tensors_tau",
        "kfold",
        "kfold_fdg_2",
        "kfold_fdg_3",
        "kfold_tau",
        "kfold_tau_2",
        "kfold_tau_3"
    ]
    splits = ["train", "test", "val"]
    
    print("Tensor Data Sample & Subject Count Summary")
    print("=" * 80)
    print("\nNOTE: Subject counts are only available in k-fold datasets (kfold_*).")
    print("Basic AD/CN tensor datasets (tensors_*) only contain 'images' and 'labels'.")
    print("=" * 80)
    
    total_train_samples = 0
    total_test_samples = 0
    total_val_samples = 0
    total_train_subjects = 0
    total_test_subjects = 0
    total_val_subjects = 0
    total_label_counts = {}
    
    for tensor_dir in tensor_dirs:
        tensor_path = data_dir / tensor_dir
        
        if not tensor_path.exists():
            continue
            
        print(f"\n{tensor_dir.upper()}:")
        print("-" * 40)
        
        tensor_total_samples = 0
        tensor_total_subjects = 0
        tensor_label_counts = {}
        
        # For kfold datasets, check for fold files directly in the directory
        if tensor_dir.startswith('kfold'):
            # K-fold datasets have files like train_fold_1.pkl, val_fold_1.pkl
            fold_files = sorted([f for f in tensor_path.glob('*.pkl') if 'fold_' in f.name])
            
            if fold_files:
                for fold_file in fold_files:
                    samples, subjects, label_counts = analyze_pickle_data(fold_file)
                    
                    # Determine split type from filename
                    if 'train' in fold_file.name:
                        split_type = 'train'
                        total_train_samples += samples
                        total_train_subjects += subjects
                    elif 'val' in fold_file.name:
                        split_type = 'val'
                        total_val_samples += samples
                        total_val_subjects += subjects
                    else:
                        split_type = 'other'
                    
                    print(f"  {fold_file.name}:")
                    print(f"    Samples:  {samples:,}")
                    if subjects > 0:
                        print(f"    Subjects: {subjects:,}")
                    print(f"    Labels:   {format_label_counts(label_counts)}")
                    
                    tensor_total_samples += samples
                    tensor_total_subjects += subjects
                    
                    # Aggregate label counts
                    for label, count in label_counts.items():
                        tensor_label_counts[label] = tensor_label_counts.get(label, 0) + count
                        total_label_counts[label] = total_label_counts.get(label, 0) + count
                
                # Tensor totals
                print(f"\n  {tensor_dir.upper()} TOTALS:")
                print(f"    Total Samples:  {tensor_total_samples:,}")
                if tensor_total_subjects > 0:
                    print(f"    Total Subjects: {tensor_total_subjects:,}")
                if tensor_label_counts:
                    print(f"    Total Labels:   {format_label_counts(tensor_label_counts)}")
                continue
        
        # For non-kfold datasets, check train/test/val subdirectories
        for split in splits:
            split_path = tensor_path / split
            total_samples, total_subjects, split_label_counts, file_details = analyze_directory(split_path)
            
            if total_samples == 0 and len(file_details) == 0:
                continue
                
            print(f"  {split.upper()}:")
            for filename, samples, subjects, label_counts in file_details:
                print(f"    {filename}:")
                print(f"      Samples:  {samples:,}")
                if subjects > 0:
                    print(f"      Subjects: {subjects:,}")
                print(f"      Labels:   {format_label_counts(label_counts)}")
            
            print(f"    Split Total: {total_samples:,} samples")
            if total_subjects > 0:
                print(f"                 {total_subjects:,} subjects")
            if split_label_counts:
                print(f"    Split Labels: {format_label_counts(split_label_counts)}")
            
            # Add to totals
            tensor_total_samples += samples
            tensor_total_subjects += total_subjects
            if split == "train":
                total_train_samples += total_samples
                total_train_subjects += total_subjects
            elif split == "test":
                total_test_samples += total_samples
                total_test_subjects += total_subjects
            elif split == "val":
                total_val_samples += total_samples
                total_val_subjects += total_subjects
            
            # Aggregate label counts
            for label, count in split_label_counts.items():
                tensor_label_counts[label] = tensor_label_counts.get(label, 0) + count
                total_label_counts[label] = total_label_counts.get(label, 0) + count
        
        # Tensor totals for non-kfold datasets
        if not tensor_dir.startswith('kfold'):
            print(f"\n  {tensor_dir.upper()} TOTALS:")
            print(f"    Total Samples:  {tensor_total_samples:,}")
            if tensor_total_subjects > 0:
                print(f"    Total Subjects: {tensor_total_subjects:,}")
            if tensor_label_counts:
                print(f"    Total Labels:   {format_label_counts(tensor_label_counts)}")
    
    print("\n" + "=" * 80)
    print("OVERALL TOTALS:")
    print("-" * 40)
    print(f"Total Train Samples:   {total_train_samples:,}")
    print(f"Total Test Samples:    {total_test_samples:,}")
    print(f"Total Val Samples:     {total_val_samples:,}")
    print(f"Grand Total Samples:   {total_train_samples + total_test_samples + total_val_samples:,}")
    if (total_train_subjects + total_test_subjects + total_val_subjects) > 0:
        print(f"\nSubject Counts (from k-fold datasets):")
        print(f"  Train Subjects: {total_train_subjects:,}")
        print(f"  Test Subjects:  {total_test_subjects:,}")
        print(f"  Val Subjects:   {total_val_subjects:,}")
        print(f"  Grand Total:    {total_train_subjects + total_test_subjects + total_val_subjects:,}")
    if total_label_counts:
        print("\nOverall Labels: " + format_label_counts(total_label_counts))

if __name__ == "__main__":
    main()
