"""
Dataset Splitting Script

Splits your dataset into train/validation/test sets with stratification
to ensure balanced class distribution across all splits.

Usage:
    python data_split.py --input_csv train.csv --train_ratio 0.7 --val_ratio 0.15
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                  label_column='label', random_state=42):
    """
    Split dataset into train/val/test with stratification.
    
    Args:
        df: DataFrame with image paths and labels
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        label_column: Name of the label column
        random_state: Random seed for reproducibility
    
    Returns:
        train_df, val_df, test_df
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df[label_column],
        random_state=random_state
    )
    
    # Second split: separate train and validation
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        stratify=train_val_df[label_column],
        random_state=random_state
    )
    
    return train_df, val_df, test_df


def print_split_stats(train_df, val_df, test_df, label_column='label'):
    """Print statistics about the dataset splits."""
    
    print("\n📊 Dataset Split Summary")
    print("=" * 60)
    
    total = len(train_df) + len(val_df) + len(test_df)
    
    print(f"\n{'Split':<15} {'Total':<10} {'Class 0':<12} {'Class 1':<12} {'Ratio'}")
    print("-" * 60)
    
    for name, df in [('Training', train_df), ('Validation', val_df), ('Test', test_df)]:
        total_count = len(df)
        class_0 = (df[label_column] == 0).sum()
        class_1 = (df[label_column] == 1).sum()
        ratio = f"{total_count/total*100:.1f}%"
        
        print(f"{name:<15} {total_count:<10} {class_0:<12} {class_1:<12} {ratio}")
    
    print("-" * 60)
    print(f"{'TOTAL':<15} {total:<10}")
    print("=" * 60)
    
    # Check balance
    train_balance = train_df[label_column].value_counts()
    print(f"\n✅ Class balance in training set:")
    print(f"   Class 0: {train_balance.get(0, 0)} ({train_balance.get(0, 0)/len(train_df)*100:.1f}%)")
    print(f"   Class 1: {train_balance.get(1, 0)} ({train_balance.get(1, 0)/len(train_df)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test')
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Input CSV file with image paths and labels')
    parser.add_argument('--output_dir', type=str, default='../processed/splits',
                        help='Directory to save split CSV files')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio (default: 0.15)')
    parser.add_argument('--label_column', type=str, default='label',
                        help='Name of label column (default: label)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Read input CSV
    print(f"📂 Reading data from: {args.input_csv}")
    try:
        df = pd.read_csv(args.input_csv)
    except Exception as e:
        print(f"❌ Error reading CSV file: {str(e)}")
        return
    
    # Validate label column
    if args.label_column not in df.columns:
        print(f"❌ Label column '{args.label_column}' not found in CSV")
        print(f"   Available columns: {', '.join(df.columns)}")
        return
    
    print(f"✅ Loaded {len(df)} samples")
    
    # Check class distribution
    class_dist = df[args.label_column].value_counts()
    print(f"\n🏷️  Original class distribution:")
    for label, count in class_dist.items():
        print(f"   Class {label}: {count} ({count/len(df)*100:.1f}%)")
    
    # Split dataset
    print(f"\n🔀 Splitting with ratios: Train={args.train_ratio}, Val={args.val_ratio}, Test={args.test_ratio}")
    
    try:
        train_df, val_df, test_df = split_dataset(
            df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            label_column=args.label_column,
            random_state=args.random_seed
        )
    except Exception as e:
        print(f"❌ Error during splitting: {str(e)}")
        return
    
    # Print statistics
    print_split_stats(train_df, val_df, test_df, args.label_column)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    train_path = output_dir / 'train.csv'
    val_path = output_dir / 'val.csv'
    test_path = output_dir / 'test.csv'
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n💾 Saved splits to:")
    print(f"   Training:   {train_path} ({len(train_df)} samples)")
    print(f"   Validation: {val_path} ({len(val_df)} samples)")
    print(f"   Test:       {test_path} ({len(test_df)} samples)")
    
    print(f"\n🎉 All done! Your data is ready for training!")


if __name__ == '__main__':
    main()
