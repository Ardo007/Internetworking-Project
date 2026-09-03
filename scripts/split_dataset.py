import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_splits(manifest_path, output_dir='data/processed/', train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Splits the dataset into stratified train, val, and test CSV sets.
    """
    if not os.path.exists(manifest_path):
        print(f"Error: Path '{manifest_path}' not found.")
        return

    df = pd.read_csv(manifest_path)
    
    train_df, temp_df = train_test_split(
        df, 
        test_size=(val_ratio + test_ratio), 
        stratify=df['label'],  
        random_state=seed
    )
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        stratify=temp_df['label'], 
        random_state=seed
    )
    
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Dataset successfully split:")
    print(f"  Train set: {len(train_df)} samples")
    print(f"  Validation set: {len(val_df)} samples")
    print(f"  Test set: {len(test_df)} samples")

if __name__ == "__main__":
    prepare_splits('capture_manifest.csv')