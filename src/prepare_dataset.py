import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Konfig
RAW_DATA_PATH = "C:/URPATH"
PROCESSED_DATA_PATH = "C:/URPATH"

TRAIN_CSV_PATH = os.path.join(RAW_DATA_PATH, "Train.csv")
TEST_CSV_PATH = os.path.join(RAW_DATA_PATH, "Test.csv")

# Anzahl der Klassen im GTSRB-Dataset
NUM_CLASSES = 43


def create_directory_structure():
    """erstellt die Zielordner für train, val und test"""
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(PROCESSED_DATA_PATH, split)
        os.makedirs(split_path, exist_ok=True)
        for class_id in range(NUM_CLASSES):
            class_dir = os.path.join(split_path, str(class_id))
            os.makedirs(class_dir, exist_ok=True)


def safe_copy(source_file, dest_file):
    """Kopiert nur, wenn die Datei noch nicht existiert"""
    if not os.path.exists(dest_file):
        shutil.copy(source_file, dest_file)


def process_training_data():
    """Liest die Trainingsdaten + teilt sie auf und verschiebt sie"""
    print("Verarbeite Trainingsdaten...")
    df_train = pd.read_csv(TRAIN_CSV_PATH)

    # split in Training 80% und Validierung 20%
    train_data, val_data = train_test_split(
        df_train,
        test_size=0.2,
        random_state=42,  # für reproduzierbare Ergebnisse
        stratify=df_train['ClassId']  # beibehält das Klassenverhältnis
    )

    # Trainingsbilder kopieren
    for _, row in train_data.iterrows():
        source_file = os.path.join(RAW_DATA_PATH, row['Path'])
        filename = os.path.basename(row['Path'])
        dest_file = os.path.join(PROCESSED_DATA_PATH, 'train', str(row['ClassId']), filename)
        safe_copy(source_file, dest_file)

    # Validierungsbilder kopieren
    for _, row in val_data.iterrows():
        source_file = os.path.join(RAW_DATA_PATH, row['Path'])
        filename = os.path.basename(row['Path'])
        dest_file = os.path.join(PROCESSED_DATA_PATH, 'val', str(row['ClassId']), filename)
        safe_copy(source_file, dest_file)

    print(f"Trainingsdaten verarbeitet: {len(train_data)} Bilder in 'train', {len(val_data)} in 'val'")


def process_test_data():
    """Liest die Testdaten und verschiebt sie"""
    print("Verarbeite Testdaten...")
    df_test = pd.read_csv(TEST_CSV_PATH)

    for _, row in df_test.iterrows():
        source_file = os.path.join(RAW_DATA_PATH, row['Path'])
        filename = os.path.basename(row['Path'])
        dest_file = os.path.join(PROCESSED_DATA_PATH, 'test', str(row['ClassId']), filename)
        safe_copy(source_file, dest_file)

    print(f"Testdaten verarbeitet: {len(df_test)} Bilder in 'test'")


def main():
    create_directory_structure()
    process_training_data()
    process_test_data()
    print("\nDatenverarbeitung abgeschlossen")


if __name__ == "__main__":
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Fehler: Pfad '{RAW_DATA_PATH}' existiert nicht")
    else:
        main()
