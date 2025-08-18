import os


def count_images_in_folder(folder_path):
    # Liste gängiger Bild-Dateiendungen
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

    files = os.listdir(folder_path)

    # Nur Dateien zählen die mit den Bild-Endungen enden
    image_count = sum(1 for file in files if file.lower().endswith(image_extensions))

    return image_count

def count_images_in_folder2(folder_path):
    # Liste gängiger Bild-Dateiendungen
    image_extensions = ('.txt')

    files = os.listdir(folder_path)

    # Nur Dateien zählen die mit den Bild-Endungen enden
    image_count = sum(1 for file in files if file.lower().endswith(image_extensions))

    return image_count

folder_path = "data/yolo_signs/train/images"
folder_path2 = "data/yolo_signs/train/labels"

num_images = count_images_in_folder(folder_path)
num_images2 = count_images_in_folder2(folder_path2)
print(f'Anzahl der Bilder im Ordner "{folder_path}": {num_images}')
print(f'Anzahl der Bilder im Ordner "{folder_path2}": {num_images2}')