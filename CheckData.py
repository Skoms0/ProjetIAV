import os

# Chemins vers les dossiers
label_dir = "ms-coco/labels/train"
image_dir = "ms-coco/images/train-resized"

# Récupérer les noms de fichiers sans extension
label_files = {os.path.splitext(f)[0] for f in os.listdir(label_dir)}
image_files = {os.path.splitext(f)[0] for f in os.listdir(image_dir)}

# Supprimer les fichiers qui ne sont pas dans l'autre dossier
for f in os.listdir(label_dir):
    if os.path.splitext(f)[0] not in image_files:
        os.remove(os.path.join(label_dir, f))
        print(f"Supprimé du label: {f}")

for f in os.listdir(image_dir):
    if os.path.splitext(f)[0] not in label_files:
        os.remove(os.path.join(image_dir, f))
        print(f"Supprimé de l'image: {f}")

print("Nettoyage terminé.")
