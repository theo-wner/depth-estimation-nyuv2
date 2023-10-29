import os
import shutil

################################################################################
# Skript, das das Löschen von Ordnern möglich macht, die mit python os 
# erstellt wurden (Sonst keine Berechtigung)
################################################################################

# Datei löschen
#os.remove('./data/seg13/class_labels.txt')

# Leeres Verzeichnis löschen
#os.rmdir('./Abbildungen/')

# Verzeichnis und seine Inhalte löschen
shutil.rmtree('logs/')

# Neues Verzeichnis erstellen
#os.mkdir('./extract_dataset/class_labels/')

# Neue Datei erstellen
#open('./data/seg13/class_labels.txt', 'w')

# Datei kopieren
#shutil.copy('./class_labels.txt', './data/seg40')