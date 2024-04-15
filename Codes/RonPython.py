#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:39:19 2024

@author: Giorgio
"""

"""import subprocess


#subprocess.call(["/Library/Frameworks/R.framework/Versions/4.3-arm64/Resources/bin/Rscript" ,
#                 "/Users/giorgiomontesi/Desktop/Universita_di_Siena/A_PhD_Project/Biomarker_Prediction/ensembleBP/Codes/EnrichPlot.R"])

a = subprocess.call(["/Library/Frameworks/R.framework/Versions/4.3-arm64/Resources/bin/Rscript" ,
                 "/Users/giorgiomontesi/Desktop/Universita_di_Siena/A_PhD_Project/Biomarker_Prediction/ensembleBP/Codes/getConsensusList.R"])
"""

import subprocess
#import json

# Definisci il percorso del file Rscript
rscript_path = "/Library/Frameworks/R.framework/Versions/4.3-arm64/Resources/bin/Rscript"

# Definisci il percorso del file R con la funzione getConsensus.list
r_file_path = "/Users/giorgiomontesi/Desktop/Universita_di_Siena/A_PhD_Project/Biomarker_Prediction/ensembleBP/Codes/getConsensusList.R"

# Esegui il comando Rscript e cattura l'output
result = subprocess.run([rscript_path, r_file_path], stdout=subprocess.PIPE)

# Ottieni l'output come stringa decodificata e rimuovi eventuali spazi bianchi aggiuntivi
output = result.stdout.decode("utf-8").strip()

"""# Trova la posizione in cui inizia il dizionario JSON
start_index = output.find('[')

# Estrai solo la parte dell'output che rappresenta il dizionario JSON
json_output = output[start_index:]

# Stampa l'output JSON
print(json_output)

# Analizza l'output JSON in un dizionario Python
try:
    consensus_list = json.loads(json_output)
    print(consensus_list)
except json.JSONDecodeError as e:
    print("Errore durante la decodifica JSON:", e)"""
    

# Stampa l'output
print(output)

# Estrai i geni dall'output
geni = [word.strip('"') for word in output.split() if word.startswith('"')]

# Stampa i geni
print(geni)
