# Utiliser une image de base Python
FROM python:3.10

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer un port (si ton projet utilise un serveur web)
EXPOSE 8000

# Commande pour exécuter le projet (adapter si besoin)
CMD ["python", "main.py"]
