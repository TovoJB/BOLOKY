import requests
from bs4 import BeautifulSoup
import re
import time
from urllib.parse import urljoin, urlparse

# --- CONFIGURATION ---
START_URL = "https://www.jw.org/mg/"
BASE_DOMAIN = "www.jw.org"
OUT_FILE = "./jw_corpus.txt"
DELAY = 0.5 # Pause de 0.5s entre chaque requête pour ne pas surcharger leur serveur

# Ensembles pour suivre notre progression
urls_visited = set()
urls_to_visit = [START_URL]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def clean_text(text):
    """Nettoie le texte (enlève les espaces multiples, retours à la ligne, etc.)"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def scrape():
    print(f"Démarrage du crawler sur {START_URL}")
    print(f"Les textes seront sauvegardés dans {OUT_FILE}")
    print("Appuyez sur Ctrl+C à tout moment pour arrêter.\n")
    
    # On ouvre le fichier en mode "append" (ajout) pour ne pas perdre les données si on relance
    with open(OUT_FILE, "a", encoding="utf-8") as f:
        
        while urls_to_visit:
            current_url = urls_to_visit.pop(0)
            
            if current_url in urls_visited:
                continue
                
            urls_visited.add(current_url)
            
            try:
                print(f"[{len(urls_visited)} visitées | {len(urls_to_visit)} en attente] Scrape : {current_url}")
                response = requests.get(current_url, headers=HEADERS, timeout=10)
                
                # Si ce n'est pas une page web (ex: PDF ou MP3), on ignore
                if 'text/html' not in response.headers.get('Content-Type', ''):
                    continue
                    
                response.encoding = 'utf-8'
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 1. EXTRACTION DU TEXTE
                # On cible principalement les balises de paragraphe et les titres
                text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'span', 'div', 'li'])
                
                page_texts = []
                for el in text_elements:
                    # On ignore les éléments de navigation, de scripts, etc.
                    if el.find_parent(['nav', 'header', 'footer', 'script', 'style']):
                        continue
                        
                    text = clean_text(el.get_text())
                    # On garde seulement les phrases qui ont du sens (au moins 3 mots)
                    if len(text.split()) > 3: 
                        page_texts.append(text)
                
                # Sauvegarde immédiate dans le fichier
                # On utilise un set local pour éviter les phrases en double sur la même page
                for text in set(page_texts):
                    f.write(text + "\n")
                    
                # 2. RECHERCHE DE NOUVEAUX LIENS
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(current_url, href)
                    
                    # On s'assure qu'on reste sur le domaine principal ET dans la section Malgache (/mg/)
                    parsed_url = urlparse(full_url)
                    if parsed_url.netloc == BASE_DOMAIN and parsed_url.path.startswith('/mg/'):
                        # On enlève les ancres (#) pour ne pas visiter la même page plusieurs fois
                        clean_url = full_url.split('#')[0]
                        if clean_url not in urls_visited and clean_url not in urls_to_visit:
                            # On ignore les liens vers des médias (pdf, mp3, mp4, etc.)
                            if not clean_url.endswith(('.pdf', '.mp3', '.mp4', '.zip', '.epub')):
                                urls_to_visit.append(clean_url)
                                
                time.sleep(DELAY)
                
            except KeyboardInterrupt:
                print("\nArrêt manuel par l'utilisateur.")
                break
            except Exception as e:
                print(f"Erreur sur {current_url} : {e}")

if __name__ == "__main__":
    scrape()
    print(f"\nTerminé ! Total des pages visitées : {len(urls_visited)}")
