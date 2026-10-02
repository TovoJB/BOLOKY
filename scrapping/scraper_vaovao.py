import requests
from bs4 import BeautifulSoup
import re
import time
from urllib.parse import urljoin, urlparse

# --- CONFIGURATION ---
START_URL = "https://vaovao.org/mg"
BASE_DOMAIN = "vaovao.org"
OUT_FILE = "./vaovao_corpus.txt"
DELAY = 0.5 # Pause de 0.5s pour ne pas surcharger le serveur

urls_visited = set()
urls_to_visit = [START_URL]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def scrape():
    print(f"Démarrage du crawler sur {START_URL}")
    print(f"Les textes seront sauvegardés dans {OUT_FILE}")
    print("Appuyez sur Ctrl+C à tout moment pour arrêter.\n")
    
    with open(OUT_FILE, "a", encoding="utf-8") as f:
        while urls_to_visit:
            current_url = urls_to_visit.pop(0)
            
            if current_url in urls_visited:
                continue
                
            urls_visited.add(current_url)
            
            try:
                print(f"[{len(urls_visited)} visitées | {len(urls_to_visit)} en attente] Scrape : {current_url}")
                response = requests.get(current_url, headers=HEADERS, timeout=10)
                
                if 'text/html' not in response.headers.get('Content-Type', ''):
                    continue
                    
                response.encoding = 'utf-8'
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 1. EXTRACTION DU TEXTE
                text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'span', 'div', 'li', 'article'])
                
                page_texts = []
                for el in text_elements:
                    # On ignore les menus, pieds de page, etc.
                    if el.find_parent(['nav', 'header', 'footer', 'script', 'style']):
                        continue
                        
                    text = clean_text(el.get_text())
                    if len(text.split()) > 3: 
                        page_texts.append(text)
                
                for text in set(page_texts):
                    f.write(text + "\n")
                    
                # 2. RECHERCHE DE NOUVEAUX LIENS
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(current_url, href)
                    
                    parsed_url = urlparse(full_url)
                    # On reste sur vaovao.org et on s'assure qu'on reste dans la section Malgache (/mg)
                    if parsed_url.netloc.endswith(BASE_DOMAIN) and (parsed_url.path == '/mg' or parsed_url.path.startswith('/mg/')):
                        clean_url = full_url.split('#')[0]
                        if clean_url not in urls_visited and clean_url not in urls_to_visit:
                            # Ignorer les partages sociaux, flux rss, médias
                            if not any(x in clean_url for x in ['/feed', 'share=', 'replytocom=']) and not clean_url.endswith(('.pdf', '.jpg', '.png')):
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
