import argparse
import sys
from utils import init_directories
from config import AUDIO_PUBLICATIONS
from scraper_bible import process_bible
from scraper_books import process_all_books

def main():
    parser = argparse.ArgumentParser(
        description="Dataset Scraper: Dialecte Vezo (skg-x-vz) & Malagasy officiel (mg)"
    )
    parser.add_argument("--all", action="store_true", help="Scrape tout (Bible + Tous les Livres Audio)")
    parser.add_argument("--bible", action="store_true", help="Scrape la Bible (26 livres du Nouveau Testament)")
    parser.add_argument("--books", action="store_true", help="Scrape tous les livres et brochures d'étude avec audio")
    parser.add_argument("--pub", type=str, default=None, help="Scrape un livre audio spécifique par son code (ex: lfb, bh, bhs, ll, fg, hf, ypq, jlp)")
    parser.add_argument("--no-audio", action="store_true", help="Ne télécharge pas les fichiers audio MP3 (texte seul)")
    parser.add_argument("--no-slice", action="store_true", help="Ne découpe pas les MP3 en segments WAV")
    parser.add_argument("--limit-books", type=int, default=None, help="Limiter le nombre de livres bibliques (pour test)")
    
    args = parser.parse_args()
    
    if not (args.all or args.bible or args.books or args.pub):
        print("=================================================================")
        print("  SCRAPER DATASET AUDIO & TEXTE : DIALECTE VEZO (skg-x-vz)")
        print("=================================================================")
        print("\nLivres avec audio disponibles :")
        for code, info in AUDIO_PUBLICATIONS.items():
            print(f"  - {code:6s} : {info['title_vz']}")
        print("\nExemples d'utilisation :")
        print("  python run_scraper.py --books           # Télécharger TOUS les livres audio")
        print("  python run_scraper.py --pub lfb         # Télécharger uniquement le livre 'lfb' (118 chapitres)")
        print("  python run_scraper.py --pub bh          # Télécharger le livre 'bh' (34 chapitres)")
        print("  python run_scraper.py --pub bhs         # Télécharger le livre 'bhs' (21 chapitres avec marqueurs)")
        print("  python run_scraper.py --bible           # Télécharger la Bible complète (26 livres)")
        print("  python run_scraper.py --all             # Tout télécharger")
        print("=================================================================")
        sys.exit(0)
        
    init_directories()
    download_audio = not args.no_audio
    slice_clips = not args.no_slice and download_audio
    
    if args.all or args.bible:
        print("\n>>> LANCEMENT DU SCRAPING BIBLE...")
        process_bible(download_audio=download_audio, slice_clips=slice_clips, limit_books=args.limit_books)
        
    if args.pub:
        pub_clean = args.pub.lower().strip()
        print(f"\n>>> LANCEMENT DU SCRAPING DU LIVRE [{pub_clean}]...")
        process_all_books(pubs_list=[pub_clean], download_audio=download_audio, slice_clips=slice_clips)
    elif args.all or args.books:
        print("\n>>> LANCEMENT DU SCRAPING DE TOUS LES LIVRES AUDIO...")
        process_all_books(download_audio=download_audio, slice_clips=slice_clips)
        
    print("\n=======================================================")
    print("✓ Opération terminée avec succès !")
    print("Dossier des données : dataset_vezo/")
    print("=======================================================")

if __name__ == "__main__":
    main()
