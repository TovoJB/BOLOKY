import re
from bs4 import BeautifulSoup

def parse_html(html_path):
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')
    content = soup.find('div', id='content')
    
    verses = {}
    current_verse = None
    verse_text = []

    # Iterate over all descendants or elements
    for el in content.descendants:
        if isinstance(el, str):
            text = el.strip()
            if text and current_verse is not None:
                # avoid adding if it is inside a span.v or div.c-drop
                parent_class = el.parent.get('class', [])
                if 'v' not in parent_class and 'c-drop' not in parent_class:
                    verse_text.append(text)
        elif el.name == 'div' and 'c-drop' in el.get('class', []):
            if current_verse:
                verses[current_verse] = " ".join(verse_text)
            current_verse = el.text.strip()
            verse_text = []
        elif el.name == 'span' and 'v' in el.get('class', []):
            if current_verse:
                verses[current_verse] = " ".join(verse_text)
            current_verse = el.text.strip()
            verse_text = []

    if current_verse:
        verses[current_verse] = " ".join(verse_text)

    # Clean up verses
    for v, text in verses.items():
        # replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        print(f"{v}: {text}")

if __name__ == "__main__":
    parse_html('source.html')
