import datetime
import pandas as pd
import pywikibot
from urllib.request import urlopen
from bs4 import BeautifulSoup


def parse_by_url(url):
    # Specify url of the web page
    source = urlopen(url).read()
    # Make a soup
    soup = BeautifulSoup(source, 'html.parser')
    text = ''
    for paragraph in soup.find_all('p'):
        text += paragraph.text
    return text


def create_data(title):
    site = pywikibot.Site("en", "wikipedia")
    page = pywikibot.Page(site, title)
    revs = page.revisions(content=False)
    texts = []
    timestamps = []

    last_timestamp = None

    base_url = f"https://en.wikipedia.org/w/index.php?title={'_'.join(title.split())}&oldid="
    for i, rev in enumerate(revs):
        if i == 0:
            continue
        rev_date = rev._data['timestamp'].date()
        if last_timestamp is None or rev_date < last_timestamp - datetime.timedelta(weeks=26):
            url = base_url + str(rev['revid'])
            text = parse_by_url(url)
            texts.append(text.split('which may differ significantly from the current revision.')[1])
            timestamps.append(rev_date)
            last_timestamp = rev_date

    # for text, date in zip(texts, timestamps):
    #     print(text[:50])
    #     print(date)

    df = pd.DataFrame({'date': timestamps, 'text': texts})
    df.sort_values(by='date', ascending=True, inplace=True)
    print(df)
    output_path = f"{''.join(title.split())}_revisions.csv"
    print(f"saving to {output_path}")
    df.to_csv(output_path, index=False)


def read_data(path):
    return pd.read_csv(path)


if __name__ == '__main__':
    title = "Black Lives Matter"
    create_data(title)


