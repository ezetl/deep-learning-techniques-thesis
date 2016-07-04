#!/usr/bin/env python2.7
import os
import sys
import subprocess
from bs4 import BeautifulSoup
import urllib

URLSNUM = 267
DST = './data/'
command = 'wget -O {} {}'

if __name__ == "__main__":
    r = urllib.urlopen('https://embed.stanford.edu/iframe/?url=https%3A%2F%2Fpurl.stanford.edu%2Fvn158kj2087')
    soup = BeautifulSoup(r, "lxml")
    a = soup.find_all('a')
    final_as = [elem for elem in a if "download" not in str(elem)]
    urls = [elem.get('href') for elem in final_as]
    urls = urls[:-1]
    assert len(urls) == URLSNUM, "A problem occured while trying to get the\
        list of url's, expected {}, got {}".format(URLSNUM, len(urls))

    if not os.path.isdir(DST):
        os.makedirs(DST)
        print("Created download dir {}\n".format(DST))

    for i, url in enumerate(urls):
        bname = os.path.basename(url)
        path = os.path.join(DST, bname)
        if os.path.isfile(path):
            continue
        print("Downloading {}/{} : {}".format(i+1, len(urls), bname))
        c = command.format(path, url)
        process = subprocess.Popen(c.split(), stdout=subprocess.PIPE)
        output = process.communicate()[0]
    print("\nFinished download. Files are in {}\n".format(DST))
    sys.exit(0)
