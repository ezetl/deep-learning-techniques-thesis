# Install dependencies to build the report (I use Emacs Orgmode to write plain
# text and then generate a .tex from it)
sudo apt-get install texlive-base texlive-latex-extra texlive-lang-spanish dvipng python-pip
# Pillow dependencies
sudo apt-get install libjpeg8 libjpeg62-dev libfreetype6 libfreetype6-dev
sudo pip install -r requirements.txt
