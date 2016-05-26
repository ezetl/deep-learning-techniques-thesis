#!/usr/bin/env bash

# compiles latex file and open it
file='MarcoTeorico'
rm -f $file.{aux,bbl,blg,log,out,pdf}

pdflatex $file'.tex'
bibtex $file'.aux' 
pdflatex $file'.tex'
pdflatex $file'.tex'

evince $file'.pdf'

