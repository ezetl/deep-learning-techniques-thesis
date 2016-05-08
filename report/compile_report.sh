#!/usr/bin/env bash

# compiles latex file and open it
file='MarcoTeorico'
pdflatex $file'.tex'
bibtex $file'.aux' 
pdflatex $file'.tex'
pdflatex $file'.tex'

evince $file'.pdf'

