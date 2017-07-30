#!/usr/bin/env bash

# compiles latex file and open it
file='slides'
rm -f $file.{aux,bbl,blg,log,out,pdf,toc} &> /dev/null

pdflatex $file'.tex'

evince $file'.pdf'

