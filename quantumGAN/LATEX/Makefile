CC=pdflatex

makepaper:
	$(CC) Rplan.tex
	$(CC) Rplan.tex
	bibtex Rplan 
	$(CC)	Rplan.tex
	$(CC) Rplan.tex
	
clean:
	rm *.aux *.log *.toc *.lof *.lot *.out *.blg *.bbl

clear:
	rm *.aux *.log *.blg *.bbl *.pdf
