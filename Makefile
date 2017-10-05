.SUFFIXES: .tex .svg .pdf

all: notes

notes: course-notes.pdf

check:
	py.test-3 || py.test

mnist.pkl.gz:
	curl -o $@ http://deeplearning.net/data/mnist/mnist.pkl.gz

.tex.pdf:
	xelatex -shell-escape $<
	biber $(basename $@)
	xelatex -shell-escape $<

.svg.pdf:
	inkscape $< -A $@

course-notes.pdf: course-notes.tex supervised.pdf unsupervised.pdf bibliography.bib

clean:
	rm -Rf __pycache__ *.pdf *.log *.aux *.bbl *.bcf *.blg *.run.xml *.out *.pyg
