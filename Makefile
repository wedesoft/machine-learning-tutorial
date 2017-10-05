.SUFFIXES: .tex .svg .pdf

all: course-notes.pdf

check:
	py.test-3 || py.test

mnist.pkl.gz:
	curl -o $@ http://deeplearning.net/data/mnist/mnist.pkl.gz

.tex.pdf:
	xelatex $<
	biber $(basename $@)
	xelatex $<

.svg.pdf:
	inkscape $< -A $@

course-notes.pdf: course-notes.tex supervised.pdf unsupervised.pdf bibliography.bib

clean:
	rm -Rf __pycache__ *.png *.pdf
