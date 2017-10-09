.SUFFIXES: .tex .svg .pdf .py

all: notes

notes: machine-learning-tutorial.pdf

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

.py.pdf:
	python $<

machine-learning-tutorial.pdf: machine-learning-tutorial.tex supervised.pdf unsupervised.pdf bibliography.bib \
	least_squares.pdf gradient_descent.pdf learning_rate.pdf sigmoid.pdf classifier.pdf \
	scaling.jpg gradientproscons.jpg decisionboundary.jpg circularboundary.jpg costy1.jpg costy0.jpg onevsall.jpg \
	convolution.py least_squares.py gradient_descent.py learning_rate.py

clean:
	rm -Rf __pycache__ *.pdf *.log *.aux *.bbl *.bcf *.blg *.run.xml *.out *.pyg
