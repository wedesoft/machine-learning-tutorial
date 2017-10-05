.SUFFIXES: .tex .pdf

all: course-notes.pdf

check:
	py.test-3 || py.test

mnist.pkl.gz:
	curl -o $@ http://deeplearning.net/data/mnist/mnist.pkl.gz

.tex.pdf:
	xelatex $<
	biber $(basename $@)
	xelatex $<

clean:
	rm -Rf __pycache__ *.png *.pdf
