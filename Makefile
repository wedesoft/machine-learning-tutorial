.SUFFIXES: .tex .svg .pdf .py

all: notes data

notes: machine-learning-tutorial.pdf

train: data
	./mnist-backprop.py

data: mnist.pkl.gz

check:
	py.test-3 -x

mnist.pkl.gz:
	curl -o $@ http://deeplearning.net/data/mnist/mnist.pkl.gz

.tex.pdf:
	xelatex -shell-escape $<
	biber $(basename $@)
	xelatex -shell-escape $<
	xelatex -shell-escape $<

.svg.pdf:
	inkscape $< -A $@

.py.pdf:
	python3 $<

machine-learning-tutorial.pdf: machine-learning-tutorial.tex supervised.pdf unsupervised.pdf bibliography.bib \
	least_squares.pdf gradient_descent.pdf learning_rate.pdf sigmoid.pdf classifier.pdf polynomial.pdf \
	overfitting.pdf logcost.pdf \
	scaling.jpg gradientproscons.jpg decisionboundary.jpg circularboundary.jpg costy1.jpg costy0.jpg onevsall.jpg \
	logisticunit.jpg hidden.jpg xnor.jpg lecun.jpg alvinn.jpg \
	convolution.py least_squares.py gradient_descent.py learning_rate.py classifier.py polynomial.py overfitting.py \
	backprop.py mnist-backprop.py

clean:
	rm -Rf __pycache__ *.pdf *.log *.aux *.bbl *.bcf *.blg *.run.xml *.out *.pyg mnist.ckpt* checkpoint
