.SUFFIXES: .tex .svg .pdf .py

all: notes data

notes: machine-learning-tutorial.pdf

train: data
	./mnist-backprop.py

data: mnist.pkl.gz shakespeare.txt

check:
	py.test-3 -x

mnist.pkl.gz:
	curl -o $@ https://github.com/mnielsen/neural-networks-and-deep-learning/raw/refs/heads/master/data/mnist.pkl.gz

shakespeare.txt:
	curl -o $@ http://www.gutenberg.org/cache/epub/100/pg100.txt

.tex.pdf:
	pdflatex -shell-escape $<
	biber $(basename $@)
	pdflatex -shell-escape $<
	pdflatex -shell-escape $<

.svg.pdf:
	rsvg-convert -f pdf -o $@ $<

.py.pdf:
	python2 $<

machine-learning-tutorial.pdf: machine-learning-tutorial.tex supervised.pdf unsupervised.pdf bibliography.bib \
	least_squares.pdf gradient_descent.pdf learning_rate.pdf sigmoid.pdf classifier.pdf polynomial.pdf \
	overfitting.pdf logcost.pdf rnnconf.pdf \
	scaling.jpg gradientproscons.jpg decisionboundary.jpg circularboundary.jpg costy1.jpg costy0.jpg onevsall.jpg \
	logisticunit.jpg hidden.jpg xnor.jpg lecun.jpg alvinn.jpg charseg.jpg ceiling.jpg sparse.jpg autoimage.jpg \
	dream0.png dream1.png dream2.png dream3.png dream4.png dream5.png twodenoised.png twonoise.png \
	convolution.py least_squares.py gradient_descent.py learning_rate.py classifier.py polynomial.py overfitting.py \
	backprop.py mnist-backprop.py mnist-run.py mnist-convolutional.py mnist-rbm.py mnist-dream.py \
	autoencoder.py autoencoder-run.py rnn.py rnn-predict.py lstm.py lstm-predict.py

clean:
	rm -Rf __pycache__ *.pdf *.log *.aux *.bbl *.bcf *.blg *.run.xml *.out *.pyg mnist.ckpt* checkpoint
