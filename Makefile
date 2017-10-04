check:
	py.test-3

mnist.pkl.gz:
	curl -o $@ http://deeplearning.net/data/mnist/mnist.pkl.gz

clean:
	rm -Rf __pycache__ *.png
