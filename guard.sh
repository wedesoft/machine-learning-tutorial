#!/bin/sh
while true; do \
  clear; \
  py.test-3 $@ || py.test $@; \
  inotifywait -e CLOSE_WRITE `git ls-files`; \
done
