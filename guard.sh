#!/bin/sh
while true; do \
  clear; \
  py.test-3 $@; \
  inotifywait -e CLOSE_WRITE `git ls-files`; \
done
