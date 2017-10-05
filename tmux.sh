#!/bin/bash
tmux set mouse-resize-pane on
tmux set mouse-select-pane on
tmux set mouse-select-window on
tmux split-window -d -h
tmux split-window -d -p 60
tmux select-pane -t0
tmux send-keys "ptpython" C-m
tmux select-pane -t2
tmux send-keys "vim" C-m
sleep 1
tmux send-keys C-O C-O
tmux select-pane -t1
