if [ -z "$TMUX" ]; then
	tmux attach -t nmt || tmux new -s nmt
fi

alias tmux="direnv exec / tmux"

eval "$(direnv hook bash)"
