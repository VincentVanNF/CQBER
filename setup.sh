#!/usr/bin/env bash

# source setup.sh
DIR_PWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export DIR_PWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$PYTHONPATH:$DIR_PWD"
echo $PYTHONPATH
