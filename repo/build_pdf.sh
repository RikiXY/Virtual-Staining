#!/bin/bash
jupyter nbconvert "$1" --to pdf --template ./template --config ./template/conf.json
