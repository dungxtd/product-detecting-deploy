#!/bin/sh
conda run --no-capture-output -n env gunicorn -b 0.0.0.0:2808 app:app