#!/bin/bash

while true; do
    free -m | awk 'NR==2 {print strftime("%H:%M:%S"), $7}' >> memory.log
    sleep 2
done
