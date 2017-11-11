#!/bin/bash
for (( i=0; i<=11; i++))
do
	echo "noise layer == $i" >> record1.txt
	for (( j=1; j<=20; j++))
	do
		python3 -W ignore main.py --noiselayer $i --std $j --load 3
	done
	echo $'\n' >> record1.txt
	echo "noise layer $i is done"
done
