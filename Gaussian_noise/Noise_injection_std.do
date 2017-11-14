#!/bin/bash
rm noise_injection_std_result1.txt
for (( i=0; i<=12; i++))
do
    echo "Noise layer == $i" >> noise_injection_std_result1.txt
	for (( j=1; j<=2; j++))
	do
    python3 -W ignore Noise_injection_std.py --noiselayer $i --load $j
	done
    echo $'\n' >> noise_injection_std_result1.txt
    echo "Noise layer $i is done"
done
