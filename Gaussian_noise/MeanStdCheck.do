#!/bin/bash
rm MeanStdCheck.txt
for (( i=0; i<=12; i++))
do
    echo "Target layer == $i" >> MeanStdCheck.txt
    python3 -W ignore MeanStdCheck.py --targetlayer $i --load 1 --filename MeanStdCheck.txt 
    echo $'\n' >> MeanStdCheck.txt
    echo "Target layer $i is done"
done
