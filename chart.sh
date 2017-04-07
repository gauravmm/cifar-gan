#!/bin/bash
GENERATOR="Simple"
DISCRIMINATOR="Simple"

CSVFILE="train_logs/loss-$GENERATOR-$DISCRIMINATOR.csv"
gnuplot -p << eor
set multiplot layout 2,1 rowsfirst
set datafile separator ','
set style data lines
set lmargin at screen 0.1; set rmargin at screen 0.9

unset xlabel
set tmargin at screen 0.9; set bmargin at screen 0.25
set xtics ('' 0)
set ytics 0.2
set ylabel "percent"
plot "$CSVFILE" using 2:4 title "generator fooling", \
     "$CSVFILE" using 2:7 title "discriminator true positive", \
     "$CSVFILE" using 2:11 title "discriminator true negative"

set tmargin at screen 0.25; set bmargin at screen 0.1
set xtics auto
set xlabel "batch"
set ytics 1
set ylabel "number"
plot "$CSVFILE" using 2:12 title "discriminator steps", \
     "$CSVFILE" using 2:13 title "generator steps"
eor