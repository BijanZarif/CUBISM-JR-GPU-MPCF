set term postscript enhanced color
set out "roofline_K20c.ps"
set title "Tesla K20c"

set grid
set logscale xy 2
set xlabel 'Operational Intensity'
set ylabel 'Gflops'
set samples 2000
ridge = 19.234973
Pmax = 3520.000000
bmax = 183.000000
roof(x) = x<ridge ? x*bmax : Pmax
ridge10 = Pmax/(10*bmax)
ridge25 = Pmax/(4*bmax)
ridge50 = Pmax/(2*bmax)
roof10(x) = x<ridge10 ? NaN : Pmax/10.
roof25(x) = x<ridge25 ? NaN : Pmax/4.
roof50(x) = x<ridge50 ? NaN : Pmax/2.

set xrange [1:32]
set key top left

set label 1 "390 Gflops" at 21,390
set label 2 "426 Gflops" at 21,426
set label 3 "OI = 19.94" at 21,136
set label 4 "10% Pmax" at 4,(Pmax/10. + 2*10)
set label 5 "25% Pmax" at 8,(Pmax/4.  + 4*10)
set label 6 "50% Pmax" at 16,(Pmax/2. + 8*10)

set arrow from 19.94,128 to 19.94,426 nohead lt -1

plot "<echo '19.94 390'" w p pt 5 title 'x/yflux', \
     "<echo '19.94 426'" w p pt 5 title 'zflux', \
     roof10(x) notitle lt -1, \
     roof25(x) notitle lt -1, \
     roof50(x) notitle lt -1, \
     roof(x) notitle lt 1 lw 2
