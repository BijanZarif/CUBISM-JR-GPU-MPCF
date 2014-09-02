set term postscript enhanced color
set out "roofline_K20c.ps"
set title "Tesla K20c"

set grid
set logscale xy 2
set xlabel 'Operational Intensity'
set ylabel 'Gflops'
set samples 2000
Pmax = 3520.000000
bmax = 208.000000
ridge = Pmax/bmax
bmax_eff = 171.0
ridge_eff = Pmax/bmax_eff
roof(x) = x<ridge ? x*bmax : Pmax
roof_eff(x) = x<ridge_eff ? x*bmax_eff : NaN
ridge10 = Pmax/(10*bmax)
ridge25 = Pmax/(4*bmax)
ridge50 = Pmax/(2*bmax)
roof10(x) = x<ridge10 ? NaN : Pmax/10.
roof25(x) = x<ridge25 ? NaN : Pmax/4.
roof50(x) = x<ridge50 ? NaN : Pmax/2.

set xrange [0.25:32]
set key top left

# set label 1 "390 Gflops" at 21,390
# set label 2 "426 Gflops" at 21,426
# set label 3 "OI = 19.94" at 21,136
set label 4 "10% Pmax" at 4,(Pmax/10. + 2*16)
set label 5 "25% Pmax" at 8,(Pmax/4.  + 4*16)
set label 6 "50% Pmax" at 16,(Pmax/2. + 8*16)

# set arrow from 19.94,128 to 19.94,426 nohead lt -1

plot "<echo '0.318 46.46'" w p pt 3 title 'CONV (46.5 Gflops)', \
     "<echo '6.0 840.8'" w p pt 7 title 'WENO (840.8 Gflops)', \
     "<echo '3.0 326.8'" w p pt 5 title 'HLLC (326.8 Gflops)', \
     roof10(x) notitle lt -1, \
     roof25(x) notitle lt -1, \
     roof50(x) notitle lt -1, \
     roof_eff(x) notitle lt 2 lw 2, \
     roof(x) notitle lt 1 lw 2
