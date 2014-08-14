# File        : roofline.py
# Creator     : Fabian Wermelinger <fabianw@student.ethz.ch>
# Created     : Tue 12 Aug 2014 08:44:37 AM CEST
# Modified    : Tue 12 Aug 2014 09:20:59 AM CEST
# Description :
import sys

if (len(sys.argv) < 4):
    print "ERROR: Need Arguments <peak> <BW> <arch>"
    sys.exit(1)

Pmax = float(sys.argv[1])
bmax = float(sys.argv[2])
arch = sys.argv[3]

ridge = Pmax/bmax

out = open("roofline_" + arch + ".gp", 'w')
out.write("set grid\n")
out.write("set logscale xy 2\n")
out.write("set xlabel 'Operational Intensity'\n")
out.write("set ylabel 'GFlop/s'\n")
out.write("set samples 2000\n")
out.write("set ridge = %f\nset Pmax = %f\nset bmax = %f\n" % (ridge, Pmax, bmax))
out.write("roof(x) = x<ridge ? x*bmax : Pmax\n")
out.close()
