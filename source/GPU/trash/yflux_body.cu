/* File        : yflux_body.cu */
/* Creator     : Fabian Wermelinger <fabianw@student.ethz.ch> */
/* Created     : Thu 14 Aug 2014 02:58:02 PM CEST */
/* Modified    : Thu 14 Aug 2014 02:58:31 PM CEST */
/* Description : Computational body of y-flux */

// 2.)
// convert to primitive variables
#pragma unroll 6
for (uint_t i = 0; i < 6; ++i)
{
    e[i] = (e[i] - 0.5f*(u[i]*u[i] + v[i]*v[i] + w[i]*w[i])/r[i] - P[i]) / G[i];
    u[i] = u[i]/r[i];
    v[i] = v[i]/r[i];
    w[i] = w[i]/r[i];
} // 6 x (8 MUL/ADD/SUB + 5 DIV) = 78 FLOPS

const Real rm = _weno_minus_clipped(r[0], r[1], r[2], r[3], r[4]); // 96 FLOP (6 DIV)
const Real rp = _weno_pluss_clipped(r[1], r[2], r[3], r[4], r[5]); // 96 FLOP (6 DIV)
assert(!isnan(rp)); assert(!isnan(rm));

const Real Gm = _weno_minus_clipped(G[0], G[1], G[2], G[3], G[4]); // 96 FLOP (6 DIV)
const Real Gp = _weno_pluss_clipped(G[1], G[2], G[3], G[4], G[5]); // 96 FLOP (6 DIV)
assert(!isnan(Gp)); assert(!isnan(Gm));

const Real Pm = _weno_minus_clipped(P[0], P[1], P[2], P[3], P[4]); // 96 FLOP (6 DIV)
const Real Pp = _weno_pluss_clipped(P[1], P[2], P[3], P[4], P[5]); // 96 FLOP (6 DIV)
assert(!isnan(Pp)); assert(!isnan(Pm));

const Real pm = _weno_minus_clipped(e[0], e[1], e[2], e[3], e[4]); // 96 FLOP (6 DIV)
const Real pp = _weno_pluss_clipped(e[1], e[2], e[3], e[4], e[5]); // 96 FLOP (6 DIV)
assert(!isnan(pp)); assert(!isnan(pm));

const Real vm = _weno_minus_clipped(v[0], v[1], v[2], v[3], v[4]); // 96 FLOP (6 DIV)
const Real vp = _weno_pluss_clipped(v[1], v[2], v[3], v[4], v[5]); // 96 FLOP (6 DIV)
assert(!isnan(vp)); assert(!isnan(vm));

const Real um = _weno_minus_clipped(u[0], u[1], u[2], u[3], u[4]); // 96 FLOP (6 DIV)
const Real up = _weno_pluss_clipped(u[1], u[2], u[3], u[4], u[5]); // 96 FLOP (6 DIV)
assert(!isnan(up)); assert(!isnan(um));

const Real wm = _weno_minus_clipped(w[0], w[1], w[2], w[3], w[4]); // 96 FLOP (6 DIV)
const Real wp = _weno_pluss_clipped(w[1], w[2], w[3], w[4], w[5]); // 96 FLOP (6 DIV)
assert(!isnan(wp)); assert(!isnan(wm));

// 3.)
Real sm, sp;
_char_vel_einfeldt(rm, rp, vm, vp, pm, pp, Gm, Gp, Pm, Pp, sm, sp); // 29 FLOP (6 DIV)
const Real ss = _char_vel_star(rm, rp, vm, vp, pm, pp, sm, sp); // 11 FLOP (1 DIV)
assert(!isnan(sm)); assert(!isnan(sp)); assert(!isnan(ss));

// 4.)
const Real fr = _hllc_rho(rm, rp, vm, vp, sm, sp, ss); // 23 FLOP (2 DIV)
const Real fu = _hllc_vel(rm, rp, um, up, vm, vp, sm, sp, ss); // 25 FLOP (2 DIV)
const Real fv = _hllc_pvel(rm, rp, vm, vp, pm, pp, sm, sp, ss); // 29 FLOP (2 DIV)
const Real fw = _hllc_vel(rm, rp, wm, wp, vm, vp, sm, sp, ss); // 25 FLOP (2 DIV)
const Real fe = _hllc_e(rm, rp, vm, vp, um, up, wm, wp, pm, pp, Gm, Gp, Pm, Pp, sm, sp, ss); // 59 FLOP (4 DIV)
const Real fG = _hllc_rho(Gm, Gp, vm, vp, sm, sp, ss); // 23 FLOP (2 DIV)
const Real fP = _hllc_rho(Pm, Pp, vm, vp, sm, sp, ss); // 23 FLOP (2 DIV)
assert(!isnan(fr)); assert(!isnan(fu)); assert(!isnan(fv)); assert(!isnan(fw)); assert(!isnan(fe)); assert(!isnan(fG)); assert(!isnan(fP));

const Real hllc_vel = _extraterm_hllc_vel(vm, vp, Gm, Gp, Pm, Pp, sm, sp, ss); // 19 FLOP (2 DIV)
