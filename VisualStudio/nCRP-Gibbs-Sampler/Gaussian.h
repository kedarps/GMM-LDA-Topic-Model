#pragma once
#include <iostream>
#include <cmath>
#include <armadillo>
#include "mex.h"

using namespace std;
using namespace arma;

class Gaussian
{
public:
	// properties of Gaussian object
	double	p_dd;		// (1x1) dimensionality
	double	p_nn;		// (1x1) number of items
	double	p_rr;		// (1x1) precision, inverse of variance (ss)
	double	p_vv;		// (1x1) degrees of freedom
	mat		p_CC;		// (dxd) chol(VV*vv + rr*uu*uu')
	colvec	p_XX;		// (dx1) rr*uu
	double	p_Z0;		// (1x1) ZZ(dd, nn, rr, vv, CC, XX)

	Gaussian();
	~Gaussian();

	void	init(double, double, mat, double, colvec);
	colvec	getMu(void);
	mat		getSigma(void);
	void	additem(colvec);
	void	delitem(colvec);
	double	logpredictive(colvec);
	double	logmarginal(void);
	void	display(void);
};