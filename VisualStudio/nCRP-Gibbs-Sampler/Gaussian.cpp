#include "stdafx.h"
#include "Gaussian.h"

//Both 'cholupdate' and 'choldowndate' are based on pseudo-code here: https://en.wikipedia.org/wiki/Cholesky_decomposition#Rank-one_update
mat cholupdate(mat R, colvec xx)
{
	uword n = xx.n_elem;
	for (uword k = 0; k < n; ++k)
	{
		double r = sqrt(pow(R(k, k), 2) + pow(xx(k), 2));
		double c = r / R(k, k);
		double s = xx(k) / R(k, k);
		R(k, k) = r;

		for (uword j = k + 1; j < n; ++j)
		{
			R(j, k) = (R(j, k) + s*xx(j)) / c;
			xx(j) = c*xx(j) - s*R(j, k);
		}
	}
	return R;
}

mat choldowndate(mat R, colvec xx)
{
	uword n = xx.n_elem;
	for (uword k = 0; k < n; ++k)
	{
		double r = sqrt(pow(R(k, k), 2) - pow(xx(k), 2));
		double c = r / R(k, k);
		double s = xx(k) / R(k, k);
		R(k, k) = r;

		for (uword j = k + 1; j < n; ++j)
		{
			R(j, k) = (R(j, k) - s*xx(j)) / c;
			xx(j) = c*xx(j) - s*R(j, k);
		}
	}
	return R;
}

double ZZ(double dd, double nn, double rr, double vv, mat SS, colvec xx)
{
	// this is a dumb way to do this, but arma::lgamma(colvec) is not working with mex
	colvec vv_d = (vv - regspace<colvec>(0, dd - 1)) / 2.0;

	double s_vv_d = 0.0;
	for (arma::colvec::iterator i = vv_d.begin(); i != vv_d.end(); ++i)
		s_vv_d += std::lgamma(*i);

	double zz =
		-nn * dd * log(datum::pi) / 2.0
		- dd * log(rr) / 2.0
		- vv*sum(log(diagvec(choldowndate(SS, xx / sqrt(rr)))))
		+ s_vv_d;

	return zz;
}

Gaussian::Gaussian()
{
}

Gaussian::~Gaussian()
{
}

void Gaussian::init(double dd, double ss, mat VV, double vv, colvec uu)
{
	///// inputs to Gaussian object /////
	//double dd;		// (1x1) dimensionality
	//double ss;		// (1x1) relative variance of mm versus data
	//mat VV;			// (dxd) mean covariance of clusters
	//double vv;		// (1x1) degrees of freedom
	//colvec uu;		// (dx1) prior mean vector
	/////////////////////////////////////
	double rr = 1 / ss;

	p_dd = dd;
	p_nn = 0;
	p_rr = rr;
	p_vv = vv;
	p_CC = chol(VV * vv + rr * uu * uu.t());
	p_XX = rr * uu;
	p_Z0 = ZZ(p_dd, p_nn, p_rr, p_vv, p_CC, p_XX);
}

colvec Gaussian::getMu(void)
{
	colvec mu = p_XX / p_rr;
	return mu;
}

mat Gaussian::getSigma(void)
{
	mat CC = choldowndate(p_CC, p_XX / sqrt(p_rr));
	mat sigma = CC.t() * CC / (p_vv - p_dd - 1);

	return sigma;
}

void Gaussian::additem(colvec xx)
{
	p_nn++;
	p_rr++;
	p_vv++;
	p_CC = cholupdate(p_CC, xx);
	p_XX = p_XX + xx;
}

void Gaussian::delitem(colvec xx)
{
	p_nn--;
	p_rr--;
	p_vv--;
	p_CC = choldowndate(p_CC, xx);
	p_XX = p_XX - xx;
}

double Gaussian::logpredictive(colvec xx)
{
	double logpred =
		ZZ(p_dd, p_nn + 1, p_rr + 1, p_vv + 1, cholupdate(p_CC, xx), p_XX + xx)
		- ZZ(p_dd, p_nn, p_rr, p_vv, p_CC, p_XX);

	return logpred;
}

double Gaussian::logmarginal(void)
{
	double logmarg =
		ZZ(p_dd, p_nn, p_rr, p_vv, p_CC, p_XX)
		- p_Z0;

	return logmarg;
}

void Gaussian::display(void)
{
	ostringstream buffer;

	buffer << "dd: " << p_dd << ", nn: " << p_nn << ", rr: " << p_rr << ", vv: " << p_vv << endl;
	buffer << "mu = " << endl;
	buffer << getMu() << endl;
	buffer << "sigma = " << endl;
	buffer << getSigma() << endl;
	mexPrintf("%s\n", buffer.str().c_str());
}