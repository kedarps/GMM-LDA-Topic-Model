// nCRP-Gibbs-Sampler.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "mex.h"
#include <iostream>
#include "armaMex.hpp"
#include <armadillo>
#include "GibbsSampler.h"
#include <string>

using namespace std;
using namespace arma;

mxArray* createCell(field<rowvec> X)
{
	mxArray *cell;
	cell = mxCreateCellMatrix(X.n_elem, 1);
	for (uword i = 0; i < X.n_elem; i++)
	{
		rowvec X_i = X(i);
		mxArray *mxMat = armaCreateMxMatrix(X_i.n_rows, X_i.n_cols);
		armaSetPr(mxMat, X_i);
		mxSetCell(cell, i, mxMat);
	}
	return cell;
}

mxArray* createCell(field< field<rowvec> > X)
{
	mxArray *cell;
	cell = mxCreateCellMatrix(X.n_elem, 1);
	for (uword i = 0; i < X.n_elem; i++)
	{
		field<rowvec> X_i = X(i);
		mxArray *thisCell = createCell(X_i);
		mxSetCell(cell, i, thisCell);
	}
	return cell;
}

mxArray* createMuCell(field< field<Gaussian> > X)
{
	mxArray *cell;
	cell = mxCreateCellMatrix(X.n_elem, 1);
	for (uword k = 0; k < X.n_elem; k++)
	{
		field<Gaussian> X_k = X(k);
		mat mu(X_k(0).p_dd, X_k.n_elem, fill::zeros);

		for (uword m = 0; m < X_k.n_elem; m++)
			mu.col(m) = X_k(m).getMu();

		mxArray *mxMat = armaCreateMxMatrix(mu.n_rows, mu.n_cols);
		armaSetPr(mxMat, mu);
		mxSetCell(cell, k, mxMat);
	}
	return cell;
}

mxArray* createSigmaCell(field< field<Gaussian> > X)
{
	mxArray *cell;
	cell = mxCreateCellMatrix(X.n_elem, 1);
	for (size_t k = 0; k < X.n_elem; k++)
	{
		field<Gaussian> X_k = X(k);
		cube sigma(X_k(0).p_dd, X_k(0).p_dd, X_k.n_elem, fill::zeros);

		for (uword m = 0; m < X_k.n_elem; m++)
			sigma.slice(m) = X_k(m).getSigma();

		mxArray *mxMat = armaCreateMxMatrix(sigma.n_rows, sigma.n_cols, sigma.n_slices);
		armaSetCubePr(mxMat, sigma);
		mxSetCell(cell, k, mxMat);
	}
	return cell;
}

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	/* check for proper number of arguments */
	if (nrhs != 5)
		mexErrMsgIdAndTxt("MyToolbox:nCRP_Gibbs_Sampler:nrhs", "Five inputs required (data, KK, MM, aa, numiters).");

	if (nlhs != 5)
		mexErrMsgIdAndTxt("MyToolbox:nCRP_Gibbs_Sampler:nlhs", "Five output arguments required.");

	if (mxGetClassID(prhs[0]) != mxCELL_CLASS)
		mexErrMsgIdAndTxt("MyToolbox:nCRP_Gibbs_Sampler:notCell", "First input argument must be a cell, with each cell element containing matrix of size(dd x Nd)");

	/*Gather data*/
	mwSize DD;
	DD = mxGetNumberOfElements(prhs[0]);

	const mxArray *allDocs;
	allDocs = prhs[0];

	field<mat> data(DD, 1);
	rowvec NN(DD);
	double dd;

	mexPrintf("Got %d docs, with sizes:\n", DD);
	for (int iDoc = 0; iDoc < DD; iDoc++)
	{
		/*extract each document from cell array*/
		const mxArray *thisDoc;
		thisDoc = mxGetCell(allDocs, iDoc);
		mat doc = armaGetPr(thisDoc);

		data(iDoc, 0) = doc;
		NN(iDoc) = mxGetN(thisDoc);

		/*get data dimensionality*/
		if (iDoc == 0)
			dd = mxGetM(thisDoc);

		mexPrintf("\tdoc %d: %d x %d\n", iDoc + 1, int(dd), int(NN(iDoc)));
	}

	if ((mxGetClassID(prhs[1]) != mxDOUBLE_CLASS) || (mxGetClassID(prhs[2]) != mxDOUBLE_CLASS))
		mexErrMsgIdAndTxt("MyToolbox:nCRP_Gibbs_Sampler:notDouble", "second and third input arguments must be double");

	double KK = mxGetScalar(prhs[1]);
	rowvec MM = armaGetPr(prhs[2]);

	if (MM.n_elem != KK)
		mexErrMsgIdAndTxt("MyToolbox:nCRP_Gibbs_Sampler:dimMismatch", "number of elements in MM should be equal to KK");

	rowvec aa = armaGetPr(prhs[3]);

	if (aa.n_elem != 2)
		mexErrMsgIdAndTxt("MyToolbox:nCRP_Gibbs_Sampler:incorrectaa", "aa should be a row-vector of length 2");

	///////////////////
	GibbsSampler gibbs;
	wall_clock timer;

	ostringstream buffer;
	buffer << "K = " << KK << ", MM = " << MM << endl;
	buffer << "prior, aa = " << aa << endl;
	timer.tic();
	gibbs.init(dd, DD, NN, KK, MM, data, aa);
	mexPrintf("%s\nInitialised in %3.2f sec\n---------------\n", buffer.str().c_str(), timer.toc());
	ioFlush();
	///////////////////

	/*Run gibbs sampler*/
	if (!mxIsScalar(prhs[4]))
		mexErrMsgIdAndTxt("MyToolbox:nCRP_Gibbs_Sampler:notScalar", "fifth input argument must be scalar");

	double numiters = mxGetScalar(prhs[4]);
	gibbs.run(numiters);
	///////////////////

	/*Return results to output*/
	plhs[0] = createCell(gibbs.p_ZZ);
	plhs[1] = createCell(gibbs.p_XiXi);
	plhs[2] = createCell(gibbs.p_nn);
	plhs[3] = createMuCell(gibbs.p_GG);
	plhs[4] = createSigmaCell(gibbs.p_GG);
	///////////////////
	return;
}