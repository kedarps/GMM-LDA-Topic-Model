#pragma once
#pragma comment(lib, "libmwservices.lib")
#include "Gaussian.h"
#include "mex.h"
#include <armadillo>
#include <random>
#include <iomanip>

using namespace std;
using namespace arma;
//so that mexPrintf prints properly
extern bool ioFlush(void);

class GibbsSampler
{
public:
	double						p_dd;	// dimensionality
	double						p_KK;	// number of GMMs
	rowvec						p_MM;	// number of components in each GMM, length should be equal to p_KK
	double						p_DD;	// number of documents
	rowvec						p_NN;	// number of words in each document
	rowvec						p_aa;	// prior for each level, should be of length 2
	field< field<Gaussian> >	p_GG;	// nested Gaussian objects
	field<mat>					p_XX;	// data
	field<rowvec>				p_ZZ;	// GMM assignment
	field<rowvec>				p_XiXi;	// GMM component assignment
	field< field<rowvec> >		p_nn;	// number of items in each GMM/component for a given document

	GibbsSampler();
	~GibbsSampler();

	void display(void);
	void init(double, double, rowvec, double, rowvec, field<mat>, rowvec);
	void run(double);
};

