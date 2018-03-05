#include "stdafx.h"
#include "GibbsSampler.h"


GibbsSampler::GibbsSampler()
{
}

GibbsSampler::~GibbsSampler()
{
}

void GibbsSampler::display(void)
{

}

void GibbsSampler::init(double dd, double DD, rowvec NN, double KK, rowvec MM, field<mat> XX, rowvec aa)
{
	/*Initialise Gaussians*/
	field< field<Gaussian> > gg(KK, 1);

	for (uword k = 0; k < KK; ++k)
	{
		field<Gaussian> g(1, MM(k));
		for (uword m = 0; m < MM(k); ++m)
		{
			// Parameters for Gaussian prior
			double var = 3.0;
			double dof = 10.0;
			mat VV(dd, dd, fill::eye);
			colvec uu(dd, fill::zeros);
			
			g(m) = Gaussian();
			g(m).init(dd, var, VV, dof, uu);
		}
		gg(k) = g;
	}
	/////////////////////////

	/*Initialise Z and Xi vectors, and add respective data items to Gaussians*/
	field<rowvec> ZZ(DD, 1);
	field<rowvec> XiXi(DD, 1);
	field< field<rowvec> > nn(DD, 1);

	arma::arma_rng::set_seed_random();
	for (uword d = 0; d < DD; ++d)
	{
		mat xx_d = XX(d);
		rowvec zz = randi<rowvec>(NN(d), distr_param(1.0, KK));
		rowvec xixi(NN(d), fill::zeros);
		field<rowvec> nn_d(KK, 1);

		for (uword k = 0; k < uword(KK); ++k)
			nn_d(k).zeros(MM(k));

		nn(d) = nn_d;

		for (uword i = 0; i < NN(d); ++i)
		{
			double k = zz(i);
			/*remember C++ uses zero indexing*/
			vec xi = randi<vec>(1, distr_param(1.0, MM(k - 1)));
			double m = xi(0);

			gg(k - 1)(m - 1).additem(xx_d.col(i));
			nn(d)(k - 1)(m - 1)++;
			xixi(i) = m;
		}
		ZZ(d) = zz;
		XiXi(d) = xixi;
	}

	/////////////////////////
	/*Initialise nCRP prior for each level*/
	p_aa = aa;
	/////////////////////////
	p_dd = dd;
	p_DD = DD;
	p_NN = NN;
	p_KK = KK;
	p_MM = MM;
	p_XX = XX;
	p_GG = gg;
	p_nn = nn;
	p_ZZ = ZZ;
	p_XiXi = XiXi;
}

void GibbsSampler::run(double numiters)
{
	field<rowvec> probs_km(p_KK, 1);
	for (uword k = 0; k < probs_km.n_elem; ++k)
	{
		rowvec p_m(p_MM(k), fill::zeros);
		probs_km(k) = p_m;
	}
	rowvec N_k(p_KK, fill::zeros);
	rowvec z_k(p_KK, fill::zeros);

	wall_clock timer;

	field<rowvec> nn_d;
	mat xx_d;

	uword this_k, this_m, new_k, new_m;
	double uu;
	vec km_u;

	/*std::ofstream outFile;
	outFile.open("C:\\Users\\ksp6\\Desktop\\mxDebug.txt", std::ofstream::out | std::ofstream::app);*/

	for (double iter = 1; iter <= numiters; ++iter)
	{
		mexPrintf("gibbs iter %d of %d: ", int(iter), int(numiters));
		timer.tic();

		for (uword dd = 0; dd < p_DD; ++dd)
		{
			arma::arma_rng::set_seed_random();
			nn_d = p_nn(dd);
			for (uword ii = 0; ii < p_NN(dd); ++ii)
			{
				/*outFile << "iter=" << iter << ", D=" << dd << ", i=" << ii << endl;*/
				
				/*get current gmm and component assignment*/
				this_k = p_ZZ(dd)(ii);
				this_m = p_XiXi(dd)(ii);
				/////////////////

				/*remove sufficient stats for given gmm and component*/
				p_GG(this_k - 1)(this_m - 1).delitem(p_XX(dd).col(ii));
				nn_d(this_k - 1)(this_m - 1)--;
				/////////////////

				/*calculate probabilities*/
				for (size_t k = 0; k < p_KK; ++k)
					N_k(k) = sum(nn_d(k));
				
				for (size_t k = 0; k < p_KK; ++k)
				{
					for (size_t m = 0; m < p_MM(k); ++m)
					{
						probs_km(k)(m) =
							log((N_k(k) + p_aa(0)) / (sum(N_k) + p_KK * p_aa(0) - 1))
							+ log((nn_d(k)(m) + p_aa(1)) / (sum(nn_d(k)) + p_MM(k) * p_aa(1) - 1))
							+ p_GG(k)(m).logpredictive(p_XX(dd).col(ii));
						
						/*outFile << "k=" << k << ", m=" << m << ", first term = " << log((N_k(k) + p_aa(0)) / (sum(N_k) + p_KK * p_aa(0) - 1)) << ", second term = " << log((nn_d(k)(m) + p_aa(1)) / (sum(nn_d(k)) + p_MM(k) * p_aa(1) - 1)) << ", logpred = " << p_GG(k)(m).logpredictive(p_XX(dd).col(ii)) << ", p_km = " << probs_km(k)(m) << endl;*/
					}
					z_k(k) = sum(probs_km(k));
				}
				/////////////////

				/*sample new gmm assignment*/
				z_k = exp(z_k - max(z_k));
				z_k = z_k / sum(z_k);
				
				km_u = randu<vec>(2);
				uu = km_u(0);
				new_k = 1 + sum(uu > cumsum(z_k));
				/*outFile << "z=" << z_k << ", k=" << new_k << endl;*/
				/////////////////

				/*given new gmm assignment, sample a component*/
				rowvec xi_m = exp(probs_km(new_k - 1) - max(probs_km(new_k - 1)));
				xi_m = xi_m / sum(xi_m);

				/*km_u = randu<vec>(1);*/
				uu = km_u(1);
				new_m = 1 + sum(uu > cumsum(xi_m));

				/*outFile << "xi=" << xi_m << ", m=" << new_m << endl;*/

				/////////////////

				/*save new assignments*/
				p_ZZ(dd)(ii) = new_k;
				p_XiXi(dd)(ii) = new_m;
				/////////////////

				/*update counts and add sufficient stats to respective Gaussian objects*/
				p_GG(new_k - 1)(new_m - 1).additem(p_XX(dd).col(ii));
				nn_d(new_k - 1)(new_m - 1)++;
				/////////////////
				/*outFile << "---------------------" << endl;*/
			}
			p_nn(dd) = nn_d;
		}

		/*mexCallMATLAB(0, NULL, 0, NULL, "clc");*/
		mexPrintf("%3.5f sec\n", timer.toc());
		ioFlush();
	}
	/*outFile.close();*/
}