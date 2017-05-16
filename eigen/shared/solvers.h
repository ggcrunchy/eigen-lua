/*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
* [ MIT license: http://www.opensource.org/licenses/mit-license.php ]
*/

#pragma once

#include "CoronaLua.h"
#include "macros.h"
#include "types.h"
#include "utils.h"
#include <Eigen/Eigen>
#include <type_traits>
#include <utility>

//
BASIC_UNRAVELER(Eigen::BDCSVD);
BASIC_UNRAVELER(Eigen::ComplexEigenSolver);
BASIC_UNRAVELER(Eigen::ColPivHouseholderQR);
BASIC_UNRAVELER(Eigen::CompleteOrthogonalDecomposition);
BASIC_UNRAVELER(Eigen::ComplexSchur);
BASIC_UNRAVELER(Eigen::EigenSolver);
BASIC_UNRAVELER(Eigen::GeneralizedEigenSolver);
BASIC_UNRAVELER(Eigen::GeneralizedSelfAdjointEigenSolver);
BASIC_UNRAVELER(Eigen::FullPivHouseholderQR);
BASIC_UNRAVELER(Eigen::FullPivLU);
BASIC_UNRAVELER(Eigen::HessenbergDecomposition);
BASIC_UNRAVELER(Eigen::HouseholderQR);
EXTENDED_UNRAVELER(Eigen::JacobiSVD);
EXTENDED_UNRAVELER(Eigen::LDLT);
EXTENDED_UNRAVELER(Eigen::LLT);
BASIC_UNRAVELER(Eigen::PartialPivLU);
BASIC_UNRAVELER(Eigen::RealQZ);
BASIC_UNRAVELER(Eigen::RealSchur);
BASIC_UNRAVELER(Eigen::SelfAdjointEigenSolver);
BASIC_UNRAVELER(Eigen::Tridiagonalization);

//
enum {
	eSolverIDs = eSolverMarker,

	eBDC_SVD,	// Bidiagonal divide-and-conquer SVD
	eCES,	// Complex eigen solver
	eCPH_QR,// Householder QR with column pivoting
	eCOD,	// Complete orthogonal decomposition
	eCS,// Complex Schur
	eES,// Eigen solver
	eFPH_QR,// Householder QR with full pivoting
	eFP_LU,	// LU with complete pivoting
	eGES,	// Generalized eigen solver
	eGSAES,	// Generalized self-adjoint eigen solver
	eHD,// Hessenberg decomposition
	eH_QR,	// Householder QR
	eJCP_SVD,	// Jacobi SVD with column-pivoting QR preconditioner
	eJFP_SVD,	// Jacobi SVD with full-pivoting QR preconditioner
	eJH_SVD,// Jacobi SVD with Householder QR preconditioner
	eJNP_SVD,	// Jacobi SVD without preconditioner
	eLDLT_L,// Lower-triangular LDLT Cholesky
	eLDLT_U,// Upper-triangular LDLT Cholesky
	eLLT_L,	// Lower-triangular LLT Cholesky
	eLLT_U,	// Upper-triangular LLT Cholesky
	ePP_LU,	// LU with partial pivoting
	eRQZ,	// Real QZ
	eRS,// Real Schur
	eSAES,	// Self-adjoint eigen solver
	eT	// Tridiagonalization
};

//
#define UNROLL_SOLVER_UNRAVELERS(T)	ADD_OBJECT_TYPE_ID(Eigen::BDCSVD<T>, eBDC_SVD, "_bdcSVD");														\
									ADD_SOLVER_IF_COMPLEX(Eigen::ComplexEigenSolver<T>, eCES, "_ces");												\
									ADD_OBJECT_TYPE_ID(Eigen::ColPivHouseholderQR<T>, eCPH_QR, "_cphQR");											\
									ADD_OBJECT_TYPE_ID(Eigen::CompleteOrthogonalDecomposition<T>, eCOD, "_cod");									\
									ADD_SOLVER_IF_COMPLEX(Eigen::ComplexSchur<T>, eCS, "_ces");														\
									ADD_SOLVER_IF_REAL(Eigen::EigenSolver<T>, eES, "_es");															\
									ADD_SOLVER_IF_REAL(Eigen::GeneralizedEigenSolver<T>, eGES, "_ges");												\
									ADD_OBJECT_TYPE_ID(Eigen::GeneralizedSelfAdjointEigenSolver<T>, eGSAES, "_gsaes");								\
									ADD_OBJECT_TYPE_ID(Eigen::FullPivHouseholderQR<T>, eFPH_QR, "_fphQR");											\
									ADD_OBJECT_TYPE_ID(Eigen::FullPivLU<T>, eFP_LU, "_fpLU");														\
									ADD_OBJECT_TYPE_ID(Eigen::HessenbergDecomposition<T>, eHD, "_hd");												\
									ADD_OBJECT_TYPE_ID(Eigen::HouseholderQR<T>, eH_QR, "_hQR");														\
									ADD_OBJECT_TYPE_ID(WRAP2(Eigen::JacobiSVD, T, Eigen::ColPivHouseholderQRPreconditioner), eJCP_SVD, "_jcpSVD");	\
									ADD_OBJECT_TYPE_ID(WRAP2(Eigen::JacobiSVD, T, Eigen::FullPivHouseholderQRPreconditioner), eJFP_SVD, "_jfpSVD");	\
									ADD_OBJECT_TYPE_ID(WRAP2(Eigen::JacobiSVD, T, Eigen::HouseholderQRPreconditioner), eJH_SVD, "_jhSVD");			\
									ADD_OBJECT_TYPE_ID(WRAP2(Eigen::JacobiSVD, T, Eigen::NoQRPreconditioner), eJNP_SVD, "_jnpSVD");					\
									ADD_OBJECT_TYPE_ID(WRAP2(Eigen::LDLT, T, Eigen::Lower), eLDLT_L, "_LDLTl");										\
									ADD_OBJECT_TYPE_ID(WRAP2(Eigen::LDLT, T, Eigen::Upper), eLDLT_U, "_LDLTu");										\
									ADD_OBJECT_TYPE_ID(WRAP2(Eigen::LLT, T, Eigen::Lower), eLLT_L, "_LLTl");										\
									ADD_OBJECT_TYPE_ID(WRAP2(Eigen::LLT, T, Eigen::Upper), eLLT_U, "_LLTu");										\
									ADD_OBJECT_TYPE_ID(Eigen::PartialPivLU<T>, ePP_LU, "_ppLU");													\
									ADD_SOLVER_IF_REAL(Eigen::RealQZ<T>, eRQZ, "_rqz");																\
									ADD_SOLVER_IF_REAL(Eigen::RealSchur<T>, eRS, "_rs");															\
									ADD_OBJECT_TYPE_ID(Eigen::SelfAdjointEigenSolver<T>, eSAES, "_saes");											\
									ADD_OBJECT_TYPE_ID(Eigen::Tridiagonalization<T>, eT, "_t")

//
#define ADD_SOLVER_IF_REAL ADD_OBJECT_TYPE_ID
#define ADD_SOLVER_IF_COMPLEX(...)

//
#ifdef WANT_FLOAT
	UNROLL_SOLVER_UNRAVELERS(Eigen::MatrixXf);
#endif

#ifdef WANT_DOUBLE
	UNROLL_SOLVER_UNRAVELERS(Eigen::MatrixXd);
#endif

//
#undef ADD_SOLVER_IF_REAL
#undef ADD_SOLVER_IF_COMPLEX
#define ADD_SOLVER_IF_REAL(...)
#define ADD_SOLVER_IF_COMPLEX ADD_OBJECT_TYPE_ID

//
#ifdef WANT_CFLOAT
	UNROLL_SOLVER_UNRAVELERS(Eigen::MatrixXcf);
#endif

#ifdef WANT_CDOUBLE
	UNROLL_SOLVER_UNRAVELERS(Eigen::MatrixXcd);
#endif

//
template<typename T, typename R, bool = Eigen::NumTraits<R::Scalar>::IsComplex> struct MakeSchur {
	static int Do (lua_State * L)
	{
		T & hd = *GetInstance<T>(L);
		Eigen::ComplexSchur<R> schur;

		schur.computeFromHessenberg(hd.matrixH(), hd.matrixQ(), !WantsBool(L, "no_u"));

		return NewMoveRet<Eigen::ComplexSchur<R>>(L, schur);
	}
};

template<typename T, typename R> struct MakeSchur<T, R, false> {
	static int Do (lua_State * L)
	{
		T & hd = *GetInstance<T>(L);
		Eigen::RealSchur<R> schur;

		schur.computeFromHessenberg(hd.matrixH(), hd.matrixQ(), !WantsBool(L, "no_u"));

		return NewMoveRet<Eigen::RealSchur<R>>(L, schur);
	}
};

//
template<typename T, typename R> struct AttachSolverMethods {
	ADD_INSTANCE_GETTERS()

	//
	template<int> void DoMethod (lua_State * L) {}

	//
	using Real = typename Eigen::NumTraits<typename R::Scalar>::Real;

	//
	template<bool = true> static int SetThreshold (lua_State * L) // dummy template parameter to conditionally compile
	{
		lua_settop(L, 2);	// solver, ..., how
		lua_pushliteral(L, "default");	// solver, ..., how, "default"

		if (!lua_equal(L, 2, 3)) GetT(L)->setThreshold(LuaXS::GetArg<Real>(L, 2));

		else GetT(L)->setThreshold(Eigen::Default_t{});

		return SelfForChaining(L);	// solver, ..., how, "default", solver
	}

	/***********************************
	* Bidiagonal divide-and-conquer SVD
	***********************************/
	template<> void DoMethod<eBDC_SVD> (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_PUSH_VALUE_METHOD(computeU)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(computeV)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixU)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixV)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(nonzeroSingularValues)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(rank)
			}, {
				"setThreshold", SetThreshold<>
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(singularValues)
			}, {
				EIGEN_MATRIX_GET_MATRIX_MATRIX_PAIR_METHOD(solve)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(threshold)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	//
	template<bool = true> static int SetMaxIterations (lua_State * L) // dummy template parameter to conditionally compile
	{
		GetT(L)->setMaxIterations(LuaXS::Int(L, 2));

		return SelfForChaining(L);	// solver, count, solver
	}

	//
	template<bool = true> static int Info (lua_State * L) // dummy template parameter to conditionally compile
	{
		switch (GetT(L)->info())
		{
		case Eigen::Success:
			lua_pushliteral(L, "success");	// ..., "success"
			break;
		case Eigen::NumericalIssue:
			lua_pushliteral(L, "numerical_issue");	// ..., "numerical_issue"
			break;
		default:	// Errors trapped by asserts
			lua_pushliteral(L, "no_convergence");	// ..., "no_convergence"
		}

		return 1;
	}

	/**********************
	* Complex eigen solver
	**********************/
	template<> void DoMethod<eCES> (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_GET_MATRIX_METHOD(eigenvalues)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(eigenvectors)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(getMaxIterations)
			}, {
				"info", Info<>
			}, {
				"setMaxIterations", SetMaxIterations<>
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	//
	void HouseholderBase (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_PUSH_VALUE_METHOD(absDeterminant)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(logAbsDeterminant)
			}, {
				EIGEN_MATRIX_GET_MATRIX_MATRIX_PAIR_METHOD(solve)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	//
	void HouseholderExtensions (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_PUSH_VALUE_METHOD(isInjective)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(isInvertible)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(isSurjective)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(maxPivot)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(nonzeroPivots)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(rank)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(threshold)
			}, {
				"setThreshold", SetThreshold<>
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	//
	void HouseholderEx (lua_State * L)
	{
		HouseholderBase(L);
		HouseholderExtensions(L);
	}

	/*************************************
	* Householder QR with column pivoting
	*************************************/
	template<> void DoMethod<eCPH_QR> (lua_State * L)
	{
		HouseholderEx(L);

		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_GET_MATRIX_METHOD(colsPermutation)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(inverse)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixQR)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixR)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	/***********************************
	* Complete orthogonal decomposition
	***********************************/
	template<> void DoMethod<eCOD> (lua_State * L)
	{
		HouseholderEx(L);

		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_GET_MATRIX_METHOD(householderQ)
			}, {
				"info", Info<>
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixQTZ)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixT)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixZ)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(pseudoInverse)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	/***************
	* Complex Schur
	***************/
	template<> void DoMethod<eCS> (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_PUSH_VALUE_METHOD(getMaxIterations)
			}, {
				"info", Info<>
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixT)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixU)
			}, {
				"setMaxIterations", SetMaxIterations<>
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	/**************
	* Eigen solver
	**************/
	template<> void DoMethod<eES> (lua_State * L)
	{
		USING_COMPLEX_TYPE();

		luaL_Reg methods[] = {
			{
				EIGEN_REAL_GET_COMPLEX_METHOD(eigenvalues)
			}, {
				EIGEN_REAL_GET_COMPLEX_METHOD(eigenvectors)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(getMaxIterations)
			}, {
				"info", Info<>
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(pseudoEigenvalueMatrix)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(pseudoEigenvectors)
			}, {
				"setMaxIterations", SetMaxIterations<>
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	/***********************************
	* Householder QR with full pivoting
	***********************************/
	template<> void DoMethod<eFPH_QR> (lua_State * L)
	{
		HouseholderEx(L);

		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_GET_MATRIX_METHOD(inverse)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixQ)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixQR)
			}, {
				"rowsTranspositions", [](lua_State * L)
				{
					auto td = GetTypeData<Eigen::MatrixXi>(L);

					luaL_argcheck(L, td, 2, "rowsTranspositions() requires int matrices");

					Eigen::MatrixXi im = GetT(L)->rowsTranspositions();

					lua_getref(L, td->mPushRef);	// m, how, push
					lua_pushlightuserdata(L, &im);	// m, how, push, transps
					lua_call(L, 1, 1);	// m, how, transps_im

					return 1;
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	/***************************
	* LU with complete pivoting
	***************************/
	template<> void DoMethod<eFP_LU> (lua_State * L)
	{
		DoMethod<ePP_LU>(L);

		HouseholderExtensions(L);

		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_PUSH_VALUE_METHOD(dimensionOfKernel)
			}, {
				"image", [](lua_State * L)
				{
					return NewMoveRet<R>(L, GetT(L)->image(*GetR(L, 2)));
				}
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(kernel)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(permutationQ)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	/**************************
	* Generalized eigen solver
	**************************/
	template<> void DoMethod<eGES> (lua_State * L)
	{
		USING_COMPLEX_TYPE();

		luaL_Reg methods[] = {
			{
				EIGEN_REAL_GET_COMPLEX_METHOD(alphas)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(betas)
			}, {
				EIGEN_REAL_GET_COMPLEX_METHOD(eigenvalues)
			}, {
				EIGEN_REAL_GET_COMPLEX_METHOD(eigenvectors)
			}, {
				"setMaxIterations", SetMaxIterations<>
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	/***************************************
	* Generalized self-adjoint eigen solver
	***************************************/
	template<> void DoMethod<eGSAES> (lua_State * L)
	{
		DoMethod<eSAES>(L);
	}

	/**************************
	* Hessenberg decomposition
	**************************/
	template<> void DoMethod<eHD> (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_GET_MATRIX_METHOD(householderCoefficients)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixH)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixQ)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(packedMatrix)
			}, {
				"schur", [](lua_State * L)
				{
					return MakeSchur<T, R>::Do(L);
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	/****************
	* Householder QR
	****************/
	template<> void DoMethod<eH_QR> (lua_State * L)
	{
		HouseholderBase(L);

		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_GET_MATRIX_METHOD(householderQ)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixQR)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	/***************************************************
	* Jacobi SVD with column-pivoting QR preconditioner
	***************************************************/
	template<> void DoMethod<eJCP_SVD> (lua_State * L)
	{
		DoMethod<eBDC_SVD>(L);
	}
	
	/*************************************************
	* Jacobi SVD with full-pivoting QR preconditioner
	*************************************************/
	template<> void DoMethod<eJFP_SVD> (lua_State * L)
	{
		DoMethod<eBDC_SVD>(L);
	}

	/***********************************************
	* Jacobi SVD with Householder QR preconditioner
	***********************************************/
	template<> void DoMethod<eJH_SVD> (lua_State * L)
	{
		DoMethod<eBDC_SVD>(L);
	}

	/***********************************
	* Jacobi SVD without preconditioner
	***********************************/
	template<> void DoMethod<eJNP_SVD> (lua_State * L)
	{
		DoMethod<eBDC_SVD>(L);
	}

	//
	void Cholesky (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"adjoint", [](lua_State * L)
				{
					return SelfForChaining(L); // already self-adjoint
				}
			}, {
				"info", Info<>
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixL)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixU)
			}, {
				"rankUpdate", [](lua_State * L)
				{
					GetT(L)->rankUpdate(AsVector<R>::To(L, 2), LuaXS::GetArg<Real>(L, 3));

					return SelfForChaining(L);
				}
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(rcond)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(reconstructedMatrix)
			}, {
				EIGEN_MATRIX_GET_MATRIX_MATRIX_PAIR_METHOD(solve)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	/********************************
	* Lower-triangular LDLT Cholesky
	********************************/
	template<> void DoMethod<eLDLT_L> (lua_State * L)
	{
		Cholesky(L);

		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_PUSH_VALUE_METHOD(isNegative)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(isPositive)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixLDLT)
			}, {
				EIGEN_MATRIX_VOID_METHOD(setZero)
			}, {
				"transpositionsP", [](lua_State * L)
				{
					auto td = GetTypeData<Eigen::MatrixXi>(L);

					luaL_argcheck(L, td, 2, "transpositionsP() requires int matrices");

					Eigen::MatrixXi im = GetT(L)->transpositionsP().indices();

					lua_getref(L, td->mPushRef);	// m, how, push
					lua_pushlightuserdata(L, &im);	// m, how, push, transps
					lua_call(L, 1, 1);	// m, how, transps_im

					return 1;
				}
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(vectorD)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	/********************************
	* Upper-triangular LDLT Cholesky
	********************************/
	template<> void DoMethod<eLDLT_U> (lua_State * L)
	{
		DoMethod<eLDLT_L>(L);
	}

	/*******************************
	* Lower-triangular LLT Cholesky
	*******************************/
	template<> void DoMethod<eLLT_L> (lua_State * L)
	{
		Cholesky(L);

		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixLLT)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	/*******************************
	* Upper-triangular LLT Cholesky
	*******************************/
	template<> void DoMethod<eLLT_U> (lua_State * L)
	{
		DoMethod<eLLT_L>(L);
	}

	/**************************
	* LU with partial pivoting
	**************************/
	template<> void DoMethod<ePP_LU> (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_PUSH_VALUE_METHOD(determinant)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(inverse)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixLU)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(permutationP)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(rcond)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(reconstructedMatrix)
			}, {
				EIGEN_MATRIX_GET_MATRIX_MATRIX_PAIR_METHOD(solve)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	/*********
	* Real QZ
	*********/
	template<> void DoMethod<eRQZ> (lua_State * L)
	{
		luaL_Reg methods[] = {
			 {
				"info", Info<>
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(iterations)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixQ)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixS)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixT)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixZ)
			}, {
				"setMaxIterations", SetMaxIterations<>
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	/************
	* Real Schur
	************/
	template<> void DoMethod<eRS> (lua_State * L)
	{
		DoMethod<eCS>(L);
	}

	/***************************
	* Self-adjoint eigen solver
	***************************/
	template<> void DoMethod<eSAES> (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_GET_MATRIX_METHOD(eigenvalues)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(eigenvectors)
			}, {
				"info", Info<>
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(operatorInverseSqrt)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(operatorSqrt)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	/********************
	* Tridiagonalization
	********************/
	template<> void DoMethod<eT> (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_GET_MATRIX_METHOD(diagonal)
			}, {
				"generalizedSelfAdjointEigenSolver", [](lua_State * L)
				{
					T & tri = *GetT(L);
					Eigen::GeneralizedSelfAdjointEigenSolver<R> gsaes;
					
					gsaes.computeFromTridiagonal(tri.diagonal(), tri.subDiagonal(), WantsBool(L, "no_eigenvectors") ? Eigen::EigenvaluesOnly : Eigen::ComputeEigenvectors);

					return NewMoveRet<Eigen::GeneralizedSelfAdjointEigenSolver<R>>(L, gsaes);
				}
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(householderCoefficients)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixQ)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixT)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(packedMatrix)
			}, {
				"selfAdjointEigenSolver", [](lua_State * L)
				{
					T & tri = *GetT(L);
					Eigen::SelfAdjointEigenSolver<R> saes;
					
					saes.computeFromTridiagonal(tri.diagonal(), tri.subDiagonal(), WantsBool(L, "no_eigenvectors") ? Eigen::EigenvaluesOnly : Eigen::ComputeEigenvectors);

					return NewMoveRet<Eigen::SelfAdjointEigenSolver<R>>(L, saes);
				}
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(subDiagonal)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	//
	AttachSolverMethods (lua_State * L)
	{
		DoMethod<ObjectTypeID<T>::kID>(L);
	}
};