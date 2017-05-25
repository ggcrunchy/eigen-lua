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
#include "types.h"
#include "utils.h"
#include "macros.h"
#include "solvers.h"
#include "self_adjoint_view.h"
#include "triangular_view.h"
#include <type_traits>

//
template<typename T, typename R, bool = !Eigen::NumTraits<T::Scalar>::IsInteger> struct NonIntMethods {
	ADD_INSTANCE_GETTERS()

	template<typename T> struct UseTemporary : std::false_type {};
	template<typename U> struct UseTemporary<Eigen::Transpose<U>> : std::true_type {};
	template<typename U, int O, typename S> struct UseTemporary<Eigen::Map<U, O, S>> : std::true_type {};

	//
	template<typename bool = UseTemporary<T>::value> struct MatrixRef {
		R mMat;

		MatrixRef (lua_State * L, int arg = 1) : mMat{*GetT(L, arg)}
		{
		}

		R & operator * (void) { return mMat; }
		R * operator -> (void) { return &mMat; }
	};

	template<> struct MatrixRef<false> {
		R * mPtr;

		MatrixRef (lua_State * L, int arg = 1) : mPtr{GetT(L, arg)}
		{
		}

		R & operator * (void) { return *mPtr; }
		R * operator -> (void) { return mPtr; }
	};

	//
	template<bool = Eigen::NumTraits<T::Scalar>::IsComplex> void IgnoreWhenComplex (lua_State * L) {}

	//
	template<> void IgnoreWhenComplex<true> (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"eigenSolver", [](lua_State * L)
				{
					return NewRet<Eigen::ComplexEigenSolver<R>>(L, Eigen::ComplexEigenSolver<R>{*MatrixRef<>{L}, !WantsBool(L, "no_eigenvectors")});
				}
			}, {
				"schur", [](lua_State * L)
				{
					return NewRet<Eigen::ComplexSchur<R>>(L, Eigen::ComplexSchur<R>{*MatrixRef<>{L}, !WantsBool(L, "no_u")});
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	template<> void IgnoreWhenComplex<false> (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"eigenSolver", [](lua_State * L)
				{
					return NewRet<Eigen::EigenSolver<R>>(L, Eigen::EigenSolver<R>{*MatrixRef<>{L}, !WantsBool(L, "no_eigenvectors")});
				}
			}, {
				"generalizedEigenSolver", [](lua_State * L)
				{
					return NewRet<Eigen::GeneralizedEigenSolver<R>>(L, Eigen::GeneralizedEigenSolver<R>{*MatrixRef<>{L}, GetR(L, 2), !WantsBool(L, "no_eigenvectors")});
				}
			}, {
				"qz", [](lua_State * L)
				{
					return NewRet<Eigen::RealQZ<R>>(L, Eigen::RealQZ<R>{*MatrixRef<>{L}, GetR(L, 2), !WantsBool(L, "no_qz")});
				}
			}, {
				"schur", [](lua_State * L)
				{
					return NewRet<Eigen::RealSchur<R>>(L, Eigen::RealSchur<R>{*MatrixRef<>{L}, !WantsBool(L, "no_u")});
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	//
	static unsigned int GetOpts (lua_State * L)
	{
		const char * names[] = { "full_u", "thin_u", "full_v", "thin_v", nullptr };
		int flags[] = { Eigen::ComputeFullU, Eigen::ComputeThinU, Eigen::ComputeFullV, Eigen::ComputeThinV };
		unsigned int opts = 0;

		for (size_t i = 1, n = lua_objlen(L, 2); i <= n; ++i, lua_pop(L, 1))
		{
			lua_rawgeti(L, 2, int(i));	// mat, t, flag

			opts |= flags[luaL_checkoption(L, 3, nullptr, names)];
		}

		return opts;
	}

	//
	NonIntMethods (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"bdcSvd", [](lua_State * L)
				{
					return NewRet<Eigen::BDCSVD<R>>(L, MatrixRef<>{L}->bdcSvd(lua_istable(L, 2) ? GetOpts(L) : 0U));
				}
			}, {
				"colPivHouseholderQr", [](lua_State * L)
				{
					return NewRet<Eigen::ColPivHouseholderQR<R>>(L, MatrixRef<>{L}->colPivHouseholderQr());
				}
			}, {
				"completeOrthogonalDecomposition", [](lua_State * L)
				{
					return NewRet<Eigen::CompleteOrthogonalDecomposition<R>>(L, MatrixRef<>{L}->completeOrthogonalDecomposition());
				}
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(determinant)
			}, {
				"fullPivHouseholderQr", [](lua_State * L)
				{
					return NewRet<Eigen::FullPivHouseholderQR<R>>(L, MatrixRef<>{L}->fullPivHouseholderQr());
				}
			}, {
				"fullPivLu", [](lua_State * L)
				{
					return NewRet<Eigen::FullPivLU<R>>(L, MatrixRef<>{L}->fullPivLu());
				}
			}, {
				"generalizedSelfAdjointEigenSolver", [](lua_State * L)
				{
					auto compute = Eigen::ComputeEigenvectors;
					auto method = Eigen::Ax_lBx;

					if (lua_istable(L, 3))
					{
						lua_getfield(L, 3, "no_eigenvectors");	// a, b, opts, no_eigenvectors?

						if (lua_toboolean(L, -1)) compute = Eigen::EigenvaluesOnly;

						lua_getfield(L, 3, "method");	// a, b, opts, no_eigenvectors?, method

						const char * names[] = { "ABx_lx", "Ax_lBx", "BAx_lx", nullptr };
						decltype(method) methods[] = {Eigen::Ax_lBx, Eigen::Ax_lBx, Eigen::BAx_lx };

						method = methods[luaL_checkoption(L, -1, "", names)];
					}

					return NewRet<Eigen::GeneralizedSelfAdjointEigenSolver<R>>(L, Eigen::GeneralizedSelfAdjointEigenSolver<R>{*MatrixRef<>{L}, GetR(L, 2), compute | method});
				}
			}, {
				"hessenbergDecomposition", [](lua_State * L)
				{
					return NewRet<Eigen::HessenbergDecomposition<R>>(L, Eigen::HessenbergDecomposition<R>{*MatrixRef<>{L}});
				}
			}, {
				"householderQr", [](lua_State * L)
				{
					return NewRet<Eigen::HouseholderQR<R>>(L, MatrixRef<>{L}->householderQr());
				}
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(inverse)
			}, {
				"jacobiSvd", [](lua_State * L)
				{
					unsigned int opts = 0U;

					if (lua_istable(L, 2))
					{
						opts = GetOpts(L);

						lua_getfield(L, 2, "preconditioner");	// mat, opts, precond
						lua_replace(L, 2);	// mat, precond
					}

					const char * choices[] = { "", "fullPiv", "householder", "none", nullptr };

					using FPSVD = Eigen::JacobiSVD<R, Eigen::FullPivHouseholderQRPreconditioner>;
					using HSVD = Eigen::JacobiSVD<R, Eigen::HouseholderQRPreconditioner>;
					using NPSVD = Eigen::JacobiSVD<R, Eigen::NoQRPreconditioner>;

					MatrixRef<> mr{L};

					switch (luaL_checkoption(L, 2, "", choices))
					{
					case 0:
						return NewRet<Eigen::JacobiSVD<R>>(L, mr->jacobiSvd(opts));
					case 1:
						return NewRet<FPSVD>(L, FPSVD{*mr, opts});
					case 2:
						return NewRet<HSVD>(L, HSVD{*mr, opts});
					default: // luaL_checkoption will catch anything else
						return NewRet<NPSVD>(L, NPSVD{*mr, opts});
					}
				}
			}, {
				"ldlt", [](lua_State * L)
				{
					lua_settop(L, 2);	// mat, how?
					lua_pushliteral(L, "upper");// mat[, how], "upper"

					MatrixRef<> mr{L};

					if (!lua_equal(L, 2, 3)) return NewRet<Eigen::LDLT<R, Eigen::Lower>>(L, mr->ldlt());

					else return NewRet<Eigen::LDLT<R, Eigen::Upper>>(L, Eigen::LDLT<R, Eigen::Upper>{*mr});
				}
			}, {
				"llt", [](lua_State * L)
				{
					lua_settop(L, 2);	// mat, how?
					lua_pushliteral(L, "upper");// mat[, how], "upper"

					MatrixRef<> mr{L};

					if (!lua_equal(L, 2, 3)) return NewRet<Eigen::LLT<R, Eigen::Lower>>(L, mr->llt());

					else return NewRet<Eigen::LLT<R, Eigen::Upper>>(L, Eigen::LLT<R, Eigen::Upper>{*mr});
				}
			}, {
				"partialPivLu", [](lua_State * L)
				{
					return NewRet<Eigen::PartialPivLU<R>>(L, MatrixRef<>{L}->partialPivLu());
				}
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(operatorNorm)
			}, {
				"selfAdjointEigenSolver", [](lua_State * L)
				{
					auto opts = WantsBool(L, "no_eigenvectors") ? Eigen::EigenvaluesOnly : Eigen::ComputeEigenvectors;

					return NewRet<Eigen::SelfAdjointEigenSolver<R>>(L, Eigen::SelfAdjointEigenSolver<R>{*MatrixRef<>{L}, opts});
				}
			}, {
				"selfadjointView", [](lua_State * L)
				{
					const char * names[] = { "lower", "upper", nullptr };

					switch (luaL_checkoption(L, 2, nullptr, names))
					{
					case 0:	// Lower-triangular
						New<Eigen::SelfAdjointView<T, Eigen::Lower>>(L, GetT(L)->selfadjointView<Eigen::Lower>());	// mat[, opt], sav
						GetTypeData<Eigen::SelfAdjointView<T, Eigen::Lower>>(L)->RefAt(L, "sav_viewed_from", 1);

						break;
					default:// Upper-triangular
						New<Eigen::SelfAdjointView<T, Eigen::Upper>>(L, GetT(L)->selfadjointView<Eigen::Upper>());	// mat[, opt], sav
						GetTypeData<Eigen::SelfAdjointView<T, Eigen::Upper>>(L)->RefAt(L, "sav_viewed_from", 1);

						break;
					}

					return 1;
				}
			}, {
				"triangularView", [](lua_State * L)
				{
					const char * names[] = { "lower", "strictly_lower", "strictly_upper", "unit_lower", "unit_upper", "upper", nullptr };

					switch (luaL_checkoption(L, 2, nullptr, names))
					{
					case 0:	// Lower-triangular
						New<Eigen::TriangularView<T, Eigen::Lower>>(L, GetT(L)->triangularView<Eigen::Lower>());// mat[, opt], tv
						GetTypeData<Eigen::TriangularView<T, Eigen::Lower>>(L)->RefAt(L, "tv_viewed_from", 1);

						break;
					case 1:// Strictly lower-triangular
						New<Eigen::TriangularView<T, Eigen::StrictlyLower>>(L, GetT(L)->triangularView<Eigen::StrictlyLower>());// mat[, opt], tv
						GetTypeData<Eigen::TriangularView<T, Eigen::StrictlyLower>>(L)->RefAt(L, "tv_viewed_from", 1);

						break;
					case 2:// Strictly upper-triangular
						New<Eigen::TriangularView<T, Eigen::StrictlyUpper>>(L, GetT(L)->triangularView<Eigen::StrictlyUpper>());// mat[, opt], tv
						GetTypeData<Eigen::TriangularView<T, Eigen::StrictlyUpper>>(L)->RefAt(L, "tv_viewed_from", 1);

						break;
					case 3:// Upper-triangular
						New<Eigen::TriangularView<T, Eigen::UnitLower>>(L, GetT(L)->triangularView<Eigen::UnitLower>());// mat[, opt], tv
						GetTypeData<Eigen::TriangularView<T, Eigen::UnitLower>>(L)->RefAt(L, "tv_viewed_from", 1);

						break;
					case 4:// Unit lower-triangular
						New<Eigen::TriangularView<T, Eigen::UnitUpper>>(L, GetT(L)->triangularView<Eigen::UnitUpper>());// mat[, opt], tv
						GetTypeData<Eigen::TriangularView<T, Eigen::UnitUpper>>(L)->RefAt(L, "tv_viewed_from", 1);

						break;
					default:// Unit upper-triangular
						New<Eigen::TriangularView<T, Eigen::Upper>>(L, GetT(L)->triangularView<Eigen::Upper>());// mat[, opt], tv
						GetTypeData<Eigen::TriangularView<T, Eigen::Upper>>(L)->RefAt(L, "tv_viewed_from", 1);

						break;
					}

					return 1;
				}
			}, {
				"tridiagonalization", [](lua_State * L)
				{
					return NewRet<Eigen::Tridiagonalization<R>>(L, Eigen::Tridiagonalization<R>{*MatrixRef<>{L}});
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
		lua_getfield(L, -1, "partialPivLu");// methods, pplu
		lua_setfield(L, -2, "lu");	// methods = { lu = pplu }

		IgnoreWhenComplex(L);
	}
};

//
template<typename T, typename R> struct NonIntMethods<T, R, false> {
	NonIntMethods (lua_State *) {}
};