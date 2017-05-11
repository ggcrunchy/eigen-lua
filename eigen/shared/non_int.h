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
#include <type_traits>

//
template<typename T, typename R> struct NonIntMethods {
	//
	ADD_INSTANCE_GETTERS()

	//
	NonIntMethods (lua_State * L)
	{
		Add(L);
	}

	//
	template<bool = std::is_integral<T::Scalar>::value> void Add (lua_State *) {}

	//
	template<bool = std::is_same<T::Scalar, int>::value, bool = Eigen::NumTraits<T::Scalar>::IsComplex> void IgnoreWhenComplex(lua_State * L) {}

	//
	template<> void IgnoreWhenComplex<false, true>(lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"eigenSolver", [](lua_State * L)
				{
					return NewMoveRet<Eigen::ComplexEigenSolver<R>>(L, Eigen::ComplexEigenSolver<R>{*GetT(L), !WantsBool(L, "no_eigenvectors")});
				}
			}, {
				"schur", [](lua_State * L)
				{					
					return NewMoveRet<Eigen::ComplexSchur<R>>(L, Eigen::ComplexSchur<R>{*GetT(L), !WantsBool(L, "no_u")});
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	template<> void IgnoreWhenComplex<false, false>(lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"eigenSolver", [](lua_State * L)
				{
					return NewMoveRet<Eigen::EigenSolver<R>>(L, Eigen::EigenSolver<R>{*GetT(L), !WantsBool(L, "no_eigenvectors")});
				}
			}, {
				"generalizedEigenSolver", [](lua_State * L)
				{
					return NewMoveRet<Eigen::GeneralizedEigenSolver<R>>(L, Eigen::GeneralizedEigenSolver<R>{*GetT(L), *GetR(L, 2), !WantsBool(L, "no_eigenvectors")});
				}
			}, {
				"qz", [](lua_State * L)
				{
					return NewMoveRet<Eigen::RealQZ<R>>(L, Eigen::RealQZ<R>{*GetT(L), *GetR(L, 2), !WantsBool(L, "no_qz")});
				}
			}, {
				"schur", [](lua_State * L)
				{
					return NewMoveRet<Eigen::RealSchur<R>>(L, Eigen::RealSchur<R>{*GetT(L), !WantsBool(L, "no_u")});
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
	template<> void Add<false> (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"bdcSvd", [](lua_State * L)
				{
					return NewMoveRet<Eigen::BDCSVD<R>>(L, GetT(L)->bdcSvd(lua_istable(L, 2) ? GetOpts(L) : 0U));
				}
			}, {
				"colPivHouseholderQr", [](lua_State * L)
				{
					return NewMoveRet<Eigen::ColPivHouseholderQR<R>>(L, GetT(L)->colPivHouseholderQr());
				}
			}, {
				"completeOrthogonalDecomposition", [](lua_State * L)
				{
					return NewMoveRet<Eigen::CompleteOrthogonalDecomposition<R>>(L, GetT(L)->completeOrthogonalDecomposition());
				}
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(determinant)
			}, {
				"fullPivHouseholderQr", [](lua_State * L)
				{
					return NewMoveRet<Eigen::FullPivHouseholderQR<R>>(L, GetT(L)->fullPivHouseholderQr());
				}
			}, {
				"fullPivLu", [](lua_State * L)
				{
					return NewMoveRet<Eigen::FullPivLU<R>>(L, GetT(L)->fullPivLu());
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

					return NewMoveRet<Eigen::GeneralizedSelfAdjointEigenSolver<R>>(L, Eigen::GeneralizedSelfAdjointEigenSolver<R>{*GetT(L), *GetR(L, 2), compute | method});
				}
			}, {
				"hessenbergDecomposition", [](lua_State * L)
				{
					return NewMoveRet<Eigen::HessenbergDecomposition<R>>(L, Eigen::HessenbergDecomposition<R>{*GetT(L)});
				}
			}, {
				"householderQr", [](lua_State * L)
				{
					return NewMoveRet<Eigen::HouseholderQR<R>>(L, GetT(L)->householderQr());
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

					switch (luaL_checkoption(L, 2, "", choices))
					{
					case 0:
						return NewMoveRet<Eigen::JacobiSVD<R>>(L, GetT(L)->jacobiSvd(opts));
					case 1:
						return NewMoveRet<FPSVD>(L, FPSVD{*GetT(L), opts});
					case 2:
						return NewMoveRet<HSVD>(L, HSVD{*GetT(L), opts});
					default: // luaL_checkoption will catch anything else
						return NewMoveRet<NPSVD>(L, NPSVD{*GetT(L), opts});
					}
				}
			}, {
				"ldlt", [](lua_State * L)
				{
					lua_settop(L, 2);	// mat, how?
					lua_pushliteral(L, "upper");// mat[, how], "upper"

					if (!lua_equal(L, 2, 3)) return NewMoveRet<Eigen::LDLT<R, Eigen::Lower>>(L, GetT(L)->ldlt());

					else
					{
						Eigen::LDLT<R, Eigen::Upper> ldlt{*GetT(L)};

						return NewMoveRet<Eigen::LDLT<R, Eigen::Upper>>(L, ldlt);
					}
				}
			}, {
				"llt", [](lua_State * L)
				{
					lua_settop(L, 2);	// mat, how?
					lua_pushliteral(L, "upper");// mat[, how], "upper"

					if (!lua_equal(L, 2, 3)) return NewMoveRet<Eigen::LLT<R, Eigen::Lower>>(L, GetT(L)->llt());

					else
					{
						Eigen::LLT<R, Eigen::Upper> llt{*GetT(L)};

						return NewMoveRet<Eigen::LLT<R, Eigen::Upper>>(L, llt);
					}
				}
			}, {
				"partialPivLu", [](lua_State * L)
				{
					return NewMoveRet<Eigen::PartialPivLU<R>>(L, GetT(L)->partialPivLu());
				}
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(operatorNorm)
			}, {
				"selfAdjointEigenSolver", [](lua_State * L)
				{
					auto opts = WantsBool(L, "no_eigenvectors") ? Eigen::EigenvaluesOnly : Eigen::ComputeEigenvectors;

					return NewMoveRet<Eigen::SelfAdjointEigenSolver<R>>(L, Eigen::SelfAdjointEigenSolver<R>{*GetT(L), opts});
				}
			}, {
				"tridiagonalization", [](lua_State * L)
				{
					return NewMoveRet<Eigen::Tridiagonalization<R>>(L, Eigen::Tridiagonalization<R>{*GetT(L)});
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