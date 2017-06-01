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
#include "xprs.h"
#include "arith_ops.h"
#include "stock_ops.h"
#include "write_ops.h"
#include "xpr_ops.h"
#include "macros.h"
#include <type_traits>

// Methods assigned to matrices in general.
// TODO: maybe a vector_dependent is in order?
template<typename T, typename R> struct CommonMethods : InstanceGetters<T, R> {
	// Helper to cast the matrix to another type, which may be in another shared library.
	template<typename U> struct Cast {
		using MT = MatrixOf<U>;

		template<bool = !Eigen::NumTraits<T::Scalar>::IsComplex, bool = !Eigen::NumTraits<U>::IsComplex> MT CastTo (lua_State * L)
		{
			return GetR(L).template cast<U>();
		}

		template<> MT CastTo<false, true> (lua_State * L)
		{
			R mat = GetR(L);
			MT out;

			out.resizeLike(mat);

			struct VisitCoeffs {
				MT & mOut;

				VisitCoeffs (MT & out) : mOut{out}
				{
				}

				inline void init (T::Scalar x, Eigen::Index i, Eigen::Index j)
				{
					mOut(i, j) = static_cast<U>(x.real());
				}

				inline void operator ()(T::Scalar x, Eigen::Index i, Eigen::Index j)
				{
					init(x, i, j);
				}
			} v{out};

			mat.visit(v);

			return out;
		}

		Cast (lua_State * L)
		{
			auto td = TypeData<MT>::Get(L, GetTypeData::eFetchIfMissing);

			luaL_argcheck(L, td, 2, "Matrix type unavailable for cast");

			MT m = CastTo(L); // mat, new_mat

			PUSH_TYPED_DATA_NO_RET(m);
		}
	};

	// Helper to query the precision used for comparing matrices from the stack.
	static typename Eigen::NumTraits<typename T::Scalar>::Real GetPrecision (lua_State * L, int arg)
	{
		auto prec = Eigen::NumTraits<typename T::Scalar>::dummy_precision();

		return !lua_isnoneornil(L, arg) ? LuaXS::GetArg<decltype(prec)>(L, arg) : prec;
	}

	// Typical form of methods returning a boolean.
	#define EIGEN_MATRIX_PREDICATE(METHOD) return LuaXS::PushArgAndReturn(L, GetT(L)->METHOD(GetPrecision(L, 2)))	
	#define EIGEN_MATRIX_PREDICATE_METHOD(NAME) EIGEN_REG(NAME, EIGEN_MATRIX_PREDICATE(NAME))

	CommonMethods (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_ARRAY_METHOD(acos)
			}, {
				"add", [](lua_State * L)
				{
					return NewRet<R>(L, *GetT(L) + GetR(L, 2));
				}
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(adjoint)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(allFinite)
			}, {
				EIGEN_ARRAY_METHOD(arg)
			}, {
				EIGEN_ARRAY_METHOD(asin)
			}, {
				EIGEN_ARRAY_METHOD(atan)
			}, {
				"cast", [](lua_State * L)
				{
					const char * types[] = { "int", "float", "double", "cfloat", "cdouble", nullptr };
					int which = luaL_checkoption(L, 2, nullptr, types);

					switch (which)
					{
					case 0:	// int
						Cast<int>{L};	// mat, im
						break;
					case 1:	// float
						Cast<float>{L};	// mat, fm
						break;
					case 2:	// double
						Cast<double>{L};// mat, dm
						break;
					case 3:	// cfloat
						Cast<std::complex<float>>{L};	// mat, cfm
						break;
					case 4:	// cdouble
						Cast<std::complex<double>>{L};	// mat, cdm
						break;
					}

					return 1;
				}
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(conjugate)
			}, {
				EIGEN_ARRAY_METHOD(cos)
			}, {
				EIGEN_ARRAY_METHOD(cosh)
			}, {
				EIGEN_ARRAY_METHOD(cube)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(cwiseAbs)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(cwiseAbs2)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(cwiseInverse)
			}, {
				EIGEN_MATRIX_GET_MATRIX_MATRIX_PAIR_METHOD(cwiseProduct)
			}, {
				EIGEN_MATRIX_GET_MATRIX_MATRIX_PAIR_METHOD(cwiseQuotient)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(cwiseSign)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(cwiseSqrt)
			}, {
				"dot", [](lua_State * L)
				{
					return LuaXS::PushArgAndReturn(L, ColumnVector<R>{L}->dot(*ColumnVector<R>{L, 2}));
				}
			}, {
				EIGEN_ARRAY_METHOD(exp)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(hasNaN)
			}, {
				"isApprox", [](lua_State * L)
				{
					return LuaXS::PushArgAndReturn(L, GetT(L)->isApprox(GetR(L, 2), GetPrecision(L, 3)));
				}
			}, {
				"isConstant", [](lua_State * L)
				{
					return LuaXS::PushArgAndReturn(L, GetT(L)->isConstant(AsScalar<R>(L, 2), GetPrecision(L, 3)));
				}
			}, {
				EIGEN_MATRIX_PREDICATE_METHOD(isDiagonal)
			}, {
				EIGEN_ARRAY_METHOD_BOOL(isFinite)
			}, {
				EIGEN_MATRIX_PREDICATE_METHOD(isIdentity)
			}, {
				EIGEN_ARRAY_METHOD_BOOL(isInf)
			}, {
				EIGEN_MATRIX_PREDICATE_METHOD(isLowerTriangular)
			}, {
				EIGEN_MATRIX_PREDICATE_METHOD(isMuchSmallerThan)
			}, {
				EIGEN_ARRAY_METHOD_BOOL(isNaN)
			}, {
				EIGEN_MATRIX_PREDICATE_METHOD(isOnes)
			}, {
				"isOrthogonal", [](lua_State * L)
				{
					return LuaXS::PushArgAndReturn(L, ColumnVector<R>{L}->isOrthogonal(*ColumnVector<R>{L, 2}, GetPrecision(L, 3)));
				}
			}, {
				EIGEN_MATRIX_PREDICATE_METHOD(isUnitary)
			}, {
				EIGEN_MATRIX_PREDICATE_METHOD(isUpperTriangular)
			}, {
				EIGEN_MATRIX_PREDICATE_METHOD(isZero)
			}, {
				EIGEN_ARRAY_METHOD(log)
			}, {
				EIGEN_ARRAY_METHOD(log10)
			}, {
				"lp1Norm", [](lua_State * L)
				{
					EIGEN_MATRIX_PUSH_VALUE(lpNorm<1>);
				}
			}, {
				"lpInfNorm", [](lua_State * L)
				{
					EIGEN_MATRIX_PUSH_VALUE(lpNorm<Eigen::Infinity>);
				}
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(mean)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(norm)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(normalized)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(prod)
			}, {
				EIGEN_ARRAY_METHOD(sin)
			}, {
				EIGEN_ARRAY_METHOD(sinh)
			}, {
				EIGEN_ARRAY_METHOD(square)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(squaredNorm)
			}, {
				"stableNorm", [](lua_State * L)
				{
					return LuaXS::PushArgAndReturn(L, ColumnVector<R>{L}->stableNorm());
				}
			}, {
				"stableNormalized", [](lua_State * L)
				{
					ColumnVector<R> cv{L};

					R nv = cv->stableNormalized();

					cv.RestoreShape(&nv);

					return NewRet<R>(L, nv);
				}
			}, {
				"sub", [](lua_State * L)
				{
					return NewRet<R>(L, *GetT(L) - GetR(L, 2));
				}
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(sum)
			}, {
				EIGEN_ARRAY_METHOD(tan)
			}, {
				EIGEN_ARRAY_METHOD(tanh)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(trace)
			}, {
				"unitOrthogonal", [](lua_State * L)
				{
					ColumnVector<R> cv{L};

					R nv = cv->stableNormalized();

					cv.RestoreShape(&nv);

					return NewRet<R>(L, nv);
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);

		ArithOps<T, R> ao{L};
		StockOps<T, R> so{L};
		WriteOps<T, R> wo{L};
		XprOps<T, R> xo{L};
	}
};