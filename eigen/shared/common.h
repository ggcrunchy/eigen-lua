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
#include "write_ops.h"
#include "xpr_ops.h"
#include "macros.h"
#include <type_traits>

// Common form of arithmetic operations that leave the matrix intact.
#define NO_MUTATE(OP)	auto how = GetVectorwiseOption(L, 3);														\
																													\
						if (how == eNotVectorwise) return NewRet<R>(L, *GetT(L) OP GetR(L, 2));						\
																													\
						else																						\
						{																							\
							if (how == eColwise) return NewRet<R>(L, GetT(L)->colwise() OP AsVector<R>::To(L, 2));	\
							else return NewRet<R>(L, GetT(L)->rowwise() OP AsVector<R>::To(L, 2).transpose());		\
						}																							\
																													\
						return 1

// Common form of operations that transform the contents of a matrix.
#define XFORM(METHOD)	auto how = GetVectorwiseOption(L, 2);										\
																									\
						if (how == eNotVectorwise) return NewRet<R>(L, GetT(L)->METHOD());			\
																									\
						else																		\
						{																			\
							if (how == eColwise) return NewRet<R>(L, GetT(L)->colwise().METHOD());	\
							else return NewRet<R>(L, GetT(L)->rowwise().METHOD());					\
						}																			\
																									\
						return 1

#define XFORM_METHOD(NAME) EIGEN_REG(NAME, XFORM(NAME))

// Methods assigned to matrices in general.
// TODO: maybe a vector_dependent is in order?
template<typename T, typename R> struct CommonMethods {
	ADD_INSTANCE_GETTERS()

	// Helper to cast the matrix to another type, which may be in another shared library.
	template<typename U> struct Cast {
		using MT = MatrixOf<U>;

		template<bool = !Eigen::NumTraits<T::Scalar>::IsComplex, bool = !Eigen::NumTraits<U>::IsComplex> MT CastTo (lua_State * L)
		{
			return GetT(L)->template cast<U>();
		}

		template<> MT CastTo<false, true> (lua_State * L)
		{
			T & m = *GetT(L);
			MT out;

			out.resizeLike(m);

			struct VisitCoeffs {
				MT & mOut;

				VisitCoeffs (MatrixType & out) : mOut{out}
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

			m.visit(v);

			return out;
		}

		Cast (lua_State * L)
		{
			auto td = GetTypeData<MT>(L, eFetchIfMissing);

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

	// Helper to transpose a matrix without needlessly creating types.
	template<typename U> struct Transposer {
		static int Do (lua_State * L)
		{
			New<Eigen::Transpose<U>>(L, GetT(L)->transpose());	// mat, transp
			GetTypeData<Eigen::Transpose<U>>(L)->RefAt(L, "transposed_from", 1);

			return 1;
		}
	};

	template<typename U> struct Transposer<Eigen::Transpose<U>> {
		static int Do (lua_State * L)
		{
			return GetTypeData<T>(L)->GetRef(L, "transposed_from", 1);
		}
	};

	CommonMethods (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_GET_MATRIX_METHOD(adjoint)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(allFinite)
			}, {
				EIGEN_ARRAY_METHOD(acos)
			}, {
				"add", [](lua_State * L)
				{
					NO_MUTATE(+);
				}
			}, {
				EIGEN_ARRAY_METHOD(arg)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(asDiagonal)
			}, {
				"asMatrix", AsMatrix<T, R>
			}, {
				EIGEN_ARRAY_METHOD(asin)
			}, {
				EIGEN_ARRAY_METHOD(atan)
			}, {
				"binaryExpr", [](lua_State * L)
				{
					return NewRet<R>(L, GetT(L)->binaryExpr(GetR(L, 2), [L](const T::Scalar & x, const T::Scalar & y) {
						LuaXS::PushMultipleArgs(L, LuaXS::StackIndex{L, 3}, x, y);	// mat1, mat2, func, func, x, y

						lua_call(L, 2, 1);	// mat1, mat2, func, result

						T::Scalar result(0);

						if (!lua_isnil(L, -1)) result = AsScalar<R>(L, -1);

						lua_pop(L, 1);	// mat1, mat2, func

						return result;
					}));
				}
			}, {
				EIGEN_MATRIX_REDUCE_METHOD(blueNorm)
			}, {
				"__call", [](lua_State * L)
				{
					T & m = *GetT(L);
					int a = LuaXS::Int(L, 2) - 1;
					T::Scalar result;

					if (lua_gettop(L) == 2)
					{
						CheckVector(L, m, 1);

						result = m.cols() == 1 ? m(a, 0) : m(0, a);
					}

					else result = m(a, LuaXS::Int(L, 3) - 1);

					return LuaXS::PushArgAndReturn(L, result);
				}
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
				EIGEN_MATRIX_PUSH_VALUE_METHOD(cols)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(colStride)
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
				EIGEN_REL_OP_METHOD(cwiseEqual, ==)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(cwiseInverse)
			}, {
				EIGEN_REL_OP_METHOD(cwiseNotEqual, !=)
			}, {
				EIGEN_MATRIX_GET_MATRIX_MATRIX_PAIR_METHOD(cwiseProduct)
			}, {
				EIGEN_MATRIX_GET_MATRIX_MATRIX_PAIR_METHOD(cwiseQuotient)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(cwiseSign)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(cwiseSqrt)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(data)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(diagonalSize)
			}, {
				"dot", [](lua_State * L)
				{
					return LuaXS::PushArgAndReturn(L, AsVector<T>::To(L).dot(AsVector<R>::To(L, 2)));
				}
			}, {
				EIGEN_ARRAY_METHOD(exp)
			}, {
				"__gc", LuaXS::TypedGC<T>
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(hasNaN)
			}, {
				EIGEN_MATRIX_REDUCE_METHOD(hypotNorm)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(innerSize)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(innerStride)
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
					return LuaXS::PushArgAndReturn(L, AsVector<T>::To(L).isOrthogonal(AsVector<R>::To(L, 2), GetPrecision(L, 3)));
				}
			}, {
				EIGEN_MATRIX_PREDICATE_METHOD(isUnitary)
			}, {
				EIGEN_MATRIX_PREDICATE_METHOD(isUpperTriangular)
			}, {
				EIGEN_MATRIX_PREDICATE_METHOD(isZero)
			}, {
				"__len", [](lua_State * L)
				{
					EIGEN_MATRIX_PUSH_VALUE(size);
				}
			}, {
				EIGEN_ARRAY_METHOD(log)
			}, {
				EIGEN_ARRAY_METHOD(log10)
			}, {
				"lp1Norm", [](lua_State * L)
				{
					EIGEN_MATRIX_REDUCE(lpNorm<1>);
				}
			}, {
				"lpInfNorm", [](lua_State * L)
				{
					EIGEN_MATRIX_REDUCE(lpNorm<Eigen::Infinity>);
				}
			}, {
				EIGEN_MATRIX_REDUCE_METHOD(mean)
			}, {
				EIGEN_MATRIX_REDUCE_METHOD(norm)
			}, {
				XFORM_METHOD(normalized)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(outerSize)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(outerStride)
			}, {
				EIGEN_MATRIX_REDUCE_METHOD(prod)
			}, {
				"redux", [](lua_State * L)
				{
					auto how = GetVectorwiseOption(L, 3);

					auto func = [L](const T::Scalar & x, const T::Scalar & y)
					{
						LuaXS::PushMultipleArgs(L, LuaXS::StackIndex{L, 2}, x, y);	// mat, func[, how], func, x, y

						lua_call(L, 2, 1);	// mat, func[, how], result

						T::Scalar result(0);

						if (!lua_isnil(L, -1)) result = AsScalar<R>(L, -1);

						lua_pop(L, 1);	// mat, func

						return result;
					};

					if (how == eNotVectorwise)
					{
						T::Scalar result = GetT(L)->redux(func);

						return LuaXS::PushArgAndReturn(L, result);
					}

					else
					{
						if (how == eColwise) return NewRet<R>(L, GetT(L)->colwise().redux(func));
						else return NewRet<R>(L, GetT(L)->rowwise().redux(func));
					}
				}
			}, {
				"replicate", [](lua_State * L)
				{
					int a = LuaXS::Int(L, 2);

					if (lua_isstring(L, 3))
					{
						switch (GetVectorwiseOption(L, 3))
						{
						case eColwise:
							return NewRet<R>(L, GetT(L)->colwise().replicate(a));
						case eRowwise:
							return NewRet<R>(L, GetT(L)->rowwise().replicate(a));
						default:
							luaL_argcheck(L, false, 3, "Expected column rather than reduction choice");

							return 0;
						}
					}

					else return NewRet<R>(L, GetT(L)->replicate(a, LuaXS::Int(L, 3)));
				}
			}, {
				XFORM_METHOD(reverse)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(rows)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(rowStride)
			}, {
				EIGEN_ARRAY_METHOD(sin)
			}, {
				EIGEN_ARRAY_METHOD(sinh)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(size)
			}, {
				EIGEN_ARRAY_METHOD(square)
			}, {
				EIGEN_MATRIX_REDUCE_METHOD(squaredNorm)
			}, {
				"stableNorm", [](lua_State * L)
				{
					return LuaXS::PushArgAndReturn(L, AsVector<R>::To(L).stableNorm());
				}
			}, {
				"stableNormalized", [](lua_State * L)
				{
					return NewRet<R>(L, AsVector<R>::To(L).stableNormalized());
				}
			}, {
				"sub", [](lua_State * L)
				{
					NO_MUTATE(-);
				}
			}, {
				EIGEN_MATRIX_REDUCE_METHOD(sum)
			}, {
				EIGEN_ARRAY_METHOD(tan)
			}, {
				EIGEN_ARRAY_METHOD(tanh)
			}, {
				"__tostring", [](lua_State * L)
				{
					return Print(L, *GetT(L));
				}
			}, {
				EIGEN_MATRIX_GET_SCALAR_METHOD(trace)
			}, {
				"tranpose", Transposer<T>::Do
			}, {
				"unaryExpr", [](lua_State * L)
				{
					return NewRet<R>(L, GetT(L)->unaryExpr([L](const T::Scalar & x) {
						LuaXS::PushMultipleArgs(L, LuaXS::StackIndex{L, 2}, x);// mat, func, func, x

						lua_call(L, 1, 1);	// mat, func, result

						T::Scalar result(0);

						if (!lua_isnil(L, -1)) result = AsScalar<R>(L, -1);

						lua_pop(L, 1);	// mat, func

						return result;
					}));
				}
			}, {
				"unitOrthogonal", [](lua_State * L)
				{
					return NewRet<R>(L, AsVector<T>::To(L).unitOrthogonal());
				}
			}, {
				EIGEN_MATRIX_GET_SCALAR_METHOD(value)
			}, {
				"visit", [](lua_State * L)
				{
					struct Visitor {
						lua_State * mL;

						Visitor (lua_State * L) : mL{L}
						{
						}

						inline void Do (int arg, const T::Scalar & x, Eigen::Index i, Eigen::Index j)
						{
							LuaXS::PushMultipleArgs(mL, LuaXS::StackIndex{mL, arg}, x, int(i + 1), int(j + 1));	// mat, init, rest, func, x, i, j

							lua_call(mL, 3, 0);	// mat, init, rest
						}

						inline void init (const T::Scalar & x, Eigen::Index i, Eigen::Index j)
						{
							Do(2, x, i, j);
						}

						inline void operator ()(const T::Scalar & x, Eigen::Index i, Eigen::Index j)
						{
							Do(3, x, i, j);
						}
					} v{L};
					
					GetT(L)->visit(v);

					return 0;
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);

		ArithOps<T, R> ao{L};
		WriteOps<T, R> wo{L};
		XprOps<T, R> xo{L};
	}
};