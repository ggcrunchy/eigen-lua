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
#include <type_traits>

#define MUTATE(OP)	ReductionOption how = GetReductionChoice(L, 3);							\
																							\
					if (how == eDefault) *GetT(L) OP *GetR(L, 2);							\
																							\
					else																	\
					{																		\
						if (how == eColwise) GetT(L)->colwise() OP AsVector<R>::To(L, 2);	\
						else GetT(L)->rowwise() OP AsVector<R>::To(L, 2).transpose();		\
					}																		\
																							\
					return SelfForChaining(L)

#define NO_MUTATE(OP)	ReductionOption how = GetReductionChoice(L, 3);													\
																														\
						if (how == eDefault) return NewMoveRet<R>(L, *GetT(L) OP *GetR(L, 2));							\
																														\
						else																							\
						{																								\
							if (how == eColwise) return NewMoveRet<R>(L, GetT(L)->colwise() OP AsVector<R>::To(L, 2));	\
							else return NewMoveRet<R>(L, GetT(L)->rowwise() OP AsVector<R>::To(L, 2).transpose());		\
						}																								\
																														\
						return 1

#define IN_PLACE_REDUCE(METHOD)	ReductionOption how = GetReductionChoice(L, 3);			\
																						\
								if (how == eDefault) GetT(L)->METHOD();					\
																						\
								else													\
								{														\
									if (how == eColwise) GetT(L)->colwise().METHOD();	\
									else GetT(L)->rowwise().METHOD();					\
								}														\
																						\
								return 0

#define XFORM_REDUCE(METHOD)	ReductionOption how = GetReductionChoice(L, 3);									\
																												\
								if (how == eDefault) return NewMoveRet<R>(L, GetT(L)->METHOD());				\
																												\
								else																			\
								{																				\
									if (how == eColwise) return NewMoveRet<R>(L, GetT(L)->colwise().METHOD());	\
									else return NewMoveRet<R>(L, GetT(L)->rowwise().METHOD());					\
								}																				\
																												\
								return 1

#define IN_PLACE_REDUCE_METHOD(NAME) EIGEN_REG(NAME, IN_PLACE_REDUCE(NAME))
#define XFORM_REDUCE_METHOD(NAME) EIGEN_REG(NAME, XFORM_REDUCE(NAME))

//
template<typename T, typename R> struct CommonMethods {
	//
	ADD_INSTANCE_GETTERS()

	//
	template<typename U> struct Cast {
		using MatrixType = Eigen::Matrix<U, Eigen::Dynamic, Eigen::Dynamic>;

		template<bool = !Eigen::NumTraits<T::Scalar>::IsComplex, bool = !Eigen::NumTraits<U>::IsComplex> MatrixType CastTo (lua_State * L)
		{
			return GetT(L)->template cast<U>();
		}

		template<> MatrixType CastTo<false, true> (lua_State * L)
		{
			T & m = *GetT(L);
			MatrixType out;

			out.resizeLike(m);

			struct VisitCoeffs {
				MatrixType & mOut;

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
			auto td = GetTypeData<MatrixType>(L);

			luaL_argcheck(L, td, 2, "Matrix type unavailable for cast");

			MatrixType m = CastTo(L);

			lua_getref(L, td->mPushRef);// mat, push_new_type
			lua_pushlightuserdata(L, &m);	// mat, push_new_type, conv_mat
			lua_call(L, 1, 1);	// mat, new_mat
		}
	};

	//
	static decltype(Eigen::NumTraits<T::Scalar>::dummy_precision()) GetPrecision (lua_State * L, int arg)
	{
		auto prec = Eigen::NumTraits<T::Scalar>::dummy_precision();

		return !lua_isnoneornil(L, arg) ? LuaXS::GetArg<decltype(prec)>(L, arg) : prec;
	}

	#define EIGEN_MATRIX_PREDICATE(METHOD)	return LuaXS::PushArgAndReturn(L, GetT(L)->METHOD(GetPrecision(L, 2)))	
	#define EIGEN_MATRIX_PREDICATE_METHOD(NAME) EIGEN_REG(NAME, EIGEN_MATRIX_PREDICATE(NAME))

	//
	CommonMethods (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_GET_MATRIX_METHOD(adjoint)
			}, {
				EIGEN_MATRIX_VOID_METHOD(adjointInPlace)
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
				"addInPlace", [](lua_State * L)
				{
					MUTATE(+=);
				}
			}, {
				EIGEN_ARRAY_METHOD(arg)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(asDiagonal)
			}, {
				EIGEN_ARRAY_METHOD(asin)
			}, {
				"assign", [](lua_State * L)
				{
					MUTATE(=);
				}
			}, {
				EIGEN_ARRAY_METHOD(atan)
			}, {
				"binaryExpr", [](lua_State * L)
				{
					return NewMoveRet<R>(L, GetT(L)->binaryExpr(*GetR(L, 2), [L](const T::Scalar & x, const T::Scalar & y)
					{
						LuaXS::PushMultipleArgs(L, LuaXS::StackIndex{L, 3}, x, y);	// mat1, mat2, func, func, x, y

						lua_call(L, 2, 1);	// mat1, mat2, func, result

						T::Scalar result(0);

						if (!lua_isnil(L, -1)) result = ArgObject<R>{}.AsScalar(L, -1);

						lua_pop(L, 1);	// mat1, mat2, func

						return result;
					}));
				}
			}, {
				"block", [](lua_State * L)
				{
					return NewMoveRet<R>(L, GetT(L)->block(LuaXS::Int(L, 2) - 1, LuaXS::Int(L, 3) - 1, LuaXS::Int(L, 4), LuaXS::Int(L, 5)));
				}
			}, {
				"blockAssign", [](lua_State * L)
				{
					GetT(L)->block(LuaXS::Int(L, 2) - 1, LuaXS::Int(L, 3) - 1, LuaXS::Int(L, 4), LuaXS::Int(L, 5)) = *GetR(L, 6);

					return 0;
				}
			}, {
				EIGEN_MATRIX_REDUCE_METHOD(blueNorm)
			}, {
				EIGEN_MATRIX_GET_MATRIX_COUNT_PAIR_METHOD(bottomLeftCorner)
			}, {
				EIGEN_MATRIX_ASSIGN_MATRIX_COUNT_PAIR_METHOD(bottomLeftCorner)
			}, {
				EIGEN_MATRIX_GET_MATRIX_COUNT_PAIR_METHOD(bottomRightCorner)
			}, {
				EIGEN_MATRIX_ASSIGN_MATRIX_COUNT_PAIR_METHOD(bottomRightCorner)
			}, {
				EIGEN_MATRIX_GET_MATRIX_COUNT_METHOD(bottomRows)
			}, {
				EIGEN_MATRIX_ASSIGN_MATRIX_COUNT_METHOD(bottomRows)
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
				"clone", [](lua_State * L)
				{
					R clone = *GetT(L);

					return NewMoveRet<R>(L, clone);
				}
			}, {
				"coeffAssign", [](lua_State * L)
				{
					T & m = *GetT(L);
					int a = LuaXS::Int(L, 2) - 1;

					if (lua_gettop(L) == 3)
					{
						CheckVector(L, m, 1);

						(m.cols() == 1 ? m(a, 0) : m(0, a)) = ArgObject<R>{}.AsScalar(L, 3);
					}

					else m(a, LuaXS::Int(L, 3) - 1) = ArgObject<R>{}.AsScalar(L, 4);

					return 0;
				}
			}, {
				EIGEN_MATRIX_GET_MATRIX_INDEX_METHOD(col)
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
				"diagonal", [](lua_State * L)
				{
					if (lua_gettop(L) == 1) return NewMoveRet<R>(L, GetT(L)->diagonal());

					else return NewMoveRet<R>(L, GetT(L)->diagonal(LuaXS::Int(L, 2)));
				}
			}, {
				"diagonalAssign", [](lua_State * L)
				{
					if (lua_gettop(L) == 2)
					{
						R & diag = *GetR(L, 2);

						CheckVector(L, diag, 2);

						GetT(L)->diagonal() = diag;
					}

					else
					{
						R & diag = *GetR(L, 3);

						CheckVector(L, diag, 3);

						GetT(L)->diagonal(LuaXS::Int(L, 2)) = diag;
					}

					return 0;
				}
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(diagonalSize)
			}, {
				"__div", [](lua_State * L)
				{
					return NewMoveRet<R>(L, *GetT(L) / ArgObject<R>{}.AsScalar(L, 2));
				}
			}, {
				"dot", [](lua_State * L)
				{
					return LuaXS::PushArgAndReturn(L, AsVector<T>::To(L).dot(AsVector<R>::To(L, 2)));
				}
			}, {
				EIGEN_ARRAY_METHOD(exp)
			}, {
				EIGEN_MATRIX_SET_SCALAR_METHOD(fill)
			}, {
				"__gc", LuaXS::TypedGC<T>
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(hasNaN)
			}, {
				"head", [](lua_State * L)
				{
					return NewMoveRet<R>(L, AsVector<T>::To(L).head(LuaXS::Int(L, 2)));
				}
			}, {
				"headAssign", [](lua_State * L)
				{
					AsVector<T>::To(L).head(LuaXS::Int(L, 2)) = AsVector<R>::To(L, 3);

					return 0;
				}
			}, {
				EIGEN_MATRIX_REDUCE_METHOD(hypotNorm)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(innerSize)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(innerStride)
			}, {
				"isApprox", [](lua_State * L)
				{
					return LuaXS::PushArgAndReturn(L, GetT(L)->isApprox(*GetR(L, 2), GetPrecision(L, 3)));
				}
			}, {
				"isConstant", [](lua_State * L)
				{
					return LuaXS::PushArgAndReturn(L, GetT(L)->isConstant(ArgObject<R>{}.AsScalar(L, 2), GetPrecision(L, 3)));
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
				EIGEN_MATRIX_GET_MATRIX_COUNT_METHOD(leftCols)
			}, {
				EIGEN_MATRIX_ASSIGN_MATRIX_COUNT_METHOD(leftCols)
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
				EIGEN_MATRIX_GET_MATRIX_INDEX_PAIR_METHOD(middleCols)
			}, {
				EIGEN_MATRIX_ASSIGN_MATRIX_INDEX_PAIR_METHOD(middleCols)
			}, {
				EIGEN_MATRIX_GET_MATRIX_INDEX_PAIR_METHOD(middleRows)
			}, {
				EIGEN_MATRIX_ASSIGN_MATRIX_INDEX_PAIR_METHOD(middleRows)
			}, {
				"__mul", [](lua_State * L)
				{
					return NewMoveRet<R>(L, WithMatrixOrScalar<T, R>(L, [](const T & m1, const R & m2){
						return m1 * m2;
					}, [](const T & m, const T::Scalar & s){
						return m * s;
					}, [](const T::Scalar & s, const R & m){
						return s * m;
					}, 1, 2));
				}
			}, {
				EIGEN_MATRIX_REDUCE_METHOD(norm)
			}, {
				XFORM_REDUCE_METHOD(normalized)
			}, {
				IN_PLACE_REDUCE_METHOD(normalize)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(outerSize)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(outerStride)
			}, {
				"__pow", [](lua_State * L)
				{
					return NewMoveRet<R>(L, WithMatrixOrScalar<T, R>(L, [](const T & m1, const R & m2){
						return m1.array().pow(m2.array());
					}, [](const T & m, const T::Scalar & s){
						return m.array().pow(s);
					}, [](const T::Scalar & s, const R & m){
						return Eigen::pow(s, m.array());
					}, 1, 2));
				}
			}, {
				EIGEN_MATRIX_REDUCE_METHOD(prod)
			}, {
				"redux", [](lua_State * L)
				{
					ReductionOption how = GetReductionChoice(L, 3);

					auto func = [L](const T::Scalar & x, const T::Scalar & y)
					{
						LuaXS::PushMultipleArgs(L, LuaXS::StackIndex{L, 2}, x, y);	// mat, func[, how], func, x, y

						lua_call(L, 2, 1);	// mat, func[, how], result

						T::Scalar result(0);

						if (!lua_isnil(L, -1)) result = ArgObject<R>{}.AsScalar(L, -1);

						lua_pop(L, 1);	// mat, func

						return result;
					};

					if (how == eDefault)
					{
						T::Scalar result = GetT(L)->redux(func);

						return LuaXS::PushArgAndReturn(L, result);
					}

					else
					{
						if (how == eColwise) return NewMoveRet<R>(L, GetT(L)->colwise().redux(func));
						else return NewMoveRet<R>(L, GetT(L)->rowwise().redux(func));
					}
				}
			}, {
				"replicate", [](lua_State * L)
				{
					int a = LuaXS::Int(L, 2);

					if (lua_isstring(L, 3))
					{
						switch (GetReductionChoice(L, 3))
						{
						case eColwise:
							return NewMoveRet<R>(L, GetT(L)->colwise().replicate(a));
						case eRowwise:
							return NewMoveRet<R>(L, GetT(L)->rowwise().replicate(a));
						default:
							luaL_argcheck(L, false, 3, "Expected column rather than reduction choice");

							return 0;
						}
					}

					else return NewMoveRet<R>(L, GetT(L)->replicate(a, LuaXS::Int(L, 3)));
				}
			}, {
				XFORM_REDUCE_METHOD(reverse)
			}, {
				IN_PLACE_REDUCE_METHOD(reverseInPlace)
			}, {
				EIGEN_MATRIX_GET_MATRIX_COUNT_METHOD(rightCols)
			}, {
				EIGEN_MATRIX_ASSIGN_MATRIX_COUNT_METHOD(rightCols)
			}, {
				EIGEN_MATRIX_GET_MATRIX_INDEX_METHOD(row)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(rows)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(rowStride)
			}, {
				"selfadjointView", [](lua_State * L)
				{
					const char * names[] = { "lower", "upper", nullptr };

					switch (luaL_checkoption(L, 2, nullptr, names))
					{
					case 0:	// Lower-triangular
						return NewMoveRet<Eigen::SelfAdjointView<T, Eigen::Lower>>(L, GetT(L)->selfadjointView<Eigen::Lower>());
					default:// Upper-triangular
						return NewMoveRet<Eigen::SelfAdjointView<T, Eigen::Upper>>(L, GetT(L)->selfadjointView<Eigen::Upper>());
					}
				}
			}, {
				"segment", [](lua_State * L)
				{
					return NewMoveRet<R>(L, AsVector<T>::To(L).segment(LuaXS::Int(L, 2) - 1, LuaXS::Int(L, 3)));
				}
			}, {
				"segmentAssign", [](lua_State * L)
				{
					AsVector<T>::To(L).segment(LuaXS::Int(L, 2) - 1, LuaXS::Int(L, 3)) = AsVector<R>::To(L, 4);

					return 0;
				}
			}, {
				EIGEN_MATRIX_SET_SCALAR_METHOD(setConstant)
			}, {
				EIGEN_MATRIX_VOID_METHOD(setIdentity)
			}, {
				"setLinSpaced", [](lua_State * L)
				{
					T & m = *GetT(L);

					CheckVector(L, m, 1);

					if (m.cols() == 1) m = LinSpacing<T, Eigen::Dynamic, 1>::Do<>(L, m.rows());
					else m = LinSpacing<T, 1, Eigen::Dynamic>::Do<>(L, m.cols());

					return 0;
				}
			}, {
				EIGEN_MATRIX_VOID_METHOD(setOnes)
			}, {
				EIGEN_MATRIX_VOID_METHOD(setRandom)
			}, {
				EIGEN_MATRIX_VOID_METHOD(setZero)
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
			}, /*{
				EIGEN_MATRIX_REDUCE_METHOD(stableNorm)
			}, {
				EIGEN_MATRIX_VOID_METHOD(stableNormalize)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(stableNormalized)
			}, */{
				"sub", [](lua_State * L)
				{
					NO_MUTATE(-);
				}
			}, {
				"subInPlace", [](lua_State * L)
				{
					MUTATE(-=);
				}
			}, {
				EIGEN_MATRIX_REDUCE_METHOD(sum)
			}, {
				EIGEN_MATRIX_PAIR_VOID_METHOD(swap)
			}, {
				"tail", [](lua_State * L)
				{
					return NewMoveRet<R>(L, AsVector<T>::To(L).tail(LuaXS::Int(L, 2)));
				}
			}, {
				"tailAssign", [](lua_State * L)
				{
					AsVector<T>::To(L).tail(LuaXS::Int(L, 2)) = AsVector<R>::To(L, 3);

					return 0;
				}
			}, {
				EIGEN_ARRAY_METHOD(tan)
			}, {
				EIGEN_ARRAY_METHOD(tanh)
			}, {
				EIGEN_MATRIX_GET_MATRIX_COUNT_PAIR_METHOD(topLeftCorner)
			}, {
				EIGEN_MATRIX_ASSIGN_MATRIX_COUNT_PAIR_METHOD(topLeftCorner)
			}, {
				EIGEN_MATRIX_GET_MATRIX_COUNT_PAIR_METHOD(topRightCorner)
			}, {
				EIGEN_MATRIX_ASSIGN_MATRIX_COUNT_PAIR_METHOD(topRightCorner)
			}, {
				EIGEN_MATRIX_GET_MATRIX_COUNT_METHOD(topRows)
			}, {
				EIGEN_MATRIX_ASSIGN_MATRIX_COUNT_METHOD(topRows)
			}, {
				"__tostring", [](lua_State * L)
				{
					return Print(L, *GetT(L));
				}
			}, {
				EIGEN_MATRIX_GET_SCALAR_METHOD(trace)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(transpose)
			}, {
				EIGEN_MATRIX_VOID_METHOD(transposeInPlace)
			}, {
				"triangularView", [](lua_State * L)
				{
					const char * names[] = { "lower", "strictly_lower", "strictly_upper", "unit_lower", "unit_upper", "upper", nullptr };

					switch (luaL_checkoption(L, 2, nullptr, names))
					{
					case 0:	// Lower-triangular
						return NewMoveRet<Eigen::TriangularView<T, Eigen::Lower>>(L, GetT(L)->triangularView<Eigen::Lower>());
					case 1:// Strictly lower-triangular
						return NewMoveRet<Eigen::TriangularView<T, Eigen::StrictlyLower>>(L, GetT(L)->triangularView<Eigen::StrictlyLower>());
					case 2:// Strictly upper-triangular
						return NewMoveRet<Eigen::TriangularView<T, Eigen::StrictlyUpper>>(L, GetT(L)->triangularView<Eigen::StrictlyUpper>());
					case 3:// Upper-triangular
						return NewMoveRet<Eigen::TriangularView<T, Eigen::UnitLower>>(L, GetT(L)->triangularView<Eigen::UnitLower>());
					case 4:// Unit lower-triangular
						return NewMoveRet<Eigen::TriangularView<T, Eigen::UnitUpper>>(L, GetT(L)->triangularView<Eigen::UnitUpper>());
					default:// Unit upper-triangular
						return NewMoveRet<Eigen::TriangularView<T, Eigen::Upper>>(L, GetT(L)->triangularView<Eigen::Upper>());
					}
				}
			}, {
				"unaryExpr", [](lua_State * L)
				{
					return NewMoveRet<R>(L, GetT(L)->unaryExpr([L](const T::Scalar & x)
					{
						LuaXS::PushMultipleArgs(L, LuaXS::StackIndex{L, 2}, x);// mat, func, func, x

						lua_call(L, 1, 1);	// mat, func, result

						T::Scalar result(0);

						if (!lua_isnil(L, -1)) result = ArgObject<R>{}.AsScalar(L, -1);

						lua_pop(L, 1);	// mat, func

						return result;
					}));
				}
			}, {
				"__unm", [](lua_State * L)
				{
					return NewMoveRet<R>(L, -*GetT(L));
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
	}
};