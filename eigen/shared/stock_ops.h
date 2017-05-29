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
#include "self_adjoint_view.h"
#include "triangular_view.h"
#include <utility>

//
template<typename T, typename R = T> struct StockOps {
	ADD_INSTANCE_GETTERS()

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

	StockOps (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_GET_MATRIX_METHOD(asDiagonal)
			}, {
				"asMatrix", AsMatrix<T, R>
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
				"__call", Call<T>
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(cols)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(colStride)
			}, {
				EIGEN_REL_OP_METHOD(cwiseEqual, ==)
			}, {
				EIGEN_REL_OP_METHOD(cwiseNotEqual, !=)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(data)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(diagonalSize)
			}, {
				"__eq", [](lua_State * L)
				{
					bool bHas1 = HasType<T>(L, 1), bHas2 = HasType<T>(L, 2), result;

					if (bHas1 && bHas2) result = *GetT(L) == *GetT(L, 2);
					else if (bHas1) result = *GetT(L) == GetR(L, 2);
					else result = GetR(L, 1) == *GetT(L, 2);

					return LuaXS::PushArgAndReturn(L, result);
				}
			}, {
				"__gc", LuaXS::TypedGC<T>
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(innerSize)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(innerStride)
			}, {
				"__len", [](lua_State * L)
				{
					EIGEN_MATRIX_PUSH_VALUE(size);
				}
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(outerSize)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(outerStride)
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
			},
		#ifdef WANT_MAP
			{
				"reshape", [](lua_State * L)
				{
					using M = MatrixOf<T::Scalar>;

					T & m = *GetT(L);
					Eigen::Map<M> map{m.data(), LuaXS::Int(L, 2), LuaXS::Int(L, 3)};

					New<Eigen::Map<M>>(L, std::move(map));	// mat, m, n, map
					GetTypeData<Eigen::Map<M>>(L)->RefAt(L, "mapped_from", 1);

					return 1;
				}
			}, 
		#endif
		#ifdef WANT_MAP_WITH_CUSTOM_STRIDE
			{
				using M = MatrixOf<T::Scalar>;

				"reshapeWithInnerStride", [](lua_State * L)
				{
					T & m = *GetT(L);
					Eigen::Map<M, 0, Eigen::InnerStride<>> map{m.data(), LuaXS::Int(L, 2), LuaXS::Int(L, 3), LuaXS::Int(L, 4)};

					New<decltype(map)>(L, std::move(map));
					GetTypeData<decltype(map)>(L)->RefAt(L, "mapped_from", 1);

					return 1;
				}
			}, {
				"reshapeWithOuterStride", [](lua_State * L)
				{
					using M = MatrixOf<T::Scalar>;

					T & m = *GetT(L);
					Eigen::Map<M, 0, Eigen::OuterStride<>> map{m.data(), LuaXS::Int(L, 2), LuaXS::Int(L, 3), LuaXS::Int(L, 4)};

					New<decltype(map)>(L, std::move(map));
					GetTypeData<decltype(map)>(L)->RefAt(L, "mapped_from", 1);

					return 1;
				}
			},
		#endif
			{
				EIGEN_XFORM_METHOD(reverse)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(rows)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(rowStride)
			}, {
				"selfadjointView", [](lua_State * L)
				{
					const char * names[] = { "Lower", "Upper", nullptr };

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
				EIGEN_MATRIX_PUSH_VALUE_METHOD(size)
			}, {
				"__tostring", [](lua_State * L)
				{
					return Print(L, *GetT(L));
				}
			}, {
				"tranpose", Transposer<T>::Do
			}, {
				"triangularView", [](lua_State * L)
				{
					const char * names[] = { "Lower", "StrictlyLower", "StrictlyUpper", "UnitLower", "UnitUpper", "Upper", nullptr };

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
				EIGEN_MATRIX_PUSH_VALUE_METHOD(value)
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
