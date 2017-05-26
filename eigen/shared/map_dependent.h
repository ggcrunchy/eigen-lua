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

// Methods assigned when the matrix is not mapped.
template<typename T, typename R> struct MapDependentMethods {
	ADD_INSTANCE_GETTERS()

	// State usable by any of the resize methods' overloads.
	struct ResizeState {
		int mDim1{1}, mDim2{1};
		bool mHas1{true}, mHas2{true};

		ResizeState (lua_State * L, const T & mat)
		{
			if (!lua_isnoneornil(L, 2))
			{
				lua_pushliteral(L, "no_change");// mat, a, b, "no_change"

				mHas1 = lua_equal(L, 2, -1) != 0;
				mHas2 = lua_equal(L, 3, -1) != 0;

				luaL_argcheck(L, mHas1 || mHas2, 1, "Must resize at least one dimension");

				if (mHas1) mDim1 = LuaXS::Int(L, 2);
				if (mHas2) mDim2 = LuaXS::Int(L, 3);
			}

			else
			{
				CheckVector(L, mat, 1);

				int a = LuaXS::Int(L, 2);

				if (mat.cols() == 1) mDim2 = a;

				else mDim1 = a;
			}
		}
	};

	// Common form of resize methods.
	#define EIGEN_MATRIX_RESIZE(METHOD)	T & m = *GetT(L);										\
										ResizeState rs{L, m};									\
																								\
										if (!rs.mHas2) m.METHOD(rs.mDim1, Eigen::NoChange);		\
										else if (!rs.mHas1) m.METHOD(Eigen::NoChange, rs.mDim2);\
										else m.METHOD(rs.mDim1, rs.mDim2);						\
																								\
										return 0
	#define EIGEN_MATRIX_RESIZE_METHOD(NAME) EIGEN_REG(NAME, EIGEN_MATRIX_RESIZE(NAME))

	// Operations added for non-transposed matrices.
	template<typename U> struct AddNonTranspose {
		AddNonTranspose (lua_State * L)
		{
			luaL_Reg methods[] = {
				{
					EIGEN_MATRIX_RESIZE_METHOD(conservativeResize)
				}, {
					EIGEN_MATRIX_PAIR_VOID_METHOD(conservativeResizeLike)
				}, 
			#ifdef WANT_MAP
				{
					"reshape", [](lua_State * L)
					{
						T & m = *GetT(L);
						Eigen::Map<T> map{m.data(), LuaXS::Int(L, 2), LuaXS::Int(L, 3)};

						New<Eigen::Map<T>>(L, std::move(map));	// mat, m, n, map
						GetTypeData<Eigen::Map<T>>(L)->RefAt(L, "mapped_from", 1);

						return 1;
					}
				}, {
					"reshapeWithInnerStride", [](lua_State * L)
					{
						T & m = *GetT(L);
						Eigen::Map<T, 0, Eigen::InnerStride<>> map{m.data(), LuaXS::Int(L, 2), LuaXS::Int(L, 3), LuaXS::Int(L, 4)};

						New<decltype(map)>(L, std::move(map));
						GetTypeData<decltype(map)>(L)->RefAt(L, "mapped_from", 1);

						return 1;
					}
				}, {
					"reshapeWithOuterStride", [](lua_State * L)
					{
						T & m = *GetT(L);
						Eigen::Map<T, 0, Eigen::OuterStride<>> map{m.data(), LuaXS::Int(L, 2), LuaXS::Int(L, 3), LuaXS::Int(L, 4)};

						New<decltype(map)>(L, std::move(map));
						GetTypeData<decltype(map)>(L)->RefAt(L, "mapped_from", 1);

						return 1;
					}
				},
			#endif
				{
					EIGEN_MATRIX_RESIZE_METHOD(resize)
				}, {
					EIGEN_MATRIX_PAIR_VOID_METHOD(resizeLike)
				},
				{ nullptr, nullptr }
			};

			luaL_register(L, nullptr, methods);
		}
	};

	// No-op when matrix is transposed.
	template<typename U> struct AddNonTranspose<Eigen::Transpose<U>> {
		AddNonTranspose (lua_State *) {}
	};

	MapDependentMethods (lua_State * L)
	{
/*
TODO: by the looks of it, this file really deals in NonMappedOrTranposedMethods (or just RawMatrixMethods?)
		luaL_Reg methods[] = {
			{
				"__add", [](lua_State * L)
				{
					TwoMatrices<R> ms{L};

					return NewRet<R>(L, *ms.mMat1 + *ms.mMat2);
				}
			}, {
				"__sub", [](lua_State * L)
				{
					TwoMatrices<R> ms{L};

					return NewRet<R>(L, *ms.mMat1 - *ms.mMat2);
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
*/
		AddNonTranspose<T> ant{L};
	}
};

// No-op when matrix is mapped.
// TODO: verify this compiles... if so simplify
// TODO: Add version for transpose
template<typename U, int O, typename S, typename R> struct MapDependentMethods<Eigen::Map<U, O, S>, R> {
	using T = Eigen::Map<U, O, S>;

	ADD_INSTANCE_GETTERS()

	MapDependentMethods (lua_State * L)
	{/*
		luaL_Reg methods[] = {
			{
				"__add", [](lua_State * L)
				{
					return NewRet<R>(L, *GetT(L) + GetR(L, 2));
				}
			}, {
				"__sub", [](lua_State * L)
				{
					return NewRet<R>(L, *GetT(L) - GetR(L, 2));
				}
			},
			{ nullptr, nullptr }
		};
		
		luaL_register(L, nullptr, methods);*/
	}
};