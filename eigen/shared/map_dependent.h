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

//
template<typename T, typename R> struct MapDependentMethods {
	//
	ADD_INSTANCE_GETTERS()

	//
	MapDependentMethods (lua_State * L)
	{
		NoMap(L);
	}

	//
	template<bool = !std::is_same<T, R>::value> void NoMap (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"__add", [](lua_State * L)
				{
					return NewMoveRet<R>(L, *GetT(L) + *GetR(L, 2));
				}
			}, {
				"__sub", [](lua_State * L)
				{
					return NewMoveRet<R>(L, *GetT(L) - *GetR(L, 2));
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	//
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

	#define EIGEN_MATRIX_RESIZE(METHOD)	T & m = *GetT(L);										\
										ResizeState rs{L, m};									\
																								\
										if (!rs.mHas2) m.METHOD(rs.mDim1, Eigen::NoChange);		\
										else if (!rs.mHas1) m.METHOD(Eigen::NoChange, rs.mDim2);\
										else m.METHOD(rs.mDim1, rs.mDim2);						\
																								\
										return 0
	#define EIGEN_MATRIX_RESIZE_METHOD(NAME) EIGEN_REG(NAME, EIGEN_MATRIX_RESIZE(NAME))

	//
	template<> void NoMap<false> (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"__add", [](lua_State * L)
				{
					TwoMatrices<R> ms{L};

					return NewMoveRet<R>(L, *ms.mMat1 + *ms.mMat2);
				}
			}, {
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

					return NewMoveRet<Eigen::Map<T>>(L, map);
				}
			}, {
				"reshapeWithInnerStride", [](lua_State * L)
				{
					T & m = *GetT(L);
					Eigen::Map<T, 0, Eigen::InnerStride<>> map{m.data(), LuaXS::Int(L, 2), LuaXS::Int(L, 3), LuaXS::Int(L, 4)};

					return NewMoveRet<decltype(map)>(L, map);
				}
			}, {
				"reshapeWithOuterStride", [](lua_State * L)
				{
					T & m = *GetT(L);
					Eigen::Map<T, 0, Eigen::OuterStride<>> map{m.data(), LuaXS::Int(L, 2), LuaXS::Int(L, 3), LuaXS::Int(L, 4)};

					return NewMoveRet<decltype(map)>(L, map);
				}
			},
		#endif
			{
				EIGEN_MATRIX_RESIZE_METHOD(resize)
			}, {
				EIGEN_MATRIX_PAIR_VOID_METHOD(resizeLike)
			}, {
				"__sub", [](lua_State * L)
				{
					TwoMatrices<R> ms{L};

					return NewMoveRet<R>(L, *ms.mMat1 - *ms.mMat2);
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);

		//
		AddTypeData<Eigen::BDCSVD<T>>(L);
		AddTypeData<Eigen::ColPivHouseholderQR<T>>(L);
		AddTypeData<Eigen::CompleteOrthogonalDecomposition<T>>(L);
		AddTypeData<Eigen::FullPivHouseholderQR<T>>(L);
		AddTypeData<Eigen::FullPivLU<T>>(L);
		AddTypeData<Eigen::HouseholderQR<T>>(L);
		AddTypeData<Eigen::JacobiSVD<T, Eigen::ColPivHouseholderQRPreconditioner>>(L);
		AddTypeData<Eigen::JacobiSVD<T, Eigen::FullPivHouseholderQRPreconditioner>>(L);
		AddTypeData<Eigen::JacobiSVD<T, Eigen::HouseholderQRPreconditioner>>(L);
		AddTypeData<Eigen::JacobiSVD<T, Eigen::NoQRPreconditioner>>(L);
		AddTypeData<Eigen::LDLT<T, Eigen::Lower>>(L);
		AddTypeData<Eigen::LDLT<T, Eigen::Upper>>(L);
		AddTypeData<Eigen::LLT<T, Eigen::Lower>>(L);
		AddTypeData<Eigen::LLT<T, Eigen::Lower>>(L);
		AddTypeData<Eigen::PartialPivLU<T>>(L);
	}
};