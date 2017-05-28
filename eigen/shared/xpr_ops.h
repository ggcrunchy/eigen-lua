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
#include "utils/Thread.h"
#include "types.h"
#include "macros.h"
#include <type_traits>

//
#define NEW_XPR(METHOD, ...)	XprSource<> xprs{L};										\
								using XprType = decltype(xprs.mPtr->METHOD(__VA_ARGS__));	\
																							\
								New<XprType>(L, xprs.mPtr->METHOD(__VA_ARGS__))
#define REF_XPR_SOURCE()	GetTypeData<XprType>(L)->RefAt(L, "xpr_from", 1);	\
																				\
							return 1

#define EIGEN_OBJECT_GET_XPR_COUNT(METHOD)	NEW_XPR(METHOD, LuaXS::Int(L, 2));	\
											REF_XPR_SOURCE()
#define EIGEN_OBJECT_GET_XPR_COUNT_PAIR(METHOD)	NEW_XPR(METHOD, LuaXS::Int(L, 2), LuaXS::Int(L, 3));\
												REF_XPR_SOURCE()
#define EIGEN_OBJECT_GET_XPR_INDEX(METHOD)	NEW_XPR(METHOD, LuaXS::Int(L, 2) - 1);	\
											REF_XPR_SOURCE()
#define EIGEN_OBJECT_GET_XPR_INDEX_PAIR(METHOD)	NEW_XPR(METHOD, LuaXS::Int(L, 2) - 1, LuaXS::Int(L, 3) - 1);\
												REF_XPR_SOURCE()

//
template<typename T> struct VectorRing {
	static const int N = 4;
 
	struct Box {
		T mMap{nullptr, 0};

		Box (void) = default;
	};
	
	Box mEntries[N];//
	int mIndex{0};	//

	T * GetEntry (void)
	{
		T * ptr = &mEntries[mIndex++].mMap;

		mIndex %= N;

		return ptr;
	}
};

//
template<typename T> static typename AsVector<T>::Type * GetVectorFromRing (lua_State * L)
{
	using VRing = VectorRing<typename AsVector<T>::Type>;

	auto td = GetTypeData<T>(L);

	//
	if (td->mVectorRingRef == LUA_NOREF)
	{
		LuaXS::NewTyped<VRing>(L);	// ..., vring

		lua_pushliteral(L, "vector_ring:");	// ..., vring, "vector_ring:"
		lua_pushstring(L, td->GetName());	// ..., vring, "vector_ring:", name
		lua_concat(L, 2);	// ..., vring, "vector_ring:" .. name
		lua_insert(L, -2);	// ..., "vector_ring:" .. name, vring

		LuaXS::AttachTypedGC<VRing>(L, lua_tostring(L, -2));

		lua_remove(L, -2);	// ..., vring

		td->mVectorRingRef = lua_ref(L, 1);	// ...
	}

	//
	lua_rawgeti(L, LUA_REGISTRYINDEX, td->mVectorRingRef);	// ..., vring

	auto * pring = LuaXS::UD<VRing>(L, -1);

	lua_pop(L, 1);	// ...

	return pring->GetEntry();
}

#define EIGEN_NEW_VECTOR_BLOCK(METHOD, ...)	AsVector<T>::Type * ring_vec = GetVectorFromRing<T>(L);	\
											AsVector<T>::New(ring_vec, GetT(L));					\
																									\
											using XprType = decltype(ring_vec->METHOD(__VA_ARGS__));\
																									\
											New<XprType>(L, ring_vec->METHOD(__VA_ARGS__))

#define EIGEN_OBJECT_GET_XPR_COUNT_METHOD(NAME) EIGEN_REG(NAME, EIGEN_OBJECT_GET_XPR_COUNT(NAME))
#define EIGEN_OBJECT_GET_XPR_COUNT_PAIR_METHOD(NAME) EIGEN_REG(NAME, EIGEN_OBJECT_GET_XPR_COUNT_PAIR(NAME))
#define EIGEN_OBJECT_GET_XPR_INDEX_METHOD(NAME)	EIGEN_REG(NAME, EIGEN_OBJECT_GET_XPR_INDEX(NAME))
#define EIGEN_OBJECT_GET_XPR_INDEX_PAIR_METHOD(NAME) EIGEN_REG(NAME, EIGEN_OBJECT_GET_XPR_INDEX_PAIR(NAME))

//
template<typename T, typename R = T> struct XprOps {
	ADD_INSTANCE_GETTERS()

	//
	template<bool = IsXpr<T>::value> struct XprSource {
		R * mPtr;

		XprSource (lua_State * L)
		{
			mPtr = New<R>(L, *GetT(L));	// xpr, ..., new_mat

			lua_replace(L, 1);	// new_mat, ...
		}
	};

	template<> struct XprSource<false> {
		T * mPtr;

		XprSource (lua_State * L) : mPtr{GetT(L)}
		{
		}
	};

	XprOps (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"block", [](lua_State * L)
				{
					NEW_XPR(block, LuaXS::Int(L, 2) - 1, LuaXS::Int(L, 3) - 1, LuaXS::Int(L, 4), LuaXS::Int(L, 5));	// mat, x, y, w, h, block
					REF_XPR_SOURCE();
				}
			}, {
				EIGEN_OBJECT_GET_XPR_COUNT_PAIR_METHOD(bottomLeftCorner)
			}, {
				EIGEN_OBJECT_GET_XPR_COUNT_PAIR_METHOD(bottomRightCorner)
			}, {
				EIGEN_OBJECT_GET_XPR_COUNT_METHOD(bottomRows)
			}, {
				EIGEN_OBJECT_GET_XPR_INDEX_METHOD(col)
			}, {
				"diagonal", [](lua_State * L)
				{
					NEW_XPR(diagonal, luaL_optint(L, 2, 0));// mat[, index], diagonal
					REF_XPR_SOURCE();
				}
			}, {
				"head", [](lua_State * L)
				{
					EIGEN_NEW_VECTOR_BLOCK(head, LuaXS::Int(L, 2));	// mat, n, hseg
					REF_XPR_SOURCE();
				}
			}, {
				EIGEN_OBJECT_GET_XPR_COUNT_METHOD(leftCols)
			}, {
				EIGEN_OBJECT_GET_XPR_INDEX_PAIR_METHOD(middleCols)
			}, {
				EIGEN_OBJECT_GET_XPR_INDEX_PAIR_METHOD(middleRows)
			}, {
				EIGEN_OBJECT_GET_XPR_COUNT_METHOD(rightCols)
			}, {
				EIGEN_OBJECT_GET_XPR_INDEX_METHOD(row)
			}, {
				"segment", [](lua_State * L)
				{
					EIGEN_NEW_VECTOR_BLOCK(segment, LuaXS::Int(L, 2) - 1, LuaXS::Int(L, 3));// mat, pos, n, seg
					REF_XPR_SOURCE();
				}
			}, {
				"tail", [](lua_State * L)
				{
					EIGEN_NEW_VECTOR_BLOCK(tail, LuaXS::Int(L, 2));	// mat, n, tseg
					REF_XPR_SOURCE();
				}
			}, {
				EIGEN_OBJECT_GET_XPR_COUNT_PAIR_METHOD(topLeftCorner)
			}, {
				EIGEN_OBJECT_GET_XPR_COUNT_PAIR_METHOD(topRightCorner)
			}, {
				EIGEN_OBJECT_GET_XPR_COUNT_METHOD(topRows)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}
};