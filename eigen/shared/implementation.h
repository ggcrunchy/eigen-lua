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
#include "CoronaLibrary.h"
#include "utils/Blob.h"
#include "utils/Thread.h"

// Propagate asserts to Lua.
#ifndef eigen_assert
	#define eigen_assert(x) if (!(x)) luaL_error(tls_LuaState, "Eigen error: " #x);
#endif

// Let eigen_assert() know the main state on a given thread.
static ThreadXS::TLS<lua_State *> tls_LuaState;

#include "config.h"
#include "matrix.h"
#include "types.h"
#include "utils.h"
#include "bool_matrix.h"
#include <Eigen/Eigen>
#include <algorithm>
#include <type_traits>

// Add Umeyama() for real floating point matrices.
template<typename M, bool = !Eigen::NumTraits<typename M::Scalar>::IsInteger && !Eigen::NumTraits<typename M::Scalar>::IsComplex> struct AddUmeyama {
	AddUmeyama (lua_State * L)
	{
		luaL_Reg funcs[] = {
			{
				"Umeyama", [](lua_State * L)
				{
					return NewRet<M>(L, Eigen::umeyama(*GetInstance<M>(L, 1), *GetInstance<M>(L, 2), !WantsBool(L, "NoSaling", 3)));// src, dst[, no_scaling], xform
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, funcs);
	}
};

template<typename M> struct AddUmeyama<M, false> {
	AddUmeyama (lua_State *) {}
};

// Supplies some front end functions for the type, in particular various matrix factories.
template<typename M> static void AddType (lua_State * L)
{
	#ifdef EIGEN_PLUGIN_BASIC
		lua_newtable(L);// eigen, module
	#endif

	luaL_Reg funcs[] = {
		{
			"Constant", [](lua_State * L)
			{
				int m = LuaXS::Int(L, 1), n = luaL_optint(L, 2, m);

				return NewRet<M>(L, M::Constant(m, n, AsScalar<M>(L, 3)));	// m[, n], v, k
			}
		}, {
			"Identity", [](lua_State * L)
			{
				int m = LuaXS::Int(L, 1), n = luaL_optint(L, 2, m);

				return NewRet<M>(L, M::Identity(m, n));	// m[, n], id
			}
		}, {
			"LinSpaced", [](lua_State * L)
			{
				return NewRet<M>(L, LinSpacing<M, Eigen::Dynamic, 1>::Make(L, LuaXS::Int(L, 1)));
			}
		}, {
			"LinSpacedRow", [](lua_State * L)
			{
				return NewRet<M>(L, LinSpacing<M, 1, Eigen::Dynamic>::Make(L, LuaXS::Int(L, 1)));
			}
		}, 
	#ifdef WANT_MAP
		{
			"MatrixFromMemory", [](lua_State * L)
			{
				BlobXS::State state{L, 1};
				
				int m = luaL_checkint(L, 2), n = luaL_optint(L, 3, m);
				auto memory = state.PointToDataIfBound(L, 0, 0, n, m, 0, sizeof(M::Scalar)); // TODO: might be resizable...

				//
				if (memory)
				{
					Eigen::Map<M> map(reinterpret_cast<M::Scalar *>(memory), m, n);

					New<Eigen::Map<M>>(L, std::move(map));	// memory, m[, n], map
					GetTypeData<Eigen::Map<M>>(L)->RefAt(L, "map_bytes", 1);
				}

				//
				else
				{
					luaL_argcheck(L, !state.Bound() && lua_objlen(L, 1) >= m * n * sizeof(M::Scalar), 1, "Not enough memory for requested dimensions");

					const char * str = luaL_checkstring(L, 1);

					Eigen::Map<const M> map(reinterpret_cast<const M::Scalar *>(str), m, n);

					New<Eigen::Map<const M>>(L, std::move(map));// memory, m[, n], map
					GetTypeData<Eigen::Map<const M>>(L)->RefAt(L, "map_bytes", 1);
				}

				return 1;
			}
		}, 
	#endif
	#ifdef WANT_MAP_WITH_CUSTOM_STRIDE
		/*{
			"MatrixFromMemoryWithInnerStride", [](lua_State * L)
			{
				lua_settop(L, 4);	// memory, m[, n], stride

				ByteReader memory{L, 1}; // see MatrixFromMemory

				if (!memory.mBytes) lua_error(L);

				int m = luaL_checkint(L, 2), arg3 = luaL_checkint(L, 3), n, stride;

				if (!lua_isnil(L, 4))
				{
					stride = luaL_checkint(L, 4);
					n = arg3;
				}

				else
				{
					stride = arg3;
					n = m;
				}

				Eigen::Map<M, 0, Eigen::InnerStride<>> map(static_cast<const M::Scalar *>(memory.mBytes), m, n, stride);

				New<decltype(map)>(L, std::move(map));	// memory, m[, n], stride, map
				GetTypeData<decltype(map)>(L)->RefAt(L, "bytes", 1);

				return 1;
			}
		}, {
			"MatrixFromMemoryWithOuterStride", [](lua_State * L)
			{
				lua_settop(L, 4);	// memory, m[, n], stride

				ByteReader memory{L, 1}; // see MatrixFromMemory

				if (!memory.mBytes) lua_error(L);

				int m = luaL_checkint(L, 2), arg3 = luaL_checkint(L, 3), n, stride;

				if (!lua_isnil(L, 4))
				{
					stride = luaL_checkint(L, 4);
					n = arg3;
				}

				else
				{
					stride = arg3;
					n = m;
				}

				Eigen::Map<M, 0, Eigen::OuterStride<>> map(static_cast<const M::Scalar *>(memory.mBytes), m, n, stride);

				New<decltype(map)>(L, std::move(map));	// memory, m[, n], stride, map					
				GetTypeData<decltype(map)>(L)->RefAt(L, "bytes", 1);

				return 1;
			}
		},*/
	#endif
		{
			"NewMatrix", [](lua_State * L)
			{
				if (!lua_isnoneornil(L, 1))
				{
					int m = LuaXS::Int(L, 1), n = luaL_optint(L, 2, m);

					New<M>(L, m, n);// m[, n], M
				}

				else New<M>(L);	// M

				return 1;
			}
		}, {
			"Ones", [](lua_State * L)
			{
				int m = LuaXS::Int(L, 1), n = luaL_optint(L, 2, m);

				return NewRet<M>(L, M::Ones(m, n));// m[, n], m1
			}
		}, {
			"Random", [](lua_State * L)
			{
				int m = LuaXS::Int(L, 1), n = luaL_optint(L, 2, m);

				return NewRet<M>(L, M::Random(m, n));	// m[, n], r
			}
		}, {
			"RandomPermutation", [](lua_State * L)
			{
				Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm{LuaXS::Int(L, 1)};

				perm.setIdentity();

				std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());

				return NewRet<M>(L, perm);// size, perm
			}
		}, {
			"RowVector", [](lua_State * L)
			{
				New<M>(L, 1, LuaXS::Int(L, 1));	// size, rv

				return 1;
			}
		}, {
			"Vector", [](lua_State * L)
			{
				New<M>(L, LuaXS::Int(L, 1), 1);	// size, v

				return 1;
			}
		}, {
			"Zero", [](lua_State * L)
			{
				int m = LuaXS::Int(L, 1), n = luaL_optint(L, 2, m);

				return NewRet<M>(L, M::Zero(m, n));// m[, n], m0
			}
		},
		{ nullptr, nullptr }
	};

	luaL_register(L, nullptr, funcs);

	AddUmeyama<M> au{L};

	#ifdef EIGEN_PLUGIN_BASIC
		lua_setfield(L, -2, GetTypeData<M>(L, eCreateIfMissing)->GetName());// eigen = { ..., name = funcs }
	#endif
}

// Build-specific library name.
#define EIGEN_LIB_NAME(name) #name

// Module entry point.
CORONA_EXPORT int PLUGIN_NAME (lua_State * L)
{
	tls_LuaState = L;

	// If this is the core or all-in-one module, create a cache.
	#if defined(EIGEN_CORE) || defined(EIGEN_PLUGIN_BASIC)
		lua_getglobal(L, "require");// ... require
		lua_pushliteral(L, "cachestack");	// ..., require, "plugin.cachestack"

		if (lua_pcall(L, 1, 1, 0) != 0) lua_error(L);	// ..., cachestack / err

		lua_getfield(L, -1, "NewCacheStack");	// ..., cachestack, NewCacheStack
		lua_call(L, 0, 2);	// ..., cachestack, NewType, WithContext
	#endif

	// Register the module with Corona.
	// TODO: make robust for other threads
	luaL_Reg no_funcs[] = { { nullptr, nullptr } };

	CoronaLibraryNew(L, EIGEN_LIB_NAME(PLUGIN_SUFFIX), "com.xibalbastudios", 1, 0, no_funcs, nullptr);	// ...[, cachestack, NewType, WithContext], M

	// If this is the core or all-in-one module, add the WithCache() routine just supplied.
	// The boolean matrix type is then created; as a side effect, it adds the cache binding
	// logic to the registry, using its type data as key, q.v. GetTypeKey().
	#if defined(EIGEN_CORE) || defined(EIGEN_PLUGIN_BASIC)
		lua_insert(L, -2);	// ..., cachestack, NewType, M, WithContext
		lua_setfield(L, -2, "WithCache");	// ..., cachestack, NewType, M = { WithCache = WithContext }
		lua_insert(L, -3);	// ..., M, cachestack, NewType

		auto td = GetTypeData<BoolMatrix>(L, eCreateIfMissing);	// ..., M, cachestack; registry = { ..., [bool_matrix_type_data] = NewType }

		lua_pushboolean(L, 1);	// ..., M, cachestack, true
		lua_rawset(L, LUA_REGISTRYINDEX);	// ..., M; registry = { ..., NewType, [cachestack] = true }
	#endif

#ifdef WANT_INT
	AddType<Eigen::MatrixXi>(L);
#endif

#ifdef WANT_FLOAT
	AddType<Eigen::MatrixXf>(L);
#endif

#ifdef WANT_DOUBLE
	AddType<Eigen::MatrixXd>(L);
#endif

#ifdef WANT_CFLOAT
	AddType<Eigen::MatrixXcf>(L);
#endif

#ifdef WANT_CDOUBLE
	AddType<Eigen::MatrixXcd>(L);
#endif

	return 1;
}