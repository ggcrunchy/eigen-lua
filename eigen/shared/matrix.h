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

#include <Eigen/Eigen>
#include "utils/LuaEx.h"
#include "config.h"
#include "complex.h"
#include "types.h"
#include "utils.h"
#include "bool_matrix.h"
#include "common.h"
#include "complex_dependent.h"
#include "map_dependent.h"
#include "non_int.h"

template<typename T, typename R> struct AttachMethods {
	static_assert(std::is_same<typename T::Scalar, typename R::Scalar>::value, "Types have mixed scalars");

	AttachMethods (lua_State * L)
	{
		CommonMethods<T, R>{L};
		ComplexDependentMethods<T, R>{L};
		MapDependentMethods<T, R>{L};
		NonIntMethods<T, R>{L};

		//
		auto td = GetTypeData<T>(L);

		lua_pushcfunction(L, [](lua_State * L)
		{
			return NewMoveRet<R>(L, *LuaXS::UD<T>(L, 1));
		});	// meta, push
		lua_pushcfunction(L, [](lua_State * L)
		{
			auto bm = AttachMethods<BoolMatrix>::GetT(L);

			return NewMoveRet<R>(L, WithMatrixOrScalar<T, R>(L, [&bm](const T & m1, const R & m2){
				return bm->select(m1, m2);
			}, [&bm](const T & m, const T::Scalar & s){
				return bm->select(m, s);
			}, [&bm](const T::Scalar & s, const R & m){
				return bm->select(s, m);
			}, 2, 3));
		});	// meta, push, select

		td->mSelectRef = lua_ref(L, 1);	// meta, push; registry = { ..., ref = select }
		td->mPushRef = lua_ref(L, 1);	// meta; registry = { ..., select, ref = push }
	}
};