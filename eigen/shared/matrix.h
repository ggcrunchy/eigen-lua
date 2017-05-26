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
#include <type_traits>

// Common matrix methods attachment body.
template<typename T, typename R> struct AttachMatrixMethods {
	static_assert(std::is_same<typename T::Scalar, typename R::Scalar>::value, "Types have mixed scalars");

	AttachMatrixMethods (lua_State * L)
	{
		CommonMethods<T, R>{L};
		ComplexDependentMethods<T, R>{L};
		MapDependentMethods<T, R>{L};
		NonIntMethods<T, R>{L};

		// Hook up routines to push a matrix, e.g. from another shared library, and with similar
		// reasoning to use such matrices in BoolMatrix::select().
		auto td = GetTypeData<T>(L);

		lua_pushcfunction(L, [](lua_State * L) {
			return NewRet<R>(L, *LuaXS::UD<T>(L, 1));
		});	// meta, push

		td->mPushRef = lua_ref(L, 1);	// meta; registry = { ...[, select], ref = push }

		// Selection simply resolves to a matrix type, so reuse the logic if available.
		if (!std::is_same<T, R>::value)
		{
			auto rtd = GetTypeData<R>(L);

			if (rtd) td->mSelectRef = rtd->mSelectRef;
		}

		// Failing the above, supply a new function.
		if (td->mSelectRef != LUA_NOREF)
		{
			lua_pushcfunction(L, [](lua_State * L) {
				auto bm = AttachMethods<BoolMatrix>::GetT(L);

				return NewRet<R>(L, WithMatrixScalarCombination<R>(L, [&bm](const R & m1, const R & m2) {
					return bm->select(m1, m2);
				}, [&bm](const R & m, const R::Scalar & s) {
					return bm->select(m, s);
				}, [&bm](const R::Scalar & s, const R & m) {
					return bm->select(s, m);
				}, 2, 3));
			});	// meta, push, select

			td->mSelectRef = lua_ref(L, 1);	// meta, push; registry = { ..., ref = select }
		}
	}
};

/*****************
* Matrix methods *
*****************/
template<typename T, int Rows, int Cols, int Options, int MaxRows, int MaxCols, typename R> struct AttachMethods<Eigen::Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>, R> : AttachMatrixMethods<Eigen::Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>, R> {
	AttachMethods (lua_State * L) : AttachMatrixMethods<Eigen::Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>, R>(L)
	{
	}
};

/************************
* Mapped matrix methods *
************************/
template<typename T, int Rows, int Cols, int Options, int MaxRows, int MaxCols, int MapOptions, typename S, typename R> struct AttachMethods<Eigen::Map<Eigen::Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>, MapOptions, S>, R> : AttachMatrixMethods<Eigen::Map<Eigen::Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>, MapOptions, S>, R> {
	AttachMethods (lua_State * L) : AttachMatrixMethods<Eigen::Map<Eigen::Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>, MapOptions, S>, R>(L)
	{
	}
};

// TODO: transpose, etc.