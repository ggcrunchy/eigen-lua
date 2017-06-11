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
#include "utils/LuaEx.h"
#include "types.h"
#include "utils.h"
#include "macros.h"

// Methods assigned when the underlying type is real.
template<typename T, typename R, bool = !Eigen::NumTraits<T::Scalar>::IsComplex> struct RealOps : InstanceGetters<T, R> {
	// Version of methods when we have a matrix or basic map.
	template<bool = HasNormalStride<T>::value> void AddIfNormalStride (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_GET_MATRIX_METHOD(asPermutation)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	// Version of methods with maps having non-trivial stride.
	template<> void AddIfNormalStride<false> (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"asPermutation", [](lua_State * L)
				{
					R temp = *GetT(L);

					return NewRet<R>(L, temp.asPermutation());
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}
		
	RealOps (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_ARRAY_METHOD(ceil)
			}, {
				EIGEN_REL_OP_METHOD(cwiseGreaterThan, >)
			}, {
				EIGEN_REL_OP_METHOD(cwiseGreaterThanOrEqual, >=)
			}, {
				EIGEN_REL_OP_METHOD(cwiseLessThan, <)
			}, {
				EIGEN_REL_OP_METHOD(cwiseLessThanOrEqual, <=)
			}, {
				EIGEN_MATRIX_GET_MATRIX_SECOND_IS_MATRIX_OR_SCALAR_METHOD(cwiseMax)
			}, {
				EIGEN_MATRIX_GET_MATRIX_SECOND_IS_MATRIX_OR_SCALAR_METHOD(cwiseMin)
			}, {
				EIGEN_ARRAY_METHOD(floor)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(maxCoeff)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(minCoeff)
			}, {
				EIGEN_ARRAY_METHOD(round)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);

		AddIfNormalStride(L);
	}
};

// No-op for complex types.
template<typename T, typename R> struct RealOps<T, R, false> {
	RealOps (lua_State *) {}
};