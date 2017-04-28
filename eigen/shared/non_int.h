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
template<typename T, typename R> struct NonIntMethods {
	//
	ADD_INSTANCE_GETTERS()

	//
	NonIntMethods (lua_State * L)
	{
		Add(L);
	}

	//
	template<bool = std::is_integral<T::Scalar>::value> void Add (lua_State * L) {}

	template<> void Add<false> (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_PUSH_VALUE_METHOD(determinant)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(inverse)
			}, {
				"ldlt_solve", [](lua_State * L)
				{
					return NewMoveRet<R>(L, GetT(L)->ldlt().solve(*GetR(L, 2)));
				}
			}, {
				"llt_solve", [](lua_State * L)
				{
					return NewMoveRet<R>(L, GetT(L)->llt().solve(*GetR(L, 2)));
				}
			}, {
				"lu_solve", [](lua_State * L)
				{
					return NewMoveRet<R>(L, GetT(L)->lu().solve(*GetR(L, 2)));
				}
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(operatorNorm)
			}, {
				"qr_solve", [](lua_State * L)
				{
					return NewMoveRet<R>(L, GetT(L)->householderQr().solve(*GetR(L, 2)));
				}
			},{
				"svd_solve", [](lua_State * L)
				{
					return NewMoveRet<R>(L, Eigen::JacobiSVD<R>{*GetT(L)}.solve(*GetR(L, 2)));
				}
			}, 
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}
};