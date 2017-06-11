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

#include "types.h"
#include "utils.h"

//
template<typename T, typename R> struct ArithOps : InstanceGetters<T, R> {
	//
	template<bool = IsXpr<T>::value> struct MatrixOps {
		static int Add (lua_State * L)
		{
			if (HasType<T>(L, 1) && HasType<T>(L, 2)) return NewRet<R>(L, *LuaXS::UD<T>(L, 1) + *LuaXS::UD<T>(L, 2));
			else return MatrixOps<true>::Add(L);
		}

		static int Mul (lua_State * L)
		{
			bool b1 = HasType<T>(L, 1), b2 = HasType<T>(L, 2);

			if (b1 && b2) return NewRet<R>(L, *LuaXS::UD<T>(L, 1) * *LuaXS::UD<T>(L, 2));

			else if (b1)
			{
				T & m = *LuaXS::UD<T>(L, 1);

				if (HasType<Eigen::Block<R>>(L, 2)) return NewRet<R>(L, m * *LuaXS::UD<Eigen::Block<R>>(L, 2));
				else if (HasType<Eigen::Transpose<R>>(L, 2)) return NewRet<R>(L, m * *LuaXS::UD<Eigen::Transpose<R>>(L, 2));
			}

			else
			{
				T & m = *LuaXS::UD<T>(L, 2);

				if (HasType<Eigen::Block<R>>(L, 1)) return NewRet<R>(L, *LuaXS::UD<Eigen::Block<R>>(L, 1) * m);
				else if (HasType<Eigen::Transpose<R>>(L, 1)) return NewRet<R>(L, *LuaXS::UD<Eigen::Transpose<R>>(L, 1) * m);
			}

			return MatrixOps<true>::Mul(L);
		}

		static int Pow (lua_State * L)
		{
			if (HasType<T>(L, 1) && HasType<T>(L, 2)) return NewRet<R>(L, LuaXS::UD<T>(L, 1)->array().pow(LuaXS::UD<T>(L, 2)->array()));
			else return MatrixOps<true>::Pow(L);
		}

		static int Sub (lua_State * L)
		{
			if (HasType<T>(L, 1) && HasType<T>(L, 2)) return NewRet<R>(L, *LuaXS::UD<T>(L, 1) - *LuaXS::UD<T>(L, 2));
			else return MatrixOps<true>::Sub(L);
		}
	};

	template<> struct MatrixOps<true> {
		static int Add (lua_State * L)
		{
			TwoMatrices<R> ms{L};

			return NewRet<R>(L, *ms.mMat1 + *ms.mMat2);
		}

		static int Mul (lua_State * L)
		{
			return NewRet<R>(L, WithMatrixScalarCombination<R>(L, [](const R & m1, const R & m2) {
				return m1 * m2;
			}, [](const R & m, const T::Scalar & s) {
				return m * s;
			}, [](const T::Scalar & s, const R & m) {
				return s * m;
			}));
		}

		static int Pow (lua_State * L)
		{
			return NewRet<R>(L, WithMatrixScalarCombination<R>(L, [](const R & m1, const R & m2) {
				return m1.array().pow(m2.array());
			}, [](const R & m, const T::Scalar & s) {
				return m.array().pow(s);
			}, [](const T::Scalar & s, const R & m) {
				return Eigen::pow(s, m.array());
			}));
		}

		static int Sub (lua_State * L)
		{
			TwoMatrices<R> ms{L};

			return NewRet<R>(L, *ms.mMat1 - *ms.mMat2);
		}
	};

	ArithOps (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"__add", MatrixOps<>::Add
			}, {
				"__div", [](lua_State * L)
				{
					return NewRet<R>(L, *GetT(L) / AsScalar<R>(L, 2));
				}
			}, {
				"__mul", MatrixOps<>::Mul
			}, {
				"__pow", MatrixOps<>::Pow
			}, {
				"__sub", MatrixOps<>::Sub
			}, {
				"__unm", [](lua_State * L)
				{
					return NewRet<R>(L, -*GetT(L));
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}
};

template<typename T> struct ArithOps<T, BoolMatrix> {
	ArithOps (lua_State *) {}
};