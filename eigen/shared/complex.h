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
#include "ByteReader.h"
#include <complex>

//
template<typename T> static std::complex<T> Complex (lua_State * L, int arg)
{
	switch (lua_type(L, arg))
	{
	case LUA_TNUMBER:
		return std::complex<T>{static_cast<T>(lua_tonumber(L, arg)), static_cast<T>(0)};
	case LUA_TTABLE:
		arg = CoronaLuaNormalize(L, arg);

		lua_rawgeti(L, arg, 1);	// ..., complex, ..., r
		lua_rawgeti(L, arg, 2);	// ..., complex, ..., r, i

		{
			double r = luaL_optnumber(L, -2, 0.0), i = luaL_optnumber(L, -1, 0.0);

			lua_pop(L, 2);	// ..., complex, ...

			return std::complex<T>{static_cast<T>(r), static_cast<T>(i)};
		}
	default:
		ByteReader reader{L, arg};

		luaL_argcheck(L, reader.mBytes && reader.mCount >= sizeof(std::complex<T>), arg, "Invalid complex number");

		std::complex<T> comp;

		memcpy(&comp, reader.mBytes, sizeof(std::complex<T>));

		return comp;
	}
}

template<> inline void LuaXS::PushArg<std::complex<double>> (lua_State * L, std::complex<double> c)
{
	lua_createtable(L, 2, 0);	// ..., c
	lua_pushnumber(L, c.real());// ..., c, c.r
	lua_rawseti(L, -2, 1);	// ..., c = { r }
	lua_pushnumber(L, c.imag());// ..., c, c.i
	lua_rawseti(L, -2, 2);	// ..., c = { r, i }
}

template<> inline void LuaXS::PushArg<std::complex<float>> (lua_State * L, std::complex<float> c)
{
	LuaXS::PushArg(L, std::complex<double>{c.real(), c.imag()});
}