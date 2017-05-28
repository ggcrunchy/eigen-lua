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
#include "arith_ops.h"
#include "write_ops.h"
#include "xpr_ops.h"
#include <Eigen/Eigen>

//
template<typename T, typename R> struct AttachBlockMethods {
	ADD_INSTANCE_GETTERS()

	//
	AttachBlockMethods (lua_State * L)
	{
		RingBufferOfMethodThunksProperty<T, R>(L);

		luaL_Reg methods[] = {
			{
				"asMatrix", AsMatrix<T, R>
			}, {
				"__gc", LuaXS::TypedGC<T>
			}, {
				"__tostring", [](lua_State * L)
				{
					AsMatrix<T, R>(L);	// xpr, mat

					return Print(L, GetR(L, 2));
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);

		ArithOps<T, R> ao{L};
		WriteOps<T, R> wo{L};
		XprOps<T, R> xo{L}; // does special handling to avoid either xprs of xprs or xprs of a common temporary
	}
};

/****************
* Block methods *
****************/
template<typename U, int Rows, int Cols, bool InnerPanel, typename R> struct AttachMethods<Eigen::Block<U, Rows, Cols, InnerPanel>, R> : AttachBlockMethods<Eigen::Block<U, Rows, Cols, InnerPanel>, R> {
	AttachMethods (lua_State * L) : AttachBlockMethods<Eigen::Block<U, Rows, Cols, InnerPanel>, R>(L)
	{
	}
};

template<typename U, int R, int C, bool InnerPanel> struct AuxTypeName<Eigen::Block<U, R, C, InnerPanel>> {
	AuxTypeName (luaL_Buffer * B, lua_State * L)
	{
		luaL_addstring(B, "Block<");

		AuxTypeName<U>(B, L);

		luaL_addstring(B, ", ");

		AddDynamicOrN(B, L, R);

		luaL_addstring(B, ", ");

		AddDynamicOrN(B, L, C);

		if (InnerPanel) luaL_addstring(B, ", true");

		luaL_addstring(B, ">");
	}
};

/*******************
* Diagonal methods *
*******************/
template<typename U, int I, typename R> struct AttachMethods<Eigen::Diagonal<U, I>, R> : AttachBlockMethods<Eigen::Diagonal<U, I>, R> {
	AttachMethods (lua_State * L) : AttachBlockMethods<Eigen::Diagonal<U, I>, R>(L)
	{
	}
};

template<typename U, int I> struct AuxTypeName<Eigen::Diagonal<U, I>> {
	AuxTypeName (luaL_Buffer * B, lua_State * L)
	{
		luaL_addstring(B, "Diagonal<");

		AuxTypeName<U>(B, L);

		luaL_addstring(B, ">");
	}
};

/**********************
* VectorBlock methods *
**********************/
template<typename U, typename R> struct AttachMethods<Eigen::VectorBlock<U>, R> : AttachBlockMethods<Eigen::VectorBlock<U>, R> {
	AttachMethods (lua_State * L) : AttachBlockMethods<Eigen::VectorBlock<U>, R>(L)
	{
	}
};

template<typename U> struct AuxTypeName<Eigen::VectorBlock<U>> {
	AuxTypeName (luaL_Buffer * B, lua_State * L)
	{
		luaL_addstring(B, "VectorBlock<");

		AuxTypeName<U>(B, L);

		luaL_addstring(B, ">");
	}
};